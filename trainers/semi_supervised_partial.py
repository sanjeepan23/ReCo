import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
from tqdm import tqdm
import numpy as np
import sys
from torchvision import models

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.cityscapes_data_loader import CityscapesLoader
from data.pascal_data_loader import PascalVOCLoader
from network.mean_ts import TeacherModel
from network.reconet import ReCoNet
from trainers.train_utils import adjust_learning_rate, calculate_unsupervised_loss, compute_iou, reco_loss_func
from trainers.wandb_utils import init_wandb, log_training_metrics, log_validation_metrics, watch_model, update_summary, finish
import utils.img_processing as img_processing
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised semantic segmentation')
    
    parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'cityscapes'],
                        help='Dataset to use (pascal or cityscapes)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--label-ratio', type=str, default="p0",
                        help='partial labels p0/p1/p5/p25')
    parser.add_argument('--model', type=str, default='reconet', 
                        choices=['fcn_resnet50', 'deeplabv3', 'deeplabv3_original', 'reconet'],
                        help='Model architecture')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for SGD optimizer')
    parser.add_argument('--power', type=float, default=0.9,
                        help='Power for polynomial decay of learning rate')
    parser.add_argument('--iterations', type=int, default=40000,
                        help='Total number of training iterations')
    parser.add_argument('--max-epochs', type=int, default=500,
                        help='Maximum number of epochs')
    
    # Model arguments
    parser.add_argument('--ema-decay', type=float, default=0.99,
                        help='EMA decay rate for teacher model')
    parser.add_argument('--conf-thresh', type=float, default=0.95,
                        help='Confidence threshold for pseudo-labels')
    parser.add_argument('--unsup-weight', type=float, default=1.0,
                        help='Weight for unsupervised loss')
    parser.add_argument('--mixing-strategy', type=str, default='classmix',
                        help='The mixing strategies to use')
    
    # ReCo loss arguments
    parser.add_argument('--reco', action='store_true', default=False,
                        help='Enable ReCo loss')
    parser.add_argument('--reco-weight', type=float, default=1.0,
                        help='Weight for ReCo loss')
    parser.add_argument('--reco-temp', type=float, default=0.5,
                        help='Temperature for ReCo loss')
    parser.add_argument('--reco-num-queries', type=int, default=256,
                        help='Number of queries for ReCo')
    parser.add_argument('--reco-num-negatives', type=int, default=256,
                        help='Number of negative keys for ReCo')
    parser.add_argument('--reco-threshold', type=float, default=0.97,
                        help='Confidence threshold for hard query sampling in ReCo')
    
    # Logging and checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--val-interval', type=int, default=2000,
                        help='Validation interval (iterations)')
    parser.add_argument('--save-interval', type=int, default=10000,
                        help='Checkpoint saving interval (iterations)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--disable-saving', action='store_true', help='Disable checkpoint saving')
    
    return parser.parse_args()

def create_model(args):
    num_classes = 21 if args.dataset == 'pascal' else 19
    
    if args.model == 'fcn_resnet50':
        model = fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif args.model == 'deeplabv3_original':
        model = deeplabv3_resnet101(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif args.model == 'reconet':
        backbone = models._utils.IntermediateLayerGetter(
            models.resnet101(pretrained=True),
            {'layer4': 'out', 'layer1': 'low_level'}
        )
        model = ReCoNet(backbone, 2048, 256, num_classes, [12, 24, 36])
    
    return model

def main():
    """Semi-Supervised Learning pipeline with ReCo, Classmix support for partial label dataset"""
    args = parse_args()

    save_stuff = not args.disable_saving
    print(f'Checkpoint saving enabled: {save_stuff}')
    
    device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    wandb_config = vars(args)
    init_wandb(
        project_name="reco_v2_semi_sup_class_mix", 
        config=wandb_config,
        run_name=f"{args.dataset}_{args.model}_labeled_{args.label_ratio}_{args.seed}_semi_supervised_partial"
    )
    
    if args.dataset == 'pascal':
        loader = PascalVOCLoader(
            data_path=args.data_path,
            label_ratio=args.label_ratio,
            seed=args.seed
        )
        num_classes = 21  # 20 classes + background
        
    elif args.dataset == 'cityscapes':
        loader = CityscapesLoader(
            data_path=args.data_path,
            label_ratio=args.label_ratio,
            seed=args.seed
        )
        num_classes = 19
    
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    train_labeled_loader, train_unlabeled_loader, val_loader = loader.create_loaders(use_unlabeled=True)
    
    student_model = create_model(args)
    student_model = student_model.to(device)
    
    teacher_model = TeacherModel(student_model, ema_decay=args.ema_decay).to(device)
    
    watch_model(student_model)
    
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_iou = 0
    total_iterations = 0
    
    pbar = tqdm(total=args.iterations)
    
    for epoch in range(args.max_epochs):
        print(f"Epoch {epoch+1}/{args.max_epochs}")
        
        labeled_iter = iter(train_labeled_loader)
        unlabeled_iter = iter(train_unlabeled_loader)
        
        for labeled_batch in train_labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            labeled_img = labeled_batch[0].float().to(device)
            labeled_mask = labeled_batch[1].long().to(device)
            unlabeled_img = unlabeled_batch[0].float().to(device)

            _pseudo_labels, conf_mask, confidence = teacher_model.generate_pseudo_labels(
                unlabeled_img, confidence_threshold=args.conf_thresh
            )
            
            unlabeled_img_aug, pseudo_labels = img_processing.augment_unlabeled_batch(
                train_unlabeled_loader.dataset, unlabeled_img, _pseudo_labels, args.mixing_strategy
            )

            current_lr = adjust_learning_rate(
                optimizer, args.lr, total_iterations, args.iterations, args.power
            )
            
            student_model.train()
            
            student_labeled_output = student_model(labeled_img)
            student_labeled_logits = student_labeled_output['out']
            
            supervised_loss = criterion(student_labeled_logits, labeled_mask)
            
            student_unlabeled_output = student_model(unlabeled_img_aug)
            student_unlabeled_logits = student_unlabeled_output['out']
            
            unsupervised_loss = calculate_unsupervised_loss(
                student_unlabeled_logits, pseudo_labels, conf_mask
            )
            
            conf_ratio = conf_mask.float().mean().item()
            
            eta = conf_ratio if conf_ratio > 0 else 0.1
            total_loss = supervised_loss +  eta * unsupervised_loss
            
            reco_loss_val = None
            if args.reco:
                labeled_rep = student_labeled_output['_reco']     
                unlabeled_rep = student_unlabeled_output['_reco'] 
                labeled_probs = F.softmax(student_labeled_output['_out'], dim=1)
                unlabeled_probs = F.softmax(student_unlabeled_output["_out"], dim=1)
                
                labeled_valid_mask = (labeled_mask != -1)
                if labeled_mask.dim() == 3:
                    labeled_mask = labeled_mask.unsqueeze(1)
                
                B = unlabeled_img.shape[0]
                if conf_mask.dim() == 1:
                    conf_mask = conf_mask.reshape(B, 1, *unlabeled_rep.shape[2:])
                elif conf_mask.dim() == 2:
                    conf_mask = conf_mask.view(B, 1, *unlabeled_rep.shape[2:])
                    

                
                if pseudo_labels.dim() == 1:
                    pseudo_labels = pseudo_labels.reshape(B, *unlabeled_rep.shape[2:])
                elif pseudo_labels.dim() == 2 and pseudo_labels.shape[0] == B:
                    pseudo_labels = pseudo_labels.reshape(B, *unlabeled_rep.shape[2:])
                
                if labeled_mask.shape[2:] != labeled_rep.shape[2:]:
                    labeled_mask = F.interpolate(labeled_mask.float(), size=labeled_rep.shape[2:], mode='nearest').long()
                    labeled_valid_mask = F.interpolate(labeled_valid_mask.unsqueeze(1).float(), size=labeled_rep.shape[2:], mode='nearest').bool().squeeze(1)
                    
                if conf_mask.shape[1:] != unlabeled_rep.shape[2:]:
                    conf_mask = conf_mask.unsqueeze(1) 
                    conf_mask = F.interpolate(conf_mask.float(), size=unlabeled_rep.shape[2:], mode='nearest').bool()
                
                if pseudo_labels.shape[1:] != unlabeled_rep.shape[2:]:
                    pseudo_labels = F.interpolate(pseudo_labels.unsqueeze(1).float(), size=unlabeled_rep.shape[2:], mode='nearest').long().squeeze(1)
                    conf_mask = F.interpolate(conf_mask.float(), size=unlabeled_rep.shape[2:], mode='nearest').bool()
                    
                if labeled_probs.shape[2:] != labeled_rep.shape[2:]:
                    labeled_probs = F.interpolate(labeled_probs, size=labeled_rep.shape[2:], mode='bilinear', align_corners=False)
                
                if unlabeled_probs.shape[2:] != unlabeled_rep.shape[2:]:
                    unlabeled_probs = F.interpolate(unlabeled_probs, size=unlabeled_rep.shape[2:], mode='bilinear', align_corners=False)
                
                combined_rep = torch.cat([labeled_rep, unlabeled_rep], dim=0)
                combined_labels = torch.cat([labeled_mask.squeeze(1), pseudo_labels], dim=0)
                combined_mask = torch.cat([labeled_valid_mask, conf_mask.squeeze(1)], dim=0)
                combined_probs = torch.cat([labeled_probs, unlabeled_probs], dim=0)
                
                reco_loss = reco_loss_func(
                    rep=combined_rep,
                    label=combined_labels,
                    mask=combined_mask,
                    prob=combined_probs,
                    strong_threshold=args.reco_threshold,
                    temp=args.reco_temp,
                    num_queries=args.reco_num_queries,
                    num_negatives=args.reco_num_negatives
                )
                
                reco_loss_val = reco_loss.item()
                total_loss = total_loss + args.reco_weight * reco_loss
                
                if total_iterations % 100 == 0:
                    print(f"ReCo loss: {reco_loss_val:.4f}")
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            teacher_model.update_weights(student_model)
            
            if total_iterations % 10 == 0:
                log_training_metrics(
                    loss=total_loss.item(),
                    current_lr=current_lr,
                    epoch=epoch + 1,
                    iteration=total_iterations,
                    reco_loss=reco_loss_val
                )
                
                metrics = {
                    "train/supervised_loss": supervised_loss.item(),
                    "train/unsupervised_loss": unsupervised_loss.item(),
                    "train/reco_loss_val": reco_loss_val if reco_loss_val else 0,
                    "train/confidence_ratio": conf_ratio
                }
                wandb.log(metrics, step=total_iterations)
            
            total_iterations += 1
            pbar.update(1)
            
            desc = f"Epoch: {epoch+1}/{args.max_epochs}, Iter: {total_iterations}/{args.iterations}, " \
                  f"Loss: {total_loss.item():.4f}, Sup: {supervised_loss.item():.4f}, " \
                  f"Unsup: {unsupervised_loss.item():.4f}, LR: {current_lr:.6f}, Conf: {conf_ratio:.2f}"
            
            if args.reco and reco_loss_val is not None:
                desc += f", ReCo: {reco_loss_val:.4f}"
                
            pbar.set_description(desc)
            
            if total_iterations % args.val_interval == 0:
                student_model.eval()
                val_running_loss = 0
                val_iou = 0
                num_batches = 0
                
                with torch.no_grad():
                    for idx, img_mask in enumerate(val_loader):
                        img = img_mask[0].float().to(device)
                        mask = img_mask[1].long().to(device)
                        
                        output = student_model(img)
                        y_pred = output['out']
                            
                        loss = criterion(y_pred, mask)
                        
                        batch_iou = compute_iou(y_pred, mask, num_classes)
                        val_iou += batch_iou
                        val_running_loss += loss.item()
                        num_batches += 1
                    
                    val_loss = val_running_loss / num_batches
                    mean_iou = val_iou / num_batches
                
                print("-"*50)
                print(f"Epoch: {epoch+1}/{args.max_epochs}, Iteration: {total_iterations}/{args.iterations}")
                print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}")
                print("-"*50)
                
                log_validation_metrics(val_loss, mean_iou, total_iterations)
                
                if save_stuff and mean_iou > best_iou:
                    best_iou = mean_iou
                    torch.save(
                        student_model.state_dict(), 
                        os.path.join(args.checkpoint_dir, "best_student_model.pth")
                    )
                    
                    update_summary(best_iou, total_iterations)
            
            if save_stuff and total_iterations % args.save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'iteration': total_iterations,
                    'student_model_state_dict': student_model.state_dict(),
                    'teacher_model_state_dict': teacher_model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, os.path.join(args.checkpoint_dir, f"checkpoint_iter_{total_iterations}.pth"))
            
            if total_iterations >= args.iterations:
                break
        
        if total_iterations >= args.iterations:
            print(f"Reached {args.iterations} iterations. Stopping training.")
            break
    
    pbar.close()
    print(f"Training completed! Best validation IoU: {best_iou:.4f}")

    if save_stuff:
        torch.save(
            student_model.state_dict(), 
            os.path.join(args.checkpoint_dir, "final_student_model.pth")
        )
    
    finish()

if __name__ == '__main__':
    main()