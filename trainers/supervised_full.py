import os
import argparse
import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
import sys
from torchvision import models
from pathlib import Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from data.cityscapes_data_loader import CityscapesDataset, CityscapesLoader
from data.pascal_data_loader import PascalVOCDataset, PascalVOCLoader
from network import ReCoNet, DeepLabv3Plus
from trainers.train_utils import adjust_learning_rate, calculate_unsupervised_loss, compute_iou, reco_loss_func
from trainers.train_utils import IOU
from trainers.wandb_utils import init_wandb, log_training_metrics, log_validation_metrics, watch_model, update_summary, finish

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training Script')
    parser.add_argument('--dataset', type=str, required=True, choices=['pascal', 'cityscapes'],
                        help='Dataset to train on (pascal or cityscapes)')
    parser.add_argument('--data-path', type=str, required=True, 
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='reconet', 
                        choices=['fcn_resnet50', 'deeplabv3', 'deeplabv3_original', 'reconet'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override default batch size')
    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of epochs')
    parser.add_argument('--total-iterations', type=int, default=40000,
                        help='Total training iterations')
    parser.add_argument('--val-interval', type=int, default=2000,
                        help='Validation interval (iterations)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-labeled', type=int, default=None,
                        help='Number of labeled samples (for semi-supervised learning)')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--disable-saving', action='store_true', help='Disable checkpoint saving')
    
    # ReCo arguments
    parser.add_argument('--reco', type=bool, default=False, 
                        help='Enable ReCo loss')
    parser.add_argument('--reco_weight', type=float, default=1.0, 
                        help='Weight for ReCo loss')
    parser.add_argument('--reco_temp', type=float, default=0.5, 
                        help='Temperature for ReCo loss')
    parser.add_argument('--reco_num_queries', type=int, default=256, 
                        help='Number of queries for ReCo')
    parser.add_argument('--reco_num_negatives', type=int, default=256, 
                        help='Number of negative keys for ReCo')
    parser.add_argument('--reco_threshold', type=float, default=0.97, 
                        help='Confidence threshold for hard query sampling in ReCo')
    
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
            models.resnet101(pretrained=True, replace_stride_with_dilation=[False, False, True]), # [False, True, True]
            {'layer4': 'out', 'layer1': 'low_level'}
        )
        model = ReCoNet(backbone, 2048, 256, num_classes, [6,12,18]) # [12, 24, 36]
    
    return model

def main():
    """Supervised Learning pipeline with ReCo support for full label dataset"""
    args = parse_args()
    print(f"Training on {args.dataset} dataset with {args.model} model")

    save_stuff = not args.disable_saving
    print(f'Checkpoint saving enabled: {save_stuff}')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wandb_config = vars(args)
    init_wandb(
        project_name="reco_v1",  
        config=wandb_config,
        run_name=f"{args.dataset}_{args.model}_{args.seed}"
    )
    
    if args.dataset == 'pascal':
        loader = PascalVOCLoader(
            data_path=args.data_path,
            num_labeled=args.num_labeled,
            seed=args.seed
        )
        if args.batch_size is not None:
            loader.batch_size = args.batch_size
        num_classes = 21
    else:  # cityscapes
        loader = CityscapesLoader(
            data_path=args.data_path,
            num_labeled=args.num_labeled,
            seed=args.seed
        )
        if args.batch_size is not None:
            loader.batch_size = args.batch_size
        num_classes = 19
    
    train_loader, val_loader = loader.create_loaders(use_unlabeled=False)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    model = create_model(args)
    model = model.to(device)
    
    watch_model(model)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    
    val_interval = args.val_interval
    best_iou = 0
    total_iterations = 0
    
    pbar = tqdm(total=args.total_iterations)
    
    for epoch in range(args.epochs):
        # print(f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (img, mask) in enumerate(train_loader):
            model.train()
            
            img = img.to(device)
            mask = mask.long().to(device)
            
            outputs = model(img)
            pred = outputs['out']

            # pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)
            loss = criterion(pred, mask)
            
            reco_loss_val = None
            if args.reco:
                reps = outputs['_reco']
                
                probs = F.softmax(outputs['_out'], dim=1)
                if reps.shape[2:] != mask.shape[1:]: 
                    mask_interp = F.interpolate(mask.unsqueeze(1).float(), 
                                            size=reps.shape[2:], 
                                            mode='nearest').squeeze(1).long()
                    
                    valid_mask = F.interpolate((mask.unsqueeze(1) != -1).float(), 
                                            size=reps.shape[2:], 
                                            mode='nearest').squeeze(1) > 0.5
                else:
                    mask_interp = mask
                    valid_mask = (mask != -1)
                
                reco_loss = reco_loss_func(
                    rep=reps,
                    label=mask_interp,
                    mask=valid_mask, 
                    prob=probs,
                    strong_threshold=args.reco_threshold,
                    temp=args.reco_temp,
                    num_queries=args.reco_num_queries,
                    num_negatives=args.reco_num_negatives
                )
                
                reco_loss_val = reco_loss.item()
                if total_iterations % 100 == 0:
                    print(f"ReCo loss: {reco_loss_val:.4f}")
                
                loss = loss + args.reco_weight * reco_loss
            
            current_lr = adjust_learning_rate(optimizer, args.lr, total_iterations, args.total_iterations)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if total_iterations % 10 == 0:
                log_training_metrics(loss.item(), current_lr, epoch + 1, total_iterations, reco_loss_val)
            
            total_iterations += 1
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch+1}/{args.epochs}, Iter: {total_iterations}/{args.total_iterations}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            
            if total_iterations % val_interval == 0:
                model.eval()
                
                val_running_loss = 0
                val_iou = 0
                num_batches = 0
                iou = IOU(num_classes, device)

                with torch.no_grad():
                    for idx, (img, mask) in enumerate(val_loader):
                        img = img.to(device)
                        mask = mask.long().to(device)
                        
                        outputs = model(img)
                        pred = outputs['out']
                        # pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)

                        loss = criterion(pred, mask)
                        
                        iou_mat = iou.update(pred.argmax(1).flatten(), mask.flatten())
                        val_running_loss += loss.item()
                        num_batches += 1
                    
                    val_loss = val_running_loss / num_batches
                    mean_iou = iou.get()
                
                print("-"*50)
                print(f"Epoch: {epoch+1}/{args.epochs}, Iteration: {total_iterations}/{args.total_iterations}")
                print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}")
                print("-"*50)
                
                log_validation_metrics(val_loss, mean_iou, total_iterations)
                
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    if save_stuff:
                        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_best.pth")
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Saved best model to {checkpoint_path}")
                        update_summary(best_iou, total_iterations)
            
            if save_stuff and total_iterations % 10000 == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_iter_{total_iterations}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'iteration': total_iterations,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'iou': best_iou,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            if total_iterations >= args.total_iterations:
                break
        
        if total_iterations >= args.total_iterations:
            print(f"Reached {args.total_iterations} iterations. Stopping training.")
            break
    
    pbar.close()
    print(f"Training completed! Best validation IoU: {best_iou:.4f}")
    
    if save_stuff:
        final_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_final.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")
    
    finish()

if __name__ == "__main__":
    main()
