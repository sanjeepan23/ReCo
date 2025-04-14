import torch
import numpy as np
import torch.nn.functional as F

def adjust_learning_rate(optimizer, initial_lr, iter, total_iter, power=0.9):
    """Adjusts learning rate using a polynomial decay schedule.
   
    Args:
        optimizer: Optimizer to update
        initial_lr: Starting learning rate
        iter: Current iteration
        total_iter: Total iterations
        power: Power for polynomial decay
    
    Returns:
        Current learning rate
    """
    lr = initial_lr * (1 - iter / total_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class IOU:
    def __init__(self, num_classes, device):
        self.n = num_classes
        self.mat = torch.zeros((self.n, self.n), dtype=torch.int64, device=device)
    def update(self, pred, target):
        with torch.no_grad():
            k = (target >= 0) & (target < self.n)
            inds = self.n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=self.n ** 2).reshape(self.n, self.n)
    def get(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()



def compute_iou(outputs, targets, num_classes):
    smooth = 1e-6
    preds = torch.argmax(outputs, dim=1)
    
    ious = []
    for cls in range(0, num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        
        if union.item() > 0:
            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou.item())
    
    return np.mean(ious) if ious else 0


def calculate_unsupervised_loss(outputs, pseudo_labels, conf_mask):
    if conf_mask.sum() > 0:
        outputs = outputs.permute(0, 2, 3, 1)
        masked_outputs = outputs[conf_mask].view(-1, outputs.size(3))
        masked_labels = pseudo_labels[conf_mask].view(-1)
        unsup_loss = F.cross_entropy(
            masked_outputs, 
            masked_labels, 
            ignore_index=-1
        )
    else:
        unsup_loss = torch.tensor(0.0).to(outputs.device)
    
    return unsup_loss

def reco_loss_func(rep, label, mask, prob=None, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    """Regional Contrast (ReCo) loss for semantic segmentation.
   
    Performs pixel-level contrastive learning with:
    - Active hard query sampling based on prediction confidence
    - Class-adaptive negative key sampling using class relation graph
    - Normalized temperature-scaled cross entropy loss
    
    Args:
        rep: Dense pixel representations [B,C,H,W]
        label: Pixel labels [B,H,W]
        mask: Valid pixel mask [B,H,W]
        prob: Prediction confidence (optional)
        strong_threshold: Threshold for hard query sampling
        temp: Temperature for contrastive loss
        num_queries: Maximum number of queries per class
        num_negatives: Maximum number of negative keys
    """
    B, C, H, W = rep.shape
    rep = rep.reshape(B, C, -1)  
    label = label.reshape(B, -1)  
    mask = mask.reshape(B, -1)  
    
    if prob is not None:
        prob = prob.reshape(B, prob.size(1), -1)
    
    # Get unique classes in the batch
    classes = torch.unique(label)
    classes = classes[classes != 255]  
    
    if len(classes) == 0:
        return torch.tensor(0.0, device=rep.device, requires_grad=True)
    
    # mean class representations (positive keys)
    r_c_plus_k = {} 
    class_pixels_dict = {}
    
    for c in classes:
        c_mask = (label == c) & mask 
        if c_mask.sum() == 0:
            continue
        
        class_pixels = []
        class_probs = []
        for b in range(B):
            b_mask = c_mask[b] 
            if b_mask.sum() > 0:
                b_rep = rep[b, :, b_mask] 
                class_pixels.append(b_rep)
                if prob is not None:
                    b_prob = prob[b, c.item(), b_mask]
                    class_probs.append(b_prob)
        
        if len(class_pixels) > 0:
            all_pixels = torch.cat(class_pixels, dim=1)  
            class_pixels_dict[c.item()] = {
                'pixels': all_pixels,
                'probs': torch.cat(class_probs) if len(class_probs) > 0 else None
            }
            # mean representation for class c (positive key)
            r_c_plus_k[c.item()] = all_pixels.mean(dim=1)
    
    # class relationship graph G (for active key sampling)
    classes_list = list(r_c_plus_k.keys())
    if len(classes_list) <= 1:  
        return torch.tensor(0.0, device=rep.device, requires_grad=True)
    
    # pairwise relationships between classes as defined in Eq. 3
    G = torch.zeros((len(classes_list), len(classes_list)), device=rep.device)
    for i, c_i in enumerate(classes_list):
        for j, c_j in enumerate(classes_list):
            if i != j:
                G[i, j] = torch.dot(
                    F.normalize(r_c_plus_k[c_i], p=2, dim=0),
                    F.normalize(r_c_plus_k[c_j], p=2, dim=0)
                )
    
    # ReCo loss for all valid classes
    total_loss = 0.0
    valid_classes = 0
    
    for idx, c in enumerate(classes_list):
        c_data = class_pixels_dict[c]
        c_pixels = c_data['pixels']
        
        # active query sampling based on prediction confidence (Eq. 4)
        if c_data['probs'] is not None:
            c_probs = c_data['probs']
            hard_mask = (c_probs < strong_threshold)
            hard_pixels = c_pixels[:, hard_mask]
            hard_probs = c_probs[hard_mask]
            
            if hard_pixels.size(1) > num_queries:
                _, indices = torch.topk(hard_probs, num_queries, largest=False)
                queries = hard_pixels[:, indices]
            else:
                queries = hard_pixels

        else:
            if c_pixels.size(1) > num_queries:
                indices = torch.randperm(c_pixels.size(1), device=c_pixels.device)[:num_queries]
                queries = c_pixels[:, indices]
            else:
                queries = c_pixels
        
        if queries.size(1) == 0:
            continue
            
        # positive key (class mean representation)
        positive_key = r_c_plus_k[c]
        
        # negative key sampling 
        with torch.no_grad():
            distribution = F.softmax(G[idx], dim=0)
            neg_indices = [j for j in range(len(classes_list)) if j != idx]
            
            neg_distribution = distribution[neg_indices]
            neg_distribution = neg_distribution / neg_distribution.sum()
            
            negative_keys = []
            neg_classes = [classes_list[j] for j in neg_indices]
            remaining_samples = num_negatives
            
            for j, neg_c in enumerate(neg_classes):
                if remaining_samples <= 0:
                    break
                    
                num_samples = min(
                    max(1, int(neg_distribution[j].item() * num_negatives)),
                    class_pixels_dict[neg_c]['pixels'].size(1),
                    remaining_samples
                )
                
                if num_samples > 0:
                    indices = torch.randperm(class_pixels_dict[neg_c]['pixels'].size(1), device=rep.device)[:num_samples]
                    sampled_neg = class_pixels_dict[neg_c]['pixels'][:, indices]
                    negative_keys.append(sampled_neg)
                    remaining_samples -= num_samples
            
            if len(negative_keys) == 0:
                continue
                
            all_negative_keys = torch.cat(negative_keys, dim=1)  # [C, N]
            
            # Combine positive and negative keys: keys = [positive key | negative keys]
            positive_feat = positive_key.unsqueeze(0).unsqueeze(0).repeat(queries.size(1), 1, 1)
            all_feat = torch.cat((positive_feat, all_negative_keys.t().unsqueeze(0).repeat(queries.size(1), 1, 1)), dim=1)
        
        #  contrastive loss as in Eq. 1
        queries = queries.t()  # [Q, C]
        
        # similarity between queries and all keys
        seg_logits = F.cosine_similarity(queries.unsqueeze(1), all_feat, dim=2) / temp
        
        # cross entropy loss with positive key as target (index 0)
        class_loss = F.cross_entropy(seg_logits, torch.zeros(queries.size(0), dtype=torch.long, device=rep.device))
        
        total_loss += class_loss
        valid_classes += 1
    
    if valid_classes > 0:
        return total_loss / valid_classes
    else:
        return torch.tensor(0.0, device=rep.device, requires_grad=True)
