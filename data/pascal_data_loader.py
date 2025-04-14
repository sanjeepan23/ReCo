import os
import random
import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import utils.img_processing as img_processing

class PascalVOCDataset(Dataset):
    """Dataset for Pascal VOC with support for partial labeling.
   
    Args:
        data_root: Dataset directory
        mode: 'train' or 'val'
        label_ratio: Ratio of labeled pixels (for partial label setting)
        random_seed: Seed for reproducibility
        input_dims: Input dimensions for cropping
        scale_range: Range for random scaling
        enable_transforms: Whether to apply data augmentation
    """
    def __init__(self, 
                 data_root, 
                 mode='train', 
                 label_ratio=None, 
                 random_seed=None,
                 input_dims=(321, 321),
                 scale_range=(0.5, 1.5),
                 enable_transforms=True):
        
        self.data_root = data_root
        self.mode = mode
        self.label_ratio = label_ratio
        self.random_seed = random_seed
        self.input_dims = input_dims
        self.scale_range = scale_range
        self.enable_transforms = enable_transforms and mode == 'train'
        
        if mode == 'train':
            id_file = os.path.join(data_root, 'train_aug.txt')
        else:
            id_file = os.path.join(data_root, 'val.txt')
            
        with open(id_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.sample_ids)
    
    def apply_transformations(self, img, mask, do_scale=False, do_randcrop=False, do_augmentation=False):
        width, height = img.size
        
        if self.enable_transforms or do_scale:
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale_factor = 1.0
            
        target_size = (int(height * scale_factor), int(width * scale_factor))
        img = TF.resize(img, target_size, Image.BILINEAR)
        mask = TF.resize(mask, target_size, Image.NEAREST)
        
        crop_h, crop_w = self.input_dims
        if crop_h > target_size[0] or crop_w > target_size[1]:
            pad_right = max(0, crop_w - target_size[1])
            pad_bottom = max(0, crop_h - target_size[0])
            
            img = TF.pad(img, padding=(0, 0, pad_right, pad_bottom), padding_mode='reflect')
            mask = TF.pad(mask, padding=(0, 0, pad_right, pad_bottom), fill=255, padding_mode='constant')
        
        if self.enable_transforms or do_randcrop:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.input_dims)
        else:
            i = (target_size[0] - crop_h) // 2 if target_size[0] > crop_h else 0
            j = (target_size[1] - crop_w) // 2 if target_size[1] > crop_w else 0
            h, w = crop_h, crop_w
            
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        if self.enable_transforms or do_augmentation:
            if random.random() > 0.2:
                brightness = random.uniform(0.75, 1.25)
                contrast = random.uniform(0.75, 1.25)
                saturation = random.uniform(0.75, 1.25)
                hue = random.uniform(-0.25, 0.25)
                
                img = transforms.ColorJitter(
                    brightness=(brightness, brightness),
                    contrast=(contrast, contrast),
                    saturation=(saturation, saturation),
                    hue=(hue, hue)
                )(img)
            
            if random.random() > 0.5:
                radius = random.uniform(0.15, 1.15)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        
        img_tensor = TF.to_tensor(img)
        mask_tensor = (TF.to_tensor(mask) * 255).long()
        
        mask_tensor[mask_tensor == 255] = -1
        
        img_tensor = img_processing.normalize(img_tensor)

        return img_tensor, mask_tensor.squeeze(0)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        img_path = os.path.join(self.data_root, 'JPEGImages', f'{sample_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        
        if self.label_ratio is None or self.mode == 'val':
            mask_dir = 'SegmentationClassAug'
        else:
            mask_dir = f'SegmentationClassAug_{self.label_ratio}_{self.random_seed}'
            
        mask_path = os.path.join(self.data_root, mask_dir, f'{sample_id}.png')
        mask = Image.open(mask_path)
        
        img_tensor, mask_tensor = self.apply_transformations(img, mask)
        
        return img_tensor, mask_tensor


def select_balanced_samples(root_dir, num_samples=None, is_training=True):
    
    if is_training:
        file_path = os.path.join(root_dir, 'train_aug.txt')
    else:
        file_path = os.path.join(root_dir, 'val.txt')
        
    with open(file_path) as f:
        all_ids = f.read().splitlines()
    
    if (not is_training) or (not num_samples):
        return all_ids
    
    selected_ids = []
    candidate_ids = all_ids.copy()
    random.shuffle(candidate_ids)
    
    class_counts = np.zeros(21)  # Pascal VOC has 21 classes (0-20)
    min_class_ids = np.arange(21)
    
    reserve_ids = []
    
    while len(selected_ids) < num_samples:
        if candidate_ids:
            current_id = candidate_ids.pop()
        else:
            candidate_ids = reserve_ids.copy()
            current_id = candidate_ids.pop()
            reserve_ids = []
        
        mask_path = os.path.join(root_dir, 'SegmentationClassAug', f'{current_id}.png')
        mask = np.array(Image.open(mask_path))
        
        unique_classes = np.unique(mask)
        if 255 in unique_classes:
            unique_classes = unique_classes[:-1]
            
        if len(unique_classes) >= 3:
            if not selected_ids or np.any(np.in1d(min_class_ids, unique_classes)):
                selected_ids.append(current_id)
                class_counts[unique_classes] += 1
                
                min_class_ids = np.where(class_counts == class_counts.min())[0]
            else:
                reserve_ids.append(current_id)
    
    unlabeled_ids = [img_id for img_id in all_ids if img_id not in selected_ids]
    return selected_ids, unlabeled_ids


class PascalVOCLoader:
    """Creates data loaders for Pascal VOC dataset.
    
    Args:
        data_path: Dataset directory
        num_labeled: Number of labeled samples
        label_ratio: Ratio of labeled pixels per image
        seed: Random seed
    """
    def __init__(self, data_path, num_labeled=None, label_ratio=None, seed=0):
        self.data_path = data_path
        self.label_ratio = label_ratio
        self.seed = seed
        
        self.eval_size = [513, 513]  # Size for evaluation
        self.train_size = [321, 321]  # Size for training
        self.num_classes = 21
        self.scale_range = (0.5, 1.5)
        self.batch_size = 10
        
        random.seed(seed)
        
        if num_labeled is not None:
            self.labeled_ids, self.unlabeled_ids = select_balanced_samples(
                data_path, num_samples=num_labeled, is_training=True
            )
        else:
            self.unlabeled_ids = select_balanced_samples(
                data_path, num_samples=None ,is_training=True
            )  
            self.labeled_ids = self.unlabeled_ids.copy()
            
        self.val_ids = select_balanced_samples(data_path, is_training=False)
    
    def create_loaders(self, use_unlabeled=True):
       
        labeled_dataset = PascalVOCDataset(
            data_root=self.data_path, 
            mode='train',
            label_ratio=self.label_ratio,
            random_seed=self.seed,
            input_dims=self.train_size,
            scale_range=self.scale_range,
            enable_transforms=True
        )
        labeled_dataset.sample_ids = self.labeled_ids
        
        if use_unlabeled:
            unlabeled_dataset = PascalVOCDataset(
                data_root=self.data_path, 
                mode='train',
                label_ratio=self.label_ratio,
                random_seed=self.seed,
                input_dims=self.train_size,
                scale_range=(1.0, 1.0),  # No scaling for unlabeled data
                enable_transforms=False
            )
            unlabeled_dataset.sample_ids = self.unlabeled_ids
        
        val_dataset = PascalVOCDataset(
            data_root=self.data_path, 
            mode='val',
            input_dims=self.eval_size,
            scale_range=(1.0, 1.0),
            enable_transforms=False
        )
        val_dataset.sample_ids = self.val_ids
        
        actual_batch_size = self.batch_size * 2 if not use_unlabeled else self.batch_size
        
        iters_per_epoch = 200
        total_samples = actual_batch_size * iters_per_epoch
        
        labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=actual_batch_size,
            sampler=sampler.RandomSampler(
                data_source=labeled_dataset,
                replacement=True,
                num_samples=total_samples
            ),
            drop_last=True,
            num_workers=4
        )
        
        if use_unlabeled:
            unlabeled_loader = DataLoader(
                unlabeled_dataset,
                batch_size=self.batch_size,
                sampler=sampler.RandomSampler(
                    data_source=unlabeled_dataset,
                    replacement=True,
                    num_samples=total_samples
                ),
                drop_last=True,
                num_workers=4
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        if use_unlabeled:
            return labeled_loader, unlabeled_loader, val_loader
        else:
            return labeled_loader, val_loader
