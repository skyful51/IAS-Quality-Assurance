import os
import cv2
import numpy as np
import random
from itertools import product
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

class MVTecADDataset(Dataset):
    """
    Standard MVTec AD Dataset Loader for Category-specific Training.
    
    Structure:
    category_root/
        train/
            good/             -> Label 0: Normal
        test/
            good/             -> Label 0: Normal (optional, included for more data)
            defect_type_1/    -> Label 1: Defect Type 1
            defect_type_2/    -> Label 2: Defect Type 2
            ...
    """
    def __init__(self, category_root, transform=None, include_test_good=True):
        """
        Args:
            category_root (str): Path to the specific category (e.g., 'data/mvtec/bottle')
            transform (callable, optional): Transforms to be applied on a sample.
            include_test_good (bool): Whether to include normal images from the test folder.
        """
        self.category_root = category_root
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # 1. Register Normal (Good) Class: Label 0
        self.class_to_idx = {'good': 0}
        
        # Add normal images from 'train/good'
        train_good_dir = os.path.join(category_root, 'train', 'good')
        self._add_images_from_dir(train_good_dir, 0)
        
        # Add normal images from 'test/good' if requested
        if include_test_good:
            test_good_dir = os.path.join(category_root, 'test', 'good')
            if os.path.exists(test_good_dir):
                self._add_images_from_dir(test_good_dir, 0)
        
        # 2. Register Defect Classes: Label 1, 2, ... from 'test' folder
        test_root = os.path.join(category_root, 'test')
        defect_types = sorted([d for d in os.listdir(test_root) 
                             if os.path.isdir(os.path.join(test_root, d)) and d != 'good'])
        
        for idx, d_type in enumerate(defect_types):
            self.class_to_idx[d_type] = idx + 1
            defect_dir = os.path.join(test_root, d_type)
            self._add_images_from_dir(defect_dir, idx + 1)
            
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]

    def _add_images_from_dir(self, directory, label):
        """Helper to add all valid images from a directory to the dataset."""
        if not os.path.exists(directory):
            return
            
        for img_name in os.listdir(directory):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(directory, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloader(category_root, batch_size=32, img_size=224, mode='train'):
    """
    Creates a DataLoader for MVTec AD category.
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = MVTecADDataset(category_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=4)

class MorphologyDataset(Dataset):
    """
    Self-Supervised Morphology Dataset.
    Only uses 'good' images and applies real-time morphological transformations.
    """
    def __init__(self, category_root, transform=None, img_size=224):
        self.category_root = category_root
        self.transform = transform
        self.img_size = img_size
        
        self.image_paths = []
        train_good_dir = os.path.join(category_root, 'train', 'good')
        if os.path.exists(train_good_dir):
            for img_name in os.listdir(train_good_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(train_good_dir, img_name))
        
        # Transformation Parameters
        self.types = ['dilation', 'erosion', 'gradient']
        self.widths = [1, 3, 7, 11]
        self.heights = [1, 3, 7, 11]
        
        # All 48 combinations (3 * 4 * 4)
        self.combinations = list(product(range(len(self.types)), 
                                        range(len(self.widths)), 
                                        range(len(self.heights))))
        
        # To ensure balanced sampling within an epoch, we can assign combinations to indices
        self.num_combos = len(self.combinations)

    def __len__(self):
        return len(self.image_paths)

    def apply_morphology(self, image, t_idx, w_idx, h_idx):
        # Convert PIL to CV2 (Ensure uint8)
        img_np = np.array(image)
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Morphology
        w = self.widths[w_idx]
        h = self.heights[h_idx]
        kernel = np.ones((h, w), np.uint8)
        
        if self.types[t_idx] == 'dilation':
            img_cv = cv2.dilate(img_cv, kernel, iterations=1)
        elif self.types[t_idx] == 'erosion':
            img_cv = cv2.erode(img_cv, kernel, iterations=1)
        elif self.types[t_idx] == 'gradient':
            img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_GRADIENT, kernel)
            
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        
        # Pick a combination.
        c_idx = random.randint(0, self.num_combos - 1)
        t_idx, w_idx, h_idx = self.combinations[c_idx]
        
        transformed_image = self.apply_morphology(image, t_idx, w_idx, h_idx)
        
        if self.transform:
            transformed_image = self.transform(transformed_image)
            
        return transformed_image, t_idx, w_idx, h_idx

def get_ssl_dataloader(category_root, batch_size=32, img_size=224, val_split=0.1):
    """
    Creates Training and Validation DataLoaders for SSL task.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = MorphologyDataset(category_root, transform=transform, img_size=img_size)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    # Shuffle for split
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
