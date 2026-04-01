import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
