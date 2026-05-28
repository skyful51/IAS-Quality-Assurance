import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datetime import datetime

# Import custom backbone and head from our project models
from models.backbone import CutPasteBackbone, ResNetBackbone
from models.heads import ArcMarginProduct

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class MVTecTestDataset(Dataset):
    """
    Dataset loader that loads ONLY the test folder of a specific MVTec category.
    This contains both the 'good' (normal) class and all actual defect classes.
    """
    def __init__(self, category_root, transform=None):
        self.category_root = category_root
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # 1. Check if 'test' directory exists
        test_dir = os.path.join(category_root, 'test')
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found at {test_dir}")
            
        # 2. Register classes: 'good' is class 0, and defect folders are sorted classes 1, 2, ...
        self.class_to_idx = {'good': 0}
        
        # Add normal images from 'test/good'
        test_good_dir = os.path.join(test_dir, 'good')
        self._add_images_from_dir(test_good_dir, 0)
        
        # Find all defect directories (excluding 'good')
        defect_types = sorted([d for d in os.listdir(test_dir) 
                             if os.path.isdir(os.path.join(test_dir, d)) and d != 'good'])
        
        for idx, d_type in enumerate(defect_types):
            self.class_to_idx[d_type] = idx + 1
            defect_dir = os.path.join(test_dir, d_type)
            self._add_images_from_dir(defect_dir, idx + 1)
            
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
        
    def _add_images_from_dir(self, directory, label):
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

def run_training_for_class(args, class_data_path, checkpoint_path):
    # Setup Experiment / Log folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_name = os.path.basename(class_data_path.rstrip("/"))
    exp_name = f"run_aml_{class_name}_{args.backbone_type}_{timestamp}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log Directory: {log_dir}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset & DataLoader (Only using MVTec test set)
    print(f"Loading MVTec test set for class '{class_name}' from: {class_data_path}")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MVTecTestDataset(class_data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} images across {num_classes} classes:")
    for c_name, c_idx in dataset.class_to_idx.items():
        count = sum(1 for label in dataset.labels if label == c_idx)
        print(f"  Class {c_idx} ({c_name}): {count} images")

    # 2. Initialize Backbone
    if args.backbone_type == 'resnet18':
        print("Initializing standard ResNet18 backbone (ImageNet pre-trained)...")
        backbone = ResNetBackbone(model_name='resnet18', pretrained=True).to(device)
    elif args.backbone_type == 'cutpaste':
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified when using 'cutpaste' backbone_type")
        print(f"Loading CutPaste model from: {checkpoint_path}")
        backbone = CutPasteBackbone(
            pretrained=False,
            head_layers=args.head_layer,
            include_head=args.include_head,
            checkpoint_path=checkpoint_path
        ).to(device)
    else:
        raise ValueError(f"Unknown backbone_type: {args.backbone_type}")
    
    # 3. Handle backbone freeze/unfreeze
    if args.freeze_backbone:
        print(f"Freezing {args.backbone_type} backbone (Using as static feature extractor)...")
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        print(f"Fine-tuning the {args.backbone_type} backbone along with AML head...")
        backbone.train()

    # 4. Initialize ArcMarginProduct (AML Head)
    print(f"Initializing ArcMarginProduct Head with in_features={backbone.embedding_dim}, out_features={num_classes}")
    head = ArcMarginProduct(
        in_features=backbone.embedding_dim,
        out_features=num_classes,
        s=args.s,
        m=args.m
    ).to(device)

    # 5. Optimizer, Criterion, Scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Optimize head parameters, and optionally backbone parameters
    params_to_optimize = [{'params': head.parameters()}]
    if not args.freeze_backbone:
        params_to_optimize.append({'params': backbone.parameters()})
        
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    # 6. Training Loop
    print(f"Starting AML training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if not args.freeze_backbone:
            backbone.train()
        head.train()
        
        total_loss = 0.0
        correct_margin = 0
        correct_cosine = 0
        total_samples = 0
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features from backbone
            embeddings = backbone(images)
            
            # Pass through ArcFace head to get logits (with margin penalty)
            logits = head(embeddings, labels)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            # Track training accuracy
            with torch.no_grad():
                # 1) Accuracy with margin penalty (highly pessimistic during training)
                preds_margin = logits.argmax(dim=1)
                correct_margin += preds_margin.eq(labels).sum().item()
                
                # 2) Raw Cosine Accuracy without margin penalty (actual classification performance)
                cosine = F.linear(F.normalize(embeddings), F.normalize(head.weight))
                preds_cosine = cosine.argmax(dim=1)
                correct_cosine += preds_cosine.eq(labels).sum().item()
                
                total_samples += labels.size(0)
                
        epoch_loss = total_loss / total_samples
        epoch_acc_margin = correct_margin / total_samples
        epoch_acc_cosine = correct_cosine / total_samples
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f} - Acc (Margin): {epoch_acc_margin:.4f} - Acc (Cosine): {epoch_acc_cosine:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")

    # 7. Save trained weights
    torch.save(head.state_dict(), os.path.join(log_dir, "aml_head_final.pth"))
    if not args.freeze_backbone:
        torch.save(backbone.state_dict(), os.path.join(log_dir, "backbone_finetuned.pth"))
    
    print(f"AML head training complete. Weights saved in: {log_dir}")

def find_checkpoint(models_dir, class_name):
    """
    Finds the latest checkpoint file named 'model-{class_name}-*.tch' or 'model-{class_name}.tch' in models_dir.
    """
    import glob
    pattern = os.path.join(models_dir, f"model-{class_name}-*.tch")
    files = glob.glob(pattern)
    if not files:
        pattern_exact = os.path.join(models_dir, f"model-{class_name}.tch")
        files = glob.glob(pattern_exact)
        
    if not files:
        return None
    # Sort by modification time to get the latest one
    files.sort(key=os.path.getmtime)
    return files[-1]

def main(args):
    # Detect if data_path points to a single category or a parent MVTec directory
    is_single_class = os.path.exists(os.path.join(args.data_path, 'test'))
    
    if is_single_class:
        class_paths = [args.data_path]
    else:
        class_paths = []
        for d in sorted(os.listdir(args.data_path)):
            full_path = os.path.join(args.data_path, d)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'test')):
                class_paths.append(full_path)
                
    if not class_paths:
        print(f"No valid MVTec category folders containing 'test' directory found in {args.data_path}")
        return

    # If checkpoint_path is None and backbone_type is 'cutpaste', use default directory
    models_dir = args.checkpoint_path if args.checkpoint_path is not None else "pytorch-cutpaste/models"

    print(f"Found {len(class_paths)} classes to train: {[os.path.basename(p.rstrip('/')) for p in class_paths]}")
    for class_path in class_paths:
        class_name = os.path.basename(class_path.rstrip("/"))
        print("\n" + "="*80)
        print(f" PROCESSING CATEGORY: {class_name.upper()} ")
        print("="*80)
        
        checkpoint_file = None
        if args.backbone_type == 'cutpaste':
            # If checkpoint_path is explicitly a file, use it (only makes sense for single class)
            if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
                checkpoint_file = args.checkpoint_path
            else:
                checkpoint_file = find_checkpoint(models_dir, class_name)
                if checkpoint_file is None:
                    print(f"Warning: No CutPaste checkpoint found for '{class_name}' in '{models_dir}'. Skipping.")
                    continue
                print(f"Found latest CutPaste checkpoint for '{class_name}': {checkpoint_file}")
                
        try:
            run_training_for_class(args, class_path, checkpoint_file)
        except Exception as e:
            print(f"Error during training on category '{class_name}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcMarginProduct (AML) head on MVTec test set using CutPaste Backbone")
    parser.add_argument('--data_path', type=str, required=True, help="Path to MVTec category or root dataset directory containing multiple categories")
    parser.add_argument('--backbone_type', type=str, default='cutpaste', choices=['cutpaste', 'resnet18'], help="Type of backbone to use (default: cutpaste)")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to trained CutPaste checkpoint (.tch) or directory containing checkpoint files")
    parser.add_argument('--head_layer', type=int, default=2, help="Number of head layers in CutPaste model (default: 2)")
    parser.add_argument('--include_head', action='store_true', help="Include CutPaste projection head (outputs 128-dim features) instead of ResNet18 raw features (512-dim)")
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--img_size', type=int, default=224, help="Image size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--lr_step_size', type=int, default=20, help="Decay LR every N epochs")
    parser.add_argument('--freeze_backbone', type=str2bool, default=True, help="Whether to freeze the backbone weights (default: True)")
    parser.add_argument('--cuda', type=str2bool, default=True, help="Use CUDA if available")
    
    # ArcFace Hyperparameters
    parser.add_argument('--s', type=float, default=30.0, help="ArcFace scale parameter")
    parser.add_argument('--m', type=float, default=0.5, help="ArcFace margin parameter")
    
    args = parser.parse_args()
    main(args)

