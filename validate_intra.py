import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
import wandb

from data.dataset import MorphologyDataset
from models.backbone import ResNetBackbone
from models.heads import MorphologySSLHeads

def get_accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / len(target)

def evaluate_source_on_targets(args, source_class, backbone_path, heads_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading weights for {source_class} backbone...")
    
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=False).to(device)
    heads = MorphologySSLHeads(backbone.embedding_dim, s=30.0, m=0.5).to(device)
    
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    heads.load_state_dict(torch.load(heads_path, map_location=device))
    
    backbone.eval()
    heads.eval()
    
    mvtec_root = "datasets/mvtec"
    all_classes = [d for d in os.listdir(mvtec_root) if os.path.isdir(os.path.join(mvtec_root, d))]
    inter_classes = [c for c in all_classes if c != source_class]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {}
    for target_class in inter_classes:
        category_root = os.path.join(mvtec_root, target_class)
        print(f"  Evaluating on {target_class}...")
        
        dataset = MorphologyDataset(category_root, transform=transform, img_size=224, is_train=False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        if len(dataset) == 0:
            print(f"  Warning: No valid training images found for {target_class}. Skipping.")
            continue
            
        acc_type, acc_width, acc_height = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, t_labels, w_labels, h_labels in dataloader:
                images = images.to(device)
                t_labels, w_labels, h_labels = t_labels.to(device), w_labels.to(device), h_labels.to(device)
                
                embeddings = backbone(images)
                l_t, l_w, l_h = heads(embeddings, t_labels, w_labels, h_labels)
                
                acc_type += get_accuracy(l_t, t_labels)
                acc_width += get_accuracy(l_w, w_labels)
                acc_height += get_accuracy(l_h, h_labels)
                
        acc_type /= len(dataloader)
        acc_width /= len(dataloader)
        acc_height /= len(dataloader)
        
        results[target_class] = {'type': acc_type, 'width': acc_width, 'height': acc_height}
        print(f"    -> Type: {acc_type:.4f}, Width: {acc_width:.4f}, Height: {acc_height:.4f}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Inter-Class Generalization via Confusion Matrix")
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18 or resnet50')
    parser.add_argument('--weights_dir', type=str, default='logs', help='Directory containing run_* folders')
    args = parser.parse_args()
    
    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Initialize Matrices completely with NaNs so the diagonal is empty
    matrices = {
        'type': np.full((num_classes, num_classes), np.nan),
        'width': np.full((num_classes, num_classes), np.nan),
        'height': np.full((num_classes, num_classes), np.nan)
    }
    
    # 1. Initialize Single Global WandB run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="IAS-Morphology-SSL", name=f"Inter_Class_Validation_Matrix_{timestamp}", config=args)
    
    # 2. Iterate over Source Classes (Trained On) horizontally
    for source_class in class_names:
        print(f"\n{'='*50}\nStarting Inter Validation for model trained on: {source_class}\n{'='*50}")
        
        log_dirs = []
        if os.path.exists(args.weights_dir):
            for d in os.listdir(args.weights_dir):
                if d.startswith(f"run_{source_class}_{args.backbone}_"):
                    log_dirs.append(os.path.join(args.weights_dir, d))
        
        if not log_dirs:
            print(f"No log directory found for {source_class}. Skipping...")
            continue
            
        log_dirs.sort(reverse=True)
        latest_log_dir = log_dirs[0]
        
        backbone_path = os.path.join(latest_log_dir, "best_backbone.pth")
        heads_path = os.path.join(latest_log_dir, "best_heads.pth")
        
        if not os.path.exists(backbone_path) or not os.path.exists(heads_path):
            print(f"Missing weights in {latest_log_dir}. Skipping...")
            continue
            
        source_idx = class_to_idx[source_class]
        
        # Get evaluation results on all Target Classes
        results = evaluate_source_on_targets(args, source_class, backbone_path, heads_path)
        
        for target_class, accs in results.items():
            if target_class in class_to_idx:
                target_idx = class_to_idx[target_class]
                # y-axis is target class, x-axis is source class
                matrices['type'][target_idx, source_idx] = accs['type']
                matrices['width'][target_idx, source_idx] = accs['width']
                matrices['height'][target_idx, source_idx] = accs['height']
            
    # 3. Draw Heatmaps and log to wandb
    print("\n" + "="*50)
    print("Generating and uploading Confusion Matrices to WandB...")
    print("="*50)
    
    for metric_name, matrix in matrices.items():
        # 1. Log Numerical Table to Wandb
        df = pd.DataFrame(matrix, index=class_names, columns=class_names)
        df.index.name = 'Target_Class'
        wandb.log({f"DataTables/{metric_name}": wandb.Table(dataframe=df.reset_index())})
        
        # 2. Draw Heatmap Image
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Accuracy'},
                    mask=np.isnan(matrix))
        
        plt.title(f"Inter-class Morphological Accuracy: {metric_name.capitalize()}", fontsize=15)
        plt.xlabel("Source Class (Trained On Model)", fontsize=12)
        plt.ylabel("Target Class (Evaluated On Dataset)", fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        wandb.log({f"Confusion_Matrix/{metric_name}": wandb.Image(plt)})
        # Save locally as well just in case
        plt.savefig(f"inter_val_matrix_{metric_name}.png", dpi=300)
        plt.close()
        
    wandb.finish()
    print("Inter-class Validation Completed Successfully.")
