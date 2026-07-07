import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import pandas as pd

# Import custom modules
from models.backbone import CutPasteBackbone
from models.heads import ArcMarginProduct
from data.dataset import MVTecTestDataset
from utils.visualize import (
    plot_center_similarity,
    plot_sample_to_centroid_similarity,
    visualize_embeddings
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# find_checkpoint helper removed as user specifies direct backbone path.

def evaluate_alignment(backbone, head, dataloader, device, classes, save_dir):
    """
    Evaluates how well the sample embeddings align with the head's centroids.
    Calculates intra-class similarity (similarity to own centroid) and 
    inter-class similarity (similarity to other centroids).
    """
    backbone.eval()
    head.eval()
    
    # Normalize class center weights (centroids) -> [num_classes, dim]
    centroids = torch.nn.functional.normalize(head.weight).detach()
    num_classes = len(classes)
    
    # Accumulate embeddings by class
    class_embeddings = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = torch.nn.functional.normalize(backbone(images)) # [B, dim]
            
            for emb, label in zip(embeddings, labels):
                class_embeddings[label.item()].append(emb)
                
    print("\n" + "="*80)
    print(" QUANTITATIVE ALIGNMENT EVALUATION ")
    print("="*80)
    
    alignment_results = []
    
    for i in range(num_classes):
        embs = class_embeddings[i]
        if len(embs) == 0:
            print(f"Class '{classes[i]}': No samples found.")
            continue
        
        embs_tensor = torch.stack(embs) # [N, dim]
        
        # Calculate similarity with all centroids -> [N, num_classes]
        similarities = torch.matmul(embs_tensor, centroids.t())
        
        # Intra-class Similarity (to own centroid)
        self_sims = similarities[:, i]
        self_mean = self_sims.mean().item()
        self_std = self_sims.std().item()
        
        print(f"Class '{classes[i]:15s}' (Samples: {len(embs):3d}) -> Self Centroid Similarity: Mean={self_mean:.4f}, Std={self_std:.4f}")
        
        # Inter-class Similarity (to other centroids)
        other_indices = [j for j in range(num_classes) if j != i]
        if other_indices:
            other_sims = similarities[:, other_indices]
            other_mean = other_sims.mean().item()
            other_std = other_sims.std().item()
            print(f"                                   -> Other Centroid Similarity: Mean={other_mean:.4f}, Std={other_std:.4f}")
        else:
            other_mean, other_std = 0.0, 0.0
            
        alignment_results.append({
            'class_name': classes[i],
            'samples': len(embs),
            'intra_similarity_mean': self_mean,
            'intra_similarity_std': self_std,
            'inter_similarity_mean': other_mean,
            'inter_similarity_std': other_std
        })
        
    print("="*80)
    
    # Save results as CSV
    df = pd.DataFrame(alignment_results)
    csv_path = os.path.join(save_dir, "alignment_evaluation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved quantitative alignment evaluation to: {csv_path}\n")
    return alignment_results

def run_experiment(args):
    # Setup log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"run_selfsup_{args.class_name}_{args.backbone}_{timestamp}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Experiment Directory: {log_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Check if backbone path exists
    if not os.path.exists(args.backbone_path):
        raise FileNotFoundError(f"Backbone checkpoint file not found at: {args.backbone_path}")
    checkpoint_file = args.backbone_path
    print(f"Loading CutPaste checkpoint from: {checkpoint_file}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    class_data_path = os.path.join(args.dataset_root, args.class_name)
    print(f"Loading MVTec test set (as train & eval) for category '{args.class_name}' from: {class_data_path}")
    
    # Note: Use the MVTecTestDataset which includes good + all defect classes
    dataset = MVTecTestDataset(class_data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} images across {num_classes} classes: {dataset.classes}")
    
    # Initialize CutPaste Backbone and load pre-trained CutPaste checkpoint
    print(f"Initializing CutPaste Backbone (Base ResNet: {args.backbone})...")
    backbone = CutPasteBackbone(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        head_layers=args.head_layer,
        include_head=args.include_head,
        checkpoint_path=checkpoint_file
    ).to(device)
    
    # Freeze or Unfreeze Backbone weights
    if args.freeze_backbone:
        print("Freezing CutPaste backbone weights...")
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        print("Unfreezing CutPaste backbone weights (Fine-tuning enabled)...")
        backbone.train()
        
    # Initialize ArcMarginProduct Head
    print(f"Initializing ArcMarginProduct Head with in_features={backbone.embedding_dim}, out_features={num_classes}")
    head = ArcMarginProduct(
        in_features=backbone.embedding_dim,
        out_features=num_classes,
        s=args.s,
        m=args.m
    ).to(device)
    
    # Optimizer, Criterion, Scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimization parameters
    params_to_optimize = [{'params': head.parameters(), 'lr': args.lr}]
    if not args.freeze_backbone:
        # Fine-tune backbone at a lower learning rate
        params_to_optimize.append({'params': backbone.parameters(), 'lr': args.lr * 0.1})
        
    optimizer = optim.Adam(params_to_optimize)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    
    # Optional Weights & Biases Logging
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project, name=exp_name, config=vars(args))
        
    print(f"Starting Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if not args.freeze_backbone:
            backbone.train()
        else:
            backbone.eval()
        head.train()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Step 1: Feature Extraction
            embeddings = backbone(images)
            
            # Step 2: ArcFace Logits
            logits = head(embeddings, labels)
            
            # Step 3: Compute Loss
            loss = criterion(logits, labels)
            
            # Step 4: Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            # Cosine-based predictions (for logging actual training accuracy)
            with torch.no_grad():
                cosine = F.linear(F.normalize(embeddings), F.normalize(head.weight))
                preds = cosine.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)
                
        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.4f} - Cosine Acc: {epoch_acc:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if args.use_wandb:
            wandb.log({
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch+1)
            
    print("Training finished.")
    
    # Save Model Weights
    torch.save(backbone.state_dict(), os.path.join(log_dir, "backbone_final.pth"))
    torch.save(head.state_dict(), os.path.join(log_dir, "head_final.pth"))
    print(f"Model saved to {log_dir}")
    
    # Final Visualizations
    print("Generating visualizations...")
    plot_center_similarity(head, dataset.classes, log_dir, args.epochs)
    plot_sample_to_centroid_similarity(backbone, head, eval_dataloader, device, dataset.classes, log_dir, args.epochs)
    visualize_embeddings(backbone, head, eval_dataloader, device, log_dir, args.epochs)
    
    # Quantitative Evaluation
    alignment_stats = evaluate_alignment(backbone, head, eval_dataloader, device, dataset.classes, log_dir)
    
    if args.use_wandb:
        # Log final statistics to WandB
        for stat in alignment_stats:
            c = stat['class_name']
            wandb.log({
                f"final_eval/intra_sim_mean/{c}": stat['intra_similarity_mean'],
                f"final_eval/intra_sim_std/{c}": stat['intra_similarity_std'],
                f"final_eval/inter_sim_mean/{c}": stat['inter_similarity_mean'],
            })
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-supervised (CutPaste) baseline training and alignment evaluation on real MVTec dataset")
    parser.add_argument('--dataset_root', type=str, default='datasets/mvtec', help='Root directory of MVTec dataset')
    parser.add_argument('--class_name', type=str, default='bottle', help='Class name (e.g. bottle)')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='base backbone model of CutPaste')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pre-trained ResNet weights for starting backbone (True/False)')
    parser.add_argument('--backbone_path', type=str, required=True, help='Path to trained CutPaste checkpoint (.tch or .pth) file')
    parser.add_argument('--head_layer', type=int, default=2, help="Number of head layers in CutPaste model (default: 2)")
    parser.add_argument('--include_head', type=str2bool, default=True, help="Include CutPaste projection head (outputs 128-dim features) instead of ResNet18 raw features")
    parser.add_argument('--freeze_backbone', type=str2bool, default=True, help="Whether to freeze the backbone weights (default: True)")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=35, help='LR decay step size')
    parser.add_argument('--img_size', type=int, default=224, help='Image resolution')
    
    # ArcMarginProduct Hyperparameters
    parser.add_argument('--s', type=float, default=30.0, help='ArcFace scale parameter')
    parser.add_argument('--m', type=float, default=0.5, help='ArcFace margin parameter')
    parser.add_argument('--cuda', type=str2bool, default=True, help='Use GPU if available (True/False)')
    
    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--project', type=str, default='IAS-SelfSup-Alignment', help='WandB project name')
    
    args = parser.parse_args()
    args.cuda = str2bool(args.cuda)
    
    run_experiment(args)
