import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_dataloader
from models.backbone import ResNetBackbone
from models.heads import ArcMarginProduct
from utils.visualize import visualize_embeddings, plot_center_similarity, plot_sample_to_centroid_similarity
import argparse
import os
from datetime import datetime

def train(args):
    # Setup Experiment Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"run_{args.data_path.split('/')[-1]}_{args.backbone}_{timestamp}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Experiment Directory: {log_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader (Step 1)
    dataloader = get_dataloader(args.data_path, args.batch_size, args.img_size)
    num_classes = len(dataloader.dataset.classes)
    print(f"Loaded {num_classes} classes from {args.data_path}: {dataloader.dataset.classes}")

    # 2. Model Initialization (Step 2 & 3)
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=True).to(device)
    head = ArcMarginProduct(backbone.embedding_dim, num_classes, s=30.0, m=0.5).to(device)

    # 3. Optimizer, Criterion & Scheduler (Step 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': backbone.parameters()},
        {'params': head.parameters()}
    ], lr=args.lr)
    
    # NEW: LR Scheduler (Decay every 50 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    print(f"Starting Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        total_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Step 2: Backbone Embedding Extraction
            embeddings = backbone(images)

            # Step 3 & 4: Head Pass & Margin Application
            logits = head(embeddings, labels)

            # Step 4: Loss Calculation
            loss = criterion(logits, labels)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], LR: {scheduler.get_last_lr()[0]:.6f}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        # Update Learning Rate
        scheduler.step()

        # Visualize results every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Saving visualizations for Epoch {epoch+1}...")
            # Visualize Centroid Similarity
            plot_center_similarity(
                head, 
                dataloader.dataset.classes, 
                log_dir, 
                epoch + 1
            )

            # Sample-to-Centroid Similarity Heatmap
            plot_sample_to_centroid_similarity(
                backbone, 
                head, 
                dataloader, 
                device, 
                dataloader.dataset.classes, 
                log_dir, 
                epoch + 1
            )

    # 5. Visualization & Saving
    print("Training finished. Generating final visualization...")
    visualize_embeddings(backbone, head, dataloader, device, log_dir, args.epochs)

    # Save Checkpoints in Log Dir
    torch.save(backbone.state_dict(), os.path.join(log_dir, "backbone_final.pth"))
    torch.save(head.state_dict(), os.path.join(log_dir, "head_final.pth"))
    print(f"Models and visualization saved in: {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to MVTec category (e.g. data/bottle)')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18 or resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=50, help='Decay LR every N epochs')
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    
    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)
    
    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    for class_name in class_names:
        args.data_path = f"datasets/mvtec/{class_name}"
        print(f"Training on {class_name}...")
        train(args)
