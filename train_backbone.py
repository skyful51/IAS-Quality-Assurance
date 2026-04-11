import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import wandb
import numpy as np

from data.dataset import get_ssl_dataloader
from models.backbone import ResNetBackbone
from models.heads import MorphologySSLHeads
from utils.visualize import visualize_morphological_transform, visualize_embeddings_ssl

def get_accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / len(target)

def train(args):
    # Setup Experiment Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_name = args.data_path.split('/')[-1]
    exp_name = f"run_{class_name}_{args.backbone}_{timestamp}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Experiment Directory: {log_dir}")

    # Setup WandB
    wandb.init(project="IAS-Morphology-SSL", name=exp_name, config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    train_loader, val_loader = get_ssl_dataloader(args.data_path, args.batch_size, args.img_size, val_split=0.1)
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples from {args.data_path}")

    # 2. Visualization before training (10 samples)
    print("Saving transformation verification images...")
    visualize_morphological_transform(train_loader.dataset, log_dir, num_samples=5)

    # 3. Model Initialization
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=True).to(device)
    heads = MorphologySSLHeads(backbone.embedding_dim, s=30.0, m=0.5).to(device)

    # 4. Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': backbone.parameters()},
        {'params': heads.parameters()}
    ], lr=args.lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    print(f"Starting Training for {args.epochs} epochs...")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        backbone.train()
        heads.train()
        
        epoch_losses = {'total': 0, 'type': 0, 'width': 0, 'height': 0, 'angle': 0}
        epoch_accs = {'type': 0, 'width': 0, 'height': 0, 'angle': 0}
        
        for i, (images, t_labels, w_labels, h_labels, a_labels) in enumerate(train_loader):
            images = images.to(device)
            t_labels, w_labels, h_labels, a_labels = t_labels.to(device), w_labels.to(device), h_labels.to(device), a_labels.to(device)

            # Backbone Embedding
            embeddings = backbone(images)

            # SSL Heads
            logits_t, logits_w, logits_h, logits_a = heads(embeddings, t_labels, w_labels, h_labels, a_labels)

            # Combined Angular Margin Loss
            loss_t = criterion(logits_t, t_labels)
            loss_w = criterion(logits_w, w_labels)
            loss_h = criterion(logits_h, h_labels)
            loss_a = criterion(logits_a, a_labels)
            
            total_loss = loss_t + loss_w + loss_h + loss_a

            # Backward and Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Stats
            epoch_losses['total'] += total_loss.item()
            epoch_losses['type'] += loss_t.item()
            epoch_losses['width'] += loss_w.item()
            epoch_losses['height'] += loss_h.item()
            epoch_losses['angle'] += loss_a.item()
            
            epoch_accs['type'] += get_accuracy(logits_t, t_labels)
            epoch_accs['width'] += get_accuracy(logits_w, w_labels)
            epoch_accs['height'] += get_accuracy(logits_h, h_labels)
            epoch_accs['angle'] += get_accuracy(logits_a, a_labels)
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

        # Average Stats
        train_stats = {f"Train/Loss_{k}": v / len(train_loader) for k, v in epoch_losses.items()}
        train_stats.update({f"Train/Acc_{k}": v / len(train_loader) for k, v in epoch_accs.items()})
        train_stats["Train/LR"] = scheduler.get_last_lr()[0]
        wandb.log(train_stats, step=epoch+1)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            backbone.eval()
            heads.eval()
            val_accs = {'type': 0, 'width': 0, 'height': 0, 'angle': 0}
            val_losses = {'total': 0, 'type': 0, 'width': 0, 'height': 0, 'angle': 0}
            
            with torch.no_grad():
                for images, t_labels, w_labels, h_labels, a_labels in val_loader:
                    images = images.to(device)
                    t_labels, w_labels, h_labels, a_labels = t_labels.to(device), w_labels.to(device), h_labels.to(device), a_labels.to(device)
                    
                    embeddings = backbone(images)
                    l_t, l_w, l_h, l_a = heads(embeddings, t_labels, w_labels, h_labels, a_labels)
                    
                    v_loss_t = criterion(l_t, t_labels)
                    v_loss_w = criterion(l_w, w_labels)
                    v_loss_h = criterion(l_h, h_labels)
                    v_loss_a = criterion(l_a, a_labels)
                    
                    val_losses['total'] += (v_loss_t + v_loss_w + v_loss_h + v_loss_a).item()
                    val_losses['type'] += v_loss_t.item()
                    val_losses['width'] += v_loss_w.item()
                    val_losses['height'] += v_loss_h.item()
                    val_losses['angle'] += v_loss_a.item()
                    
                    val_accs['type'] += get_accuracy(l_t, t_labels)
                    val_accs['width'] += get_accuracy(l_w, w_labels)
                    val_accs['height'] += get_accuracy(l_h, h_labels)
                    val_accs['angle'] += get_accuracy(l_a, a_labels)
            
            val_stats = {f"Val/Loss_{k}": v / len(val_loader) for k, v in val_losses.items()}
            val_stats.update({f"Val/Acc_{k}": v / len(val_loader) for k, v in val_accs.items()})
            wandb.log(val_stats, step=epoch+1)
            
            # Save Best Model based on overall Accuracy
            current_val_acc = (val_accs['type'] + val_accs['width'] + val_accs['height'] + val_accs['angle']) / (4 * len(val_loader))
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save(backbone.state_dict(), os.path.join(log_dir, f"best_backbone_epoch_{epoch+1}.pth"))
                torch.save(heads.state_dict(), os.path.join(log_dir, f"best_heads_epoch_{epoch+1}.pth"))
                print(f"New best model saved at epoch {epoch+1} with Accuracy: {best_val_acc:.4f}")

        # Save weights every epoch
        torch.save({
            'epoch': epoch + 1,
            'backbone_state_dict': backbone.state_dict(),
            'heads_state_dict': heads.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        # Last weights
        torch.save(backbone.state_dict(), os.path.join(log_dir, "last_backbone.pth"))
        torch.save(heads.state_dict(), os.path.join(log_dir, "last_heads.pth"))

        # Visualize embeddings every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Generating SSL t-SNE visualization for epoch {epoch+1}...")
            visualize_embeddings_ssl(backbone, val_loader, device, log_dir, epoch + 1)

        # Update Learning Rate
        scheduler.step()

    wandb.finish()
    print(f"Training finished. Models and logs saved in: {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Backbone with Morphological Proxy Tasks')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MVTec category (e.g. datasets/mvtec/bottle)')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18 or resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=50, help='Decay LR every N epochs')
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    
    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)
    
    train(args)
