import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from data.dataset import get_ssl_dataloader
from models.backbone import ResNetBackbone
from models.heads import MorphologySSLHeads
from utils.visualize import visualize_morphological_transform, visualize_embeddings_ssl

class SSLTypeHeadCAMWrapper(torch.nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        
    def forward(self, x):
        emb = self.backbone(x)
        l_t, _, _ = self.heads(emb)
        return l_t

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
        
        epoch_losses = {'total': 0, 'type': 0, 'width': 0, 'height': 0}
        epoch_accs = {'type_total': 0, 'width': 0, 'height': 0, 'type_normal': 0, 'type_defect': 0}
        epoch_counts = {'normal': 0, 'defect': 0, 'total': 0}
        
        for i, (images, id_labels, t_labels, w_labels, h_labels) in enumerate(train_loader):
            images = images.to(device)
            id_labels = id_labels.to(device)
            t_labels, w_labels, h_labels = t_labels.to(device), w_labels.to(device), h_labels.to(device)

            # Backbone Embedding
            embeddings = backbone(images)

            # SSL Heads
            logits_t, logits_w, logits_h = heads(embeddings, t_labels, w_labels, h_labels)

            # Combined Angular Margin Loss
            loss_t = criterion(logits_t, t_labels)
            loss_w = criterion(logits_w, w_labels)
            loss_h = criterion(logits_h, h_labels)
            
            total_loss = loss_t + loss_w + loss_h

            # Backward and Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Stats
            epoch_losses['total'] += total_loss.item()
            epoch_losses['type'] += loss_t.item()
            epoch_losses['width'] += loss_w.item()
            epoch_losses['height'] += loss_h.item()
            
            epoch_counts['total'] += 1
            epoch_accs['type_total'] += get_accuracy(logits_t, t_labels)
            epoch_accs['width'] += get_accuracy(logits_w, w_labels)
            epoch_accs['height'] += get_accuracy(logits_h, h_labels)
            
            with torch.no_grad():
                pred_t = logits_t.argmax(dim=1)
                norm_mask = id_labels == 0
                def_mask = id_labels > 0
                
                if norm_mask.any():
                    epoch_accs['type_normal'] += pred_t[norm_mask].eq(t_labels[norm_mask]).sum().item() / norm_mask.sum().item()
                    epoch_counts['normal'] += 1
                if def_mask.any():
                    epoch_accs['type_defect'] += pred_t[def_mask].eq(t_labels[def_mask]).sum().item() / def_mask.sum().item()
                    epoch_counts['defect'] += 1
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

        # Average Stats
        # Average Stats
        train_stats = {f"Train/Loss_{k}": v / len(train_loader) for k, v in epoch_losses.items()}
        train_stats["Train/Acc_width"] = epoch_accs['width'] / epoch_counts['total']
        train_stats["Train/Acc_height"] = epoch_accs['height'] / epoch_counts['total']
        train_stats["Train/Acc_Total_SSL"] = epoch_accs['type_total'] / epoch_counts['total']
        train_stats["Train/Acc_Normal_SSL"] = epoch_accs['type_normal'] / epoch_counts['normal'] if epoch_counts['normal'] > 0 else 0
        train_stats["Train/Acc_Defect_SSL"] = epoch_accs['type_defect'] / epoch_counts['defect'] if epoch_counts['defect'] > 0 else 0
        train_stats["Train/LR"] = scheduler.get_last_lr()[0]
        wandb.log(train_stats, step=epoch+1)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            backbone.eval()
            heads.eval()
            val_accs = {'type_total': 0, 'width': 0, 'height': 0, 'type_normal': 0, 'type_defect': 0}
            val_counts = {'normal': 0, 'defect': 0, 'total': 0}
            val_losses = {'total': 0, 'type': 0, 'width': 0, 'height': 0}
            
            with torch.no_grad():
                for images, id_labels, t_labels, w_labels, h_labels in val_loader:
                    images = images.to(device)
                    id_labels = id_labels.to(device)
                    t_labels, w_labels, h_labels = t_labels.to(device), w_labels.to(device), h_labels.to(device)
                    
                    embeddings = backbone(images)
                    l_t, l_w, l_h = heads(embeddings, t_labels, w_labels, h_labels)
                    
                    v_loss_t = criterion(l_t, t_labels)
                    v_loss_w = criterion(l_w, w_labels)
                    v_loss_h = criterion(l_h, h_labels)
                    
                    val_losses['total'] += (v_loss_t + v_loss_w + v_loss_h).item()
                    val_losses['type'] += v_loss_t.item()
                    val_losses['width'] += v_loss_w.item()
                    val_losses['height'] += v_loss_h.item()
                    
                    val_counts['total'] += 1
                    val_accs['type_total'] += get_accuracy(l_t, t_labels)
                    val_accs['width'] += get_accuracy(l_w, w_labels)
                    val_accs['height'] += get_accuracy(l_h, h_labels)
                    
                    pred_t = l_t.argmax(dim=1)
                    norm_mask = id_labels == 0
                    def_mask = id_labels > 0
                    
                    if norm_mask.any():
                        val_accs['type_normal'] += pred_t[norm_mask].eq(t_labels[norm_mask]).sum().item() / norm_mask.sum().item()
                        val_counts['normal'] += 1
                    if def_mask.any():
                        val_accs['type_defect'] += pred_t[def_mask].eq(t_labels[def_mask]).sum().item() / def_mask.sum().item()
                        val_counts['defect'] += 1
            
            val_stats = {f"Val/Loss_{k}": v / len(val_loader) for k, v in val_losses.items()}
            val_stats["Val/Acc_width"] = val_accs['width'] / val_counts['total']
            val_stats["Val/Acc_height"] = val_accs['height'] / val_counts['total']
            val_stats["Val/Acc_Total_SSL"] = val_accs['type_total'] / val_counts['total']
            val_stats["Val/Acc_Normal_SSL"] = val_accs['type_normal'] / val_counts['normal'] if val_counts['normal'] > 0 else 0
            val_stats["Val/Acc_Defect_SSL"] = val_accs['type_defect'] / val_counts['defect'] if val_counts['defect'] > 0 else 0
            
            # Save Best Model based on overall Accuracy
            current_val_acc = (val_accs['type_total'] + val_accs['width'] + val_accs['height']) / (3 * len(val_loader))
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save(backbone.state_dict(), os.path.join(log_dir, "best_backbone.pth"))
                torch.save(heads.state_dict(), os.path.join(log_dir, "best_heads.pth"))
                print(f"New best model saved at epoch {epoch+1} with Accuracy: {best_val_acc:.4f}")
                
            # Grad-CAM visualization
            print(f"Generating Grad-CAM visualization for epoch {epoch+1}...")
            wrapper = SSLTypeHeadCAMWrapper(backbone, heads)
            target_layers = [backbone.model.layer4[-1]]
            cam_tool = GradCAM(model=wrapper, target_layers=target_layers)
            
            # Fetch a sample batch
            vis_images, vis_id_labels, vis_t_labels, vis_w_labels, vis_h_labels = next(iter(val_loader))
            vis_images, vis_id_labels, vis_t_labels = vis_images.to(device), vis_id_labels.to(device), vis_t_labels.to(device)
            
            num_viz = min(4, vis_images.size(0))
            figs = []
            m_types = ['dilation', 'erosion', 'gradient']
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            
            # Allow grad for GradCAM wrapper inputs
            for v_i in range(num_viz):
                input_tensor = vis_images[v_i].unsqueeze(0).requires_grad_(True)
                target_t = vis_t_labels[v_i].item()
                
                with torch.no_grad():
                    logits = wrapper(input_tensor)
                    pred_idx = logits.argmax(dim=1).item()
                    
                targets = [ClassifierOutputTarget(pred_idx)]
                grayscale_cam = cam_tool(input_tensor=input_tensor, targets=targets)[0, :]
                
                rgb_img = input_tensor[0].detach() * std + mean
                rgb_img = rgb_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(rgb_img)
                ax[0].set_title(f"Input (Label: {vis_id_labels[v_i].item()})")
                ax[0].axis('off')
                
                ax[1].imshow(cam_image)
                ax[1].set_title(f"Grad-CAM (True {m_types[target_t]} -> Pred {m_types[pred_idx] if pred_idx < 3 else pred_idx})")
                ax[1].axis('off')
                
                plt.tight_layout()
                figs.append(wandb.Image(fig))
                plt.close(fig)
            
            val_stats["Val/GradCAM"] = figs
            wandb.log(val_stats, step=epoch+1)


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
    
    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    for class_name in class_names:
        args.data_path = f"datasets/mvtec/{class_name}"
        if not os.path.exists(args.data_path):
            print(f"Directory {args.data_path} not found. Skipping {class_name}...")
            continue
        print(f"\n" + "="*50)
        print(f"Training on {class_name}...")
        print("="*50 + "\n")
        train(args)
