import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import MVTecTestDataset
from models.backbone import CutPasteBackbone

class BaselineClassifier(nn.Module):
    def __init__(self, backbone_name="resnet18", num_classes=5):
        super(BaselineClassifier, self).__init__()
        # Load unfrozen backbone with ImageNet pretrained weights (pretrained=True is default)
        self.backbone = CutPasteBackbone(backbone_name=backbone_name, include_head=False, pretrained=True)
        
        if backbone_name == "resnet18":
            feature_dim = 512
        elif backbone_name == "resnet50":
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")
            
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handlers = []
        
        # Register forward and backward hooks to capture intermediate maps
        self.handlers.append(target_layer.register_forward_hook(self._save_activations))
        self.handlers.append(target_layer.register_full_backward_hook(self._save_gradients))
        
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        
        # Backward on the target class logit
        loss = output[0, class_idx]
        loss.backward()
        
        # Compute weights by Global Average Pooling the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [1, C, 1, 1]
        
        # Apply weighted sum and ReLU
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [1, 1, H, W]
        cam = nn.functional.relu(cam)
        
        # Squeeze down to 2D
        cam = cam.squeeze(0).squeeze(0) # [H, W]
        return cam
        
    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()

def save_epoch_gradcam(model, epoch, data_loader, save_dir="logs/baseline_cam"):
    # Ensure target output directory exists
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    dataset = data_loader.dataset
    
    # 1. Identify target layer for Grad-CAM
    if hasattr(model.backbone, 'resnet50'):
        target_layer = model.backbone.resnet50.layer4[-1]
    else:
        target_layer = model.backbone.resnet18.layer4[-1]
        
    # 2. Temporarily set model to evaluation mode
    model.eval()
    
    # Standard normalizer statistics to reverse-normalize back to original RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Register hooks
    grad_cam = GradCAM(model, target_layer)
    
    for idx in range(len(dataset)):
        label = dataset.labels[idx]
        # Get raw data and reconstruct original image
        img_tensor, _ = dataset[idx] # [3, H, W]
        
        # De-normalize tensor back to RGB for visualization
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0) # [H, W, 3]
        img_orig = (std * img_np) + mean
        img_orig = np.clip(img_orig, 0.0, 1.0)
        
        # Add batch dimension and transfer to device
        img_input = img_tensor.unsqueeze(0).to(device)
        img_input.requires_grad = True # Required for backward hook
        
        # Extract activation map
        cam = grad_cam(img_input, class_idx=label)
        cam_np = cam.cpu().numpy()
        
        # Normalize Grad-CAM map to [0, 1] range
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)
            
        # Resize CAM to match original image size (224, 224)
        # Using PIL to resize the heatmap to original dimensions
        cam_pil = Image.fromarray((cam_normalized * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        cam_resized = np.array(cam_pil) / 255.0
        
        # Generate Jet colormap image
        colormap = plt.get_cmap('jet')
        cam_colored = colormap(cam_resized)[:, :, :3] # [H, W, 3]
        
        # Overlay original image and heatmap (0.5 Alpha blending)
        overlayed = 0.5 * img_orig + 0.5 * cam_colored
        
        # Concat original image and overlay image side-by-side
        concat_img = np.concatenate([img_orig, overlayed], axis=1)
        concat_uint8 = (concat_img * 255).astype(np.uint8)
        
        # Save composite image
        class_name = dataset.idx_to_class[label]
        original_path = dataset.image_paths[idx]
        file_name = os.path.basename(original_path)
        base_name = os.path.splitext(file_name)[0]
        
        save_path = os.path.join(epoch_dir, f"{class_name}_{base_name}.png")
        Image.fromarray(concat_uint8).save(save_path)
        
    # 4. Clean up hooks and restore training mode
    grad_cam.remove_hooks()
    model.train()

def main():
    parser = argparse.ArgumentParser(description="Train a baseline CNN classifier on MVTec AD category test set.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MVTec AD category folder (e.g. datasets/mvtec/bottle)")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Backbone architecture (resnet18 or resnet50)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    args = parser.parse_args()

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transform: resize and normalize as standard ImageNet inputs
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & DataLoader
    dataset = MVTecTestDataset(args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} images from {args.data_path}")
    print(f"Number of classes: {num_classes} ({dataset.classes})")

    # Model
    model = BaselineClassifier(backbone_name=args.backbone, num_classes=num_classes)
    model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} (LR: {current_lr:.4f})")
        
        # Trigger visual tracking at specific epoch milestones
        if epoch in [1, 10, 30, 50]:
            save_epoch_gradcam(model, epoch, dataloader)
            
        scheduler.step()

    # Save Model Weights
    save_path = f"baseline_{args.backbone}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Weights saved to {save_path}")

if __name__ == "__main__":
    main()
