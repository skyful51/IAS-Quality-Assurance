import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_dataloader
from models.backbone import ResNetBackbone
from models.heads import ArcMarginProduct
import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader (Step 1)
    dataloader = get_dataloader(args.data_path, args.batch_size, args.img_size)
    num_classes = len(dataloader.dataset.classes)
    print(f"Loaded {num_classes} classes from {args.data_path}")

    # 2. Model Initialization (Step 2 & 3)
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=True).to(device)
    head = ArcMarginProduct(backbone.embedding_dim, num_classes, s=30.0, m=0.5).to(device)

    # 3. Optimizer & Criterion (Step 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': backbone.parameters()},
        {'params': head.parameters()}
    ], lr=args.lr)

    print("Starting Training...")
    
    backbone.train()
    head.train()

    for epoch in range(args.epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Step 2: Backbone Embedding Extraction
            # Input: [B, 3, 224, 224] -> Output: [B, 512]
            embeddings = backbone(images)

            # Step 3 & 4: Head Pass & Margin Application
            # Logits are calculated with margin applied to target class internally
            logits = head(embeddings, labels)

            # Step 4: Loss Calculation
            loss = criterion(logits, labels)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

    # Save Checkpoints
    torch.save(backbone.state_dict(), "backbone_final.pth")
    torch.save(head.state_dict(), "head_final.pth")
    print("Training Finished and Models Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/mvtec_toy', help='Path to dataset')
    parser.add_argument('--backbone', type=str, default='resnet18', help='resnet18 or resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    # Simple check for path
    # if not os.path.exists(args.data_path):
    #     print(f"Creating dummy data structure at {args.data_path} for demonstration.")
    #     os.makedirs(os.path.join(args.data_path, "good"), exist_ok=True)
    #     os.makedirs(os.path.join(args.data_path, "crack"), exist_ok=True)
    #     os.makedirs(os.path.join(args.data_path, "scratch"), exist_ok=True)

    #     dummy_img = Image.new('RGB', (args.img_size, args.img_size), color = 'red')
    #     for i in range(10):
    #         dummy_img.save(os.path.join(args.data_path, "good", f"img_{i:03d}.png"))
    #         dummy_img.save(os.path.join(args.data_path, "crack", f"img_{i:03d}.png"))
    #         dummy_img.save(os.path.join(args.data_path, "scratch", f"img_{i:03d}.png"))

    train(args)
