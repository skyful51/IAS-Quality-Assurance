import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import pandas as pd
from models.backbone import ResNetBackbone
from models.heads import ArcMarginProduct
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2

class ArcFaceCAMWrapper(nn.Module):
    """
    Wrapper for Grad-CAM to track gradients from cosine similarity 'logits'.
    """
    def __init__(self, backbone, centroids):
        super().__init__()
        self.backbone = backbone
        self.centroids = centroids # [num_classes, dim]
        
    def forward(self, x):
        features = self.backbone(x)
        features = torch.nn.functional.normalize(features)
        # Calculate Cosine Similarity to all centroids
        logits = torch.matmul(features, self.centroids.t())
        return logits

def load_models(backbone_type, backbone_path, head_path, num_classes, device):
    """
    Loads trained backbone and head, and extracts the centroids.
    """
    # Initialize Models
    backbone = ResNetBackbone(model_name=backbone_type, pretrained=False).to(device)
    # ArcMarginProduct needs num_classes to load weights correctly
    head = ArcMarginProduct(backbone.embedding_dim, num_classes).to(device)
    
    # Load Weights
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    head.load_state_dict(torch.load(head_path, map_location=device))
    
    backbone.eval()
    head.eval()
    
    # Extract Centroids (normalized head weights)
    with torch.no_grad():
        centroids = torch.nn.functional.normalize(head.weight).detach()
        
    return backbone, centroids

def get_scores_and_cam(cam_tool, img_tensor, rgb_img):
    """
    Calculates similarity and generates Grad-CAM for the top prediction.
    """
    # 1. Forward Pass
    logits = cam_tool.model(img_tensor)
    similarities = logits.squeeze(0)
    
    # 2. Get Top Prediction
    pred_idx = torch.argmax(similarities).item()
    anomaly_score = 1 - similarities[0].item()
    
    # 3. Generate Grad-CAM for the predicted class
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_tool(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 4. Create Overlay
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return anomaly_score, similarities.detach().cpu().numpy(), pred_idx, visualization

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for Grad-CAM display (no normalization, 0~1 RGB)
    display_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print(f"Loading weights from {args.backbone_path}...")
    head_state = torch.load(args.head_path, map_location='cpu')
    num_classes = head_state['weight'].shape[0]
    
    backbone, centroids = load_models(args.backbone, args.backbone_path, args.head_path, num_classes, device)
    
    # Initialize Grad-CAM
    # For ResNet, we target the last convolutional layer
    target_layers = [backbone.model.layer4[-1]]
    cam_wrapper = ArcFaceCAMWrapper(backbone, centroids).to(device)
    cam_tool = GradCAM(model=cam_wrapper, target_layers=target_layers)
    
    # Path Logic: {dataset_root}/{class_name}/*/image
    target_root = os.path.join(args.dataset_root, args.class_name)
    cam_root = os.path.join(os.path.dirname(args.head_path), "cam")
    os.makedirs(cam_root, exist_ok=True)
    
    print(f"Searching for generated images in: {target_root}/*/image")
    print(f"Grad-CAM images will be saved to: {cam_root}")
    
    # Derive Class Names
    if os.path.exists(target_root):
        defect_names = sorted([d for d in os.listdir(target_root) 
                            if os.path.isdir(os.path.join(target_root, d)) and d != 'good'])
        class_names = ['good'] + defect_names
    else:
        class_names = []
    
    if len(class_names) != num_classes:
        print(f"Warning: Inferred {len(class_names)} classes but model has {num_classes}. Using generic names.")
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    results = []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for defect_type in os.listdir(target_root):
        defect_type_path = os.path.join(target_root, defect_type)
        if not os.path.isdir(defect_type_path): continue
        
        image_folder = os.path.join(defect_type_path, "image")
        search_path = image_folder if os.path.isdir(image_folder) else defect_type_path
        
        # Folder for this defect type CAMs
        defect_cam_dir = os.path.join(cam_root, defect_type)
        os.makedirs(defect_cam_dir, exist_ok=True)
        
        count = 0
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.lower().endswith(image_exts):
                    img_path = os.path.join(root, file)
                    
                    # Process Image
                    img_pil = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    rgb_img = np.array(display_transform(img_pil).permute(1, 2, 0).cpu().numpy())
                    
                    # Inference + Grad-CAM
                    anomaly_score, sims, pred_idx, cam_image = get_scores_and_cam(cam_tool, img_tensor, rgb_img)
                    
                    # Save Side-by-Side Image (Original | CAM)
                    img_uint8 = (rgb_img * 255).astype(np.uint8)
                    combined = cv2.hconcat([img_uint8, cam_image])
                    
                    cam_filename = f"{os.path.splitext(file)[0]}_cam.jpg"
                    cam_save_path = os.path.join(defect_cam_dir, cam_filename)
                    cv2.imwrite(cam_save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    
                    res_dict = {
                        'defect_group': defect_type,
                        'file_name': file,
                        'anomaly_score': f"{anomaly_score:.4f}",
                        'pred_class': class_names[pred_idx] if pred_idx < len(class_names) else pred_idx,
                        'cam_path': cam_save_path
                    }
                    for idx, s in enumerate(sims):
                        c_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                        res_dict[f'sim_{c_name}'] = f"{s:.4f}"
                    
                    results.append(res_dict)
                    count += 1
        print(f"Processed {count} images for {defect_type}")
    
    # 3. Output Results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='anomaly_score', ascending=False)
        output_file = os.path.join(os.path.dirname(args.head_path), f"inference_{args.class_name}_results.csv")
        df.to_csv(output_file, index=False)
        
        print(f"\n--- Inference Results for {args.class_name} (Top 10) ---")
        print(df.head(10).drop(columns=['cam_path']).to_string(index=False))
        print(f"\nFull results (including CAM paths) saved to: {output_file}")
    else:
        print(f"No images found in {target_root} patterns.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18 or resnet50')
    parser.add_argument('--backbone_path', type=str, required=True, help='Path to backbone_final.pth')
    parser.add_argument('--head_path', type=str, required=True, help='Path to head_final.pth')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root dir of generated dataset')
    parser.add_argument('--class_name', type=str, required=True, help='Class name (e.g. bottle)')
    
    args = parser.parse_args()
    run_inference(args)
