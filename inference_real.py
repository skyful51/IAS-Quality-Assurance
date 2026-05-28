import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import pandas as pd
from models.backbone import ResNetBackbone, CutPasteBackbone
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

def load_models(args, num_classes, device):
    """
    Loads trained backbone and head, and extracts the centroids.
    """
    # Initialize Models
    if args.backbone == 'cutpaste':
        backbone = CutPasteBackbone(
            pretrained=False,
            head_layers=args.head_layer,
            include_head=args.include_head,
            checkpoint_path=args.backbone_path
        ).to(device)
    else:
        backbone = ResNetBackbone(model_name=args.backbone, pretrained=False).to(device)
        backbone.load_state_dict(torch.load(args.backbone_path, map_location=device))
        
    # ArcMarginProduct needs num_classes to load weights correctly
    head = ArcMarginProduct(backbone.embedding_dim, num_classes).to(device)
    head.load_state_dict(torch.load(args.head_path, map_location=device))
    
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
    
    print(f"Loading weights...")
    head_state = torch.load(args.head_path, map_location='cpu')
    num_classes = head_state['weight'].shape[0]
    
    backbone, centroids = load_models(args, num_classes, device)
    
    # Initialize Grad-CAM
    # Target the last convolutional layer of the ResNet part
    if args.backbone == 'cutpaste':
        target_layers = [backbone.resnet18.layer4[-1]]
    else:
        target_layers = [backbone.model.layer4[-1]]
    cam_wrapper = ArcFaceCAMWrapper(backbone, centroids).to(device)
    cam_tool = GradCAM(model=cam_wrapper, target_layers=target_layers)
    
    # Path Logic for Real MVTec: {dataset_root}/{class_name}/test
    test_root = os.path.join(args.dataset_root, args.class_name, 'test')
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"MVTec test folder not found at: {test_root}")
        
    cam_root = os.path.join(os.path.dirname(args.head_path), "cam_real")
    os.makedirs(cam_root, exist_ok=True)
    
    # Get sorted class names (good is 0, rest are sorted alphabetically)
    defect_names = sorted([d for d in os.listdir(test_root) 
                           if os.path.isdir(os.path.join(test_root, d)) and d != 'good'])
    class_names = ['good'] + defect_names
    
    if len(class_names) != num_classes:
        print(f"Warning: Found {len(class_names)} folders in test set, but model has {num_classes} classes.")
        if len(class_names) < num_classes:
            class_names += [f"class_{i}" for i in range(len(class_names), num_classes)]
        else:
            class_names = class_names[:num_classes]
            
    print(f"Detected class mappings: {class_names}")
    
    results = []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    correct_predictions = 0
    total_predictions = 0
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}
    
    # Iterate through each folder inside the test directory
    for defect_type in os.listdir(test_root):
        defect_type_path = os.path.join(test_root, defect_type)
        if not os.path.isdir(defect_type_path):
            continue
            
        defect_cam_dir = os.path.join(cam_root, defect_type)
        os.makedirs(defect_cam_dir, exist_ok=True)
        
        count = 0
        for file in os.listdir(defect_type_path):
            if file.lower().endswith(image_exts):
                img_path = os.path.join(defect_type_path, file)
                
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
                
                pred_class_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
                
                # Track classification accuracy (Ground truth is defect_type)
                is_correct = (pred_class_name == defect_type)
                if is_correct:
                    correct_predictions += 1
                    if defect_type in class_correct:
                        class_correct[defect_type] += 1
                if defect_type in class_total:
                    class_total[defect_type] += 1
                total_predictions += 1
                
                res_dict = {
                    'actual_class': defect_type,
                    'file_name': file,
                    'anomaly_score': f"{anomaly_score:.4f}",
                    'pred_class': pred_class_name,
                    'is_correct': is_correct,
                    'cam_path': cam_save_path
                }
                for idx, s in enumerate(sims):
                    c_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                    res_dict[f'sim_{c_name}'] = f"{s:.4f}"
                
                results.append(res_dict)
                count += 1
        print(f"Processed {count} images for real class '{defect_type}'")
        
    # Output Results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='anomaly_score', ascending=False)
        output_file = os.path.join(os.path.dirname(args.head_path), f"inference_{args.class_name}_real_results.csv")
        df.to_csv(output_file, index=False)
        
        overall_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        print("\n" + "="*60)
        print(f" REAL MVTEC EVALUATION SUMMARY FOR '{args.class_name.upper()}' ")
        print("="*60)
        print(f"Overall Classification Accuracy: {overall_acc:.4f} ({correct_predictions}/{total_predictions})")
        print("\nClass-specific Accuracy:")
        for c in class_names:
            if class_total.get(c, 0) > 0:
                c_acc = class_correct[c] / class_total[c]
                print(f"  - {c:15s}: {c_acc:.4f} ({class_correct[c]}/{class_total[c]})")
            else:
                print(f"  - {c:15s}: No samples found in test directory")
        print("="*60)
        
        print(f"\n--- Top 10 Predictions by Anomaly Score ---")
        print(df.head(10).drop(columns=['cam_path']).to_string(index=False))
        print(f"\nFull results (including CAM paths) saved to: {output_file}")
    else:
        print(f"No images found in {test_root} matching test folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['cutpaste', 'resnet18', 'resnet50'], help='backbone type')
    parser.add_argument('--backbone_path', type=str, required=True, help='Path to backbone_final.pth or CutPaste checkpoint')
    parser.add_argument('--head_path', type=str, required=True, help='Path to head_final.pth')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root dir of real MVTec dataset')
    parser.add_argument('--class_name', type=str, required=True, help='Class name (e.g. bottle)')
    parser.add_argument('--head_layer', type=int, default=2, help="Number of head layers in CutPaste model (default: 2)")
    parser.add_argument('--include_head', action='store_true', help="Include CutPaste projection head (outputs 128-dim features) instead of ResNet18 raw features (512-dim)")
    
    args = parser.parse_args()
    run_inference(args)
