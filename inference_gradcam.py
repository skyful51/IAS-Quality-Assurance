import os
import argparse
import random
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import wandb

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.backbone import ResNetBackbone
from models.heads import MorphologySSLHeads

class SSLTypeHeadCAMWrapper(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        
    def forward(self, x):
        emb = self.backbone(x)
        l_t, _, _ = self.heads(emb)
        return l_t

def apply_morphology(img_cv, t_type, w, h):
    kernel = np.ones((h, w), np.uint8)
    if t_type == 'dilation':
        return cv2.dilate(img_cv, kernel, iterations=1)
    elif t_type == 'erosion':
        return cv2.erode(img_cv, kernel, iterations=1)
    elif t_type == 'gradient':
        return cv2.morphologyEx(img_cv, cv2.MORPH_GRADIENT, kernel)
    return img_cv

def process_pair(args, source_class, target_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Basic path validation
    target_good_dir = os.path.join("datasets/mvtec", target_class, "train", "good")
    if not os.path.exists(target_good_dir):
        print(f"Directory {target_good_dir} not found.")
        return
        
    image_files = [f for f in os.listdir(target_good_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {target_good_dir}.")
        return
    
    # Pick exactly one random image
    random.seed(42)
    target_filename = random.choice(image_files)
    target_filepath = os.path.join(target_good_dir, target_filename)
    print(f"[{source_class} -> {target_class}] Selected Base Image: {target_filepath}")

    # Set up WandB
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"GradCAM_{source_class}_to_{target_class}_{timestamp}"
    wandb.init(project="IAS-Morphology-SSL", name=run_name, config=args)
    
    # 2. Get the latest logs for the source class
    log_dirs = []
    if os.path.exists(args.weights_dir):
        for d in os.listdir(args.weights_dir):
            if d.startswith(f"run_{source_class}_{args.backbone}_"):
                log_dirs.append(os.path.join(args.weights_dir, d))
    
    if not log_dirs:
        print(f"No logs found for {source_class}.")
        wandb.finish()
        return
        
    log_dirs.sort(reverse=True)
    latest_log_dir = log_dirs[0]
    backbone_path = os.path.join(latest_log_dir, "best_backbone.pth")
    heads_path = os.path.join(latest_log_dir, "best_heads.pth")
    
    if not os.path.exists(backbone_path) or not os.path.exists(heads_path):
        print(f"Missing weights for {source_class} in {latest_log_dir}.")
        wandb.finish()
        return
    
    # 3. Load Models
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=False).to(device)
    heads = MorphologySSLHeads(backbone.embedding_dim, s=30.0, m=0.5).to(device)
    
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    heads.load_state_dict(torch.load(heads_path, map_location=device))
    
    backbone.eval()
    heads.eval()
    
    wrapper = SSLTypeHeadCAMWrapper(backbone, heads)
    target_layers = [backbone.model.layer4[-1]]
    cam_tool = GradCAM(model=wrapper, target_layers=target_layers)

    # 4. Prepare Transforms
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    display_transform = transforms.Compose([transforms.ToTensor()])
    
    base_pil = Image.open(target_filepath).convert('RGB')
    base_pil = base_pil.resize((224, 224))
    
    base_np = np.array(base_pil)
    base_cv = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR)

    m_types = ['dilation', 'erosion', 'gradient']
    m_widths = [3, 7, 11, 13]
    m_heights = [3, 7, 11, 13]
    
    def process_and_log(img_cv, log_name, title):
        trans_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        input_tensor = tensor_transform(trans_pil).unsqueeze(0).to(device)
        rgb_img = np.array(display_transform(trans_pil).permute(1, 2, 0).cpu().numpy())
        
        with torch.no_grad():
            logits = wrapper(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam_tool(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(trans_pil)
        ax[0].set_title(f"Input ({title})")
        ax[0].axis('off')
        
        ax[1].imshow(cam_image)
        ax[1].set_title(f"Grad-CAM (Pred Type: {m_types[pred_idx] if pred_idx < 3 else pred_idx})")
        ax[1].axis('off')
        
        plt.tight_layout()
        wandb.log({log_name: wandb.Image(fig)})
        plt.close()

    for t_type in m_types:
        for w in m_widths:
            for h in m_heights:
                trans_cv = apply_morphology(base_cv, t_type, w, h)
                title = f"{t_type} w={w} h={h}"
                log_name = f"GradCAM/{t_type}/w{w}_h{h}"
                process_and_log(trans_cv, log_name, title)
                
    all_morph_cv = base_cv.copy()
    all_morph_cv = apply_morphology(all_morph_cv, 'dilation', 7, 7)
    all_morph_cv = apply_morphology(all_morph_cv, 'erosion', 7, 7)
    all_morph_cv = apply_morphology(all_morph_cv, 'gradient', 7, 7)
    process_and_log(all_morph_cv, "GradCAM/all_transforms_applied", "All Sequential Morph")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Morphology GradCAM visualizer")
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weights_dir', type=str, default='logs')
    args = parser.parse_args()
    
    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    for source_class in class_names:
        for target_class in class_names:
            if source_class == target_class:
                continue
            print(f"\n{'='*50}\nEvaluating Source: {source_class} --> Target: {target_class}\n{'='*50}")
            process_pair(args, source_class, target_class)
            
    print("All combinations executed successfully.")
