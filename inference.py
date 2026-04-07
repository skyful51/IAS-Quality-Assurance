import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import pandas as pd
from models.backbone import ResNetBackbone
from models.heads import ArcMarginProduct

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

def get_scores(backbone, centroids, image_path, transform, device):
    """
    Calculates similarity to ALL centroids.
    Returns: (anomaly_score, similarities_list, pred_class_idx)
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = torch.nn.functional.normalize(backbone(img_tensor))
        
        # Calculate similarities to all centroids
        # similarities: [1, num_classes]
        similarities = torch.matmul(embedding, centroids.t()).squeeze(0)
        
        # Anomaly Score = 1 - Similarity to 'Good' (index 0)
        anomaly_score = 1 - similarities[0].item()
        
        # Predicted Class (Max Similarity)
        pred_class_idx = torch.argmax(similarities).item()
        
    return anomaly_score, similarities.cpu().numpy(), pred_class_idx

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading weights from {args.backbone_path}...")
    head_state = torch.load(args.head_path, map_location='cpu')
    num_classes = head_state['weight'].shape[0]
    
    backbone, centroids = load_models(args.backbone, args.backbone_path, args.head_path, num_classes, device)
    
    # Path Logic: {dataset_root}/{class_name}/*/image
    target_root = os.path.join(args.dataset_root, args.class_name)
    print(f"Searching for generated images in: {target_root}/*/image")
    
    results = []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Find all 'image' subdirectories within the class root
    if not os.path.exists(target_root):
        print(f"Error: Path {target_root} does not exist.")
        return

    # 2. Derive Class Names (Matches MVTecADDataset logic: 'good' at 0, others alpha-sorted)
    # We check the directory structure to infer labels
    defect_names = sorted([d for d in os.listdir(target_root) 
                          if os.path.isdir(os.path.join(target_root, d)) and d != 'good'])
    class_names = ['good'] + defect_names
    
    if len(class_names) != num_classes:
        print(f"Warning: Inferred {len(class_names)} classes but model has {num_classes}. Using generic names.")
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    results = []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for defect_type in os.listdir(target_root):
        defect_type_path = os.path.join(target_root, defect_type)
        if not os.path.isdir(defect_type_path): continue
        
        # Check for 'image' subfolder (standard Anomaly Diffusion output)
        image_folder = os.path.join(defect_type_path, "image")
        search_path = image_folder if os.path.isdir(image_folder) else defect_type_path
        
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.lower().endswith(image_exts):
                    img_path = os.path.join(root, file)
                    anomaly_score, sims, pred_idx = get_scores(backbone, centroids, img_path, transform, device)
                    
                    res_dict = {
                        'defect_group': defect_type,
                        'file_name': file,
                        'anomaly_score': f"{anomaly_score:.4f}",
                        'pred_class': class_names[pred_idx] if pred_idx < len(class_names) else pred_idx,
                    }
                    # Add individual class similarities with REAL NAMES
                    for idx, s in enumerate(sims):
                        c_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                        res_dict[f'sim_{c_name}'] = f"{s:.4f}"
                    
                    results.append(res_dict)
    
    # 3. Output Results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='anomaly_score', ascending=False)
        output_file = os.path.join(os.path.dirname(args.head_path), f"inference_{args.class_name}_results.csv")
        df.to_csv(output_file, index=False)
        
        print(f"\n--- Inference Results for {args.class_name} (Top 10) ---")
        print(df.head(10).to_string(index=False))
        print(f"\nFull results saved to: {output_file}")
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
