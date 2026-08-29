import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import warnings

# Custom module import
from models.backbone import ResNetBackbone

warnings.filterwarnings("ignore")

ALL_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MVTecMaskedTestDataset(Dataset):
    """Dataset loader for real MVTec AD test/gt set."""
    def __init__(self, category_root, img_transform=None, mask_transform=None):
        self.category_root = category_root
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.is_good = []
        
        test_dir = os.path.join(category_root, 'test')
        gt_dir = os.path.join(category_root, 'ground_truth')
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found at {test_dir}")
            
        self.class_to_idx = {'good': 0}
        
        # Load Normal (good) images
        test_good_dir = os.path.join(test_dir, 'good')
        if os.path.exists(test_good_dir):
            for img_name in sorted(os.listdir(test_good_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(test_good_dir, img_name))
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.is_good.append(True)
                    
        # Load Defect types
        defect_types = sorted([d for d in os.listdir(test_dir) 
                             if os.path.isdir(os.path.join(test_dir, d)) and d != 'good'])
        
        for idx, d_type in enumerate(defect_types):
            class_idx = idx + 1
            self.class_to_idx[d_type] = class_idx
            defect_img_dir = os.path.join(test_dir, d_type)
            defect_gt_dir = os.path.join(gt_dir, d_type) if os.path.exists(gt_dir) else None
            
            for img_name in sorted(os.listdir(defect_img_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(defect_img_dir, img_name)
                    base_name = os.path.splitext(img_name)[0]
                    
                    mask_path = None
                    if defect_gt_dir and os.path.exists(defect_gt_dir):
                        possible_masks = [
                            os.path.join(defect_gt_dir, f"{base_name}_mask.png"),
                            os.path.join(defect_gt_dir, f"{base_name}.png"),
                            os.path.join(defect_gt_dir, f"{base_name}_mask.jpg")
                        ]
                        for pm in possible_masks:
                            if os.path.exists(pm):
                                mask_path = pm
                                break
                                
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(class_idx)
                    self.is_good.append(False)
                    
        self.num_classes = len(self.class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(self.num_classes)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        is_good = self.is_good[idx]
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image) if self.img_transform else transforms.ToTensor()(image)
            
        if is_good or mask_path is None:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            mask_tensor = self.mask_transform(mask) if self.mask_transform else transforms.ToTensor()(mask)
            mask_tensor = (mask_tensor > 0.5).float()
            
        return image_tensor, mask_tensor, label, is_good, idx


class SyntheticAnomalyDataset(Dataset):
    """Generic dataset loader for synthetic anomaly datasets (realnet_paired, anomaly_diffusion)."""
    def __init__(self, syn_cat_dir, class_to_idx, img_transform=None, mask_transform=None):
        self.syn_cat_dir = syn_cat_dir
        self.class_to_idx = class_to_idx
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []
        self.file_names = []
        
        if not os.path.exists(syn_cat_dir):
            return
            
        subdirs = sorted([d for d in os.listdir(syn_cat_dir) if os.path.isdir(os.path.join(syn_cat_dir, d))])
        
        for d_type in subdirs:
            class_idx = class_to_idx.get(d_type, -1)
            img_dir = os.path.join(syn_cat_dir, d_type, 'image')
            mask_dir = os.path.join(syn_cat_dir, d_type, 'mask')
            
            if not os.path.exists(img_dir):
                continue
                
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(img_dir, fname)
                    mask_path = os.path.join(mask_dir, fname) if os.path.exists(mask_dir) else None
                    if mask_path and os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        self.labels.append(class_idx)
                        self.defect_types.append(d_type)
                        self.file_names.append(fname)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        file_name = self.file_names[idx]
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image) if self.img_transform else transforms.ToTensor()(image)
            
        mask = Image.open(mask_path).convert('L')
        mask_tensor = self.mask_transform(mask) if self.mask_transform else transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor, label, defect_type, file_name, idx


def extract_masked_embeddings(backbone, images, masks, device):
    """Extracts Masked Average Pooled (MAP) feature embeddings at layer4 of ResNet."""
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        resnet = backbone.model
        x = resnet.conv1(images)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)
        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        feature_map = resnet.layer4(x)  # [B, C, H_f, W_f]
        
        B, C, H_f, W_f = feature_map.shape
        mask_resized = F.interpolate(masks, size=(H_f, W_f), mode='bilinear', align_corners=False) # [B, 1, H_f, W_f]
        
        spatial_sum = (feature_map * mask_resized).sum(dim=(2, 3)) # [B, C]
        mask_sum = mask_resized.sum(dim=(2, 3)) # [B, 1]
        
        zero_mask = (mask_sum < 1e-6)
        mask_sum_clamped = torch.clamp(mask_sum, min=1e-8)
        
        embeddings = spatial_sum / mask_sum_clamped # [B, C]
        if zero_mask.any():
            gap_embeddings = feature_map.mean(dim=(2, 3))
            embeddings[zero_mask.squeeze(1)] = gap_embeddings[zero_mask.squeeze(1)]
            
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
    return normalized_embeddings


def compute_masked_centroids(backbone, dataset, indices, num_classes, device, seed=42):
    """Computes real class centroids from 60% real training split."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=2)
    
    backbone.eval()
    class_feats = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for images, masks, targets, is_goods, _ in loader:
            embeddings = extract_masked_embeddings(backbone, images, masks, device)
            for i in range(images.size(0)):
                label = targets[i].item()
                class_feats[label].append(embeddings[i])
                
    centroids = []
    for c in range(num_classes):
        if len(class_feats[c]) > 0:
            c_tensor = torch.stack(class_feats[c], dim=0) # [N_c, C]
            c_mean = c_tensor.mean(dim=0)
            c_centroid = F.normalize(c_mean, p=2, dim=0) # [C]
            centroids.append(c_centroid)
        else:
            feat_dim = backbone.embedding_dim
            centroids.append(torch.zeros(feat_dim, device=device))
            
    return torch.stack(centroids, dim=0) # [num_classes, C]


def evaluate_category_similarity(cat, algo, mvtec_root, syn_root, weights_dir, device):
    """Evaluates Masked Cosine Similarity for a single category and algorithm."""
    cat_dir = os.path.join(mvtec_root, cat)
    syn_cat_dir = os.path.join(syn_root, algo, cat)
    
    if not os.path.exists(cat_dir) or not os.path.exists(syn_cat_dir):
        return None
        
    # Transforms
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    # Load Model Backbone
    backbone = ResNetBackbone(model_name="resnet18", pretrained=False).to(device)
    weights_path = os.path.join(weights_dir, f"{cat}.pth")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k.replace('model.', '')] = v
            else:
                clean_state_dict[k] = v
        backbone.model.load_state_dict(clean_state_dict, strict=False)
    backbone.eval()
    
    # Real dataset & 60% split
    real_dataset = MVTecMaskedTestDataset(cat_dir, img_transform=img_transform, mask_transform=mask_transform)
    num_samples = len(real_dataset)
    indices = list(range(num_samples))
    set_seed(42)
    random.shuffle(indices)
    train_size = int(0.6 * num_samples)
    train_indices = indices[:train_size]
    
    # Real Centroids
    centroids = compute_masked_centroids(backbone, real_dataset, train_indices, real_dataset.num_classes, device).cpu()
    
    # Synthetic Dataset
    syn_dataset = SyntheticAnomalyDataset(syn_cat_dir, class_to_idx=real_dataset.class_to_idx, img_transform=img_transform, mask_transform=mask_transform)
    if len(syn_dataset) == 0:
        return None
        
    syn_loader = DataLoader(syn_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Evaluate similarities
    target_sims = []
    good_sims = []
    quality_scores = []
    high_q_flags = []
    
    with torch.no_grad():
        for images, masks, targets, defect_types, file_names, idxs in syn_loader:
            embs = extract_masked_embeddings(backbone, images, masks, device).cpu()
            sims = torch.matmul(embs, centroids.t()).numpy() # [B, num_classes]
            
            for i in range(images.size(0)):
                t_label = targets[i].item()
                if t_label >= 1:
                    s_target = sims[i, t_label]
                else:
                    # Generic combined defect: match to max defect centroid (c >= 1)
                    s_target = np.max(sims[i, 1:])
                    
                s_good = sims[i, 0]
                q_score = s_target - s_good
                is_high_q = (s_target >= 0.6) and (s_target > s_good)
                
                target_sims.append(s_target)
                good_sims.append(s_good)
                quality_scores.append(q_score)
                high_q_flags.append(is_high_q)
                
    mean_target_sim = float(np.mean(target_sims))
    mean_good_sim = float(np.mean(good_sims))
    mean_quality_score = float(np.mean(quality_scores))
    high_q_ratio = float(np.mean(high_q_flags)) * 100.0
    
    return {
        "algorithm": algo,
        "category": cat,
        "num_samples": len(syn_dataset),
        "target_similarity": mean_target_sim,
        "good_leakage": mean_good_sim,
        "net_quality_score": mean_quality_score,
        "high_quality_ratio_pct": high_q_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="Feature-Level Masked Similarity Benchmark")
    parser.add_argument("--mvtec_root", type=str, default="datasets/mvtec")
    parser.add_argument("--syn_root", type=str, default="datasets/generated_dataset")
    parser.add_argument("--weights_dir", type=str, default="logs/resnet18_baseline_0819")
    parser.add_argument("--algorithms", nargs="+", default=["realnet_paired", "anomaly_diffusion"])
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="mask_similarity_benchmark_results.csv")
    args = parser.parse_args()
    
    target_categories = ALL_MVTEC_CATEGORIES if "all" in args.categories else args.categories
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"MVTec Root     : {args.mvtec_root}")
    print(f"Synthetic Root : {args.syn_root}")
    print(f"Weights Dir    : {args.weights_dir}")
    print(f"Algorithms     : {args.algorithms}")
    print(f"Categories ({len(target_categories)}): {target_categories}")
    print(f"Device         : {device}\n")
    
    all_records = []
    summary_data = []
    
    for algo in args.algorithms:
        print(f"==================================================")
        print(f"   Evaluating Masked Similarity: {algo}")
        print(f"==================================================")
        algo_records = []
        for cat in target_categories:
            res = evaluate_category_similarity(cat, algo, args.mvtec_root, args.syn_root, args.weights_dir, device)
            if res:
                algo_records.append(res)
                print(f"Category: {cat:12s} | TargetSim: {res['target_similarity']:.4f} | GoodLeak: {res['good_leakage']:.4f} | NetQuality (Q): {res['net_quality_score']:.4f} | HighQ%: {res['high_quality_ratio_pct']:.1f}%")
            else:
                print(f"Category: {cat:12s} | Skipped (Not found)")
                
        if algo_records:
            df_algo = pd.DataFrame(algo_records)
            all_records.append(df_algo)
            m_target_sim = df_algo["target_similarity"].mean()
            m_good_leak = df_algo["good_leakage"].mean()
            m_quality = df_algo["net_quality_score"].mean()
            m_high_q = df_algo["high_quality_ratio_pct"].mean()
            
            summary_data.append({
                "algorithm": algo,
                "mTarget_Similarity": m_target_sim,
                "mGood_Leakage": m_good_leak,
                "mSimilarity (Net Quality Q)": m_quality,
                "mHigh_Quality_Ratio (%)": m_high_q,
                "categories_evaluated": len(df_algo)
            })
            print(f"\n---> {algo} Macro-Average mSimilarity (Net Quality Q): {m_quality:.4f} | TargetSim: {m_target_sim:.4f} | HighQ%: {m_high_q:.1f}%\n")

    if all_records:
        df_all = pd.concat(all_records, ignore_index=True)
        df_all.to_csv(args.output_csv, index=False)
        print(f"\nResults exported to {args.output_csv}")
        
        print("\n==================================================")
        print("     FINAL MASKED SIMILARITY BENCHMARK SUMMARY    ")
        print("==================================================")
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        pivot_df = df_all.pivot(index="category", columns="algorithm", values="net_quality_score")
        print("\nCategory-level Net Quality Score (Q = TargetSim - GoodLeak) Comparison:")
        print(pivot_df.to_string())

if __name__ == "__main__":
    main()
