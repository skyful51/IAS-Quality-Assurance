import os
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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Import custom modules
from models.backbone import ResNetBackbone

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MVTecMaskedTestDataset(Dataset):
    """
    Dataset loader for real MVTec AD dataset to compute reference class centroids.
    """
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
        
        # 1. Load Normal (good) images
        test_good_dir = os.path.join(test_dir, 'good')
        if os.path.exists(test_good_dir):
            for img_name in sorted(os.listdir(test_good_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(test_good_dir, img_name))
                    self.mask_paths.append(None)
                    self.labels.append(0)
                    self.is_good.append(True)
                    
        # 2. Collect all available defect mask paths for virtual mask sampling on good images
        self.all_defect_mask_paths = []
        
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
                                self.all_defect_mask_paths.append(pm)
                                break
                                
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(class_idx)
                    self.is_good.append(False)
                    
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        is_good = self.is_good[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image_tensor = self.img_transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        if is_good or mask_path is None:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask_tensor = self.mask_transform(mask)
            else:
                mask_tensor = transforms.ToTensor()(mask)
            mask_tensor = (mask_tensor > 0.5).float()
            
        return image_tensor, mask_tensor, label, is_good, idx

    def load_mask_from_path(self, mask_path):
        """Helper to load and binarize a mask from a given path."""
        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = transforms.ToTensor()(mask)
        return (mask_tensor > 0.5).float()


class AnomalyDiffusionDataset(Dataset):
    """
    Dataset loader for synthetic AnomalyDiffusion dataset.
    
    Structure:
    syn_category_root/
        defect_type_1/
            image/  -> 1.jpg, 2.jpg...
            mask/   -> 1.jpg, 2.jpg...
        defect_type_2/
            ...
    """
    def __init__(self, syn_category_root, class_to_idx, img_transform=None, mask_transform=None):
        self.syn_category_root = syn_category_root
        self.class_to_idx = class_to_idx
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []
        self.file_names = []
        
        if not os.path.exists(syn_category_root):
            raise FileNotFoundError(f"Synthetic dataset directory not found at {syn_category_root}")
            
        subdirs = sorted([d for d in os.listdir(syn_category_root) if os.path.isdir(os.path.join(syn_category_root, d))])
        
        for d_type in subdirs:
            if d_type in class_to_idx:
                class_idx = class_to_idx[d_type]
            else:
                # Generic synthetic defect category (e.g. 'combined', 'sdas')
                # Map to -1 indicating evaluated against all defect centroids
                class_idx = -1
                
            img_dir = os.path.join(syn_category_root, d_type, 'image')
            mask_dir = os.path.join(syn_category_root, d_type, 'mask')
            
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
                    else:
                        print(f"[Warning] Mask missing for synthetic image: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        file_name = self.file_names[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image_tensor = self.img_transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor, label, defect_type, file_name, idx


def extract_masked_embeddings(backbone, images, masks, device):
    """
    Extract feature maps from backbone and perform feature-map level spatial masked mean pooling.
    """
    images = images.to(device)
    masks = masks.to(device)
    
    resnet = backbone.model
    x = resnet.conv1(images)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    feature_map = resnet.layer4(x)  # Shape: [B, C, H_f, W_f]
    
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


def compute_masked_centroids(backbone, dataset, train_indices, num_classes, num_restarts, device, seed=42):
    """
    Compute class centroids on the training split of real MVTec dataset.
    Identical logic to eval-mask-similarity.py.
    """
    backbone.eval()
    class_embeddings = {c: [] for c in range(num_classes)}
    
    train_subset = Subset(dataset, train_indices)
    dataloader = DataLoader(train_subset, batch_size=16, shuffle=False, num_workers=2)
    
    defect_mask_paths = dataset.all_defect_mask_paths
    
    with torch.no_grad():
        for images, masks, labels, is_goods, idxs in dataloader:
            defect_mask = (~is_goods)
            if defect_mask.any():
                def_imgs = images[defect_mask]
                def_msks = masks[defect_mask]
                def_lbls = labels[defect_mask]
                
                embs = extract_masked_embeddings(backbone, def_imgs, def_msks, device)
                for emb, lbl in zip(embs, def_lbls):
                    class_embeddings[lbl.item()].append(emb.cpu())
                    
            good_mask_sample = is_goods
            if good_mask_sample.any():
                good_imgs = images[good_mask_sample]
                num_good = good_imgs.size(0)
                
                good_restart_embs = []
                for r in range(num_restarts):
                    rng = random.Random(seed + r * 1000)
                    sampled_mask_paths = [rng.choice(defect_mask_paths) for _ in range(num_good)]
                    
                    batch_masks = []
                    for mp in sampled_mask_paths:
                        m_tensor = dataset.load_mask_from_path(mp)
                        batch_masks.append(m_tensor)
                    batch_masks = torch.stack(batch_masks)
                    
                    embs_r = extract_masked_embeddings(backbone, good_imgs, batch_masks, device)
                    good_restart_embs.append(embs_r.cpu())
                    
                avg_good_embs = torch.stack(good_restart_embs, dim=0).mean(dim=0)
                avg_good_embs = F.normalize(avg_good_embs, p=2, dim=1)
                
                for emb in avg_good_embs:
                    class_embeddings[0].append(emb)
                    
    centroids = []
    for c in range(num_classes):
        embs = class_embeddings[c]
        if len(embs) == 0:
            raise ValueError(f"No training samples found for class label {c}!")
        embs_tensor = torch.stack(embs)
        mean_emb = embs_tensor.mean(dim=0)
        centroid = F.normalize(mean_emb, p=2, dim=0)
        centroids.append(centroid)
        
    return torch.stack(centroids)


def evaluate_anomaly_diffusion_similarity(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_ad_sim_{args.class_name}_{args.backbone}"
    save_dir = args.save_dir if args.save_dir else os.path.join("logs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results and plots will be saved in: {save_dir}")
    
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project, name=exp_name, config=vars(args), reinit=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Trained Backbone Checkpoint
    if os.path.isfile(args.weights_dir):
        weight_path = args.weights_dir
    else:
        weight_path = os.path.join(args.weights_dir, f"{args.class_name}.pth")
        if not os.path.exists(weight_path):
            candidates = [
                os.path.join(args.weights_dir, args.class_name, "backbone_finetuned.pth"),
                os.path.join(args.weights_dir, args.class_name, "backbone_final.pth"),
                os.path.join(args.weights_dir, "backbone_finetuned.pth"),
                os.path.join(args.weights_dir, "backbone_final.pth"),
            ]
            found = False
            for cand in candidates:
                if os.path.exists(cand):
                    weight_path = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Checkpoint weight file not found for class '{args.class_name}' in {args.weights_dir}")
            
    print(f"Loading trained backbone checkpoint from: {weight_path}")
    
    # 2. Data Transforms
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    # 3. Load Real MVTec dataset to compute reference centroids
    class_real_data_path = os.path.join(args.dataset_root, args.class_name)
    print(f"Loading real MVTec dataset from: {class_real_data_path}")
    real_dataset = MVTecMaskedTestDataset(class_real_data_path, img_transform=img_transform, mask_transform=mask_transform)
    num_classes = len(real_dataset.classes)
    classes = real_dataset.classes
    class_to_idx = real_dataset.class_to_idx
    
    real_labels = np.array(real_dataset.labels)
    real_indices = np.arange(len(real_labels))
    
    train_indices = []
    rng_split = np.random.default_rng(args.seed)
    for c in range(num_classes):
        c_indices = real_indices[real_labels == c]
        rng_split.shuffle(c_indices)
        split_idx = int(len(c_indices) * 0.6)
        train_indices.extend(c_indices[:split_idx])
        
    # 4. Initialize Backbone Model
    print(f"Initializing {args.backbone} backbone...")
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=False).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    if hasattr(backbone, 'model'):
        backbone.model.load_state_dict(clean_state_dict, strict=False)
    else:
        backbone.load_state_dict(clean_state_dict, strict=False)
    backbone.eval()
    
    # 5. Compute Real Reference Centroids
    print("\nComputing real class centroids from MVTec 60% train split...")
    centroids = compute_masked_centroids(backbone, real_dataset, train_indices, num_classes, args.num_restarts, device, seed=args.seed)
    centroids_cpu = centroids.cpu()
    
    # 6. Load Synthetic AnomalyDiffusion Dataset
    syn_category_path = os.path.join(args.syn_root, args.class_name)
    print(f"\nLoading synthetic AnomalyDiffusion dataset from: {syn_category_path}")
    syn_dataset = AnomalyDiffusionDataset(syn_category_path, class_to_idx=class_to_idx, img_transform=img_transform, mask_transform=mask_transform)
    
    if len(syn_dataset) == 0:
        raise ValueError(f"No synthetic images found in {syn_category_path}")
        
    print(f"Found {len(syn_dataset)} synthetic AnomalyDiffusion samples for category '{args.class_name}'.")
    
    syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 7. Evaluate Synthetic Masked Similarity against Real Centroids
    syn_sim_vectors = []
    syn_labels = []
    syn_defect_types = []
    syn_file_names = []
    syn_embeddings_list = []
    
    correct_preds = 0
    total_syn_samples = 0
    
    with torch.no_grad():
        for images, masks, targets, defect_types, file_names, idxs in syn_loader:
            batch_size = images.size(0)
            
            embs = extract_masked_embeddings(backbone, images, masks, device).cpu()
            sims = torch.matmul(embs, centroids_cpu.t()) # [batch_size, num_classes]
            
            preds = sims.argmax(dim=1)
            for p, t in zip(preds, targets):
                if t >= 0:
                    if p == t:
                        correct_preds += 1
                else:
                    if p > 0: # Correct if predicted as any defect class
                        correct_preds += 1
            total_syn_samples += batch_size
            
            syn_sim_vectors.append(sims)
            syn_labels.append(targets)
            syn_defect_types.extend(defect_types)
            syn_file_names.extend(file_names)
            syn_embeddings_list.append(embs)
            
    syn_acc = correct_preds / total_syn_samples
    print("\n" + "="*60)
    print(" ANOMALYDIFFUSION SYNTHETIC FEATURE QUALITY EVALUATION ")
    print("="*60)
    print(f"Synthetic Nearest Centroid Accuracy: {syn_acc * 100:.2f}%")
    print("="*60 + "\n")
    
    syn_sim_vectors_np = torch.cat(syn_sim_vectors, dim=0).numpy() # [N, num_classes]
    syn_labels_np = torch.cat(syn_labels, dim=0).numpy() # [N]
    syn_embeddings_np = torch.cat(syn_embeddings_list, dim=0).numpy() # [N, dim]
    
    # Scaled Softmax Probabilities
    scaled_logits = syn_sim_vectors_np * args.s
    syn_probs_np = F.softmax(torch.from_numpy(scaled_logits), dim=1).numpy()
    
    # Build Sample-Level Detailed Quality Records for CSV Export
    csv_records = []
    for i in range(len(syn_labels_np)):
        t_label = syn_labels_np[i]
        d_type = syn_defect_types[i]
        f_name = syn_file_names[i]
        
        if t_label >= 0:
            sim_to_target = syn_sim_vectors_np[i, t_label]
            prob_to_target = syn_probs_np[i, t_label]
        else:
            # Generic combined defect type: target is nearest defect centroid (class >= 1)
            defect_sims = syn_sim_vectors_np[i, 1:]
            nearest_defect_idx = np.argmax(defect_sims) + 1
            sim_to_target = syn_sim_vectors_np[i, nearest_defect_idx]
            prob_to_target = syn_probs_np[i, nearest_defect_idx]
            
        sim_to_good = syn_sim_vectors_np[i, 0] # Class 0 is 'good'
        is_high_quality = (sim_to_target >= args.quality_threshold) and (sim_to_target > sim_to_good)
        
        record = {
            'category': args.class_name,
            'defect_type': d_type,
            'file_name': f_name,
            'similarity_to_target': round(float(sim_to_target), 4),
            'similarity_to_good': round(float(sim_to_good), 4),
            'softmax_target_prob': round(float(prob_to_target), 4),
            'is_high_quality': bool(is_high_quality)
        }
        
        for c in range(num_classes):
            c_name = classes[c]
            record[f'similarity_to_{c_name}'] = round(float(syn_sim_vectors_np[i, c]), 4)
            record[f'prob_to_{c_name}'] = round(float(syn_probs_np[i, c]), 4)
            
        csv_records.append(record)
        
    df_results = pd.DataFrame(csv_records)
    csv_path = os.path.join(save_dir, "similarity_results_anomaly_diffusion.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved detailed sample-level quality CSV to: {csv_path}")
    
    # Print Summary Analysis by Defect Type
    print("\n" + "="*80)
    print(" SYNTHETIC DEFECT TYPE QUALITY SUMMARY ")
    print("="*80)
    
    unique_d_types = df_results['defect_type'].unique()
    for d_type in unique_d_types:
        sub_df = df_results[df_results['defect_type'] == d_type]
        total_count = len(sub_df)
        if total_count == 0:
            continue
            
        sub_sims = sub_df['similarity_to_target'].to_numpy()
        sub_good_sims = sub_df['similarity_to_good'].to_numpy()
        sub_probs = sub_df['softmax_target_prob'].to_numpy()
        high_q_count = sub_df['is_high_quality'].sum()
        
        print(f"Defect Type: '{d_type:15s}' (Total Samples: {total_count:3d})")
        print(f"  -> Similarity to Target Centroid: Mean={sub_sims.mean():.4f}, Std={sub_sims.std():.4f}")
        print(f"  -> Good Leakage Similarity:       Mean={sub_good_sims.mean():.4f}, Std={sub_good_sims.std():.4f}")
        print(f"  -> Softmax Target Probability:    Mean={sub_probs.mean():.4f}, Std={sub_probs.std():.4f}")
        print(f"  -> High Quality Samples Ratio:    {high_q_count}/{total_count} ({high_q_count/total_count*100:.1f}%)")
        print("-" * 80)
        
    # 8. Plotting Heatmaps and Plots
    avg_sim_matrix = np.zeros((num_classes, num_classes))
    avg_prob_matrix = np.zeros((num_classes, num_classes))
    
    for c in range(num_classes):
        c_mask = (syn_labels_np == c)
        if c_mask.any():
            avg_sim_matrix[c] = syn_sim_vectors_np[c_mask].mean(axis=0)
            avg_prob_matrix[c] = syn_probs_np[c_mask].mean(axis=0)
            
    # Heatmap 1: Cosine Similarity
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(avg_sim_matrix, annot=True, cmap='coolwarm', xticklabels=classes, yticklabels=classes, fmt=".4f", vmin=0.0, vmax=1.0)
    plt.title(f"Average AnomalyDiffusion Feature Cosine Similarity ({args.class_name.upper()})\n(Evaluated against Real MVTec Centroids)", fontsize=14, pad=15)
    plt.xlabel("Real Class Centroids ($C_j$)", fontsize=12)
    plt.ylabel("Synthetic Defect Types", fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "sample_centroid_similarity_heatmap_masked_ad.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    
    # Heatmap 2: Softmax Probability Matrix
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(avg_prob_matrix, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes, fmt=".2f", vmin=0.0, vmax=1.0)
    plt.title(f"Average AnomalyDiffusion Softmax Probability Matrix ({args.class_name.upper()} - s={args.s})\n(Evaluated against Real MVTec Centroids)", fontsize=14, pad=15)
    plt.xlabel("Real Class Centroids ($C_j$)", fontsize=12)
    plt.ylabel("Synthetic Defect Types", fontsize=12)
    plt.tight_layout()
    softmax_heatmap_path = os.path.join(save_dir, "sample_centroid_similarity_heatmap_masked_softmax_ad.png")
    plt.savefig(softmax_heatmap_path, dpi=150)
    plt.close()
    
    # Heatmap 3: Confusion Matrix
    syn_preds = syn_sim_vectors_np.argmax(axis=1)
    cm = confusion_matrix(syn_labels_np, syn_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"AnomalyDiffusion Nearest Centroid Confusion Matrix ({args.class_name.upper()})", fontsize=14, pad=15)
    plt.xlabel("Predicted Class (Real Centroid)", fontsize=12)
    plt.ylabel("Synthetic Target Class", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_masked_ad.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    print(f"\nSaved plots to:\n  - {heatmap_path}\n  - {softmax_heatmap_path}\n  - {cm_path}")
    
    if args.use_wandb:
        import wandb
        wandb.log({
            "ad_eval/accuracy": float(syn_acc),
            "plots/heatmap_ad": wandb.Image(heatmap_path),
            "plots/softmax_ad": wandb.Image(softmax_heatmap_path),
            "plots/confusion_matrix_ad": wandb.Image(cm_path),
            "ad_eval/sample_results_table": wandb.Table(dataframe=df_results)
        })

    return {
        'category': args.class_name,
        'accuracy': syn_acc,
        'save_dir': save_dir,
        'df_results': df_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AnomalyDiffusion Synthetic Quality using Masked Feature Similarity")
    parser.add_argument('--class_name', type=str, default='all', help="MVTec category name (e.g. bottle, screw, hazelnut) or 'all' to evaluate all categories")
    parser.add_argument('--dataset_root', type=str, default='datasets/mvtec', help='Path to real MVTec dataset root')
    parser.add_argument('--syn_root', type=str, default='datasets/generated_dataset/anomaly_diffusion', help='Path to AnomalyDiffusion dataset root')
    parser.add_argument('--weights_dir', type=str, default='logs/resnet18_baseline_0819', help='Directory containing trained backbone weights (.pth)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save output plots and CSV logs')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='Backbone architecture')
    parser.add_argument('--img_size', type=int, default=224, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--s', type=float, default=30.0, help='Scale factor for ArcFace softmax probabilities')
    parser.add_argument('--quality_threshold', type=float, default=0.70, help='Similarity threshold for high quality sample classification')
    parser.add_argument('--num_restarts', type=int, default=10, help='Ensemble restarts for virtual mask sampling on good images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Log metrics to WandB')
    parser.add_argument('--project', type=str, default='IAS-Quality-Assurance', help='WandB project name')
    
    args = parser.parse_args()
    
    ALL_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    if args.class_name == 'all':
        classes_to_run = ALL_CLASSES
    else:
        classes_to_run = [args.class_name]
        
    all_summary_results = []
    all_df_records = []
    
    for c_name in classes_to_run:
        real_c_path = os.path.join(args.dataset_root, c_name)
        syn_c_path = os.path.join(args.syn_root, c_name)
        
        if not os.path.exists(real_c_path):
            print(f"Real MVTec path {real_c_path} not found. Skipping '{c_name}'...")
            continue
        if not os.path.exists(syn_c_path):
            print(f"Synthetic AnomalyDiffusion path {syn_c_path} not found. Skipping '{c_name}'...")
            continue
            
        print(f"\n" + "="*80)
        print(f" RUNNING ANOMALYDIFFUSION QUALITY EVALUATION ON CATEGORY: {c_name.upper()} ")
        print("="*80 + "\n")
        
        args.class_name = c_name
        set_seed(args.seed)
        
        try:
            res = evaluate_anomaly_diffusion_similarity(args)
            all_summary_results.append(res)
            if 'df_results' in res:
                all_df_records.append(res['df_results'])
        except Exception as e:
            print(f"Error evaluating category '{c_name}': {e}")
            import traceback
            traceback.print_exc()
            
    if len(all_df_records) > 0 and len(classes_to_run) > 1:
        combined_df = pd.concat(all_df_records, ignore_index=True)
        summary_save_dir = args.save_dir if args.save_dir else os.path.join("logs", f"ad_sim_all_classes_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(summary_save_dir, exist_ok=True)
        combined_csv_path = os.path.join(summary_save_dir, "similarity_results_anomaly_diffusion_all_classes.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print("\n" + "="*80)
        print(f" ALL CLASSES EVALUATION COMPLETE ")
        print(f" Saved combined multi-category quality CSV to: {combined_csv_path}")
        print("="*80 + "\n")
