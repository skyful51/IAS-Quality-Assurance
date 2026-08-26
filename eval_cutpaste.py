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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Import custom modules
from models.backbone import CutPasteBackbone

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
    Dataset loader for MVTec AD test set with Ground Truth Defect Masks.
    
    Structure:
    category_root/
        test/
            good/             -> Class 0
            defect_type_1/    -> Class 1
            defect_type_2/    -> Class 2
            ...
        ground_truth/
            defect_type_1/    -> Defect masks (*_mask.png or *.png)
            defect_type_2/
            ...
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
                    
                    # Find corresponding ground truth mask file
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
            # Placeholder mask for good samples (overridden during virtual mask sampling)
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


def find_cutpaste_checkpoint(cutpaste_dir, class_name):
    """Find the checkpoint path for the specified class in cutpaste_dir."""
    patterns = [
        os.path.join(cutpaste_dir, f"model-{class_name}-*.tch"),
        os.path.join(cutpaste_dir, f"model_{class_name}_*.tch"),
        os.path.join(cutpaste_dir, f"model-{class_name}.tch"),
        os.path.join(cutpaste_dir, f"{class_name}.tch"),
        os.path.join(cutpaste_dir, f"{class_name}.pth"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        raise FileNotFoundError(f"CutPaste model checkpoint for class '{class_name}' not found in '{cutpaste_dir}'!")
    candidates.sort()
    return candidates[-1]


def verify_model_shapes(backbone, img_size, device):
    """
    Verifies input and output shapes of the CutPaste backbone model.
    """
    print("\n" + "="*65)
    print(" MODEL INPUT / OUTPUT TENSOR SIZE VERIFICATION ")
    print("="*65)
    
    dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
    dummy_mask = torch.ones(1, 1, img_size, img_size).to(device)
    
    resnet = backbone.resnet18
    x = resnet.conv1(dummy_img)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    f_map = resnet.layer4(x)
    
    H_f, W_f = f_map.shape[2], f_map.shape[3]
    mask_resized = F.interpolate(dummy_mask, size=(H_f, W_f), mode='bilinear', align_corners=False)
    spatial_pooled = (f_map * mask_resized).sum(dim=(2, 3)) / mask_resized.sum(dim=(2, 3))
    
    if backbone.include_head:
        out_emb = backbone.head(spatial_pooled)
    else:
        out_emb = spatial_pooled
    out_emb_norm = F.normalize(out_emb, p=2, dim=1)
    
    print(f"  Input Image Tensor Shape:            {list(dummy_img.shape)}")
    print(f"  ResNet layer4 Feature Map Shape:     {list(f_map.shape)}")
    print(f"  Downsampled Mask Tensor Shape:       {list(mask_resized.shape)}")
    print(f"  Spatial Masked Pooled Feature Shape: {list(spatial_pooled.shape)}")
    print(f"  Output Embedding Tensor Shape:       {list(out_emb.shape)}")
    print(f"  L2-Normalized Embedding Shape:       {list(out_emb_norm.shape)}")
    print("="*65 + "\n")


def extract_cutpaste_masked_embeddings(backbone, images, masks, device):
    """
    Extract feature maps from CutPaste backbone and perform feature-map level spatial masked mean pooling.
    
    Args:
        backbone (CutPasteBackbone): Model backbone
        images (torch.Tensor): Input images [B, 3, H, W]
        masks (torch.Tensor): Input binary masks [B, 1, H, W]
    Returns:
        torch.Tensor: L2-normalized feature map level masked embeddings [B, embedding_dim]
    """
    images = images.to(device)
    masks = masks.to(device)
    
    resnet = backbone.resnet18
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
    pooled_feat = spatial_sum / mask_sum_clamped # [B, C]
    
    if zero_mask.any():
        gap_feat = feature_map.mean(dim=(2, 3))
        pooled_feat[zero_mask.squeeze(1)] = gap_feat[zero_mask.squeeze(1)]
        
    if backbone.include_head:
        out_emb = backbone.head(pooled_feat)
    else:
        out_emb = pooled_feat
        
    return F.normalize(out_emb, p=2, dim=1)


def compute_cutpaste_masked_centroids(backbone, dataset, train_indices, num_classes, num_restarts, device, seed=42):
    """
    Compute class centroids on the training split using feature map level masking with CutPaste backbone.
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
                
                embs = extract_cutpaste_masked_embeddings(backbone, def_imgs, def_msks, device)
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
                    
                    embs_r = extract_cutpaste_masked_embeddings(backbone, good_imgs, batch_masks, device)
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
        embs_tensor = torch.stack(embs) # [N, embedding_dim]
        mean_emb = embs_tensor.mean(dim=0)
        centroid = F.normalize(mean_emb, p=2, dim=0)
        centroids.append(centroid)
        
    return torch.stack(centroids) # [num_classes, embedding_dim]


def evaluate_cutpaste_similarity(args):
    # Setup output log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_cutpaste_masked_{args.class_name}_{args.backbone}"
    save_dir = args.save_dir if args.save_dir else os.path.join("logs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results and plots will be saved in: {save_dir}")
    
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project, name=exp_name, config=vars(args), reinit=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Find CutPaste weight checkpoint path
    cutpaste_checkpoint = find_cutpaste_checkpoint(args.cutpaste_dir, args.class_name)
    print(f"Loading CutPaste backbone checkpoint from: {cutpaste_checkpoint}")
    
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
    
    class_data_path = os.path.join(args.dataset_root, args.class_name)
    print(f"Loading MVTec dataset from: {class_data_path}")
    
    dataset = MVTecMaskedTestDataset(class_data_path, img_transform=img_transform, mask_transform=mask_transform)
    num_classes = len(dataset.classes)
    classes = dataset.classes
    labels = np.array(dataset.labels)
    indices = np.arange(len(labels))
    
    # Deterministic 60% Train / 40% Eval split matching baseline
    train_indices = []
    eval_indices = []
    
    rng_split = np.random.default_rng(args.seed)
    for c in range(num_classes):
        c_indices = indices[labels == c]
        rng_split.shuffle(c_indices)
        
        split_idx = int(len(c_indices) * 0.6)
        train_indices.extend(c_indices[:split_idx])
        eval_indices.extend(c_indices[split_idx:])
        
    print(f"\nDataset Split Summary (Deterministic Split with Seed {args.seed}):")
    for c in range(num_classes):
        c_train = sum(1 for idx in train_indices if labels[idx] == c)
        c_eval = sum(1 for idx in eval_indices if labels[idx] == c)
        print(f"  Class {c} ({classes[c]:15s}): Train={c_train:2d}, Eval={c_eval:2d}")
        
    # 3. Instantiate & Freeze CutPaste Backbone Model
    print(f"\nInitializing CutPaste ({args.backbone}) backbone with include_head={args.include_head}...")
    backbone = CutPasteBackbone(
        backbone_name=args.backbone,
        pretrained=False,
        head_layers=args.head_layers,
        include_head=args.include_head,
        checkpoint_path=cutpaste_checkpoint
    ).to(device)
    
    # Freeze backbone parameters explicitly as per experiment design
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print("CutPaste backbone parameters successfully frozen (requires_grad = False).")
    
    # 4. Perform Input/Output Size Verification
    verify_model_shapes(backbone, args.img_size, device)
    
    # 5. Compute Masked Centroids on Train Split
    print("Computing CutPaste masked class centroids from the train split...")
    centroids = compute_cutpaste_masked_centroids(backbone, dataset, train_indices, num_classes, args.num_restarts, device, seed=args.seed)
    centroids_cpu = centroids.cpu()
    
    # 6. Masked Similarity Evaluation on Eval Split (40%)
    print(f"\nEvaluating Masked Cosine Similarity on unseen 40% eval split with {args.num_restarts} ensemble restarts for good images...")
    
    eval_subset = Subset(dataset, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    defect_mask_paths = dataset.all_defect_mask_paths
    
    sample_sim_vectors = []
    eval_labels = []
    eval_is_good = []
    eval_embeddings_list = []
    
    correct_centroid = 0
    total_eval_samples = 0
    
    with torch.no_grad():
        for images, masks, targets, is_goods, idxs in eval_loader:
            batch_size = images.size(0)
            
            defect_mask_in_batch = (~is_goods)
            good_mask_in_batch = is_goods
            
            batch_sims = torch.zeros((batch_size, num_classes), dtype=torch.float32)
            batch_embs = torch.zeros((batch_size, backbone.embedding_dim), dtype=torch.float32)
            
            if defect_mask_in_batch.any():
                def_imgs = images[defect_mask_in_batch]
                def_msks = masks[defect_mask_in_batch]
                
                def_embs = extract_cutpaste_masked_embeddings(backbone, def_imgs, def_msks, device).cpu()
                def_sims = torch.matmul(def_embs, centroids_cpu.t())
                
                batch_sims[defect_mask_in_batch] = def_sims
                batch_embs[defect_mask_in_batch] = def_embs
                
            if good_mask_in_batch.any():
                good_imgs = images[good_mask_in_batch]
                num_good = good_imgs.size(0)
                
                good_sims_restarts = []
                good_embs_restarts = []
                
                for r in range(args.num_restarts):
                    rng = random.Random(args.seed + r * 1000 + 7)
                    sampled_mask_paths = [rng.choice(defect_mask_paths) for _ in range(num_good)]
                    
                    batch_masks = []
                    for mp in sampled_mask_paths:
                        m_tensor = dataset.load_mask_from_path(mp)
                        batch_masks.append(m_tensor)
                    batch_masks = torch.stack(batch_masks)
                    
                    g_embs_r = extract_cutpaste_masked_embeddings(backbone, good_imgs, batch_masks, device).cpu()
                    g_sims_r = torch.matmul(g_embs_r, centroids_cpu.t())
                    
                    good_sims_restarts.append(g_sims_r)
                    good_embs_restarts.append(g_embs_r)
                    
                avg_g_sims = torch.stack(good_sims_restarts, dim=0).mean(dim=0)
                avg_g_embs = torch.stack(good_embs_restarts, dim=0).mean(dim=0)
                avg_g_embs = F.normalize(avg_g_embs, p=2, dim=1)
                
                batch_sims[good_mask_in_batch] = avg_g_sims
                batch_embs[good_mask_in_batch] = avg_g_embs
                
            preds_centroid = batch_sims.argmax(dim=1)
            correct_centroid += preds_centroid.eq(targets).sum().item()
            total_eval_samples += batch_size
            
            sample_sim_vectors.append(batch_sims)
            eval_labels.append(targets)
            eval_is_good.append(is_goods)
            eval_embeddings_list.append(batch_embs)
            
    eval_acc_centroid = correct_centroid / total_eval_samples
    
    print("\n" + "="*65)
    print(" CUTPASTE MASKED EVALUATION ACCURACY ON UNSEEN DATA (40% SPLIT)")
    print("="*65)
    print(f"Masked Nearest Centroid Accuracy:  {eval_acc_centroid * 100:.2f}%")
    print("="*65 + "\n")
    
    eval_sim_vectors = torch.cat(sample_sim_vectors, dim=0).numpy()
    eval_labels = torch.cat(eval_labels, dim=0).numpy()
    eval_embeddings = torch.cat(eval_embeddings_list, dim=0)
    
    class_similarities = {c: [] for c in range(num_classes)}
    for i in range(len(eval_labels)):
        c_label = eval_labels[i]
        class_similarities[c_label].append(eval_sim_vectors[i])
        
    avg_sim_matrix = np.zeros((num_classes, num_classes))
    results_records = []
    
    print("="*80)
    print(" CUTPASTE MASKED FEATURE SIMILARITY DISTRIBUTION & ANALYSIS ")
    print("="*80)
    
    for c in range(num_classes):
        sims_list = class_similarities[c]
        if len(sims_list) == 0:
            continue
        sims_array = np.stack(sims_list)
        avg_sims = sims_array.mean(axis=0)
        avg_sim_matrix[c] = avg_sims
        
        self_sim = sims_array[:, c]
        self_mean = self_sim.mean()
        self_std = self_sim.std()
        self_min = self_sim.min()
        self_max = self_sim.max()
        
        other_cols = [j for j in range(num_classes) if j != c]
        other_sim = sims_array[:, other_cols]
        other_mean = other_sim.mean()
        other_std = other_sim.std()
        
        good_sim = sims_array[:, 0]
        good_mean = good_sim.mean()
        good_std = good_sim.std()
        
        print(f"Class: '{classes[c]:15s}' (Samples: {len(sims_list):2d})")
        print(f"  -> Similarity to OWN Centroid:       Mean={self_mean:.4f}, Std={self_std:.4f}, Range=[{self_min:.4f}, {self_max:.4f}]")
        print(f"  -> Similarity to OTHER Centroids:    Mean={other_mean:.4f}, Std={other_std:.4f}")
        print(f"  -> Similarity to 'good' (Normal):    Mean={good_mean:.4f}, Std={good_std:.4f}")
        print("-" * 80)
        
        if args.use_wandb:
            import wandb
            c_name = classes[c]
            wandb.log({
                f"cutpaste_eval/intra_sim_mean/{c_name}": self_mean,
                f"cutpaste_eval/intra_sim_std/{c_name}": self_std,
                f"cutpaste_eval/inter_sim_mean/{c_name}": other_mean,
                f"cutpaste_eval/good_sim_mean/{c_name}": good_mean
            })
            
        for s_idx in range(len(sims_list)):
            record = {
                'class_name': classes[c],
                'sample_idx': s_idx,
                'similarity_to_own': self_sim[s_idx],
                'similarity_to_good': good_sim[s_idx]
            }
            for c_target in range(num_classes):
                record[f'similarity_to_{classes[c_target]}'] = sims_array[s_idx, c_target]
            results_records.append(record)
            
    print("="*80)
    
    # Save quantitative data to CSV
    df_results = pd.DataFrame(results_records)
    csv_path = os.path.join(save_dir, "similarity_results_cutpaste_masked.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved quantitative similarity analysis to: {csv_path}\n")
    
    # 7. Plotting CutPaste Masked Heatmap
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        avg_sim_matrix, 
        annot=True, 
        cmap='coolwarm', 
        xticklabels=classes, 
        yticklabels=classes,
        fmt=".4f",
        vmin=0.0, vmax=1.0
    )
    plt.title(f"CutPaste Masked Feature Cosine Similarity Heatmap ({args.class_name.upper()})\n(Self-Supervised Backbone + Feature Map Masked Input)", fontsize=13, pad=15)
    plt.xlabel("Class Centroids ($C_j$)", fontsize=12)
    plt.ylabel("Evaluation Samples ($x_i$)", fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "sample_centroid_similarity_heatmap_cutpaste_masked.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved similarity heatmap to: {heatmap_path}")
    
    # 7-2. Confusion Matrix Plotting
    eval_labels_np = eval_labels.numpy() if isinstance(eval_labels, torch.Tensor) else np.array(eval_labels)
    eval_preds = eval_sim_vectors.argmax(axis=1)
    cm = confusion_matrix(eval_labels_np, eval_preds, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap='Blues', 
        xticklabels=classes, 
        yticklabels=classes
    )
    plt.title(f"Nearest Centroid Confusion Matrix ({args.class_name.upper()} - CutPaste)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_cutpaste_masked.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix plot to: {cm_path}")

    # 8. Plotting CutPaste Masked Similarity Distribution (KDE / Histogram)
    plt.figure(figsize=(12, 6))
    
    own_sims = []
    other_sims = []
    defect_to_good_sims = []
    
    for c in range(num_classes):
        sims_list = class_similarities[c]
        if len(sims_list) == 0:
            continue
        sims_array = np.stack(sims_list)
        own_sims.extend(sims_array[:, c].tolist())
        
        other_cols = [j for j in range(num_classes) if j != c]
        other_sims.extend(sims_array[:, other_cols].flatten().tolist())
        
        if c > 0:
            defect_to_good_sims.extend(sims_array[:, 0].tolist())
            
    sns.kdeplot(own_sims, fill=True, color="blue", label="Intra-Class (Samples to Own Centroid)", bw_adjust=0.5, alpha=0.4)
    sns.kdeplot(other_sims, fill=True, color="red", label="Inter-Class (Samples to Other Centroids)", bw_adjust=0.5, alpha=0.4)
    if len(defect_to_good_sims) > 0:
        sns.kdeplot(defect_to_good_sims, fill=True, color="green", label="Defect Samples to 'good' Centroid", bw_adjust=0.5, alpha=0.4)
        
    plt.xlim(0.0, 1.05)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Distribution of CutPaste Cosine Similarities (Feature Map Masked Input - {args.class_name.upper()})", fontsize=13, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    dist_path = os.path.join(save_dir, "similarity_distribution_cutpaste_masked.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved similarity distribution plot to: {dist_path}")
    
    # 9. Embedding Space Visualization (t-SNE)
    print("Generating t-SNE projection of CutPaste masked embeddings and centroids...")
    centroids_cpu_np = centroids_cpu.numpy()
    eval_embeddings_np = eval_embeddings.numpy()
    centroid_sim_matrix = np.dot(centroids_cpu_np, centroids_cpu_np.T)
    
    combined_data = np.concatenate([eval_embeddings_np, centroids_cpu_np], axis=0)
    perp = min(15, len(combined_data) - 1)
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=perp)
    combined_2d = tsne.fit_transform(combined_data)
    
    eval_2d = combined_2d[:-num_classes]
    centroids_2d = combined_2d[-num_classes:]
    
    plt.figure(figsize=(10, 8))
    if num_classes <= 10:
        colors = plt.cm.tab10(np.arange(num_classes))
    else:
        colors = plt.cm.tab20(np.arange(num_classes))
        
    for c in range(num_classes):
        mask = (eval_labels == c)
        plt.scatter(
            eval_2d[mask, 0], 
            eval_2d[mask, 1], 
            color=colors[c], 
            label=f"{classes[c]} (Eval Samples)", 
            alpha=0.6, 
            s=40
        )
        plt.scatter(
            centroids_2d[c, 0], 
            centroids_2d[c, 1], 
            color=colors[c], 
            marker='*', 
            s=350, 
            edgecolors='black', 
            linewidths=1.5,
            label=f"{classes[c]} (Centroid)"
        )
        
    plt.title(f"t-SNE Visualization of CutPaste Masked Embeddings & Class Centroids ({args.class_name.upper()})", fontsize=13, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, "tsne_embeddings_cutpaste_masked.png")
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE visualization to: {tsne_path}")
    
    if args.use_wandb:
        import wandb
        
        columns = ["True_Class"] + classes
        # 1. Similarity Matrix Table
        sim_table = wandb.Table(
            data=[[classes[c]] + [float(val) for val in avg_sim_matrix[c]] for c in range(num_classes)],
            columns=columns
        )

        # 2. Centroid-to-Centroid Similarity Matrix Table
        centroid_columns = ["Class"] + classes
        centroid_sim_table = wandb.Table(
            data=[[classes[i]] + [float(val) for val in centroid_sim_matrix[i]] for i in range(num_classes)],
            columns=centroid_columns
        )

        # 3. Confusion Matrix Tables (Raw Counts & Normalized Recall)
        cm_table = wandb.Table(
            data=[[classes[i]] + [int(val) for val in cm[i]] for i in range(num_classes)],
            columns=columns
        )
        cm_norm_table = wandb.Table(
            data=[[classes[i]] + [float(val) for val in cm_norm[i]] for i in range(num_classes)],
            columns=columns
        )

        # 4. Distribution Summary Statistics & Table
        dist_summary_data = []
        own_sims_np = np.array(own_sims) if len(own_sims) > 0 else np.array([])
        other_sims_np = np.array(other_sims) if len(other_sims) > 0 else np.array([])
        defect_sims_np = np.array(defect_to_good_sims) if len(defect_to_good_sims) > 0 else np.array([])

        def get_dist_row(name, arr):
            if len(arr) == 0:
                return [name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return [
                name,
                int(len(arr)),
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.percentile(arr, 25)),
                float(np.median(arr)),
                float(np.percentile(arr, 75)),
                float(np.max(arr))
            ]

        dist_summary_data.append(get_dist_row("Intra-Class (Own Centroid)", own_sims_np))
        dist_summary_data.append(get_dist_row("Inter-Class (Other Centroids)", other_sims_np))
        if len(defect_sims_np) > 0:
            dist_summary_data.append(get_dist_row("Defect Samples to Good Centroid", defect_sims_np))

        dist_summary_table = wandb.Table(
            data=dist_summary_data,
            columns=["Distribution", "Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"]
        )

        # 5. t-SNE 2D Coordinates Table
        tsne_table_data = []
        for i in range(len(eval_2d)):
            lbl = int(eval_labels_np[i])
            tsne_table_data.append([float(eval_2d[i, 0]), float(eval_2d[i, 1]), lbl, classes[lbl], "Eval Sample"])
        for c in range(num_classes):
            tsne_table_data.append([float(centroids_2d[c, 0]), float(centroids_2d[c, 1]), c, classes[c], "Centroid"])

        tsne_coords_table = wandb.Table(
            data=tsne_table_data,
            columns=["dim_1", "dim_2", "class_idx", "class_name", "type"]
        )

        # 6. Sample-level Quantitative Results Table
        sample_results_table = wandb.Table(dataframe=df_results)

        # 7. Log all Numeric Tables, Metrics, Histograms, and Plots to WandB
        wandb_log_dict = {
            # Accuracy Metric
            "cutpaste_eval/nearest_centroid_acc": float(eval_acc_centroid),
            # Static Plot Images
            "plots/cutpaste_masked_similarity_heatmap": wandb.Image(heatmap_path),
            "plots/cutpaste_masked_confusion_matrix": wandb.Image(cm_path),
            "plots/cutpaste_masked_similarity_distribution": wandb.Image(dist_path),
            "plots/cutpaste_masked_tsne_embeddings": wandb.Image(tsne_path),
            # Interactive Confusion Matrix Plot
            "plots/cutpaste_masked_interactive_confusion_matrix": wandb.plot.confusion_matrix(
                preds=eval_preds.tolist(),
                y_true=eval_labels_np.tolist(),
                class_names=classes,
                title=f"Nearest Centroid Confusion Matrix ({args.class_name.upper()} - CutPaste)"
            ),
            # Numeric Tables
            "cutpaste_eval/similarity_matrix": sim_table,
            "cutpaste_eval/centroid_similarity_matrix": centroid_sim_table,
            "cutpaste_eval/confusion_matrix_table": cm_table,
            "cutpaste_eval/confusion_matrix_normalized": cm_norm_table,
            "cutpaste_eval/distribution_summary_table": dist_summary_table,
            "cutpaste_eval/tsne_coordinates": tsne_coords_table,
            "cutpaste_eval/sample_similarity_results": sample_results_table,
            # Histograms for Numeric Distribution
            "cutpaste_eval/intra_sim_distribution": wandb.Histogram(own_sims_np),
            "cutpaste_eval/inter_sim_distribution": wandb.Histogram(other_sims_np),
            # Overall Distribution Scalars
            "cutpaste_eval/overall_intra_sim_mean": float(np.mean(own_sims_np)) if len(own_sims_np) > 0 else 0.0,
            "cutpaste_eval/overall_intra_sim_std": float(np.std(own_sims_np)) if len(own_sims_np) > 0 else 0.0,
            "cutpaste_eval/overall_intra_sim_median": float(np.median(own_sims_np)) if len(own_sims_np) > 0 else 0.0,
            "cutpaste_eval/overall_inter_sim_mean": float(np.mean(other_sims_np)) if len(other_sims_np) > 0 else 0.0,
            "cutpaste_eval/overall_inter_sim_std": float(np.std(other_sims_np)) if len(other_sims_np) > 0 else 0.0,
            "cutpaste_eval/overall_inter_sim_median": float(np.median(other_sims_np)) if len(other_sims_np) > 0 else 0.0
        }
        if len(defect_sims_np) > 0:
            wandb_log_dict["cutpaste_eval/defect_to_good_sim_distribution"] = wandb.Histogram(defect_sims_np)
            wandb_log_dict["cutpaste_eval/overall_defect_to_good_sim_mean"] = float(np.mean(defect_sims_np))
            wandb_log_dict["cutpaste_eval/overall_defect_to_good_sim_std"] = float(np.std(defect_sims_np))
            wandb_log_dict["cutpaste_eval/overall_defect_to_good_sim_median"] = float(np.median(defect_sims_np))

        wandb.log(wandb_log_dict)
        wandb.finish()
        
    print("\n" + "="*80)
    print(" CUTPASTE MASKED EVALUATION EXPERIMENT COMPLETE ")
    print("="*80 + "\n")
    
    return {
        'category': args.class_name,
        'accuracy': eval_acc_centroid,
        'save_dir': save_dir
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Defect Pattern Cosine Similarity using CutPaste Backbone with Feature Map Masking")
    parser.add_argument('--cutpaste_dir', type=str, default='pytorch-cutpaste/models', help='Directory containing CutPaste model checkpoints (model-<class_name>-*.tch)')
    parser.add_argument('--dataset_root', type=str, default='datasets/mvtec', help='Root path to MVTec dataset')
    parser.add_argument('--class_name', type=str, default='all', help="Category name (e.g. bottle) or 'all' to run all 15 MVTec categories")
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='ResNet backbone architecture')
    parser.add_argument('--head_layers', type=int, default=2, help='Number of hidden layers in CutPaste projection head MLP')
    parser.add_argument('--include_head', type=str2bool, default=True, help='Whether to include CutPaste projection head (output dim 128 if True, 512 if False)')
    parser.add_argument('--img_size', type=int, default=224, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_restarts', type=int, default=5, help='Number of random ensemble restarts for normal (good) sample virtual masks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU if available')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save output files and plots')
    parser.add_argument('--use_wandb', action='store_true', help='Log results to Weights & Biases')
    parser.add_argument('--project', type=str, default='IAS-CutPaste-Masked-Similarity', help='WandB project name')
    
    args = parser.parse_args()
    
    ALL_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    if args.class_name == 'all':
        classes_to_run = ALL_CLASSES
    else:
        classes_to_run = [args.class_name]
        
    summary_results = []
    
    for class_name in classes_to_run:
        class_path = os.path.join(args.dataset_root, class_name)
        if not os.path.exists(class_path):
            print(f"Directory {class_path} not found. Skipping category '{class_name}'...")
            continue
            
        print(f"\n" + "="*80)
        print(f" RUNNING CUTPASTE MASKED EVALUATION ON CATEGORY: {class_name.upper()} ")
        print("="*80 + "\n")
        
        args.class_name = class_name
        set_seed(args.seed)
        
        try:
            res = evaluate_cutpaste_similarity(args)
            summary_results.append(res)
        except Exception as e:
            print(f"Error occurred while evaluating CutPaste for class '{class_name}': {e}")
            import traceback
            traceback.print_exc()
            
    if len(summary_results) > 1:
        print("\n" + "="*80)
        print(" OVERALL ALL-CLASSES CUTPASTE MASKED EVALUATION SUMMARY ")
        print("="*80)
        df_summary = pd.DataFrame(summary_results)
        print(df_summary.to_string(index=False))
        print(f"\nAverage Nearest Centroid Accuracy Across All Categories: {df_summary['accuracy'].mean() * 100:.2f}%")
        print("="*80 + "\n")
