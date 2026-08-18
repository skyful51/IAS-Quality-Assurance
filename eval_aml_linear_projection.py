import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import custom modules
from models.backbone import CutPasteBackbone, ResNetBackbone
from models.heads import ArcMarginProduct
from eval_aml import MVTecMaskedTestDataset, find_cutpaste_checkpoint, str2bool, set_seed


class MultiScaleFPNBackbone(nn.Module):
    """
    CutPaste Backbone (ResNet18 or ResNet50) + Multi-Scale Feature Extractor
    Layer-wise L2-Norm Pooling + Optional Direct Concatenation (896d) or Learnable Projection (896d -> proj_dim)
    """
    def __init__(self, cutpaste_backbone, proj_dim=512, direct_concat=False):
        super().__init__()
        self.cutpaste_backbone = cutpaste_backbone
        self.direct_concat = direct_concat
        
        if hasattr(cutpaste_backbone, 'resnet18'):
            self.resnet = cutpaste_backbone.resnet18
        elif hasattr(cutpaste_backbone, 'model'):
            self.resnet = cutpaste_backbone.model
        else:
            self.resnet = cutpaste_backbone
            
        backbone_type = getattr(cutpaste_backbone, 'backbone_name', 'resnet18')
        if backbone_type == 'resnet50':
            c2, c3, c4 = 512, 1024, 2048
        else:
            c2, c3, c4 = 128, 256, 512
            
        in_dim = c2 + c3 + c4  # 896 for ResNet18, 3584 for ResNet50
        
        if self.direct_concat:
            self.proj = None
            self.embedding_dim = in_dim
        else:
            # Learnable Linear Projection block (896d -> 512d)
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(inplace=True)
            )
            self.embedding_dim = proj_dim

    def extract_multiscale_masked_feat(self, images, masks, device):
        images = images.to(device)
        masks = masks.to(device)
        
        # 1. Base ResNet Feature Map Extraction (Frozen)
        with torch.no_grad():
            x = self.resnet.conv1(images)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            x = self.resnet.layer1(x)
            f2 = self.resnet.layer2(x)  # [B, c2, 28, 28]
            f3 = self.resnet.layer3(f2) # [B, c3, 14, 14]
            f4 = self.resnet.layer4(f3) # [B, c4, 7, 7]
            
        # Helper function for layer-wise spatial masked mean pooling and independent L2-norm
        def pool_layer(f_map, mask):
            B, C, H_f, W_f = f_map.shape
            m_res = F.interpolate(mask, size=(H_f, W_f), mode='bilinear', align_corners=False)
            spatial_sum = (f_map * m_res).sum(dim=(2, 3))
            mask_sum = m_res.sum(dim=(2, 3))
            zero_mask = (mask_sum < 1e-6)
            mask_sum_clamped = torch.clamp(mask_sum, min=1e-8)
            pooled = spatial_sum / mask_sum_clamped
            if zero_mask.any():
                gap_feat = f_map.mean(dim=(2, 3))
                pooled[zero_mask.squeeze(1)] = gap_feat[zero_mask.squeeze(1)]
            return F.normalize(pooled, p=2, dim=1) # Independent L2-Norm per layer
            
        p2 = pool_layer(f2, masks) # [B, c2]
        p3 = pool_layer(f3, masks) # [B, c3]
        p4 = pool_layer(f4, masks) # [B, c4]
        
        # 2. Concatenate Multi-Scale Features (e.g. 128 + 256 + 512 = 896d)
        concat_feat = torch.cat([p2, p3, p4], dim=1) # [B, in_dim]
        
        # 3. Direct Concatenation or Pass through Projection Layer
        if self.direct_concat:
            return F.normalize(concat_feat, p=2, dim=1) # Direct [B, in_dim]
            
        out_emb = self.proj(concat_feat)              # [B, proj_dim]
        return F.normalize(out_emb, p=2, dim=1)        # [B, proj_dim]


def verify_model_shapes_proj(multi_scale_model, num_classes, img_size, device):
    """
    Verifies input and output tensor shapes of MultiScaleFPNBackbone and AML setup.
    """
    print("\n" + "="*65)
    print(" MODEL INPUT / OUTPUT TENSOR SIZE VERIFICATION (PROJECTION SETUP) ")
    print("="*65)
    
    dummy_img = torch.randn(2, 3, img_size, img_size).to(device)
    dummy_mask = torch.ones(2, 1, img_size, img_size).to(device)
    
    with torch.no_grad():
        x = multi_scale_model.resnet.conv1(dummy_img)
        x = multi_scale_model.resnet.bn1(x)
        x = multi_scale_model.resnet.relu(x)
        x = multi_scale_model.resnet.maxpool(x)
        x = multi_scale_model.resnet.layer1(x)
        f2 = multi_scale_model.resnet.layer2(x)
        f3 = multi_scale_model.resnet.layer3(f2)
        f4 = multi_scale_model.resnet.layer4(f3)
        
    pooled = multi_scale_model.extract_multiscale_masked_feat(dummy_img, dummy_mask, device)
    
    print(f"  Input Image Tensor Shape:                  {list(dummy_img.shape)}")
    print(f"  ResNet layer2 f2 Feature Map Shape:        {list(f2.shape)}")
    print(f"  ResNet layer3 f3 Feature Map Shape:        {list(f3.shape)}")
    print(f"  ResNet layer4 f4 Feature Map Shape:        {list(f4.shape)}")
    print(f"  Multi-Scale Concatenated Embedding Dim:    896d")
    print(f"  Projected & L2-Normalized Embedding Shape: {list(pooled.shape)}")
    print("="*65 + "\n")


def fit_aml_head_proj(multi_scale_model, train_loader, num_classes, args, device):
    """
    Fits ArcMarginProduct (AML) head on the 60% train split.
    Optionally fits Learnable Projection layer if direct_concat is False.
    Keeps base ResNet backbone strictly frozen.
    """
    emb_dim = multi_scale_model.embedding_dim
    print(f"\nInitializing ArcMarginProduct (AML) Head: in_features={emb_dim}, out_features={num_classes}, s={args.s}, m={args.m}")
    head = ArcMarginProduct(in_features=emb_dim, out_features=num_classes, s=args.s, m=args.m).to(device)
    
    criterion = nn.CrossEntropyLoss()
    if getattr(multi_scale_model, 'direct_concat', False):
        trainable_params = list(head.parameters())
        print(f"Fitting AML Head Only (Direct Concatenation Mode, 896d) for {args.epochs} epochs on 60% train split (Frozen Base ResNet)...")
    else:
        trainable_params = list(head.parameters()) + list(multi_scale_model.proj.parameters())
        print(f"Fitting Projection Layer & AML Head for {args.epochs} epochs on 60% train split (Frozen Base ResNet)...")
        
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=weight_decay)
    
    head.train()
    multi_scale_model.train()
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, masks, labels, is_goods, idxs in train_loader:
            if images.size(0) <= 1:
                # Skip single-sample batch during training to prevent BatchNorm1d error
                continue
            images, labels = images.to(device), labels.to(device)
            masks = masks.to(device)
            
            embeddings = multi_scale_model.extract_multiscale_masked_feat(images, masks, device)
            
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch [{epoch+1:02d}/{args.epochs:02d}] - AML Train Loss: {epoch_loss:.4f} - AML Train Acc: {epoch_acc * 100:.2f}%")
            
    head.eval()
    multi_scale_model.eval()
    with torch.no_grad():
        aml_centroids = F.normalize(head.weight, p=2, dim=1).detach().cpu()
        
    print("AML Projection Head fitting completed. Extracted normalized class centroids.")
    return head, aml_centroids


def evaluate_aml_proj_similarity(args):
    # Setup output log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "aml_direct" if getattr(args, 'direct_concat', False) else "aml_proj"
    exp_name = f"{timestamp}_{mode_tag}_{args.class_name}_{args.backbone}"
    save_dir = args.save_dir if args.save_dir else os.path.join("logs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results and plots will be saved in: {save_dir}")
    
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project, name=exp_name, config=vars(args), reinit=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Transforms
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
    
    # 2. Deterministic 60% Train / 40% Eval Stratified Split (Seed 42)
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
        
    # 3. Load Backbone & Construct MultiScaleFPNBackbone
    if args.backbone == 'cutpaste':
        cutpaste_checkpoint = find_cutpaste_checkpoint(args.cutpaste_dir, args.class_name)
        print(f"\nLoading CutPaste backbone checkpoint from: {cutpaste_checkpoint}")
        cutpaste_base = CutPasteBackbone(
            backbone_name='resnet18',
            pretrained=False,
            head_layers=args.head_layers,
            include_head=False,
            checkpoint_path=cutpaste_checkpoint
        )
    else:
        print(f"\nInitializing standard {args.backbone} backbone (ImageNet pre-trained)...")
        cutpaste_base = ResNetBackbone(model_name=args.backbone, pretrained=True)
        
    # 4. Instantiate MultiScaleFPNBackbone Model
    print("\nInitializing MultiScaleFPNBackbone...")
    multi_scale_model = MultiScaleFPNBackbone(
        cutpaste_base, 
        proj_dim=args.proj_dim, 
        direct_concat=args.direct_concat
    ).to(device)
    
    # Enforce strict freezing on base ResNet parameters
    multi_scale_model.eval()
    if args.freeze_backbone:
        for param in multi_scale_model.resnet.parameters():
            param.requires_grad = False
        print("Base ResNet parameters strictly frozen (freeze_backbone = True, requires_grad = False).")
        
    # 4. Perform Tensor Shape Verification
    verify_model_shapes_proj(multi_scale_model, num_classes, args.img_size, device)
    
    # 5. AML Fitting Phase on 60% Train Split
    train_subset = Subset(dataset, train_indices)
    drop_last = (len(train_subset) > args.batch_size)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
    
    aml_head, aml_centroids = fit_aml_head_proj(multi_scale_model, train_loader, num_classes, args, device)
    
    # Save AML checkpoint
    head_save_path = os.path.join(save_dir, "aml_linear_projection_fitted.pth")
    save_dict = {'head': aml_head.state_dict()}
    if multi_scale_model.proj is not None:
        save_dict['proj'] = multi_scale_model.proj.state_dict()
    torch.save(save_dict, head_save_path)
    print(f"Saved fitted AML checkpoint to: {head_save_path}")
    
    # 6. Masked Similarity Evaluation on Unseen 40% Eval Split
    print(f"\nEvaluating Cosine Similarity on unseen 40% eval split with {args.num_restarts} ensemble restarts for good images...")
    
    eval_subset = Subset(dataset, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    defect_mask_paths = dataset.all_defect_mask_paths
    
    sample_sim_vectors = []
    eval_labels = []
    eval_embeddings_list = []
    
    correct_centroid = 0
    total_eval_samples = 0
    
    with torch.no_grad():
        for images, masks, targets, is_goods, idxs in eval_loader:
            batch_size = images.size(0)
            
            defect_mask_in_batch = (~is_goods)
            good_mask_in_batch = is_goods
            
            batch_sims = torch.zeros((batch_size, num_classes), dtype=torch.float32)
            batch_embs = torch.zeros((batch_size, multi_scale_model.embedding_dim), dtype=torch.float32)
            
            # Defect samples: multi-scale masked feature extraction
            if defect_mask_in_batch.any():
                def_imgs = images[defect_mask_in_batch]
                def_msks = masks[defect_mask_in_batch]
                
                def_embs = multi_scale_model.extract_multiscale_masked_feat(def_imgs, def_msks, device).cpu()
                def_sims = torch.matmul(def_embs, aml_centroids.t())
                
                batch_sims[defect_mask_in_batch] = def_sims
                batch_embs[defect_mask_in_batch] = def_embs
                
            # Good samples: virtual defect mask sampling across restarts
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
                    
                    g_embs_r = multi_scale_model.extract_multiscale_masked_feat(good_imgs, batch_masks, device).cpu()
                    g_sims_r = torch.matmul(g_embs_r, aml_centroids.t())
                    
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
            eval_embeddings_list.append(batch_embs)
            
    eval_acc_centroid = correct_centroid / total_eval_samples
    
    print("\n" + "="*65)
    print(" AML PROJECTION EVALUATION ACCURACY ON UNSEEN DATA (40% SPLIT)")
    print("="*65)
    print(f"AML Projection Nearest Centroid Accuracy:  {eval_acc_centroid * 100:.2f}%")
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
    print(" AML PROJECTION COSINE SIMILARITY DISTRIBUTION & ANALYSIS ")
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
                f"aml_proj_eval/intra_sim_mean/{c_name}": self_mean,
                f"aml_proj_eval/intra_sim_std/{c_name}": self_std,
                f"aml_proj_eval/inter_sim_mean/{c_name}": other_mean,
                f"aml_proj_eval/good_sim_mean/{c_name}": good_mean
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
    csv_path = os.path.join(save_dir, "similarity_results_aml_linear_projection.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved quantitative similarity analysis to: {csv_path}\n")
    
    # Calculate Softmax Probabilities from Raw Cosine Similarities (Scaled by s=30.0)
    scaled_logits = eval_sim_vectors * args.s
    probs_tensor = F.softmax(torch.from_numpy(scaled_logits), dim=1)
    eval_prob_vectors = probs_tensor.numpy()
    
    class_probabilities = {c: [] for c in range(num_classes)}
    for i in range(len(eval_labels)):
        c_label = eval_labels[i]
        class_probabilities[c_label].append(eval_prob_vectors[i])
        
    avg_prob_matrix = np.zeros((num_classes, num_classes))
    for c in range(num_classes):
        probs_list = class_probabilities[c]
        if len(probs_list) > 0:
            avg_prob_matrix[c] = np.stack(probs_list).mean(axis=0)
            
    title_tag = "AML Direct Concat (896d)" if args.direct_concat else "AML Projection"
    
    # 7-1. Plotting Pre-Softmax Raw Cosine Similarity Heatmap (Single-hue Blues Colormap)
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        avg_sim_matrix, 
        annot=True, 
        cmap='Blues',           # Single-hue sequential colormap: low=light blue, high=dark navy
        xticklabels=classes, 
        yticklabels=classes,
        fmt=".2f",
        vmin=-0.2, 
        vmax=1.0
    )
    plt.title(f"Raw Cosine Similarity Heatmap (Pre-Softmax {title_tag} - {args.class_name.upper()})", fontsize=14, pad=15)
    plt.xlabel("AML Class Centroids ($C_j$)", fontsize=12)
    plt.ylabel("Evaluation Samples ($x_i$)", fontsize=12)
    plt.tight_layout()
    raw_prefix = "sample_centroid_similarity_heatmap_aml_direct_concat_raw.png" if args.direct_concat else "sample_centroid_similarity_heatmap_aml_linear_projection_raw.png"
    raw_heatmap_path = os.path.join(save_dir, raw_prefix)
    plt.savefig(raw_heatmap_path, dpi=150)
    plt.close()
    print(f"Saved Raw Cosine Similarity heatmap to: {raw_heatmap_path}")
    
    # 7-2. Plotting AML Softmax Probability Heatmap (Single-hue Blues Colormap)
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        avg_prob_matrix, 
        annot=True, 
        cmap='Blues',           # Single-hue sequential colormap: low=light blue, high=dark navy
        xticklabels=classes, 
        yticklabels=classes,
        fmt=".2f",
        vmin=0.0, 
        vmax=1.0
    )
    plt.title(f"Average {title_tag} Softmax Probability Heatmap (Scaled by s={args.s})", fontsize=14, pad=15)
    plt.xlabel("AML Class Centroids ($C_j$)", fontsize=12)
    plt.ylabel("Evaluation Samples ($x_i$)", fontsize=12)
    plt.tight_layout()
    soft_prefix = "sample_centroid_similarity_heatmap_aml_direct_concat.png" if args.direct_concat else "sample_centroid_similarity_heatmap_aml_linear_projection.png"
    heatmap_path = os.path.join(save_dir, soft_prefix)
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved Softmax Probability heatmap to: {heatmap_path}")
    
    # 8. Plotting AML Projection Similarity Distribution (KDE)
    plt.figure(figsize=(12, 6))
    
    own_sims = []
    other_sims = []
    defect_to_good_sims = []
    
    for c in range(num_classes):
        sims_list = class_similarities[c]
        if len(sims_list) == 0:
            continue
        sims_array = np.stack(sims_list)
        own_sims.extend(sims_array[:, c])
        
        other_cols = [j for j in range(num_classes) if j != c]
        other_sims.extend(sims_array[:, other_cols].flatten())
        
        if c > 0:
            defect_to_good_sims.extend(sims_array[:, 0])
            
    sns.kdeplot(own_sims, fill=True, color="blue", label="Intra-Class (Samples to Own AML Centroid)", bw_adjust=0.5, alpha=0.4)
    sns.kdeplot(other_sims, fill=True, color="red", label="Inter-Class (Samples to Other AML Centroids)", bw_adjust=0.5, alpha=0.4)
    if len(defect_to_good_sims) > 0:
        sns.kdeplot(defect_to_good_sims, fill=True, color="green", label="Defect Samples to 'good' AML Centroid", bw_adjust=0.5, alpha=0.4)
        
    plt.xlim(0.0, 1.05)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Distribution of AML Projection Cosine Similarities ({args.class_name.upper()} - s={args.s}, m={args.m})", fontsize=13, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    dist_path = os.path.join(save_dir, "similarity_distribution_aml_linear_projection.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved similarity distribution plot to: {dist_path}")
    
    # 9. Embedding Space Visualization (t-SNE)
    print("Generating t-SNE projection of masked embeddings and AML centroids...")
    centroids_cpu_np = aml_centroids.numpy()
    eval_embeddings_np = eval_embeddings.numpy()
    
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
            label=f"{classes[c]} (AML Centroid)"
        )
        
    plt.title(f"t-SNE Visualization of Multi-Scale Masked Embeddings & AML Centroids ({args.class_name.upper()})", fontsize=13, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, "tsne_embeddings_aml_linear_projection.png")
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE visualization to: {tsne_path}")
    
    if args.use_wandb:
        import wandb
        wandb.log({
            "plots/aml_proj_raw_cosine_heatmap": wandb.Image(raw_heatmap_path),
            "plots/aml_proj_softmax_prob_heatmap": wandb.Image(heatmap_path),
            "plots/aml_proj_similarity_distribution": wandb.Image(dist_path),
            "plots/aml_proj_tsne_embeddings": wandb.Image(tsne_path)
        })
        wandb.finish()
        
    print("\n" + "="*80)
    print(" AML PROJECTION EVALUATION EXPERIMENT COMPLETE ")
    print("="*80 + "\n")
    
    return {
        'category': args.class_name,
        'accuracy': eval_acc_centroid,
        'save_dir': save_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Defect Pattern Cosine Similarity using Multi-Scale CutPaste Backbone and Learnable Projection AML Head")
    parser.add_argument('--cutpaste_dir', type=str, default='pytorch-cutpaste/models', help='Directory containing CutPaste model checkpoints')
    parser.add_argument('--dataset_root', type=str, default='datasets/mvtec', help='Root path to MVTec dataset')
    parser.add_argument('--class_name', type=str, default='all', help="Category name or 'all'")
    parser.add_argument('--backbone', type=str, default='cutpaste', choices=['cutpaste', 'resnet18', 'resnet50'], help='Backbone architecture')
    parser.add_argument('--proj_dim', type=int, default=512, help='Output dimension for learnable linear projection block (default: 512)')
    parser.add_argument('--direct_concat', action='store_true', help='Directly concatenate 896d multi-scale features to AML head without 1x1 convs or linear projection')
    parser.add_argument('--freeze_backbone', type=str2bool, default=True, help='Freeze base ResNet backbone parameters (default: True)')
    parser.add_argument('--head_layers', type=int, default=2, help='Number of hidden layers in CutPaste projection head MLP')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to fit AML head & projection layer on 60% train split (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for Adam optimizer (default: 1e-4)')
    parser.add_argument('--s', type=float, default=30.0, help='ArcMarginProduct scale factor (default: 30.0)')
    parser.add_argument('--m', type=float, default=0.30, help='ArcMarginProduct margin factor (default: 0.30)')
    parser.add_argument('--img_size', type=int, default=224, help='Image resolution (default: 224)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--num_restarts', type=int, default=5, help='Number of random ensemble restarts for normal sample virtual masks (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--cuda', type=str2bool, default=True, help='Use GPU if available')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save output files and plots')
    parser.add_argument('--use_wandb', action='store_true', help='Log results to Weights & Biases')
    parser.add_argument('--project', type=str, default='IAS-AML-LinearProjection-Evaluation', help='WandB project name')
    
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
        print(f" RUNNING AML PROJECTION HEAD FITTING & EVALUATION ON CATEGORY: {class_name.upper()} ")
        print("="*80 + "\n")
        
        args.class_name = class_name
        set_seed(args.seed)
        
        try:
            res = evaluate_aml_proj_similarity(args)
            summary_results.append(res)
        except Exception as e:
            print(f"Error occurred while evaluating AML Projection for class '{class_name}': {e}")
            import traceback
            traceback.print_exc()
            
    if len(summary_results) > 1:
        print("\n" + "="*80)
        print(" OVERALL ALL-CLASSES AML PROJECTION EVALUATION SUMMARY ")
        print("="*80)
        df_summary = pd.DataFrame(summary_results)
        print(df_summary.to_string(index=False))
        print(f"\nAverage AML Projection Nearest Centroid Accuracy Across All Categories: {df_summary['accuracy'].mean() * 100:.2f}%")
        print("="*80 + "\n")
