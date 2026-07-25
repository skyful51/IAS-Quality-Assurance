import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import custom modules
from models.backbone import ResNetBackbone
from data.dataset import MVTecTestDataset

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility.")

def compute_centroids(backbone, dataloader, num_classes, device):
    """Compute the centroid for each class on the training split."""
    backbone.eval()
    class_embeddings = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = backbone(images) # [B, embedding_dim]
            # Normalize each sample embedding
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            
            for emb, label in zip(normalized_embeddings, labels):
                class_embeddings[label.item()].append(emb)
                
    centroids = []
    for c in range(num_classes):
        embs = class_embeddings[c]
        if len(embs) == 0:
            raise ValueError(f"No training samples found for class label {c} during centroid calculation!")
        embs_tensor = torch.stack(embs) # [N, embedding_dim]
        # Calculate mean embedding
        mean_emb = embs_tensor.mean(dim=0)
        # Normalize the mean embedding to get centroid
        centroid = F.normalize(mean_emb, p=2, dim=0)
        centroids.append(centroid)
        
    return torch.stack(centroids) # [num_classes, embedding_dim]

def train_and_evaluate(args):
    # Setup log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_baseline_{args.class_name}_{args.backbone}"
    save_dir = args.save_dir if args.save_dir else os.path.join("logs", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results and plots will be saved in: {save_dir}")
    
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project, name=exp_name, config=vars(args), reinit=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Transforms
    class AddGaussianNoise(object):
        """Custom transform to add Gaussian Noise to a tensor."""
        def __init__(self, mean=0.0, std=0.01):
            self.mean = mean
            self.std = std
        def __call__(self, tensor):
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
            
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # ① 미세 회전 및 이동 & ② 미세 크기 조절
        transforms.RandomAffine(
            degrees=(-5, 5),
            translate=(0.02, 0.02),
            scale=(0.95, 1.05)
        ),
        # ③ 미세한 조도/대비 변경 (±10% 범위, 색조 변경은 억제)
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.0,
            hue=0.0
        ),
        # ④ 가우시안 블러
        # transforms.GaussianBlur(
        #     kernel_size=3,
        #     sigma=(0.1, 1.0)
        # ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ④ 가우시안 노이즈
        AddGaussianNoise(mean=0.0, std=0.01),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_data_path = os.path.join(args.dataset_root, args.class_name)
    print(f"Loading dataset from: {class_data_path}")
    
    # Instantiate two datasets to apply different transforms
    full_train_dataset = MVTecTestDataset(class_data_path, transform=train_transform)
    full_eval_dataset = MVTecTestDataset(class_data_path, transform=eval_transform)
    
    num_classes = len(full_train_dataset.classes)
    classes = full_train_dataset.classes
    labels = np.array(full_train_dataset.labels)
    indices = np.arange(len(labels))
    
    # Stratified deterministic split: 60% train (for tuning and centroid calculation), 40% eval (similarity distribution measurement)
    train_indices = []
    eval_indices = []
    
    # Use numpy generator with seed 42 to make the split deterministic
    rng = np.random.default_rng(42)
    for c in range(num_classes):
        c_indices = indices[labels == c]
        rng.shuffle(c_indices)
        
        split_idx = int(len(c_indices) * 0.6)
        train_indices.extend(c_indices[:split_idx])
        eval_indices.extend(c_indices[split_idx:])
        
    print(f"Dataset Split Summary (Deterministic Split with Seed 42):")
    for c in range(num_classes):
        c_train = sum(1 for idx in train_indices if labels[idx] == c)
        c_eval = sum(1 for idx in eval_indices if labels[idx] == c)
        print(f"  Class {c} ({classes[c]:15s}): Train={c_train:2d}, Eval={c_eval:2d}")
        
    # Subsets
    train_subset = Subset(full_train_dataset, train_indices)
    eval_subset = Subset(full_eval_dataset, eval_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # Loader for centroid calculation (no augmentation)
    train_eval_subset = Subset(full_eval_dataset, train_indices)
    train_eval_loader = DataLoader(train_eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Initialize Model
    print(f"Initializing standard {args.backbone} backbone with ImageNet weights...")
    backbone = ResNetBackbone(model_name=args.backbone, pretrained=True).to(device)
    
    # Temporary classification head for fine-tuning backbone
    classifier = nn.Linear(backbone.embedding_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tuning: optimize backbone and classification head
    optimizer = optim.Adam([
        {'params': backbone.parameters(), 'lr': args.lr * 0.1},  # lower lr for pretrained backbone
        {'params': classifier.parameters(), 'lr': args.lr}
    ])
    
    # 3. Fine-tuning
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        backbone.train()
        classifier.train()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            embeddings = backbone(images)
            logits = classifier(embeddings)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total_samples += targets.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        
        # Epoch-wise validation
        eval_loss = 0.0
        eval_correct = 0
        eval_total_samples = 0
        
        backbone.eval()
        classifier.eval()
        with torch.no_grad():
            for images, targets in eval_loader:
                images, targets = images.to(device), targets.to(device)
                embeddings = backbone(images)
                logits = classifier(embeddings)
                
                loss = criterion(logits, targets)
                eval_loss += loss.item() * images.size(0)
                
                preds = logits.argmax(dim=1)
                eval_correct += preds.eq(targets).sum().item()
                eval_total_samples += targets.size(0)
                
        epoch_eval_loss = eval_loss / eval_total_samples
        epoch_eval_acc = eval_correct / eval_total_samples
        
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_acc:.4f} - Eval Loss: {epoch_eval_loss:.4f} - Eval Accuracy: {epoch_eval_acc:.4f}")
            
        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_loss,
                "train/acc": epoch_acc,
                "eval/loss": epoch_eval_loss,
                "eval/acc": epoch_eval_acc
            })
            
    # Save checkpoint
    torch.save(backbone.state_dict(), os.path.join(save_dir, "backbone_finetuned.pth"))
    print("Fine-tuning completed. Saved model weights.")
    
    # 4. Centroid Extraction
    print("Computing class centroids from the train split...")
    centroids = compute_centroids(backbone, train_eval_loader, num_classes, device)
    
    # 5. Inference & Similarity Evaluation on the Eval Split
    backbone.eval()
    classifier.eval()
    
    eval_embeddings = []
    eval_labels = []
    
    # Calculate classification accuracy on evaluation split
    eval_correct_clf = 0
    eval_correct_centroid = 0
    total_eval_samples = 0
    
    with torch.no_grad():
        for images, targets in eval_loader:
            images = images.to(device)
            embeddings = backbone(images)
            normalized_embs = F.normalize(embeddings, p=2, dim=1)
            
            # Classifier head prediction
            logits = classifier(embeddings)
            preds_clf = logits.argmax(dim=1).cpu()
            eval_correct_clf += preds_clf.eq(targets).sum().item()
            
            # Centroid-based prediction (nearest cosine similarity)
            # normalized_embs: [B, dim], centroids: [num_classes, dim]
            sim_matrix = torch.matmul(normalized_embs, centroids.t()) # [B, num_classes]
            preds_centroid = sim_matrix.argmax(dim=1).cpu()
            eval_correct_centroid += preds_centroid.eq(targets).sum().item()
            
            eval_embeddings.append(normalized_embs.cpu())
            eval_labels.append(targets)
            total_eval_samples += targets.size(0)
            
    eval_acc_clf = eval_correct_clf / total_eval_samples
    eval_acc_centroid = eval_correct_centroid / total_eval_samples
    
    print("\n" + "="*60)
    print(" EVALUATION ACCURACY ON UNSEEN DATA (40% SPLIT)")
    print("="*60)
    print(f"Classifier Head Accuracy:    {eval_acc_clf * 100:.2f}%")
    print(f"Nearest Centroid Accuracy:  {eval_acc_centroid * 100:.2f}%")
    print("="*60 + "\n")
    
    if args.use_wandb:
        import wandb
        wandb.log({
            "final_eval/classifier_acc": eval_acc_clf,
            "final_eval/nearest_centroid_acc": eval_acc_centroid
        })
    
    # Concatenate all eval embeddings
    eval_embeddings = torch.cat(eval_embeddings, dim=0) # [M, dim]
    eval_labels = torch.cat(eval_labels, dim=0) # [M]
    
    # Calculate pairwise similarities for each eval class to all centroids
    centroids_cpu = centroids.cpu()
    class_similarities = {c: [] for c in range(num_classes)}
    for emb, label in zip(eval_embeddings, eval_labels):
        # Cosine similarity to all centroids
        # centroids: [num_classes, dim]
        sims = torch.matmul(emb, centroids_cpu.t()) # [num_classes]
        class_similarities[label.item()].append(sims.numpy())
        
    # Calculate average sample-to-centroid similarity matrix
    avg_sim_matrix = np.zeros((num_classes, num_classes))
    results_records = []
    
    print("="*80)
    print(" QUANTITATIVE SIMILARITY DISTRIBUTION & INFLATION ANALYSIS ")
    print("="*80)
    
    for c in range(num_classes):
        sims_list = class_similarities[c]
        if len(sims_list) == 0:
            continue
        sims_array = np.stack(sims_list) # [num_samples_in_class, num_classes]
        avg_sims = sims_array.mean(axis=0)
        avg_sim_matrix[c] = avg_sims
        
        # Intra-class (similarity to own centroid)
        self_sim = sims_array[:, c]
        self_mean = self_sim.mean()
        self_std = self_sim.std()
        self_min = self_sim.min()
        self_max = self_sim.max()
        
        # Inter-class (similarity to other centroids)
        other_cols = [j for j in range(num_classes) if j != c]
        other_sim = sims_array[:, other_cols]
        other_mean = other_sim.mean()
        other_std = other_sim.std()
        
        # Similarity to 'good' class centroid (essential to demonstrate background inflation)
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
                f"final_eval/intra_sim_mean/{c_name}": self_mean,
                f"final_eval/intra_sim_std/{c_name}": self_std,
                f"final_eval/inter_sim_mean/{c_name}": other_mean,
                f"final_eval/good_sim_mean/{c_name}": good_mean
            })
        
        # Log records for CSV saving
        for s_idx in range(len(sims_list)):
            record = {
                'class_name': classes[c],
                'sample_idx': s_idx,
                'similarity_to_own': self_sim[s_idx],
                'similarity_to_good': good_sim[s_idx]
            }
            # Add similarities to each individual centroid
            for c_target in range(num_classes):
                record[f'similarity_to_{classes[c_target]}'] = sims_array[s_idx, c_target]
            results_records.append(record)
            
    print("="*80)
    
    # Save quantitative data to CSV
    df_results = pd.DataFrame(results_records)
    csv_path = os.path.join(save_dir, "similarity_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved quantitative similarity analysis to: {csv_path}\n")
    
    if args.use_wandb:
        import wandb
        # 1. Log sample-to-centroid similarity matrix as a table
        columns = ["True_Class"] + classes
        table_data = []
        for c in range(num_classes):
            row_data = [classes[c]] + [float(val) for val in avg_sim_matrix[c]]
            table_data.append(row_data)
        
        sim_table = wandb.Table(data=table_data, columns=columns)
        
        # 2. Log centroid-to-centroid pairwise similarity matrix as a table
        centroids_np = centroids.cpu().numpy() # [num_classes, dim], already L2-normalized
        centroid_sim_matrix = np.dot(centroids_np, centroids_np.T)
        
        centroid_columns = ["Class"] + classes
        centroid_table_data = []
        for i in range(num_classes):
            row_data = [classes[i]] + [float(val) for val in centroid_sim_matrix[i]]
            centroid_table_data.append(row_data)
            
        centroid_sim_table = wandb.Table(data=centroid_table_data, columns=centroid_columns)
        
        # Log tables to WandB
        wandb.log({
            "final_eval/similarity_matrix": sim_table,
            "final_eval/centroid_similarity_matrix": centroid_sim_table
        })
    
    # 6. Plotting Heatmap
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
    plt.title(f"Average Sample-to-Centroid Cosine Similarity Heatmap\n(Global Input / Similarity Inflation Visualized)", fontsize=14, pad=15)
    plt.xlabel("Class Centroids ($C_j$)", fontsize=12)
    plt.ylabel("Real Evaluation Samples ($x_i$)", fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "sample_centroid_similarity_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved similarity heatmap to: {heatmap_path}")
    
    # 7. Plotting Similarity Distribution (Density/Histogram)
    plt.figure(figsize=(12, 6))
    
    # Collect flat lists of own similarities and other similarities for plot
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
        
        # If this is a defect class, record its similarity to the good centroid
        if c > 0:
            defect_to_good_sims.extend(sims_array[:, 0])
            
    # Plotting KDE curves to show how compressed the similarities are
    sns.kdeplot(own_sims, fill=True, color="blue", label="Intra-Class (Samples to Own Centroid)", bw_adjust=0.5, alpha=0.4)
    sns.kdeplot(other_sims, fill=True, color="red", label="Inter-Class (Samples to Other Centroids)", bw_adjust=0.5, alpha=0.4)
    sns.kdeplot(defect_to_good_sims, fill=True, color="green", label="Defect Samples to 'good' Centroid", bw_adjust=0.5, alpha=0.4)
    
    plt.xlim(0.0, 1.05)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Cosine Similarities (Global Input)\nExposing Severity of Similarity Inflation", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    dist_path = os.path.join(save_dir, "similarity_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved similarity distribution plot to: {dist_path}")
    
    # 8. Embedding Space Visualization (t-SNE)
    print("Generating t-SNE projection of embeddings and centroids...")
    centroids_cpu = centroids.cpu().numpy()
    eval_embeddings_cpu = eval_embeddings.numpy()
    
    # Combine evaluation embeddings and centroids
    combined_data = np.concatenate([eval_embeddings_cpu, centroids_cpu], axis=0)
    
    # Fit t-SNE
    # Set perplexity carefully to prevent error with small dataset size
    perp = min(15, len(combined_data) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    combined_2d = tsne.fit_transform(combined_data)
    
    eval_2d = combined_2d[:-num_classes]
    centroids_2d = combined_2d[-num_classes:]
    
    plt.figure(figsize=(10, 8))
    # Generate distinct colors dynamically based on the number of classes
    if num_classes <= 10:
        colors = plt.cm.tab10(np.arange(num_classes))
    else:
        colors = plt.cm.tab20(np.arange(num_classes))
    
    for c in range(num_classes):
        mask = (eval_labels == c).numpy()
        # Plot evaluation samples
        plt.scatter(
            eval_2d[mask, 0], 
            eval_2d[mask, 1], 
            color=colors[c], 
            label=f"{classes[c]} (Eval Samples)", 
            alpha=0.6, 
            s=40
        )
        # Plot corresponding class centroid
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
        
    plt.title("t-SNE Visualization of Evaluation Embeddings & Class Centroids", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, "tsne_embeddings.png")
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE visualization to: {tsne_path}")
    
    if args.use_wandb:
        import wandb
        wandb.log({
            "plots/similarity_heatmap": wandb.Image(heatmap_path),
            "plots/similarity_distribution": wandb.Image(dist_path),
            "plots/tsne_embeddings": wandb.Image(tsne_path)
        })
        wandb.finish()
    
    print("\n" + "="*80)
    print(" EXPERIMENT COMPLETE ")
    print("="*80)
    print("Interpretation of the similarity results:")
    print("1. If the model classifies defect classes on unseen evaluation split with high accuracy,")
    print("   it proves the model has a discriminative embedding space (unseen defects group together).")
    print("2. If the cosine similarity scores of all defect classes to the 'good' class and to each other")
    print("   are extremely high (e.g. >0.90) and highly compressed/overlapping, it proves the dominant")
    print("   influence of the common normal background (Similarity Inflation / 배경 지배 현상).")
    print("3. Consequently, using this model as an evaluation backbone on full images will lead to")
    print("   False Positive Evaluations: a synthetic image with distorted defects but clean backgrounds")
    print("   will score extremely close to 1.0, rating it as highly realistic.")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment: Similarity Inflation Exposure under Global Input")
    parser.add_argument('--dataset_root', type=str, default='datasets/mvtec', help='Root path to MVTec dataset')
    parser.add_argument('--class_name', type=str, default='all', help="Category name (e.g. bottle) or 'all' to run all classes")
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='ResNet backbone to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to fine-tune the backbone')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for classifier head')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--img_size', type=int, default=224, help='Resolution to resize images')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use GPU if available')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save output files and plots')
    parser.add_argument('--use_wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--project', type=str, default='IAS-Baseline-Inflation', help='WandB project name')
    
    args = parser.parse_args()
    
    # List of all standard MVTec categories
    ALL_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    if args.class_name == 'all':
        classes_to_run = ALL_CLASSES
    else:
        classes_to_run = [args.class_name]
        
    for class_name in classes_to_run:
        # Check if dataset path exists
        class_path = os.path.join(args.dataset_root, class_name)
        if not os.path.exists(class_path):
            print(f"Directory {class_path} not found. Skipping class '{class_name}'...")
            continue
            
        print(f"\n" + "="*80)
        print(f" RUNNING EXPERIMENT ON CATEGORY: {class_name.upper()} ")
        print("="*80 + "\n")
        
        # Override class_name in args
        args.class_name = class_name
        
        # Set seed 42 for each category to ensure independent split reproducibility
        set_seed(42)
        
        try:
            train_and_evaluate(args)
        except Exception as e:
            print(f"Error occurred while processing class '{class_name}': {e}")
            import traceback
            traceback.print_exc()
