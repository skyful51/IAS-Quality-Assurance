import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from PIL import Image

def plot_center_similarity(head, classes, save_path, epoch):
    """
    Plots the cosine similarity between class center weights (centroids).
    Assumes head has a 'weight' parameter of shape [num_classes, embedding_dim].
    """
    # 1. Normalize class center weights
    weights = torch.nn.functional.normalize(head.weight).detach().cpu().numpy()
    
    # 2. Calculate pairwise cosine similarity matrix: W * W.T
    similarity_matrix = np.dot(weights, weights.T)

    # 3. Visualization using Seaborn Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        cmap='coolwarm', 
        xticklabels=classes, 
        yticklabels=classes,
        fmt=".2f",
        vmin=-1, vmax=1
    )
    plt.title(f"Class Centroid Cosine Similarity - Epoch {epoch}")
    plt.tight_layout()

    # Save Plot
    save_file = os.path.join(save_path, f"centroid_similarity_epoch_{epoch}.png")
    plt.savefig(save_file)
    plt.close()
    print(f"Centroid similarity heatmap saved to {save_file}")

def plot_sample_to_centroid_similarity(backbone, head, dataloader, device, classes, save_path, epoch):
    """
    Plots average cosine similarity between all samples of each class and all centroids.
    This verifies intra-class compactness vs inter-class separation.
    """
    backbone.eval()
    head.eval()
    
    # [num_classes, embedding_dim]
    centroids = torch.nn.functional.normalize(head.weight).detach()
    num_classes = len(classes)
    
    # Accumulators for average similarity
    # similarity_sums[i][j] = sum of similarities of samples from class i to centroid j
    similarity_sums = torch.zeros(num_classes, num_classes).to(device)
    class_counts = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # [Batch, 512]
            embeddings = torch.nn.functional.normalize(backbone(images))
            
            # Calculate similarity matrix between [Batch, 512] and [num_classes, 512]
            # Result: [Batch, num_classes]
            sim_matrix = torch.matmul(embeddings, centroids.t())
            
            for i in range(num_classes):
                mask = (labels == i)
                if mask.any():
                    similarity_sums[i] += sim_matrix[mask].sum(dim=0)
                    class_counts[i] += mask.sum()

    # Calculate average
    avg_similarity = (similarity_sums / class_counts.unsqueeze(1)).cpu().numpy()

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_similarity, 
        annot=True, 
        cmap='YlGnBu', 
        xticklabels=classes, 
        yticklabels=classes,
        fmt=".4f",
        vmin=0, vmax=1
    )
    plt.xlabel("Centroids (W)")
    plt.ylabel("Real Samples (X)")
    plt.title(f"Avg Sample-to-Centroid Cosine Similarity - Epoch {epoch}")
    plt.tight_layout()

    save_file = os.path.join(save_path, f"sample_centroid_sim_epoch_{epoch}.png")
    plt.savefig(save_file)
    plt.close()
    print(f"Sample-to-Centroid heatmap saved to {save_file}")

def visualize_embeddings(backbone, head, dataloader, device, save_path, epoch):
    """
    Collects embeddings and centroids, then visualizes them using t-SNE.
    Centroids are marked with a distinct 'star' marker.
    """
    backbone.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = torch.nn.functional.normalize(backbone(images))
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Get Centroids
    centroids = torch.nn.functional.normalize(head.weight).detach().cpu().numpy()
    num_centroids = centroids.shape[0]

    # Combine embeddings and centroids for joint t-SNE
    combined_data = np.concatenate([all_embeddings, centroids], axis=0)

    print(f"Running t-SNE on {len(all_embeddings)} samples + {num_centroids} centroids...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_data)-1))
    combined_2d = tsne.fit_transform(combined_data)
    
    embeddings_2d = combined_2d[:-num_centroids]
    centroids_2d = combined_2d[-num_centroids:]

    # Plotting
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(all_labels)
    classes = getattr(dataloader.dataset, 'classes', [f"Class {i}" for i in unique_labels])
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        # 1. Plot Samples
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            color=colors[i],
            label=f"{classes[label]} samples",
            alpha=0.4,
            s=20
        )
        # 2. Plot Corresponding Centroid
        plt.scatter(
            centroids_2d[label, 0],
            centroids_2d[label, 1],
            color=colors[i],
            marker='*',
            s=300,
            edgecolors='black',
            linewidths=2,
            label=f"{classes[label]} centroid"
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Embedding & Centroid Visualization (t-SNE) - Epoch {epoch}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_file = os.path.join(save_path, f"embedding_vix_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Enhanced visualization saved to {plot_file}")

def visualize_morphological_transform(dataset, save_path, num_samples=10):
    """
    Saves a few examples of morphological transformations to verify correctness.
    """
    os.makedirs(save_path, exist_ok=True)
    
    types = ['dilation', 'erosion', 'gradient']
    widths = [3, 7, 11, 13]
    heights = [3, 7, 11, 13]

    plt.figure(figsize=(15, 6 * num_samples))
    
    for i in range(num_samples):
        # Get a sample
        idx = np.random.randint(len(dataset))
        
        # Handle Subset correctly
        morph_ds = dataset
        if isinstance(dataset, torch.utils.data.Subset):
            morph_ds = dataset.dataset
            actual_idx = dataset.indices[idx]
        else:
            actual_idx = idx
            
        # Use modulo to map back to real image paths
        real_img_idx = actual_idx % len(morph_ds.image_paths)
        img_path = morph_ds.image_paths[real_img_idx]
        orig_pil = Image.open(img_path).convert('RGB')
        orig_image = np.array(orig_pil)
        
        # Pick random combo
        c_idx = np.random.randint(len(morph_ds.combinations))
        t_idx, w_idx, h_idx = morph_ds.combinations[c_idx]
        
        transformed_image = morph_ds.apply_morphology(orig_pil, t_idx, w_idx, h_idx)
        transformed_image = np.array(transformed_image)
        
        # Plot
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(orig_image)
        plt.title(f"Original: {os.path.basename(img_path)}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(transformed_image)
        plt.title(f"Type: {types[t_idx]}, W: {widths[w_idx]}, H: {heights[h_idx]}")
        plt.axis('off')
        
    plt.tight_layout()
    save_file = os.path.join(save_path, "morphology_verification.png")
    plt.savefig(save_file)
    plt.close()
    print(f"Morphological transformation verification saved to {save_file}")

def visualize_embeddings_ssl(backbone, dataloader, device, save_path, epoch):
    """
    Collects embeddings and visualizes them using t-SNE, colored by transformation type.
    """
    backbone.eval()
    all_embeddings = []
    all_types = []

    with torch.no_grad():
        for images, t_labels, w_labels, h_labels in dataloader:
            images = images.to(device)
            embeddings = torch.nn.functional.normalize(backbone(images))
            all_embeddings.append(embeddings.cpu().numpy())
            all_types.append(t_labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_types = np.concatenate(all_types, axis=0)
    
    # Run t-SNE
    print(f"Running t-SNE for SSL on {len(all_embeddings)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    types = ['Dilation', 'Erosion', 'Gradient']
    colors = ['red', 'blue', 'green']
    
    for i in range(len(types)):
        mask = all_types == i
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            color=colors[i],
            label=types[i],
            alpha=0.6,
            s=20
        )

    plt.legend()
    plt.title(f"SSL Embedding Visualization (t-SNE) - Epoch {epoch}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_file = os.path.join(save_path, f"ssl_embedding_vix_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"SSL visualization saved to {plot_file}")
