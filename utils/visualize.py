import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize_embeddings(backbone, dataloader, device, save_path, epoch):
    """
    Collects embeddings from the dataloader and visualizes them using t-SNE.
    """
    backbone.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Get normalized embeddings as ArcFace works on hypersphere
            embeddings = torch.nn.functional.normalize(backbone(images))
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Dimensionality Reduction using t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    
    # Use idx_to_class if available in dataset
    classes = getattr(dataloader.dataset, 'classes', [f"Class {i}" for i in unique_labels])

    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            label=classes[label],
            alpha=0.6,
            edgecolors='w'
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Embedding Visualization (t-SNE) - Epoch {epoch}")
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(save_path, f"embedding_vix_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Visualization saved to {plot_file}")
