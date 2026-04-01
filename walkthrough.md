# Walkthrough: Backbone Training with Angular Margin Loss

This document summarizes the implementation of the backbone training framework using ArcFace (Angular Margin) for anomaly detection.

## 🏗️ Project Architecture

The code is organized into modular components for clarity and reusability:

- **[data/dataset.py](file:///home/jhkang51/Public/IAS-Quality-Assurance/data/dataset.py)**: Loads MVTec AD style images and converts folder names into integer class labels.
- **[models/backbone.py](file:///home/jhkang51/Public/IAS-Quality-Assurance/models/backbone.py)**: Wraps ResNet (18/50) and extracts 1D embedding vectors.
- **[models/heads.py](file:///home/jhkang51/Public/IAS-Quality-Assurance/models/heads.py)**: Implements the ArcFace logic, calculating angles between embeddings and class centers.
- **[train.py](file:///home/jhkang51/Public/IAS-Quality-Assurance/train.py)**: The main entry point that orchestrates the training loop (Forward/Backward pass).

## 🚀 Step-by-Step Execution Flow

The implementation follows the requested Phase 1 flow exactly:

1.  **Step 1: Data Prep**: The [MVTecDataset](file:///home/jhkang51/Public/IAS-Quality-Assurance/data/dataset.py#6-51) reads images from `good`, `crack`, etc.
2.  **Step 2: Embedding Extraction**: Input images are passed through the backbone to get a 512D vector.
3.  **Step 3: Angular Margin Head**: The head calculates the cosine similarity with learned class center weights.
4.  **Step 4: Margin & Loss**: During training, a margin $m$ is added to the target class angle, and `CrossEntropyLoss` is applied to optimize the weights.

## 🛠️ How to Run

To start training with default parameters:

```bash
python train.py --data_path /path/to/your/dataset --epochs 10 --batch_size 32
```

> [!TIP]
> You can easily swap architectures by using the `--backbone resnet50` flag.

## 📊 Verification Results

- **Feature Vectors**: Verified that the final output dimension is correct for both ResNet18 and ResNet50.
- **Training Loop**: Orchestrated to handle normalized embeddings and weights as required by the ArcFace algorithm.
- **Loss Function**: Integrated `nn.CrossEntropyLoss` which correctly handles the angular logits produced by the head.
