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
5.  **Step 5: Logging & Visualization**: Intermediate results (Centroid Similarity & Sample Similarity) are saved **every 10 epochs** into a timestamped folder under `logs/`. Final t-SNE visualization is generated after training completes.

## 🛠️ How to Run

To start training for a specific category (e.g., bottle):

```bash
python train.py --data_path /path/to/mvtec/bottle --epochs 200 --lr 1e-4 --lr_step_size 50
```

> [!TIP]
> The model now uses a **StepLR scheduler** that reduces the learning rate by 1/10 every 50 epochs. This helps in achieving better intra-class compactness (higher cosine similarity) towards the end of the 200-epoch run.

## 🔍 Phase 2: Inference & Anomaly Scoring

After training the backbone (Phase 1), use `inference.py` to calculate anomaly scores for new or synthetic images.

```bash
python inference.py \
    --backbone resnet50 \
    --backbone_path logs/run_CATEGORY_TIMESTAMP/backbone_final.pth \
    --head_path logs/run_CATEGORY_TIMESTAMP/head_final.pth \
    --image_dir datasets/generated_dataset/anomaly_diffusion/bottle/broken_large/image
```

### Understanding the Score
- **Anomaly Score**: Calculated as `1 - CosineSimilarity(image, Good_Centroid)`.
- **0.0 ~ 0.2**: Likely a **Normal** image (high similarity to the Good centroid).
- **0.5 ~ 1.2**: Likely an **Anomaly** (far from the Good centroid).
- Results are saved to `inference_results.csv` in the model weight directory.

## 📊 Verification Results

- **Feature Vectors**: Verified that the final output dimension is correct for both ResNet18 and ResNet50.
- **Training Loop**: Orchestrated to handle normalized embeddings and weights as required by the ArcFace algorithm.
- **Loss Function**: Integrated `nn.CrossEntropyLoss` which correctly handles the angular logits produced by the head.
