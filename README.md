# IAS Quality Assurance (Anomaly Detection with AML)

This project implements a modular training pipeline and inference system for **Anomaly Detection** on the **MVTec AD** dataset using **Angular Margin (ArcFace) Loss**.

## 🏗️ Project Structure
- `train.py`: Main training script for Phase 1 (Backbone learning).
- `inference.py`: Inference script for Phase 2 (Anomaly scoring).
- `models/`:
  - `backbone.py`: ResNet (18/50) feature extractors.
  - `heads.py`: ArcMarginProduct (ArcFace) head for angular clustering.
- `data/dataset.py`: MVTec AD dataset loader with deterministic train/val splitting.
- `utils/visualize.py`: Visualization tools (t-SNE, Heatmaps).

## 🚀 Phase 1: Training
The goal is to learn discriminative embeddings where normal images ('good') and defect types form distinct radial clusters in 512D/2048D space.

```bash
# Basic training on a single category
python train.py --data_path /path/to/mvtec/bottle --backbone resnet50 --epochs 200

# Additional Arguments:
# --lr: Initial learning rate (default: 1e-4)
# --lr_step_size: Decay LR every N epochs (default: 50)
# --seed: Random seed for reproducibility (default: 42)
```
*   **Outputs**: Checkpoints and visualization results (t-SNE, Centroid Heatmaps) are saved in the `logs/run_CATEGORY_TIMESTAMP/` directory every 10 epochs.

## 🔍 Phase 2: Inference & Anomaly Scoring
Use the trained centroids to calculate how "realistic" a new or synthetic image is compared to the 'Good' class.

```bash
python inference.py \
    --backbone resnet50 \
    --backbone_path logs/run_bottle_.../backbone_final.pth \
    --head_path logs/run_bottle_.../head_final.pth \
    --dataset_root datasets/generated_dataset/ \
    --class_name bottle
```

*   **Logic**: Score = $1 - \text{Similarity}(x, \text{Centroid}_{\text{good}})$.
*   **Results**: A CSV file (`inference_bottle_results.csv`) is generated, showing the anomaly score and individual class similarities for each image found in `{dataset_root}/{class_name}/*/image`.

## 📦 Requirements
- PyTorch / Torchvision
- PIL (Pillow)
- Pandas & NumPy
- Matplotlib & Seaborn
- scikit-learn (for t-SNE)

---
Documentation generated for IAS-Quality-Assurance.
