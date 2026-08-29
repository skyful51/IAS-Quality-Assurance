import os
import glob
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd
import torch
from cleanfid import fid

ALL_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def collect_images_to_temp_dir(image_paths, temp_dir):
    """
    Collects a list of image paths into a flat temporary directory for cleanfid computation.
    """
    os.makedirs(temp_dir, exist_ok=True)
    for idx, path in enumerate(image_paths):
        ext = os.path.splitext(path)[1]
        dst = os.path.join(temp_dir, f"img_{idx:05d}{ext}")
        shutil.copy2(path, dst)

def get_real_image_paths(mvtec_root, category, split="test"):
    """
    Returns image paths for real MVTec dataset category.
    split options: 'test' (all test images including defects), 'train' (clean train images), 'all'
    """
    cat_dir = os.path.join(mvtec_root, category)
    image_paths = []
    
    if split in ["test", "all"]:
        test_dir = os.path.join(cat_dir, "test")
        if os.path.exists(test_dir):
            image_paths.extend(glob.glob(os.path.join(test_dir, "*", "*.png")))
            image_paths.extend(glob.glob(os.path.join(test_dir, "*", "*.jpg")))
            
    if split in ["train", "all"]:
        train_good_dir = os.path.join(cat_dir, "train", "good")
        if os.path.exists(train_good_dir):
            image_paths.extend(glob.glob(os.path.join(train_good_dir, "*.png")))
            image_paths.extend(glob.glob(os.path.join(train_good_dir, "*.jpg")))
            
    return sorted(image_paths)

def get_synthetic_image_paths(syn_algo_dir):
    """
    Finds all synthetic anomaly images inside a synthetic algorithm directory for a category.
    Handles flat structures (combined/image) and nested structures (defect_type/image).
    """
    image_paths = []
    # Flat structure: combined/image
    flat_img_dir = os.path.join(syn_algo_dir, "combined", "image")
    if os.path.exists(flat_img_dir):
        image_paths.extend(glob.glob(os.path.join(flat_img_dir, "*.png")))
        image_paths.extend(glob.glob(os.path.join(flat_img_dir, "*.jpg")))
        image_paths.extend(glob.glob(os.path.join(flat_img_dir, "*.jpeg")))
    
    # Subfolder structure: {defect_type}/image
    sub_img_paths = glob.glob(os.path.join(syn_algo_dir, "*", "image", "*.[jp][pn]g"))
    sub_img_paths += glob.glob(os.path.join(syn_algo_dir, "*", "image", "*.jpeg"))
    image_paths.extend(sub_img_paths)
    
    # Direct files inside syn_algo_dir
    direct_img_paths = glob.glob(os.path.join(syn_algo_dir, "*.[jp][pn]g"))
    image_paths.extend(direct_img_paths)

    # Filter unique and sort
    return sorted(list(set(image_paths)))

def compute_fid_for_category(real_paths, syn_paths, category, device="cuda"):
    """
    Computes FID score between real images and synthetic images for a single category using cleanfid.
    """
    if len(real_paths) == 0:
        print(f"  [Warning] No real images found for category '{category}'. Skipping...")
        return None
    if len(syn_paths) == 0:
        print(f"  [Warning] No synthetic images found for category '{category}'. Skipping...")
        return None

    with tempfile.TemporaryDirectory() as temp_real_dir, tempfile.TemporaryDirectory() as temp_syn_dir:
        collect_images_to_temp_dir(real_paths, temp_real_dir)
        collect_images_to_temp_dir(syn_paths, temp_syn_dir)

        # Compute FID using cleanfid
        score = fid.compute_fid(
            fdir1=temp_real_dir,
            fdir2=temp_syn_dir,
            mode="clean",
            device=torch.device(device if torch.cuda.is_available() else "cpu")
        )
        return float(score)

def evaluate_algorithm(algo_name, syn_root, mvtec_root, categories, split="test", device="cuda"):
    """
    Evaluates FID across all specified categories for a synthesis algorithm and calculates mFID.
    """
    print(f"\n==================================================")
    print(f"   Evaluating Algorithm: {algo_name}")
    print(f"==================================================")

    results = []
    for cat in categories:
        syn_cat_dir = os.path.join(syn_root, algo_name, cat)
        if not os.path.exists(syn_cat_dir):
            print(f"Category folder '{syn_cat_dir}' not found. Skipping {cat}...")
            continue

        real_paths = get_real_image_paths(mvtec_root, cat, split=split)
        syn_paths = get_synthetic_image_paths(syn_cat_dir)

        print(f"Category: {cat:12s} | Real Images: {len(real_paths):4d} | Syn Images: {len(syn_paths):4d}", end="", flush=True)
        
        fid_score = compute_fid_for_category(real_paths, syn_paths, cat, device=device)
        if fid_score is not None:
            print(f" | FID: {fid_score:.4f}")
            results.append({"algorithm": algo_name, "category": cat, "fid": fid_score})
        else:
            print(" | FID: N/A")

    df_algo = pd.DataFrame(results)
    if not df_algo.empty:
        mfid = df_algo["fid"].mean()
        print(f"\n---> {algo_name} Macro-Average mFID ({len(df_algo)} categories): {mfid:.4f}")
    else:
        mfid = None
        print(f"\n---> {algo_name} Macro-Average mFID: N/A")

    return df_algo, mfid

def main():
    parser = argparse.ArgumentParser(description="ASBench Intrinsic Quality Evaluation: FID & mFID Benchmark")
    parser.add_argument("--mvtec_root", type=str, default="datasets/mvtec", help="Path to real MVTec AD dataset")
    parser.add_argument("--syn_root", type=str, default="datasets/generated_dataset", help="Path to synthetic datasets root")
    parser.add_argument("--algorithms", nargs="+", default=["realnet_paired", "anomaly_diffusion"], help="List of synthetic algorithm folder names")
    parser.add_argument("--categories", nargs="+", default=["all"], help="List of categories or 'all'")
    parser.add_argument("--real_split", type=str, default="test", choices=["test", "train", "all"], help="Real image reference split (test/train/all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output_csv", type=str, default="fid_benchmark_results.csv", help="Path to export CSV results")
    args = parser.parse_args()

    # Determine target categories
    if "all" in args.categories:
        target_categories = ALL_MVTEC_CATEGORIES
    else:
        target_categories = args.categories

    print(f"MVTec AD Root : {args.mvtec_root}")
    print(f"Synthetic Root : {args.syn_root}")
    print(f"Target Algorithms: {args.algorithms}")
    print(f"Target Categories ({len(target_categories)}): {target_categories}")
    print(f"Real Image Split : {args.real_split}")

    all_results = []
    summary_data = []

    for algo in args.algorithms:
        df_algo, mfid = evaluate_algorithm(
            algo_name=algo,
            syn_root=args.syn_root,
            mvtec_root=args.mvtec_root,
            categories=target_categories,
            split=args.real_split,
            device=args.device
        )
        if not df_algo.empty:
            all_results.append(df_algo)
            summary_data.append({"algorithm": algo, "mFID": mfid, "categories_evaluated": len(df_algo)})

    if len(all_results) > 0:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(args.output_csv, index=False)
        print(f"\nResults exported to {args.output_csv}")

        # Summary Table Display
        print("\n==================================================")
        print("           FINAL FID BENCHMARK SUMMARY            ")
        print("==================================================")
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))

        # Pivot Table Display
        pivot_df = df_all.pivot(index="category", columns="algorithm", values="fid")
        print("\nCategory-level FID Comparison:")
        print(pivot_df.to_string())

if __name__ == "__main__":
    main()
