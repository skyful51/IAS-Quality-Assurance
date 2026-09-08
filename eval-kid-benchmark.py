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
    """Collects a list of image paths into a flat temporary directory for cleanfid computation."""
    os.makedirs(temp_dir, exist_ok=True)
    for idx, path in enumerate(image_paths):
        ext = os.path.splitext(path)[1]
        dst = os.path.join(temp_dir, f"img_{idx:05d}{ext}")
        shutil.copy2(path, dst)

def get_real_image_paths(mvtec_root, category, split="test"):
    """Returns image paths for real MVTec dataset category."""
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
    """Finds all synthetic anomaly images inside a synthetic algorithm directory for a category."""
    image_paths = []
    
    # 1. Direct subfolder: image/ (e.g. syn_algo_dir/image)
    direct_img_dir = os.path.join(syn_algo_dir, "image")
    if os.path.exists(direct_img_dir):
        image_paths.extend(glob.glob(os.path.join(direct_img_dir, "*.png")))
        image_paths.extend(glob.glob(os.path.join(direct_img_dir, "*.jpg")))
        image_paths.extend(glob.glob(os.path.join(direct_img_dir, "*.jpeg")))

    # 2. Flat structure: combined/image
    flat_img_dir = os.path.join(syn_algo_dir, "combined", "image")
    if os.path.exists(flat_img_dir):
        image_paths.extend(glob.glob(os.path.join(flat_img_dir, "*.png")))
        image_paths.extend(glob.glob(os.path.join(flat_img_dir, "*.jpg")))
        image_paths.extend(glob.glob(os.path.join(flat_img_dir, "*.jpeg")))
    
    # 3. Subfolder structure: {defect_type}/image
    sub_dirs = [d for d in glob.glob(os.path.join(syn_algo_dir, "*")) if os.path.isdir(d) and os.path.basename(d) not in ["image", "mask"]]
    for sd in sub_dirs:
        sub_img_dir = os.path.join(sd, "image")
        if os.path.exists(sub_img_dir):
            image_paths.extend(glob.glob(os.path.join(sub_img_dir, "*.png")))
            image_paths.extend(glob.glob(os.path.join(sub_img_dir, "*.jpg")))
            image_paths.extend(glob.glob(os.path.join(sub_img_dir, "*.jpeg")))
    
    # 4. Direct files inside syn_algo_dir
    direct_img_paths = glob.glob(os.path.join(syn_algo_dir, "*.[jp][pn]g"))
    image_paths.extend(direct_img_paths)

    return sorted(list(set(image_paths)))

def compute_kid_for_category(real_paths, syn_paths, category, device="cuda"):
    """Computes KID score between real images and synthetic images for a category using cleanfid."""
    if len(real_paths) == 0 or len(syn_paths) == 0:
        return None

    with tempfile.TemporaryDirectory() as temp_real_dir, tempfile.TemporaryDirectory() as temp_syn_dir:
        collect_images_to_temp_dir(real_paths, temp_real_dir)
        collect_images_to_temp_dir(syn_paths, temp_syn_dir)

        # Compute KID score (returned as float)
        score = fid.compute_kid(
            fdir1=temp_real_dir,
            fdir2=temp_syn_dir,
            mode="clean",
            device=torch.device(device if torch.cuda.is_available() else "cpu")
        )
        return float(score)

def evaluate_algorithm(algo_name, syn_root, mvtec_root, categories, split="test", device="cuda"):
    """Evaluates KID across all specified categories for a synthesis algorithm and calculates mKID."""
    print(f"\n==================================================")
    print(f"   Evaluating KID Benchmark: {algo_name}")
    print(f"==================================================")

    results = []
    for cat in categories:
        syn_cat_dir = os.path.join(syn_root, algo_name, cat)
        if not os.path.exists(syn_cat_dir):
            continue

        real_paths = get_real_image_paths(mvtec_root, cat, split=split)
        syn_paths = get_synthetic_image_paths(syn_cat_dir)

        print(f"Category: {cat:12s} | Real Images: {len(real_paths):4d} | Syn Images: {len(syn_paths):4d}", end="", flush=True)
        
        kid_score = compute_kid_for_category(real_paths, syn_paths, cat, device=device)
        if kid_score is not None:
            print(f" | KID: {kid_score:.6f}")
            results.append({"algorithm": algo_name, "category": cat, "kid": kid_score})
        else:
            print(" | KID: N/A")

    df_algo = pd.DataFrame(results)
    if not df_algo.empty:
        mkid = df_algo["kid"].mean()
        print(f"\n---> {algo_name} Macro-Average mKID ({len(df_algo)} categories): {mkid:.6f}")
    else:
        mkid = None

    return df_algo, mkid

def main():
    parser = argparse.ArgumentParser(description="ASBench Intrinsic Quality Evaluation: KID & mKID Benchmark")
    parser.add_argument("--mvtec_root", type=str, default="datasets/mvtec")
    parser.add_argument("--syn_root", type=str, default="datasets/generated_dataset")
    parser.add_argument("--algorithms", nargs="+", default=["realnet_paired", "anomaly_diffusion", "cutpaste"])
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--real_split", type=str, default="test", choices=["test", "train", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="kid_benchmark_results.csv")
    args = parser.parse_args()

    target_categories = ALL_MVTEC_CATEGORIES if "all" in args.categories else args.categories

    print(f"MVTec AD Root : {args.mvtec_root}")
    print(f"Synthetic Root : {args.syn_root}")
    print(f"Target Algorithms: {args.algorithms}")
    print(f"Target Categories ({len(target_categories)}): {target_categories}")

    all_results = []
    summary_data = []

    for algo in args.algorithms:
        df_algo, mkid = evaluate_algorithm(
            algo_name=algo,
            syn_root=args.syn_root,
            mvtec_root=args.mvtec_root,
            categories=target_categories,
            split=args.real_split,
            device=args.device
        )
        if not df_algo.empty:
            all_results.append(df_algo)
            summary_data.append({"algorithm": algo, "mKID": mkid, "categories_evaluated": len(df_algo)})

    if len(all_results) > 0:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(args.output_csv, index=False)
        print(f"\nResults exported to {args.output_csv}")

        print("\n==================================================")
        print("           FINAL KID BENCHMARK SUMMARY            ")
        print("==================================================")
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))

        pivot_df = df_all.pivot(index="category", columns="algorithm", values="kid")
        print("\nCategory-level KID Comparison:")
        print(pivot_df.to_string())

if __name__ == "__main__":
    main()
