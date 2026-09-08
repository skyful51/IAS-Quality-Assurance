import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
import lpips
from PIL import Image
from torchvision import transforms

ALL_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def get_real_image_paths(mvtec_root, category, split="test"):
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

def load_and_preprocess_image(img_path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def compute_lpips_for_category(real_paths, syn_paths, category, loss_fn, device="cuda", max_pairs=500, seed=42):
    """Computes mean LPIPS perceptual distance for a category."""
    if len(real_paths) == 0 or len(syn_paths) == 0:
        return None

    random.seed(seed)
    # Sample synthetic images up to max_pairs
    syn_sampled = syn_paths if len(syn_paths) <= max_pairs else random.sample(syn_paths, max_pairs)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1] for LPIPS
    ])

    lpips_scores = []
    
    with torch.no_grad():
        for syn_path in syn_sampled:
            real_path = random.choice(real_paths)
            img_syn = load_and_preprocess_image(syn_path, transform).unsqueeze(0).to(device)
            img_real = load_and_preprocess_image(real_path, transform).unsqueeze(0).to(device)

            dist = loss_fn(img_syn, img_real)
            lpips_scores.append(float(dist.item()))

    return float(np.mean(lpips_scores))

def evaluate_algorithm(algo_name, syn_root, mvtec_root, categories, loss_fn, split="test", device="cuda"):
    print(f"\n==================================================")
    print(f"   Evaluating LPIPS Benchmark: {algo_name}")
    print(f"==================================================")

    results = []
    for cat in categories:
        syn_cat_dir = os.path.join(syn_root, algo_name, cat)
        if not os.path.exists(syn_cat_dir):
            continue

        real_paths = get_real_image_paths(mvtec_root, cat, split=split)
        syn_paths = get_synthetic_image_paths(syn_cat_dir)

        print(f"Category: {cat:12s} | Real Images: {len(real_paths):4d} | Syn Images: {len(syn_paths):4d}", end="", flush=True)
        
        lpips_score = compute_lpips_for_category(real_paths, syn_paths, cat, loss_fn, device=device)
        if lpips_score is not None:
            print(f" | LPIPS: {lpips_score:.4f}")
            results.append({"algorithm": algo_name, "category": cat, "lpips": lpips_score})
        else:
            print(" | LPIPS: N/A")

    df_algo = pd.DataFrame(results)
    if not df_algo.empty:
        mlpips = df_algo["lpips"].mean()
        print(f"\n---> {algo_name} Macro-Average mLPIPS ({len(df_algo)} categories): {mlpips:.4f}")
    else:
        mlpips = None

    return df_algo, mlpips

def main():
    parser = argparse.ArgumentParser(description="ASBench Intrinsic Quality Evaluation: LPIPS & mLPIPS Benchmark")
    parser.add_argument("--mvtec_root", type=str, default="datasets/mvtec")
    parser.add_argument("--syn_root", type=str, default="datasets/generated_dataset")
    parser.add_argument("--algorithms", nargs="+", default=["realnet_paired", "anomaly_diffusion", "cutpaste"])
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--real_split", type=str, default="test", choices=["test", "train", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="lpips_benchmark_results.csv")
    args = parser.parse_args()

    target_categories = ALL_MVTEC_CATEGORIES if "all" in args.categories else args.categories
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"MVTec AD Root : {args.mvtec_root}")
    print(f"Synthetic Root : {args.syn_root}")
    print(f"Target Algorithms: {args.algorithms}")
    print(f"Target Categories ({len(target_categories)}): {target_categories}")
    print(f"Device         : {device}")

    # Initialize LPIPS model (AlexNet backbone)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    all_results = []
    summary_data = []

    for algo in args.algorithms:
        df_algo, mlpips = evaluate_algorithm(
            algo_name=algo,
            syn_root=args.syn_root,
            mvtec_root=args.mvtec_root,
            categories=target_categories,
            loss_fn=loss_fn,
            split=args.real_split,
            device=args.device
        )
        if not df_algo.empty:
            all_results.append(df_algo)
            summary_data.append({"algorithm": algo, "mLPIPS": mlpips, "categories_evaluated": len(df_algo)})

    if len(all_results) > 0:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(args.output_csv, index=False)
        print(f"\nResults exported to {args.output_csv}")

        print("\n==================================================")
        print("          FINAL LPIPS BENCHMARK SUMMARY           ")
        print("==================================================")
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))

        pivot_df = df_all.pivot(index="category", columns="algorithm", values="lpips")
        print("\nCategory-level LPIPS Comparison:")
        print(pivot_df.to_string())

if __name__ == "__main__":
    main()
