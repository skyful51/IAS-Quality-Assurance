import os
import glob
import random
import math
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def lerp_np(x, y, w):
    return (y - x) * w + x

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, shift: (
        np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), axis=-1) * grad[:shape[0], :shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

def generate_perlin_noise_mask(shape=(224, 224), min_perlin_scale=0, perlin_scale=6, perlin_noise_threshold=0.5):
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))
    
    # Rotate using OpenCV
    angle = float(np.random.uniform(-90, 90))
    h, w = perlin_noise.shape
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    perlin_noise = cv2.warpAffine(perlin_noise, M_rot, (w, h))

    mask_noise = np.where(
        perlin_noise > perlin_noise_threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise)
    )
    return mask_noise

def generate_target_foreground_mask(img: np.ndarray, subclass: str) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
        return np.ones_like(img_gray)
    elif subclass == 'pill':
        _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(np.uint8)
    elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
        _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(np.uint8)
    elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(np.uint8)
        target_foreground_mask = (1 - target_background_mask).astype(np.uint8)
    else:
        target_foreground_mask = np.ones_like(img_gray, dtype=np.uint8)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    target_foreground_mask = cv2.morphologyEx(target_foreground_mask, cv2.MORPH_CLOSE, kernel)
    target_foreground_mask = cv2.morphologyEx(target_foreground_mask, cv2.MORPH_OPEN, kernel)
    return target_foreground_mask

def blend_sdas_anomaly(normal_img, sdas_img, mask, transparency_range=(0.15, 0.85)):
    factor = np.random.uniform(*transparency_range, size=1)[0]
    mask_expanded = np.expand_dims(mask, axis=2)
    
    normal_img = normal_img.astype(np.float32)
    sdas_img = sdas_img.astype(np.float32)
    
    anomaly_source_img = factor * (mask_expanded * sdas_img) + (1 - factor) * (mask_expanded * normal_img)
    blended = ((- mask_expanded + 1) * normal_img) + anomaly_source_img
    return np.clip(blended, 0, 255).astype(np.uint8)

def generate_paired_dataset_for_category(category, dataset_root, syn_realnet_root, output_root, num_samples=1500, img_size=224):
    print(f"\n[RealNet Paired Generation] Category: {category}")
    
    # 1. Collect Normal Images from real MVTec dataset (train/good + test/good)
    train_good_dir = os.path.join(dataset_root, category, 'train', 'good')
    test_good_dir = os.path.join(dataset_root, category, 'test', 'good')
    
    normal_paths = []
    if os.path.exists(train_good_dir):
        normal_paths.extend(glob.glob(os.path.join(train_good_dir, '*.png')) + glob.glob(os.path.join(train_good_dir, '*.jpg')))
    if os.path.exists(test_good_dir):
        normal_paths.extend(glob.glob(os.path.join(test_good_dir, '*.png')) + glob.glob(os.path.join(test_good_dir, '*.jpg')))
        
    if len(normal_paths) == 0:
        print(f"[Warning] No normal images found for category '{category}' at {dataset_root}. Skipping...")
        return
        
    # 2. Collect SDAS synthetic images from RealNet pre-generated folder
    sdas_dir = os.path.join(syn_realnet_root, category)
    if not os.path.exists(sdas_dir):
        print(f"[Warning] RealNet SDAS dir {sdas_dir} not found. Skipping...")
        return
        
    sdas_paths = glob.glob(os.path.join(sdas_dir, '*.jpg')) + glob.glob(os.path.join(sdas_dir, '*.png'))
    if len(sdas_paths) == 0:
        print(f"[Warning] No SDAS images found in {sdas_dir}. Skipping...")
        return
        
    # Output directory setup
    out_img_dir = os.path.join(output_root, category, 'combined', 'image')
    out_mask_dir = os.path.join(output_root, category, 'combined', 'mask')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    print(f"Generating {num_samples} paired (Image, Mask) samples -> {out_img_dir}")
    
    saved_count = 0
    for i in range(num_samples):
        # Pick normal image and sdas candidate image
        norm_path = random.choice(normal_paths)
        sdas_path = random.choice(sdas_paths)
        
        norm_img = cv2.imread(norm_path)
        norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
        norm_img = cv2.resize(norm_img, (img_size, img_size))
        
        sdas_img = cv2.imread(sdas_path)
        sdas_img = cv2.cvtColor(sdas_img, cv2.COLOR_BGR2RGB)
        sdas_img = cv2.resize(sdas_img, (img_size, img_size))
        
        # Generate Perlin & Foreground mask
        fg_mask = generate_target_foreground_mask(norm_img, category)
        perlin_mask = generate_perlin_noise_mask(shape=(img_size, img_size))
        combined_mask = (perlin_mask * fg_mask).astype(np.float32)
        
        # If mask happens to be empty, force non-empty perlin mask
        if combined_mask.sum() < 10:
            combined_mask = perlin_mask.astype(np.float32)
            
        # Blend anomaly
        blended_img = blend_sdas_anomaly(norm_img, sdas_img, combined_mask)
        mask_vis = (combined_mask * 255.0).astype(np.uint8)
        
        fname = f"{i:04d}.png"
        cv2.imwrite(os.path.join(out_img_dir, fname), cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_mask_dir, fname), mask_vis)
        saved_count += 1
        
    print(f"Successfully saved {saved_count} paired samples for '{category}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Paired RealNet Synthetic Images and GT Binary Masks")
    parser.add_argument('--class_name', type=str, default='all', help="Category name (e.g. bottle, screw) or 'all'")
    parser.add_argument('--dataset_root', type=str, default='datasets/mvtec', help='Path to MVTec dataset root')
    parser.add_argument('--syn_realnet_root', type=str, default='datasets/generated_dataset/realnet', help='Path to RealNet SDAS synthetic folder')
    parser.add_argument('--output_root', type=str, default='datasets/generated_dataset/realnet_paired', help='Path to save paired RealNet dataset')
    parser.add_argument('--num_samples', type=int, default=1500, help='Number of synthetic paired samples to generate per category')
    parser.add_argument('--img_size', type=int, default=224, help='Target resolution')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    ALL_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    if args.class_name == 'all':
        categories = ALL_CLASSES
    else:
        categories = [args.class_name]
        
    for cat in categories:
        generate_paired_dataset_for_category(
            category=cat,
            dataset_root=args.dataset_root,
            syn_realnet_root=args.syn_realnet_root,
            output_root=args.output_root,
            num_samples=args.num_samples,
            img_size=args.img_size
        )
