#!/usr/bin/env python3
"""
Process a 3D TIFF stack for Scribble2Label training.

Samples every N slices, binarizes, generates scribble labels via skeletonization,
and saves in the expected directory structure.

Usage:
  python process_3d_tiff.py
      --tiff_path /data/datahere/Todd/data/xy_assemble_0/xystd_assemble_0_512_cubic_big_neuron/xystd_assemble_0_mask.tif \
      --output_dir ./examples \
      --modality xyvar_assemble_0_512_cubic \
      --step 10 \
      --ratio 0.5
"""

import os
import csv
import argparse
import random

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage import convolve
from tqdm import tqdm
import tifffile


def trim_skeleton_endpoints(skeleton, n_pixels=5):
    """Trim n_pixels from both ends of skeleton lines by iteratively
    removing endpoints (pixels with only 1 neighbor in 8-connected nbhd)."""
    if n_pixels <= 0:
        return skeleton
    sk = skeleton.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    for _ in range(n_pixels):
        neighbors = convolve(sk, kernel, mode='constant', cval=0)
        endpoints = (sk == 1) & (neighbors == 1)
        if not endpoints.any():
            break
        sk[endpoints] = 0
    return sk


def remove_corner(mask, coords):
    h, w = mask.shape
    for coord in coords:
        x, y = coord
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    mask[nx][ny] = 0
    return mask


def scribblize(mask, ratio=1.0, max_dim=1024, trim=0):
    original_height, original_width = mask.shape
    downscale_factor = 1.0
    if max(original_height, original_width) > max_dim:
        downscale_factor = max_dim / max(original_height, original_width)
        resized_height = int(original_height * downscale_factor)
        resized_width = int(original_width * downscale_factor)
        downscaled_mask = np.array(
            Image.fromarray(mask).resize((resized_width, resized_height), Image.NEAREST)
        )
    else:
        downscaled_mask = mask

    # Foreground skeleton
    foreground_skeleton = skeletonize(downscaled_mask).astype(np.uint8)
    if trim > 0:
        foreground_skeleton = trim_skeleton_endpoints(foreground_skeleton, trim)

    # Background skeleton
    background_mask = 1 - downscaled_mask
    background_skeleton = skeletonize(background_mask).astype(np.uint8)
    corner_coords = corner_peaks(corner_harris(background_skeleton), min_distance=5)
    background_skeleton = remove_corner(background_skeleton, corner_coords)
    if trim > 0:
        background_skeleton = trim_skeleton_endpoints(background_skeleton, trim)

    # Control foreground skeleton ratio: randomly remove connected components
    foreground_components = label(foreground_skeleton)
    num_foreground_components = np.max(foreground_components)
    if num_foreground_components > 0:
        num_foreground_to_remove = int(num_foreground_components * (1 - ratio))
        if num_foreground_to_remove > 0:
            components_to_remove = random.sample(range(1, num_foreground_components + 1), num_foreground_to_remove)
            for component_id in components_to_remove:
                foreground_components[foreground_components == component_id] = 0
    foreground_skeleton = (foreground_components > 0).astype('uint8')

    # Balance background components to match foreground count
    num_foreground_kept = np.max(label(foreground_skeleton))
    background_components = label(background_skeleton)
    num_background_components = np.max(background_components)
    if num_background_components > 0 and num_foreground_kept > 0:
        num_background_to_remove = max(0, num_background_components - num_foreground_kept)
        if num_background_to_remove > 0:
            components_to_remove = random.sample(range(1, num_background_components + 1), num_background_to_remove)
            for component_id in components_to_remove:
                background_components[background_components == component_id] = 0
    background_skeleton = (background_components > 0).astype('uint8')

    # Upscale back to original size if downscaled
    if downscale_factor < 1.0:
        foreground_skeleton = np.array(Image.fromarray(foreground_skeleton).resize((original_width, original_height), Image.NEAREST))
        background_skeleton = np.array(Image.fromarray(background_skeleton).resize((original_width, original_height), Image.NEAREST))

    return foreground_skeleton, background_skeleton



def main():
    parser = argparse.ArgumentParser(description='Process 3D TIFF for Scribble2Label')
    parser.add_argument('--tiff_path', required=True, help='Path to 3D TIFF file')
    parser.add_argument('--output_dir', default='./examples', help='Output directory')
    parser.add_argument('--modality', default='fiji_BC', help='Modality name')
    parser.add_argument('--step', type=int, default=10, help='Sample every N slices')
    parser.add_argument('--ratio', type=float, default=1.0, help='Scribble ratio')
    parser.add_argument('--trim', type=int, default=0,
                        help='Trim N pixels from both ends of each skeleton line (default: 0)')
    #parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    random.seed(args.seed)

    scr_name = f'scribble{int(args.ratio * 100)}'

    # Output paths
    img_out = os.path.join(args.output_dir, 'images', args.modality)
    scr_out = os.path.join(args.output_dir, 'labels', args.modality, scr_name)
    full_out = os.path.join(args.output_dir, 'labels', args.modality, 'full')
    csv_dir = os.path.join(args.output_dir, 'labels', args.modality)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(scr_out, exist_ok=True)
    os.makedirs(full_out, exist_ok=True)

    # Read 3D TIFF
    print(f"Reading {args.tiff_path} ...")
    stack = tifffile.imread(args.tiff_path)
    n_slices = stack.shape[0]
    print(f"Stack shape: {stack.shape}, dtype: {stack.dtype}")

    # Sample slices
    slice_indices = list(range(0, n_slices, args.step))
    print(f"Sampling every {args.step} slices: {len(slice_indices)} slices from {n_slices}")

    image_ids = []
    skipped = 0

    for si in tqdm(slice_indices, desc="Processing slices"):
        arr = stack[si]
        image_id = f'slice_{si:04d}'

        # Skip if already processed
        if os.path.exists(os.path.join(img_out, f'{image_id}.png')):
            image_ids.append(image_id)
            continue

        # Binarize
        mask = (arr > 220).astype('uint8')

        # Skip empty or fully filled slices
        fg_ratio = mask.sum() / mask.size
        if fg_ratio == 0 or fg_ratio == 1:
            skipped += 1
            continue

        # Save RGB image (grayscale -> 3ch)
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            rgb = arr[:, :, :3]
        Image.fromarray(rgb).save(os.path.join(img_out, f'{image_id}.png'))

        # Save full mask (0/255 for validation)
        mask_255 = (mask * 255).astype('uint8')
        Image.fromarray(mask_255).save(os.path.join(full_out, f'{image_id}.png'))

        # Generate scribble label
        sk, i_sk = scribblize(mask, ratio=args.ratio, trim=args.trim)
        scr = np.full_like(mask, 250, dtype=np.uint8)
        scr[i_sk == 1] = 0
        scr[sk == 1] = 1
        Image.fromarray(scr).save(os.path.join(scr_out, f'{image_id}.png'))

        image_ids.append(image_id)

    if not image_ids:
        print("No valid slices found.")
        return
    """
    # Generate CSV
    folds, actual_n_folds = simple_kfold(len(image_ids), n_splits=args.n_folds)
    test_fold = actual_n_folds - 1

    train_path = os.path.join(csv_dir, 'train.csv')
    test_path = os.path.join(csv_dir, 'test.csv')

    with open(train_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageID', 'fold'])
        for img_id, fold in zip(image_ids, folds):
            if fold != test_fold:
                writer.writerow([img_id, fold])

    with open(test_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageID', 'fold'])
        for img_id, fold in zip(image_ids, folds):
            if fold == test_fold:
                writer.writerow([img_id, fold])

    n_train = sum(1 for f in folds if f != test_fold)
    n_test = sum(1 for f in folds if f == test_fold)
    """
    print(f"\nDone! Processed {len(image_ids)} slices (skipped {skipped} empty/full)")
    print(f"  Images:    {img_out}/")
    print(f"  Scribbles: {scr_out}/  (ratio={args.ratio})")
    print(f"  Full masks:{full_out}/")
    #print(f"  train.csv: {train_path}  ({n_train} images)")
    #print(f"  test.csv:  {test_path}  ({n_test} images)")
    print(f"\nNext steps:")
    print(f"  1. Tile: python tile_dataset.py --input_dir {args.output_dir} --modality {args.modality} --tile_size 512 --overlap 64 --scribble_name {scr_name}")
    print(f"  2. Update Train.py config name to '{args.modality}_tile512'")


if __name__ == '__main__':
    main()
