#!/usr/bin/env python3
"""
Generate Scribble2Label training data from binary masks using skeletonization.

Takes binary mask TIF files (0/255) and generates:
  - RGB images (grayscale -> 3ch)
  - Scribble labels via skeletonize (0=bg, 1=fg, 250=unlabeled)
  - Full masks for validation
  - train/test CSV

Usage:
  python generate_from_masks.py \
      --images_dir /path/to/tif/images/ \
      --output_dir ./examples \
      --modality custom \
      --ratio 0.3
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
from tqdm import tqdm


def remove_corner(mask, coords):
    """Remove corner artifacts from skeleton."""
    h, w = mask.shape
    for coord in coords:
        x, y = coord
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    mask[nx][ny] = 0
    return mask


def scribblize(mask, ratio=0.3, max_dim=1024):
    """
    Generate scribble labels from a binary mask via skeletonization.
    Adapted from preprocess.py to handle 0/255 masks.
    Downscales large images for speed, then maps results back.

    :param mask: binary mask (0=background, 1=foreground)
    :param ratio: fraction of skeleton to keep (0.0-1.0)
    :param max_dim: max dimension for skeletonization (larger images get downscaled)
    :return: (foreground_scribble, background_scribble) as binary arrays
    """
    orig_h, orig_w = mask.shape
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        # Downscale using PIL for speed
        mask_small = np.array(
            Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)
        )
    else:
        mask_small = mask

    # Foreground skeleton
    sk = skeletonize(mask_small)

    # Background skeleton
    i_mask = 1 - mask_small
    i_sk = skeletonize(i_mask)
    coords = corner_peaks(corner_harris(i_sk), min_distance=5)
    i_sk = remove_corner(i_sk, coords)

    # Randomly keep a fraction of foreground skeleton components
    label_sk = label(sk)
    n_sk = np.max(label_sk)
    if n_sk > 0:
        n_remove = int(n_sk * (1 - ratio))
        removes = random.sample(range(1, n_sk + 1), n_remove)
        for i in removes:
            label_sk[label_sk == i] = 0
    sk = (label_sk > 0).astype('uint8')

    # Keep same number of background skeleton components
    n_keep = np.max(label(sk))  # how many fg components we kept
    label_i_sk = label(i_sk)
    n_i_sk = np.max(label_i_sk)
    if n_i_sk > 0 and n_keep > 0:
        n_i_remove = max(0, n_i_sk - n_keep)
        if n_i_remove > 0:
            removes = random.sample(range(1, n_i_sk + 1), n_i_remove)
            for i in removes:
                label_i_sk[label_i_sk == i] = 0
    i_sk = (label_i_sk > 0).astype('uint8')

    # Upscale back to original size if we downscaled
    if scale < 1.0:
        sk = np.array(
            Image.fromarray(sk).resize((orig_w, orig_h), Image.NEAREST)
        )
        i_sk = np.array(
            Image.fromarray(i_sk).resize((orig_w, orig_h), Image.NEAREST)
        )

    return sk, i_sk


def simple_kfold(n, n_splits=5, seed=42):
    
    n_splits = min(n_splits, n)
    if n_splits < 2:
        return [0] * n, 1
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    folds = [0] * n
    for i, idx in enumerate(indices):
        folds[idx] = i % n_splits
    return folds, n_splits


def main():
    parser = argparse.ArgumentParser(
        description='Generate scribble labels from binary masks'
    )
    parser.add_argument(
        '--images_dir', required=True,
        help='Directory containing *_seg.tif binary masks (which are also the images)'
    )
    parser.add_argument(
        '--output_dir', default='./examples',
        help='Output directory (default: ./examples)'
    )
    parser.add_argument(
        '--modality', default='custom',
        help='Modality name (default: custom)'
    )
    parser.add_argument(
        '--ratio', type=float, default=0.3,
        help='Scribble ratio: fraction of skeleton to keep (default: 0.3)'
    )
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()
    random.seed(args.seed)

    scr_name = f'scribble{int(args.ratio * 100)}'

    # Find all TIF files
    tif_files = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    print(f"Found {len(tif_files)} TIF files in {args.images_dir}")

    # Setup output dirs
    img_out = os.path.join(args.output_dir, 'images', args.modality)
    scr_out = os.path.join(args.output_dir, 'labels', args.modality, scr_name)
    full_out = os.path.join(args.output_dir, 'labels', args.modality, 'full')
    csv_dir = os.path.join(args.output_dir, 'labels', args.modality)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(scr_out, exist_ok=True)
    os.makedirs(full_out, exist_ok=True)

    image_ids = []
    skipped = 0

    for fname in tqdm(tif_files, desc="Processing"):
        image_id = os.path.splitext(fname)[0]
        src_path = os.path.join(args.images_dir, fname)

        arr = np.array(Image.open(src_path))

        # Binarize: 0/255 -> 0/1
        mask = (arr > 0).astype('uint8')

        # Skip if mask is all background or all foreground
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

        # Save full mask (keep original 0/255 for validation)
        Image.fromarray(arr).save(os.path.join(full_out, f'{image_id}.png'))

        # Generate and save scribble label
        sk, i_sk = scribblize(mask, ratio=args.ratio)
        scr = np.full_like(mask, 250, dtype=np.uint8)
        scr[i_sk == 1] = 0   # background scribble
        scr[sk == 1] = 1     # foreground scribble
        Image.fromarray(scr).save(os.path.join(scr_out, f'{image_id}.png'))

        image_ids.append(image_id)

    if not image_ids:
        print("No valid images found.")
        return

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

    print(f"\nDone! Processed {len(image_ids)} images (skipped {skipped} empty/full masks)")
    print(f"  Images:    {img_out}/")
    print(f"  Scribbles: {scr_out}/  (ratio={args.ratio})")
    print(f"  Full masks:{full_out}/")
    print(f"  train.csv: {train_path}  ({n_train} images)")
    print(f"  test.csv:  {test_path}  ({n_test} images)")
    print(f"\nUpdate Train.py config:")
    print(f"  name = '{args.modality}'")
    print(f"  scr_dir = './examples/labels/{args.modality}/{scr_name}/'")


if __name__ == '__main__':
    main()
