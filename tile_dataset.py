#!/usr/bin/env python3
"""
Tile large images, scribble labels, and full masks into smaller patches.

Takes the existing converted dataset and splits each image into tiles
for more consistent train/valid behavior.

Usage:
  python tile_dataset.py \
      --input_dir ./examples \
      --modality custom \
      --tile_size 512 \
      --overlap 64
"""

import os
import csv
import argparse
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


def tile_image(arr, tile_size, overlap):
    """
    Split a 2D (or 3D) array into overlapping tiles.
    Returns list of (tile, row_idx, col_idx).
    """
    if arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        h, w = arr.shape

    step = tile_size - overlap
    tiles = []

    for row_idx, y in enumerate(range(0, h - tile_size + 1, step)):
        for col_idx, x in enumerate(range(0, w - tile_size + 1, step)):
            if arr.ndim == 3:
                tile = arr[y:y + tile_size, x:x + tile_size, :]
            else:
                tile = arr[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, row_idx, col_idx))

    return tiles


def has_content(scr_tile, mask_tile, min_labeled_ratio=0.001):
    """
    Check if a tile has enough labeled content to be useful for training.
    Skip tiles where scribble is entirely unlabeled (all 250) or mask is all black.
    """
    labeled = np.sum(scr_tile != 250)
    fg = np.sum(mask_tile > 0)
    return labeled > scr_tile.size * min_labeled_ratio or fg > mask_tile.size * min_labeled_ratio


def simple_kfold(n, n_splits=5, seed=42):
    """Simple KFold split."""
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
        description='Tile dataset into smaller patches'
    )
    parser.add_argument(
        '--input_dir', default='./examples',
        help='Base directory containing images/ and labels/ (default: ./examples)'
    )
    parser.add_argument(
        '--modality', default='custom',
        help='Source modality name (default: custom)'
    )
    parser.add_argument(
        '--output_modality', default=None,
        help='Output modality name (default: {modality}_tile{tile_size})'
    )
    parser.add_argument(
        '--tile_size', type=int, default=512,
        help='Tile size in pixels (default: 512)'
    )
    parser.add_argument(
        '--overlap', type=int, default=64,
        help='Overlap between tiles in pixels (default: 64)'
    )
    parser.add_argument(
        '--scribble_name', default='scribble30',
        help='Scribble directory name (default: scribble30)'
    )
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--skip_empty', action='store_true', default=True,
        help='Skip tiles with no labeled content (default: True)'
    )
    args = parser.parse_args()

    if args.output_modality is None:
        args.output_modality = f'{args.modality}_tile{args.tile_size}'

    # Input paths
    img_in = os.path.join(args.input_dir, 'images', args.modality)
    scr_in = os.path.join(args.input_dir, 'labels', args.modality, args.scribble_name)
    full_in = os.path.join(args.input_dir, 'labels', args.modality, 'full')

    # Output paths
    img_out = os.path.join(args.input_dir, 'images', args.output_modality)
    scr_out = os.path.join(args.input_dir, 'labels', args.output_modality, args.scribble_name)
    full_out = os.path.join(args.input_dir, 'labels', args.output_modality, 'full')
    csv_dir = os.path.join(args.input_dir, 'labels', args.output_modality)

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(scr_out, exist_ok=True)
    os.makedirs(full_out, exist_ok=True)

    # Get list of images
    img_files = sorted([f for f in os.listdir(img_in) if f.endswith('.png')])
    print(f"Found {len(img_files)} images in {img_in}")
    print(f"Tile size: {args.tile_size}, overlap: {args.overlap}")
    print(f"Output modality: {args.output_modality}")

    tile_ids = []
    total_tiles = 0
    skipped_tiles = 0

    for fname in tqdm(img_files, desc="Tiling"):
        image_id = os.path.splitext(fname)[0]

        # Load image, scribble, and mask
        img = np.array(Image.open(os.path.join(img_in, fname)))
        scr = np.array(Image.open(os.path.join(scr_in, fname)))
        mask = np.array(Image.open(os.path.join(full_in, fname)))

        # Tile all three together
        img_tiles = tile_image(img, args.tile_size, args.overlap)
        scr_tiles = tile_image(scr, args.tile_size, args.overlap)
        mask_tiles = tile_image(mask, args.tile_size, args.overlap)

        for (img_t, r, c), (scr_t, _, _), (mask_t, _, _) in zip(img_tiles, scr_tiles, mask_tiles):
            total_tiles += 1

            if args.skip_empty and not has_content(scr_t, mask_t):
                skipped_tiles += 1
                continue

            tile_id = f'{image_id}_r{r:02d}_c{c:02d}'
            tile_ids.append(tile_id)

            Image.fromarray(img_t).save(os.path.join(img_out, f'{tile_id}.png'))
            Image.fromarray(scr_t).save(os.path.join(scr_out, f'{tile_id}.png'))
            Image.fromarray(mask_t).save(os.path.join(full_out, f'{tile_id}.png'))

    if not tile_ids:
        print("No tiles generated.")
        return

    # Generate CSV
    folds, actual_n_folds = simple_kfold(len(tile_ids), n_splits=args.n_folds)
    test_fold = actual_n_folds - 1

    train_path = os.path.join(csv_dir, 'train.csv')
    test_path = os.path.join(csv_dir, 'test.csv')

    with open(train_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageID', 'fold'])
        for tid, fold in zip(tile_ids, folds):
            if fold != test_fold:
                writer.writerow([tid, fold])

    with open(test_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageID', 'fold'])
        for tid, fold in zip(tile_ids, folds):
            if fold == test_fold:
                writer.writerow([tid, fold])

    n_train = sum(1 for f in folds if f != test_fold)
    n_test = sum(1 for f in folds if f == test_fold)

    print(f"\nDone!")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Skipped (empty): {skipped_tiles}")
    print(f"  Saved: {len(tile_ids)} tiles")
    print(f"  Train: {n_train}, Test: {n_test}")
    print(f"\n  Images:    {img_out}/")
    print(f"  Scribbles: {scr_out}/")
    print(f"  Full masks:{full_out}/")
    print(f"  train.csv: {train_path}")
    print(f"  test.csv:  {test_path}")
    print(f"\nUpdate Train.py config:")
    print(f"  name = '{args.output_modality}'")


if __name__ == '__main__':
    main()
