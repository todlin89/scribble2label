#!/usr/bin/env python3
"""
Morphological closing + hole filling for binary segmentation masks.

1. Closing (dilation then erosion): connects nearby borders / fills small gaps
2. Fill holes: fills interior regions completely surrounded by foreground

Usage:
  # Closing (kernel=5) + fill holes:
  python fill_holes.py \
      --input_dir ./logs/fiji_BC_r50_tile512/predictions_xyvar_assemble_0/ \
      --output_dir ./logs/fiji_BC_r50_tile512/predictions_xyvar_assemble_0_filled/ \
      --closing 5

  # Fill holes only (no closing):
  python fill_holes.py --input_dir ./path/to/masks/

  # Preview without saving:
  python fill_holes.py --input_dir ./path/to/masks/ --closing 5 --preview
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import disk, binary_closing
from tqdm import tqdm
from glob import glob


def process_mask(mask, closing_radius=0):
    """
    Apply morphological closing and hole filling to a binary mask.

    Args:
        mask: 2D numpy array, binary (0 and 255)
        closing_radius: radius of disk kernel for closing (0 = skip closing)
    Returns:
        result: 2D numpy array with closing applied and holes filled
    """
    binary = (mask > 0)

    # Step 1: Morphological closing (dilation → erosion)
    # Connects nearby edges and fills small gaps
    if closing_radius > 0:
        kernel = disk(closing_radius)
        binary = binary_closing(binary, kernel)

    # Step 2: Fill holes
    # Fills 0-regions completely enclosed by foreground
    result = ndimage.binary_fill_holes(binary).astype(np.uint8)

    return (result * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Morphological closing + hole filling for binary masks')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing binary mask PNGs (0/255)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: input_dir + _filled)')
    parser.add_argument('--closing', type=int, default=0,
                        help='Disk radius for morphological closing (0 = skip, default: 0)')
    parser.add_argument('--preview', action='store_true',
                        help='Preview mode: show stats without saving')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip('/') + '_filled'

    # Find all PNG files
    file_paths = sorted(glob(os.path.join(args.input_dir, '*.png')))
    print(f'Found {len(file_paths)} mask files in {args.input_dir}')
    print(f'Closing radius: {args.closing} (0 = disabled)')

    if len(file_paths) == 0:
        print('No PNG files found.')
        return

    if not args.preview:
        os.makedirs(args.output_dir, exist_ok=True)

    total_modified = 0
    total_pixels_changed = 0

    for fpath in tqdm(file_paths, desc='Processing'):
        mask = np.array(Image.open(fpath))
        result = process_mask(mask, closing_radius=args.closing)

        # Count changed pixels
        pixels_changed = np.sum((result > 0).astype(int) - (mask > 0).astype(int) > 0)
        if pixels_changed > 0:
            total_modified += 1
            total_pixels_changed += pixels_changed

        if not args.preview:
            fname = os.path.basename(fpath)
            Image.fromarray(result).save(os.path.join(args.output_dir, fname))

    print(f'\nDone!')
    print(f'  Closing radius: {args.closing}')
    print(f'  Images modified: {total_modified} / {len(file_paths)}')
    print(f'  Total pixels filled: {total_pixels_changed:,}')
    if not args.preview:
        print(f'  Output: {args.output_dir}')


if __name__ == '__main__':
    main()
