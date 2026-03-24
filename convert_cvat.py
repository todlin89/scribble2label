#!/usr/bin/env python3
"""
Convert CVAT XML (for images 1.1) annotations to Scribble2Label format.

Reads mask annotations from CVAT annotations.xml and converts them to the
scribble label PNG format expected by Scribble2Label.

CVAT RLE details:
  - Alternating counts of 0s and 1s, starting with 0s
  - Actual dimensions are (width+1) x (height+1)
  - Row-major (C) order

Output structure:
  output_dir/
    images/{modality}/{image_id}.png        (RGB 8-bit)
    labels/{modality}/scribble100/{image_id}.png  (0=bg, 1=fg, 250=unlabeled)
    labels/{modality}/full/{image_id}.png    (placeholder, all zeros)
    labels/{modality}/train.csv
    labels/{modality}/test.csv

Usage:
  python convert_cvat.py \\
      --annotations /path/to/annotations.xml \\
      --images_dir /path/to/tif/images/ \\
      --output_dir ./examples \\
      --modality custom
"""

import os
import csv
import argparse
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm


def decode_cvat_rle(rle_str, width, height):
    """
    Decode CVAT mask RLE to a 2D binary numpy array.

    CVAT RLE encodes masks as alternating run-lengths of 0s and 1s,
    starting with 0. The actual pixel count is (width+1)*(height+1).
    """
    counts = [int(x.strip()) for x in rle_str.split(',')]
    actual_w = width + 1
    actual_h = height + 1
    expected = actual_w * actual_h

    flat = np.zeros(expected, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        val = i % 2  # 0 for even indices, 1 for odd
        flat[pos:pos + count] = val
        pos += count

    assert pos == expected, (
        f"RLE decoded {pos} pixels but expected {expected} "
        f"(w+1={actual_w}, h+1={actual_h})"
    )

    mask = flat.reshape((actual_h, actual_w))
    return mask


def parse_cvat_xml(xml_path):
    """Parse CVAT annotations.xml and return per-image annotation info."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect label names
    labels = []
    for label_elem in root.findall('.//task/labels/label'):
        labels.append(label_elem.find('name').text)
    print(f"Labels in annotation: {labels}")

    images = {}
    for img_elem in root.findall('.//image'):
        img_name = img_elem.get('name')
        img_width = int(img_elem.get('width'))
        img_height = int(img_elem.get('height'))

        masks = []
        for mask_elem in img_elem.findall('mask'):
            masks.append({
                'label': mask_elem.get('label'),
                'rle': mask_elem.get('rle'),
                'left': int(mask_elem.get('left')),
                'top': int(mask_elem.get('top')),
                'width': int(mask_elem.get('width')),
                'height': int(mask_elem.get('height')),
            })

        images[img_name] = {
            'width': img_width,
            'height': img_height,
            'masks': masks,
        }

    return images, labels


def create_scribble_label(img_info, fg_label='object', bg_label='background'):
    """
    Create a scribble label image from CVAT mask annotations.

    Pixel values:
      0   = background scribble
      1   = foreground scribble
      250 = unlabeled
    """
    h, w = img_info['height'], img_info['width']
    scribble = np.full((h, w), 250, dtype=np.uint8)

    for mask_info in img_info['masks']:
        bbox_mask = decode_cvat_rle(
            mask_info['rle'],
            mask_info['width'],
            mask_info['height'],
        )

        left = mask_info['left']
        top = mask_info['top']
        mh, mw = bbox_mask.shape

        # Clip to image bounds
        paste_h = min(mh, h - top)
        paste_w = min(mw, w - left)

        if mask_info['label'] == fg_label:
            region = scribble[top:top + paste_h, left:left + paste_w]
            region[bbox_mask[:paste_h, :paste_w] == 1] = 1
        elif mask_info['label'] == bg_label:
            region = scribble[top:top + paste_h, left:left + paste_w]
            region[bbox_mask[:paste_h, :paste_w] == 1] = 0

    return scribble


def load_image_as_rgb(path):
    """Load an image file and convert to RGB uint8."""
    img = Image.open(path)
    arr = np.array(img)

    # Handle 16-bit
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float64) / arr.max() * 255).astype(np.uint8)

    # Handle grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha

    return arr


def simple_kfold(n, n_splits=5, seed=42):
    """Simple KFold split without sklearn dependency."""
    # Cap folds to number of samples
    n_splits = min(n_splits, n)
    if n_splits < 2:
        # Not enough data to split; put everything in fold 0 (train)
        return [0] * n, n_splits

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    folds = [0] * n
    for i, idx in enumerate(indices):
        folds[idx] = i % n_splits
    return folds, n_splits


def main():
    parser = argparse.ArgumentParser(
        description='Convert CVAT XML annotations to Scribble2Label format'
    )
    parser.add_argument(
        '--annotations', required=True,
        help='Path to CVAT annotations.xml'
    )
    parser.add_argument(
        '--images_dir', required=True,
        help='Directory containing original images (TIF/PNG)'
    )
    parser.add_argument(
        '--output_dir', default='./examples',
        help='Output directory (default: ./examples)'
    )
    parser.add_argument(
        '--modality', default='custom',
        help='Modality name for subdirectory (default: custom)'
    )
    parser.add_argument(
        '--fg_label', default='object',
        help='Foreground label name in CVAT (default: object)'
    )
    parser.add_argument(
        '--bg_label', default='background',
        help='Background label name in CVAT (default: background)'
    )
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    args = parser.parse_args()

    # Parse XML
    print(f"Parsing {args.annotations} ...")
    images, labels = parse_cvat_xml(args.annotations)

    # Filter to images with at least one mask annotation
    annotated = {
        name: info for name, info in images.items()
        if info['masks']
    }
    print(f"Total images in XML: {len(images)}")
    print(f"Images with annotations: {len(annotated)}")

    if not annotated:
        print("No annotated images found. Nothing to convert.")
        return

    # Setup output directories
    img_out = os.path.join(args.output_dir, 'images', args.modality)
    scr_out = os.path.join(args.output_dir, 'labels', args.modality, 'scribble100')
    full_out = os.path.join(args.output_dir, 'labels', args.modality, 'full')
    csv_dir = os.path.join(args.output_dir, 'labels', args.modality)

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(scr_out, exist_ok=True)
    os.makedirs(full_out, exist_ok=True)

    # Convert each annotated image
    image_ids = []
    for img_name, img_info in tqdm(annotated.items(), desc="Converting"):
        image_id = os.path.splitext(img_name)[0]

        # Load source image
        src_path = os.path.join(args.images_dir, img_name)
        if not os.path.exists(src_path):
            print(f"  Warning: image not found, skipping: {src_path}")
            continue

        rgb = load_image_as_rgb(src_path)
        Image.fromarray(rgb).save(os.path.join(img_out, f'{image_id}.png'))

        # Create scribble label
        scribble = create_scribble_label(img_info, args.fg_label, args.bg_label)
        Image.fromarray(scribble).save(
            os.path.join(scr_out, f'{image_id}.png')
        )

        # Create placeholder full mask (all zeros — no ground truth available)
        full_mask = np.zeros(
            (img_info['height'], img_info['width']), dtype=np.uint8
        )
        Image.fromarray(full_mask).save(
            os.path.join(full_out, f'{image_id}.png')
        )

        image_ids.append(image_id)

    if not image_ids:
        print("No images were successfully converted.")
        return

    # Generate train/test CSV
    folds, actual_n_folds = simple_kfold(len(image_ids), n_splits=args.n_folds)
    if actual_n_folds < args.n_folds:
        print(f"\n  Note: n_folds reduced from {args.n_folds} to {actual_n_folds} "
              f"(only {len(image_ids)} images)")
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

    print(f"\nDone! Converted {len(image_ids)} images.")
    print(f"  Images:    {img_out}/")
    print(f"  Scribbles: {scr_out}/")
    print(f"  Full masks:{full_out}/  (placeholder — all zeros)")
    print(f"  train.csv: {train_path}  ({n_train} images)")
    print(f"  test.csv:  {test_path}  ({n_test} images)")
    print()
    print("To train, update Train.py config:")
    print(f"  name = '{args.modality}'")
    print(f"  data_dir = '{img_out}/'")
    print(f"  scr_dir  = '{scr_out}/'")
    print(f"  mask_dir = '{full_out}/'")
    print(f"  df_path  = '{train_path}'")
    print()
    print("WARNING: Full masks are placeholders (all zeros).")
    print("  Validation metrics will NOT be meaningful.")
    print("  Provide real ground truth masks if you need proper evaluation.")


if __name__ == '__main__':
    main()
