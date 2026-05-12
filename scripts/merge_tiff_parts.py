#!/usr/bin/env python3
"""Concatenate per-GPU partition TIFFs (from run_multi_gpu_inference.sh) along z.

Usage:
  python scripts/merge_tiff_parts.py \
      --parts_dir ./logs/thres_90_3d_basicunet/inference_mg \
      --out_path  ./logs/thres_90_3d_basicunet/inference_mg/pred_mask.tif

Scans <parts_dir>/part_0, part_1, ... in numeric order and writes a single
BigTIFF. Each partition should already contain the output slices for its
assigned z-range (no overlap between partitions).
"""

import argparse
import glob
import os
import re

import tifffile


def natural_part_key(path):
    m = re.search(r'part_(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parts_dir', required=True,
                   help='Directory containing part_0, part_1, ... subdirs')
    p.add_argument('--out_path', required=True,
                   help='Output BigTIFF path')
    p.add_argument('--filename', default='pred_mask.tif',
                   help='Per-partition TIFF filename (default: pred_mask.tif)')
    args = p.parse_args()

    parts = sorted(
        glob.glob(os.path.join(args.parts_dir, 'part_*')),
        key=natural_part_key,
    )
    parts = [p for p in parts if os.path.isdir(p)]
    if not parts:
        raise SystemExit(f'No part_* directories under {args.parts_dir}')

    total = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or '.', exist_ok=True)
    with tifffile.TiffWriter(args.out_path, bigtiff=True) as writer:
        for part in parts:
            src = os.path.join(part, args.filename)
            if not os.path.isfile(src):
                raise SystemExit(f'Missing {src}')
            # Iterate pages directly: memmap/imread collapse single-page or
            # writer-generated multi-page TIFFs inconsistently. Reading each
            # IFD page keeps the count correct regardless of how they were written.
            with tifffile.TiffFile(src) as tf:
                n = len(tf.pages)
                print(f'  {src}: {n} pages, page shape {tf.pages[0].shape}, dtype {tf.pages[0].dtype}')
                for page in tf.pages:
                    writer.write(page.asarray())
            total += n

    print(f'Wrote {total} slices -> {args.out_path}')


if __name__ == '__main__':
    main()
