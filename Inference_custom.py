"""
- Loads model trained on fluorescence example data
- Reads grayscale TIF/TIFF images (including multi-frame & 16-bit)
- Tiles large images into 256x256 patches, runs inference, then stitches back

Optimization changelog (based on Nsight Systems profiling):
==========================================================
Baseline profile (single_gpu_profile, 56 frames): avg 1.894s per frame

1. stitch_tiles — cache count_map + reciprocal (0.318s → 0.151s per frame, 2.11x)
   - WHY: count_map only depends on image size + tile layout, which is identical
     across all frames. Previously re-computed from scratch every frame.
   - WHAT: _get_count_map() caches the reciprocal of count_map on first call.
     Subsequent frames skip the loop entirely and use multiplication instead
     of division.

2. H2D transfer — pin_memory() + non_blocking=True (0.149s → 0.067s per frame, 2.27x)
   - WHY: Standard .to(device) allocates pageable host memory, which requires
     an extra CPU→pinned copy before DMA to GPU. This blocks the CPU.
   - WHAT: pin_memory() pre-allocates in pinned (page-locked) memory, and
     non_blocking=True lets the DMA proceed asynchronously so CPU can continue
     preparing the next batch. cudaStreamSynchronize dropped 97% (0.28s → 0.01s).

After optimization (optimized_v2_profile, 69 frames): avg 1.262s per frame
Overall speedup: 1.894s → 1.262s per frame (33% faster, 1.50x)
Estimated full run: 4160 frames × 1.262s ≈ 87 min (was ~131 min, saves ~44 min)
Remaining bottleneck: GPU compute itself (visible in D2H_transfer NVTX range
due to CUDA async — the .cpu() call waits for GPU to finish).
"""

import os
import argparse
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
import tifffile
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

from segmentation_models_pytorch import Unet


def get_args():
    parser = argparse.ArgumentParser(description='Scribble2Label Custom Inference')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input images (supports .tif, .tiff, .png, .jpg)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to save predictions (default: input_dir/predictions)')
    parser.add_argument('--model_path', type=str, default='./logs/fiji_BC_r50_tile512/best_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Tile size for inference (default: 512)')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap between tiles to reduce edge artifacts (default: 32)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for single GPU (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for tile inference (default: 16)')
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Use all available GPUs for inference')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU IDs to use (default: all available, e.g. 0,1,2,3)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable NVTX profiling markers for Nsight Systems')
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained U-Net model."""
    model = Unet(encoder_name='resnet50', encoder_weights='imagenet',
                 decoder_use_batchnorm=True, decoder_attention_type='scse',
                 classes=2, activation=None)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Model loaded: {model_path}')
    return model


def normalize_to_uint8(arr):
    """Normalize any dtype array to uint8 (0-255)."""
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def load_single_image(img):
    """Convert a PIL Image to RGB uint8 numpy array (H, W, 3)."""
    arr = np.array(img)
    # 16-bit or other non-uint8 dtypes
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    # grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def load_frames(path):
    """
    Load image file and yield (frame_index, RGB uint8 numpy array) tuples.
    Handles multi-frame TIFF stacks (via tifffile) and single images.
    """
    # Try tifffile first for 3D TIFF stacks
    if path.lower().endswith(('.tif', '.tiff')):
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[1] > 1 and arr.shape[2] > 1:
            # Could be 3D stack (Z, H, W) or single RGB (H, W, 3)
            if arr.shape[2] <= 4:
                # Likely single RGB/RGBA image (H, W, 3 or 4)
                yield 0, load_single_image(Image.fromarray(arr))
            else:
                # 3D stack (Z, H, W)
                for i in range(arr.shape[0]):
                    yield i, load_single_image(Image.fromarray(arr[i]))
            return
        elif arr.ndim == 2:
            # Single grayscale image
            yield 0, load_single_image(Image.fromarray(arr))
            return

    # Fallback to PIL for other formats
    img = Image.open(path)
    n_frames = getattr(img, 'n_frames', 1)

    if n_frames == 1:
        yield 0, load_single_image(img)
    else:
        for i in range(n_frames):
            img.seek(i)
            yield i, load_single_image(img)


def extract_tiles(image, tile_size, overlap):
    """
    Extract overlapping tiles from a large image.
    Returns list of (tile, row_start, col_start) tuples and padded image shape.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    # Pad image so tiles cover the entire area
    pad_h = (stride - (h - tile_size) % stride) % stride if h > tile_size else tile_size - h
    pad_w = (stride - (w - tile_size) % stride) % stride if w > tile_size else tile_size - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    tiles = []
    padded_h, padded_w = padded.shape[:2]
    for r in range(0, padded_h - tile_size + 1, stride):
        for c in range(0, padded_w - tile_size + 1, stride):
            tile = padded[r:r + tile_size, c:c + tile_size]
            tiles.append((tile, r, c))
    return tiles, padded.shape[:2], (h, w)


_stitch_cache = {}

def _get_count_map(padded_shape, positions, tile_size):
    """Cache count_map — same image size + tile layout always produces the same map."""
    cache_key = (padded_shape, tuple(positions), tile_size)
    if cache_key not in _stitch_cache:
        padded_h, padded_w = padded_shape
        count_map = np.zeros((padded_h, padded_w), dtype=np.float32)
        for r, c in positions:
            count_map[r:r + tile_size, c:c + tile_size] += 1.0
        np.maximum(count_map, 1.0, out=count_map)
        # Pre-compute reciprocal to replace division with multiplication
        np.reciprocal(count_map, out=count_map)
        _stitch_cache[cache_key] = count_map
    return _stitch_cache[cache_key]


def stitch_tiles(tiles_with_preds, padded_shape, original_shape, tile_size, overlap):
    """
    Stitch predicted tiles back into a full image using averaging in overlap regions.
    """
    padded_h, padded_w = padded_shape
    orig_h, orig_w = original_shape

    # Collect positions and build/cache count map
    positions = [(r, c) for _, r, c in tiles_with_preds]
    inv_count_map = _get_count_map(padded_shape, positions, tile_size)

    # Accumulate predictions into pre-allocated buffer
    prediction_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    for pred, r, c in tiles_with_preds:
        prediction_sum[r:r + tile_size, c:c + tile_size] += pred

    # Multiply by cached instead of dividing
    prediction_sum *= inv_count_map

    # Crop to original size and threshold
    return (prediction_sum[:orig_h, :orig_w] > 0.5).astype(np.uint8)


from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

# Module-level singleton — avoid re-creating per call
_INFERENCE_TRANSFORM = Compose([Normalize(), ToTensorV2()])


def inference_tiles(model, tiles, device, batch_size, tile_size, profile=False):
    """Run inference on a list of tiles in batches."""
    all_preds = []
    tile_batch = []
    positions = []

    for tile, r, c in tiles:
        if profile:
            nvtx.range_push("tile_preprocess")
        augmented = _INFERENCE_TRANSFORM(image=tile)
        tile_tensor = augmented['image']
        tile_batch.append(tile_tensor)
        positions.append((r, c))
        if profile:
            nvtx.range_pop()

        if len(tile_batch) == batch_size:
            if profile:
                nvtx.range_push(f"batch_inference_{len(tile_batch)}")
            preds = _run_batch(model, tile_batch, device, profile)
            if profile:
                nvtx.range_pop()
            for pred, (pr, pc) in zip(preds, positions):
                all_preds.append((pred, pr, pc))
            tile_batch = []
            positions = []

    # Handle remaining tiles
    if tile_batch:
        if profile:
            nvtx.range_push(f"batch_inference_{len(tile_batch)}")
        preds = _run_batch(model, tile_batch, device, profile)
        if profile:
            nvtx.range_pop()
        for pred, (pr, pc) in zip(preds, positions):
            all_preds.append((pred, pr, pc))

    return all_preds


def _run_batch(model, tile_batch, device, profile=False):
    """Run a batch of tiles through the model."""
    if profile:
        nvtx.range_push("H2D_transfer")
    # batch = torch.stack(tile_batch).to(device)
    batch = torch.stack(tile_batch).pin_memory().to(device, non_blocking=True)
    if profile:
        nvtx.range_pop()

    with torch.no_grad():
        if profile:
            nvtx.range_push("forward_pass")
        outputs = model(batch)
        if profile:
            nvtx.range_pop()

        if profile:
            nvtx.range_push("softmax")
        probs = F.softmax(outputs, dim=1)
        if profile:
            nvtx.range_pop()

    if profile:
        nvtx.range_push("D2H_transfer")
    result = probs[:, 1].cpu().numpy()
    if profile:
        nvtx.range_pop()
    # Return foreground probability (class 1)
    return result


def process_single_frame(image, model, device, args):
    """Run tiling inference on a single frame and return the stitched result."""
    profile = getattr(args, 'profile', False)

    if profile:
        nvtx.range_push("extract_tiles")
    tiles, padded_shape, original_shape = extract_tiles(
        image, args.tile_size, args.overlap)
    if profile:
        nvtx.range_pop()

    if profile:
        nvtx.range_push("inference_tiles")
    preds_with_pos = inference_tiles(
        model, tiles, device, args.batch_size, args.tile_size, profile)
    if profile:
        nvtx.range_pop()

    if profile:
        nvtx.range_push("stitch_tiles")
    result = stitch_tiles(
        preds_with_pos, padded_shape, original_shape,
        args.tile_size, args.overlap)
    if profile:
        nvtx.range_pop()

    return result


def gpu_worker(gpu_id, frame_indices, stack, output_dir, args, result_dict):
    """Worker function: each GPU processes its assigned frames."""
    device = torch.device(f'cuda:{gpu_id}')
    model = load_model(args.model_path, device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    profile = getattr(args, 'profile', False)
    processed = 0
    worker_time = 0.0
    for frame_idx in tqdm(frame_indices, desc=f'GPU {gpu_id}', position=gpu_id):
        if profile:
            nvtx.range_push(f"gpu{gpu_id}_frame_{frame_idx}")

        if profile:
            nvtx.range_push("load_image")
        image = load_single_image(Image.fromarray(stack[frame_idx]))
        if profile:
            nvtx.range_pop()

        torch.cuda.synchronize(device)
        t_start = time.time()
        result = process_single_frame(image, model, device, args)
        torch.cuda.synchronize(device)
        elapsed = time.time() - t_start
        worker_time += elapsed

        if profile:
            nvtx.range_push("save_image")
        save_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        Image.fromarray(result * 255).save(save_path)
        if profile:
            nvtx.range_pop()

        processed += 1
        if profile:
            nvtx.range_pop()

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    result_dict[gpu_id] = {
        'processed': processed,
        'time': worker_time,
        'peak_mem': peak_mem,
        'peak_reserved': peak_reserved,
    }


def main():
    args = get_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'predictions')
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all images
    extensions = ['*.tif', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(image_paths)
    print(f'Found {len(image_paths)} image file(s) in {args.input_dir}')

    if len(image_paths) == 0:
        print('No images found. Check your input directory.')
        return

    # Determine GPU list
    if args.multi_gpu:
        if args.gpu_ids:
            gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        print(f'Multi-GPU mode: using GPUs {gpu_ids}')
    else:
        gpu_ids = None

    total_frames = 0
    total_time = 0.0

    # Process each image file
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Check number of frames
        if img_path.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(img_path) as tif:
                shape = tif.series[0].shape
            if len(shape) == 3 and shape[2] > 4:
                n_frames = shape[0]
            else:
                n_frames = 1
        else:
            img_check = Image.open(img_path)
            n_frames = getattr(img_check, 'n_frames', 1)
            img_check.close()

        if n_frames == 1:
            # Single frame - always single GPU
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            model = load_model(args.model_path, device)

            print(f'Processing: {os.path.basename(img_path)} (single frame)')
            _, image = next(load_frames(img_path))

            if args.profile:
                nvtx.range_push(f"frame_{basename}")
            t_start = time.time()
            result = process_single_frame(image, model, device, args)
            elapsed = time.time() - t_start
            if args.profile:
                nvtx.range_pop()
            total_time += elapsed
            total_frames += 1
            print(f'  -> Inference time: {elapsed:.3f}s')

            save_path = os.path.join(args.output_dir, f'{basename}.png')
            Image.fromarray(result * 255).save(save_path)

        elif gpu_ids and len(gpu_ids) > 1:
            # Multi-frame + Multi-GPU
            print(f'Processing: {os.path.basename(img_path)} ({n_frames} frames) on {len(gpu_ids)} GPUs')
            frame_dir = os.path.join(args.output_dir, basename)
            os.makedirs(frame_dir, exist_ok=True)

            # Read entire stack into shared memory
            print(f'Reading stack...')
            stack = tifffile.imread(img_path)

            # Split frames across GPUs
            all_frame_indices = list(range(n_frames))
            chunks = [[] for _ in gpu_ids]
            for i, idx in enumerate(all_frame_indices):
                chunks[i % len(gpu_ids)].append(idx)

            for i, gpu_id in enumerate(gpu_ids):
                print(f'  GPU {gpu_id}: {len(chunks[i])} frames')

            t_start = time.time()

            # Launch processes with shared result dict
            mp.set_start_method('spawn', force=True)
            manager = mp.Manager()
            result_dict = manager.dict()
            processes = []
            for i, gpu_id in enumerate(gpu_ids):
                p = mp.Process(target=gpu_worker,
                               args=(gpu_id, chunks[i], stack, frame_dir, args, result_dict))
                p.start()
                processes.append(p)

            # Wait for all to finish
            for p in processes:
                p.join()

            elapsed = time.time() - t_start
            total_time += elapsed
            total_frames += n_frames

            # Print per-GPU stats
            print(f'\n  Per-GPU stats:')
            for gpu_id in gpu_ids:
                if gpu_id in result_dict:
                    r = result_dict[gpu_id]
                    print(f'    GPU {gpu_id}: {r["processed"]} frames, '
                          f'{r["time"]:.3f}s, '
                          f'peak alloc {r["peak_mem"]:.1f} MB, '
                          f'peak reserved {r["peak_reserved"]:.1f} MB')
            print(f'  -> Total wall time: {elapsed:.3f}s ({elapsed/n_frames:.3f}s per frame)')

            del stack

        else:
            # Multi-frame + Single GPU
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            model = load_model(args.model_path, device)

            print(f'Processing: {os.path.basename(img_path)} ({n_frames} frames)')
            frame_dir = os.path.join(args.output_dir, basename)
            os.makedirs(frame_dir, exist_ok=True)

            for frame_idx, image in tqdm(load_frames(img_path),
                                         total=n_frames,
                                         desc=f'{basename}'):
                if args.profile:
                    nvtx.range_push(f"frame_{frame_idx}")
                t_start = time.time()
                result = process_single_frame(image, model, device, args)
                elapsed = time.time() - t_start
                if args.profile:
                    nvtx.range_pop()
                total_time += elapsed
                total_frames += 1

                save_path = os.path.join(frame_dir, f'frame_{frame_idx:04d}.png')
                Image.fromarray(result * 255).save(save_path)

    # Print profiling summary
    print(f'\n{"="*50}')
    print(f'Profiling Summary')
    print(f'{"="*50}')
    print(f'Total frames processed : {total_frames}')
    print(f'Total inference time   : {total_time:.3f}s')
    if total_frames > 0:
        print(f'Avg time per frame     : {total_time / total_frames:.3f}s')
    if not (args.multi_gpu and gpu_ids and len(gpu_ids) > 1):
        # Single GPU memory stats
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            current_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            print(f'GPU peak memory alloc  : {peak_mem:.1f} MB')
            print(f'GPU current memory     : {current_mem:.1f} MB')
            print(f'GPU peak memory reserve: {peak_reserved:.1f} MB')
    print(f'{"="*50}')

    print(f'\nDone! Predictions saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
