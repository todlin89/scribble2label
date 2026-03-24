"""
Inference_custom_gpu.py — All computation on GPU, minimal CPU↔GPU transfers.

Two modes:
  --mode full    : Send entire image to GPU, no tiling (1x H2D + 1x D2H per frame)
  --mode tiling  : Tile on GPU using unfold (1x H2D + 1x D2H per frame, same logic as tiling)

Both modes only transfer data twice per frame:
  CPU: read TIFF → H2D (once) → GPU: normalize + inference + post-process → D2H (once) → CPU: save PNG

Compared to Inference_custom.py which transfers 6-7 times per frame due to CPU-based tiling.
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
import torch.cuda.nvtx as nvtx

from segmentation_models_pytorch import Unet


# ImageNet normalization constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args():
    parser = argparse.ArgumentParser(description='Scribble2Label GPU Inference')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='./logs/fiji_BC_r50_tile512/best_model.pth')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'tiling'],
                        help='full: no tiling, send whole image. tiling: tile on GPU with unfold.')
    parser.add_argument('--profile', action='store_true', default=False)
    return parser.parse_args()


def load_model(model_path, device):
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
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def load_single_image(img):
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def load_frames(path):
    if path.lower().endswith(('.tif', '.tiff')):
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[1] > 1 and arr.shape[2] > 1:
            if arr.shape[2] <= 4:
                yield 0, load_single_image(Image.fromarray(arr))
            else:
                for i in range(arr.shape[0]):
                    yield i, load_single_image(Image.fromarray(arr[i]))
            return
        elif arr.ndim == 2:
            yield 0, load_single_image(Image.fromarray(arr))
            return
    img = Image.open(path)
    n_frames = getattr(img, 'n_frames', 1)
    if n_frames == 1:
        yield 0, load_single_image(img)
    else:
        for i in range(n_frames):
            img.seek(i)
            yield i, load_single_image(img)


def gpu_normalize(image_tensor, device):
    """Normalize uint8 tensor to ImageNet stats, entirely on GPU."""
    # (H, W, 3) uint8 → (1, 3, H, W) float32, normalized
    x = image_tensor.to(device).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # HWC → NCHW
    mean = _MEAN.to(device)
    std = _STD.to(device)
    x = (x - mean) / std
    return x


def pad_to_multiple(x, multiple=32):
    """Pad tensor so H and W are multiples of `multiple` (required by UNet encoder)."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, h, w


# ============================================================
# Mode: full — no tiling, entire image in one pass
# ============================================================
def inference_full(model, image_np, device, profile=False):
    """
    CPU: read image
    H2D: send entire image to GPU (once)
    GPU: normalize + pad + forward + softmax + threshold + crop
    D2H: send result back (once)
    """
    if profile:
        nvtx.range_push("H2D_full_image")
    image_tensor = torch.from_numpy(image_np).to(device)
    if profile:
        nvtx.range_pop()

    if profile:
        nvtx.range_push("gpu_normalize")
    x = gpu_normalize(image_tensor, device)
    del image_tensor
    if profile:
        nvtx.range_pop()

    if profile:
        nvtx.range_push("gpu_pad")
    x, orig_h, orig_w = pad_to_multiple(x, 32)
    if profile:
        nvtx.range_pop()

    with torch.no_grad():
        if profile:
            nvtx.range_push("forward_pass")
        outputs = model(x)
        if profile:
            nvtx.range_pop()
        del x

        if profile:
            nvtx.range_push("gpu_postprocess")
        probs = F.softmax(outputs, dim=1)
        pred = (probs[:, 1, :orig_h, :orig_w] > 0.5).byte().squeeze(0)
        del outputs, probs
        if profile:
            nvtx.range_pop()

    if profile:
        nvtx.range_push("D2H_result")
    result = pred.cpu().numpy()
    if profile:
        nvtx.range_pop()

    return result


# ============================================================
# Mode: tiling — tile on GPU, same logic but no CPU↔GPU bouncing
# ============================================================
def inference_tiling_gpu(model, image_np, device, tile_size, overlap, batch_size, profile=False):
    """
    CPU: read image
    H2D: send entire image to GPU (once)
    GPU: normalize + pad + unfold tiles + batch inference + stitch
    D2H: send result back (once)
    """
    h, w = image_np.shape[:2]
    stride = tile_size - overlap

    # === H2D: one transfer ===
    if profile:
        nvtx.range_push("H2D_full_image")
    image_tensor = torch.from_numpy(image_np).to(device)
    if profile:
        nvtx.range_pop()

    # === GPU: normalize ===
    if profile:
        nvtx.range_push("gpu_normalize")
    x = gpu_normalize(image_tensor, device)  # (1, 3, H, W)
    del image_tensor
    if profile:
        nvtx.range_pop()

    # === GPU: pad for tiling ===
    if profile:
        nvtx.range_push("gpu_pad")
    _, _, ch, cw = x.shape
    pad_h = (stride - (ch - tile_size) % stride) % stride if ch > tile_size else tile_size - ch
    pad_w = (stride - (cw - tile_size) % stride) % stride if cw > tile_size else tile_size - cw
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    del x
    _, _, ph, pw = x_padded.shape
    if profile:
        nvtx.range_pop()

    # === GPU: extract tiles using unfold ===
    if profile:
        nvtx.range_push("gpu_unfold_tiles")
    # unfold H dimension, then W dimension
    # (1, 3, pH, pW) → (1, 3, nH, tile_size, pW) → (1, 3, nH, tile_size, nW, tile_size)
    tiles = x_padded.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
    del x_padded
    # tiles shape: (1, 3, nH, nW, tile_size, tile_size)
    n_h, n_w = tiles.shape[2], tiles.shape[3]
    # Reshape to (nH*nW, 3, tile_size, tile_size)
    tiles = tiles.contiguous().view(1, 3, n_h, n_w, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, tile_size, tile_size)
    n_tiles = tiles.shape[0]
    if profile:
        nvtx.range_pop()

    # === GPU: batch inference ===
    if profile:
        nvtx.range_push("gpu_batch_inference")
    all_preds = []
    for i in range(0, n_tiles, batch_size):
        batch = tiles[i:i + batch_size]
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)[:, 1]  # foreground probability
            all_preds.append(probs)
    all_preds = torch.cat(all_preds, dim=0)  # (n_tiles, tile_size, tile_size)
    del tiles
    if profile:
        nvtx.range_pop()

    # === GPU: stitch tiles ===
    if profile:
        nvtx.range_push("gpu_stitch")
    prediction_sum = torch.zeros(ph, pw, device=device, dtype=torch.float32)
    count_map = torch.zeros(ph, pw, device=device, dtype=torch.float32)

    idx = 0
    for r_idx in range(n_h):
        for c_idx in range(n_w):
            r = r_idx * stride
            c = c_idx * stride
            prediction_sum[r:r + tile_size, c:c + tile_size] += all_preds[idx]
            count_map[r:r + tile_size, c:c + tile_size] += 1.0
            idx += 1

    del all_preds
    count_map = torch.clamp(count_map, min=1.0)
    result_gpu = ((prediction_sum / count_map)[:h, :w] > 0.5).byte()
    del prediction_sum, count_map
    if profile:
        nvtx.range_pop()

    # === D2H: one transfer ===
    if profile:
        nvtx.range_push("D2H_result")
    result = result_gpu.cpu().numpy()
    if profile:
        nvtx.range_pop()

    return result


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'predictions')
    os.makedirs(args.output_dir, exist_ok=True)

    extensions = ['*.tif', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(image_paths)
    print(f'Found {len(image_paths)} image file(s) in {args.input_dir}')

    if len(image_paths) == 0:
        print('No images found.')
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    print(f'Mode: {args.mode}')

    total_frames = 0
    total_time = 0.0

    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Count frames
        if img_path.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(img_path) as tif:
                shape = tif.series[0].shape
            n_frames = shape[0] if len(shape) == 3 and shape[2] > 4 else 1
        else:
            img_check = Image.open(img_path)
            n_frames = getattr(img_check, 'n_frames', 1)
            img_check.close()

        if n_frames == 1:
            print(f'Processing: {os.path.basename(img_path)} (single frame)')
            _, image = next(load_frames(img_path))

            if args.profile:
                nvtx.range_push(f"frame_{basename}")
            t_start = time.time()
            if args.mode == 'full':
                result = inference_full(model, image, device, args.profile)
            else:
                result = inference_tiling_gpu(model, image, device,
                                              args.tile_size, args.overlap, args.batch_size, args.profile)
            elapsed = time.time() - t_start
            if args.profile:
                nvtx.range_pop()

            total_time += elapsed
            total_frames += 1
            print(f'  -> {elapsed:.3f}s')
            save_path = os.path.join(args.output_dir, f'{basename}.png')
            Image.fromarray(result * 255).save(save_path)

        else:
            print(f'Processing: {os.path.basename(img_path)} ({n_frames} frames)')
            frame_dir = os.path.join(args.output_dir, basename)
            os.makedirs(frame_dir, exist_ok=True)

            for frame_idx, image in tqdm(load_frames(img_path),
                                          total=n_frames, desc=f'{basename}'):
                if args.profile:
                    nvtx.range_push(f"frame_{frame_idx}")
                t_start = time.time()
                if args.mode == 'full':
                    result = inference_full(model, image, device, args.profile)
                else:
                    result = inference_tiling_gpu(model, image, device,
                                                  args.tile_size, args.overlap, args.batch_size, args.profile)
                elapsed = time.time() - t_start
                if args.profile:
                    nvtx.range_pop()

                total_time += elapsed
                total_frames += 1
                save_path = os.path.join(frame_dir, f'frame_{frame_idx:04d}.png')
                Image.fromarray(result * 255).save(save_path)

    # Summary
    print(f'\n{"="*50}')
    print(f'Profiling Summary (mode={args.mode})')
    print(f'{"="*50}')
    print(f'Total frames processed : {total_frames}')
    print(f'Total inference time   : {total_time:.3f}s')
    if total_frames > 0:
        print(f'Avg time per frame     : {total_time / total_frames:.3f}s')
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f'GPU peak memory alloc  : {peak_mem:.1f} MB')
        print(f'GPU peak memory reserve: {peak_reserved:.1f} MB')
    print(f'{"="*50}')
    print(f'\nDone! Predictions saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
