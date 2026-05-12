import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.metric_mdice import Evaluator as mdice_evaluator
from scripts.metric import Evaluator as iou_evaluator
from Train import MODELS_3D, build_model_3d


def get_config(mode, model_name='basicunet'):
    class config:
        pass

    config.mode = mode
    config.model_name = model_name
    config.seed = 42
    config.save_result = True

    if mode == '2d':
        config.name = 'custom_r100_tile512'
        config.device = torch.device('cuda:0')
        config.data_dir = f'./examples/images/{config.name}/'
        config.mask_dir = f'./examples/labels/{config.name}/full/'
        config.df_path = f'./examples/labels/{config.name}/test.csv'
        config.model_path = f'./logs/{config.name}/best_model.pth'
        config.input_size = 512
        config.batch_size = 1
        config.num_workers = 8

    elif mode == '3d':
        config.name = f'thres_90_3d_{model_name}'
        config.device = torch.device('cuda:1')
        config.input_path = '/data/datahere/Todd/data/xy_assemble_0/xystd_assemble_0_512_cubic/image/test_cubic.tif'
        config.mask_path = '/data/datahere/Todd/data/ImagesTr-1/full.tif'
        config.model_path = f'./logs/{config.name}/best_model.pth'
        config.out_dir = f'./logs/{config.name}/inference_512'
        config.use_amp = True
        config.sw_roi = (256, 256, 256)
        config.sw_overlap = 0.5
        config.sw_batch_size = 1
        # Keep sliding-window patches on GPU, but stitch the full chunk output on CPU
        # so CUDA memory stays bounded even for large XY dimensions.
        config.sw_stitch_device = 'cpu'
        # Streamed inference settings to avoid sending the full 3D volume to GPU at once.
        # stream_depth must be >= sw_roi[0]; otherwise the per-chunk roi is clamped
        # smaller than what the model was trained on and large-object recall collapses.
        config.stream_depth = 256
        config.stream_overlap = 64
        # pred_prob is optional because full-volume probabilities can be very large.
        config.save_prob = False
        config.prob_dtype = 'float16'
        config.downscale_xy = 1.0

    return config


# ─── 2D inference ────────────────────────────────────────────────────────────

def inference_image_2d(net, images, device):
    with torch.no_grad():
        predictions = net(images.to(device))
        predictions = F.softmax(predictions, dim=1)
    return predictions.detach().cpu().numpy()


def inference_2d(net, test_loader, device, save_dir=None):
    semantic_eval, instance_eval = iou_evaluator(), mdice_evaluator()
    semantic_eval.reset()
    instance_eval.reset()
    for image_names, images, masks in tqdm(test_loader):
        masks = masks.numpy()
        predictions = inference_image_2d(net, images, device)
        predictions = np.argmax(predictions, axis=1).astype('uint8')
        semantic_eval.add_batch((masks > 0).astype('uint8'), predictions)
        for image_name, pred, mask in zip(image_names, predictions, masks):
            instance_eval.add_pred(mask, pred)
            if save_dir:
                Image.fromarray(pred * 255).save(os.path.join(save_dir, f'{image_name}.png'))
    return semantic_eval.IoU, instance_eval.Dice


# ─── 3D inference ────────────────────────────────────────────────────────────

def _chunk_starts(length, chunk, overlap):
    # Build z-start positions for chunked streaming with overlap.
    chunk = int(chunk)
    overlap = int(overlap)
    if chunk <= 0:
        raise ValueError(f'stream_depth must be > 0, got {chunk}')
    if overlap < 0:
        raise ValueError(f'stream_overlap must be >= 0, got {overlap}')
    if overlap >= chunk:
        raise ValueError(f'stream_overlap ({overlap}) must be < stream_depth ({chunk})')

    starts = []
    stride = chunk - overlap
    start = 0
    while start < length:
        starts.append(start)
        if start + chunk >= length:
            break
        start += stride
    return starts


def _resize_xy_volume(volume, scale_factor, mode):
    """Resize a (D, H, W) volume only in XY, keeping Z unchanged."""
    if scale_factor == 1.0:
        return volume

    depth, height, width = volume.shape
    new_h = max(1, int(round(height * scale_factor)))
    new_w = max(1, int(round(width * scale_factor)))

    tensor = torch.from_numpy(volume).unsqueeze(1)
    resized = F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode=mode,
        align_corners=False if mode in ('bilinear', 'bicubic') else None,
    )
    return resized.squeeze(1).numpy()


def _resize_pred_xy_to_original(volume, original_hw, is_prob=False):
    """Restore a predicted (D, H, W) slab back to the original XY size."""
    orig_h, orig_w = original_hw
    tensor = torch.from_numpy(volume.astype(np.float32, copy=False)).unsqueeze(1)
    mode = 'bilinear' if is_prob else 'nearest'
    resized = F.interpolate(
        tensor,
        size=(orig_h, orig_w),
        mode=mode,
        align_corners=False if mode == 'bilinear' else None,
    ).squeeze(1).numpy()

    if is_prob:
        return np.clip(resized, 0.0, 1.0)
    return (resized > 0.5).astype(np.uint8) * 255

def inference_3d(model, config):
    import tifffile
    from monai.inferers import sliding_window_inference

    os.makedirs(config.out_dir, exist_ok=True)

    print(f"Reading {config.input_path} ...")
    # Use memmap when possible to avoid loading the entire TIFF into RAM.
    try:
        vol = tifffile.memmap(config.input_path)
    except Exception:
        vol = tifffile.imread(config.input_path)
    print(f"  shape: {vol.shape}, dtype: {vol.dtype}")
    if vol.ndim != 3:
        raise ValueError(f'Expected 3D volume, got shape={vol.shape}')
    if config.downscale_xy <= 0:
        raise ValueError(f'downscale_xy must be > 0, got {config.downscale_xy}')
    if config.downscale_xy != 1.0:
        print(f"  experimental XY downscale: {config.downscale_xy}")

    depth = vol.shape[0]

    # Multi-GPU partitioning: each process handles a global z-range [out_lo, out_hi).
    # A guard buffer on each side gives the sliding window / chunk trim logic
    # enough context to produce seamless output when parts are later concatenated.
    out_lo, out_hi = getattr(config, 'z_range', (0, depth))
    out_lo = max(0, int(out_lo))
    out_hi = min(depth, int(out_hi))
    if out_lo >= out_hi:
        raise ValueError(f'z_range produces empty output: [{out_lo}, {out_hi})')

    guard = int(getattr(config, 'z_guard', 128))
    in_lo = 0 if out_lo == 0 else max(0, out_lo - guard)
    in_hi = depth if out_hi == depth else min(depth, out_hi + guard)
    sub_depth = in_hi - in_lo

    print(f"  processing input z=[{in_lo},{in_hi}) → writing output z=[{out_lo},{out_hi}) "
          f"(guard={guard}, sub_depth={sub_depth})")

    starts = _chunk_starts(sub_depth, config.stream_depth, config.stream_overlap)
    n_chunks = len(starts)

    mask_out = os.path.join(config.out_dir, 'pred_mask.tif')
    prob_out = os.path.join(config.out_dir, 'pred_prob.tif')
    save_prob = bool(getattr(config, 'save_prob', False))
    prob_dtype = np.float16 if getattr(config, 'prob_dtype', 'float16') == 'float16' else np.float32

    gt = None
    has_gt = hasattr(config, 'mask_path') and os.path.isfile(config.mask_path)
    if has_gt:
        # Keep GT loading lazy too; IoU is accumulated chunk-by-chunk.
        try:
            gt = tifffile.memmap(config.mask_path)
        except Exception:
            gt = tifffile.imread(config.mask_path)
        if gt.shape != vol.shape:
            print(f"Warning: mask shape {gt.shape} != input shape {vol.shape}; skipping IoU.")
            has_gt = False

    use_amp = config.use_amp
    stitch_device = torch.device(getattr(config, 'sw_stitch_device', config.device))
    print(f"Running sliding_window_inference "
          f"(roi={config.sw_roi}, overlap={config.sw_overlap}, amp={use_amp}, "
          f"stream_depth={config.stream_depth}, stream_overlap={config.stream_overlap}, "
          f"sw_batch_size={config.sw_batch_size}, stitch_device={stitch_device}) ...")

    inter_sum = 0
    union_sum = 0
    fg_sum = 0
    voxel_sum = 0
    written = 0

    # Stream inference chunk-by-chunk and write TIFF slices directly.
    with tifffile.TiffWriter(mask_out, bigtiff=True) as mask_writer:
        prob_writer = tifffile.TiffWriter(prob_out, bigtiff=True) if save_prob else None
        try:
            for chunk_idx, z0_sub in enumerate(starts):
                z1_sub = min(z0_sub + config.stream_depth, sub_depth)
                z0_global = in_lo + z0_sub
                z1_global = in_lo + z1_sub

                # Only current slab goes to GPU, preventing full-volume CUDA OOM.
                slab = (np.asarray(vol[z0_global:z1_global], dtype=np.float32) / 255.0)
                original_hw = slab.shape[1:]
                if config.downscale_xy != 1.0:
                    slab = _resize_xy_volume(slab, config.downscale_xy, mode='bilinear')
                img_dtype = torch.float16 if use_amp else torch.float32
                img_t = torch.from_numpy(slab).unsqueeze(0).unsqueeze(0).to(config.device, dtype=img_dtype)
                roi_size = (
                    min(config.sw_roi[0], img_t.shape[2]),
                    min(config.sw_roi[1], img_t.shape[3]),
                    min(config.sw_roi[2], img_t.shape[4]),
                )

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    logits = sliding_window_inference(
                        inputs=img_t,
                        roi_size=roi_size,
                        sw_batch_size=config.sw_batch_size,
                        predictor=model,
                        overlap=config.sw_overlap,
                        mode='gaussian',
                        sw_device=config.device,
                        device=stitch_device,
                    )
                # Free the input slab before the peak post-processing allocation.
                del img_t

                if save_prob:
                    with torch.no_grad():
                        prob_t = F.softmax(logits.to(torch.float32), dim=1)[0, 1]
                    pred_u8 = ((prob_t > 0.5).to(torch.uint8).cpu().numpy() * 255)
                    prob_np = prob_t.to(torch.float32).cpu().numpy()
                    del prob_t
                else:
                    # Binary argmax without softmax: skip the extra fp16 tensor alloc.
                    pred_u8 = (logits[0, 1] > logits[0, 0]).to(torch.uint8).cpu().numpy() * 255
                    prob_np = None
                del logits, slab
                # Release fragmented cache so late chunks don't OOM.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if config.downscale_xy != 1.0:
                    pred_u8 = _resize_pred_xy_to_original(pred_u8, original_hw, is_prob=False)
                    if prob_np is not None:
                        prob_np = _resize_pred_xy_to_original(prob_np, original_hw, is_prob=True)

                # Trim overlaps so each z-slice is written exactly once within sub-volume.
                left_trim = 0 if chunk_idx == 0 else config.stream_overlap // 2
                right_trim = 0 if chunk_idx == (n_chunks - 1) else (config.stream_overlap - config.stream_overlap // 2)
                z_len = z1_sub - z0_sub
                if left_trim + right_trim >= z_len:
                    left_trim = 0
                    right_trim = 0
                local_end = z_len - right_trim if right_trim > 0 else z_len

                # Clip chunk's covered z-range [z0_global+left_trim, z0_global+local_end)
                # to the output window [out_lo, out_hi). Guard zones fall outside and are skipped.
                g_start = z0_global + left_trim
                g_end = z0_global + local_end
                eff_start = max(g_start, out_lo)
                eff_end = min(g_end, out_hi)
                if eff_start >= eff_end:
                    continue

                lo_idx = eff_start - z0_global
                hi_idx = eff_end - z0_global
                pred_write = pred_u8[lo_idx:hi_idx]

                for z_local in range(pred_write.shape[0]):
                    mask_writer.write(pred_write[z_local])
                    if prob_writer is not None:
                        prob_writer.write(prob_np[lo_idx + z_local].astype(prob_dtype, copy=False))

                # Incremental IoU accumulation avoids full-volume temporary arrays.
                if has_gt:
                    gt_chunk = (np.asarray(gt[eff_start:eff_end]) > 0).astype(np.uint8)
                    pred_bin = (pred_write > 0).astype(np.uint8)
                    inter_sum += np.logical_and(gt_chunk, pred_bin).sum(dtype=np.int64)
                    union_sum += np.logical_or(gt_chunk, pred_bin).sum(dtype=np.int64)

                fg_sum += (pred_write > 0).sum(dtype=np.int64)
                voxel_sum += pred_write.size
                written += pred_write.shape[0]
                print(f"  chunk {chunk_idx + 1}/{n_chunks}: sub-z[{z0_sub}:{z1_sub}] "
                      f"-> global z[{eff_start}:{eff_end}] wrote {pred_write.shape[0]} slices")
        finally:
            if prob_writer is not None:
                prob_writer.close()

    expected = out_hi - out_lo
    if written != expected:
        print(f"Warning: wrote {written} slices but output range depth is {expected}.")
    if has_gt:
        iou = inter_sum / max(union_sum, 1)
        print(f"IoU vs full.tif: {iou:.4f}")

    out_shape = (out_hi - out_lo,) + tuple(vol.shape[1:])
    print(f"Saved:")
    print(f"  {mask_out}  (uint8 0/255, shape {out_shape}, global z=[{out_lo},{out_hi}))")
    if save_prob:
        print(f"  {prob_out}  ({prob_dtype.__name__} 0.0-1.0, shape {out_shape})")
    print(f"Foreground fraction: {fg_sum / max(voxel_sum, 1):.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model_help = '\n'.join(f'  {k:15s} {v}' for k, v in MODELS_3D.items())
    parser = argparse.ArgumentParser(
        description='Scribble2Label Inference',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'],
                        help='Inference mode: 2d or 3d (default: 3d)')
    parser.add_argument('--model', type=str, default='basicunet',
                        choices=list(MODELS_3D.keys()),
                        help=f'3D model architecture (ignored for 2d):\n{model_help}')
    # 3D chunk-stream controls for large-volume inference.
    parser.add_argument('--stream_depth', type=int, default=None,
                        help='3D mode: depth per streamed chunk (default from config)')
    parser.add_argument('--stream_overlap', type=int, default=None,
                        help='3D mode: overlap between streamed chunks (default from config)')
    parser.add_argument('--save_prob', action='store_true',
                        help='3D mode: also save pred_prob.tif (very large for huge volumes)')
    parser.add_argument('--prob_dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='3D mode: dtype for pred_prob.tif when --save_prob is set')
    parser.add_argument('--sw_batch_size', type=int, default=None,
                        help='3D mode: MONAI sliding-window batch size (default from config)')
    parser.add_argument('--sw_stitch_device', type=str, default=None,
                        help='3D mode: where MONAI assembles chunk logits, e.g. cpu or cuda:0')
    # Multi-GPU partitioning: one process per GPU, each on a disjoint global z-range.
    parser.add_argument('--z_range', nargs=2, type=int, default=None, metavar=('LO', 'HI'),
                        help='3D mode: process only global z in [LO, HI); output TIFF has HI-LO slices')
    parser.add_argument('--z_guard', type=int, default=128,
                        help='3D mode: extra z-slices read on each side of z_range as U-Net / trim context')
    parser.add_argument('--device', type=str, default=None,
                        help='Override config device (e.g. cuda:0, cuda:1)')
    parser.add_argument('--input', type=str, default=None,
                        help='3D mode: override config.input_path')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='3D mode: override config.out_dir')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override config.model_path (checkpoint .pth to load)')
    parser.add_argument('--downscale_xy', type=float, default=None,
                        help='3D mode experimental: downscale each chunk only in XY before inference, '
                             'then resize predictions back to original XY')
    args = parser.parse_args()

    config = get_config(args.mode, args.model)
    if args.device is not None:
        config.device = torch.device(args.device)
    if args.checkpoint is not None:
        config.model_path = args.checkpoint
    if args.mode == '3d':
        if args.stream_depth is not None:
            config.stream_depth = args.stream_depth
        if args.stream_overlap is not None:
            config.stream_overlap = args.stream_overlap
        if args.sw_batch_size is not None:
            config.sw_batch_size = args.sw_batch_size
        config.save_prob = args.save_prob
        config.prob_dtype = args.prob_dtype
        if args.sw_stitch_device is not None:
            config.sw_stitch_device = args.sw_stitch_device
        if args.input is not None:
            config.input_path = args.input
        if args.out_dir is not None:
            config.out_dir = args.out_dir
        if args.z_range is not None:
            config.z_range = tuple(args.z_range)
        config.z_guard = args.z_guard
        if args.downscale_xy is not None:
            config.downscale_xy = args.downscale_xy

    if config.mode == '2d':
        from scripts.dataset import get_transforms, dsbTestDataset
        from segmentation_models_pytorch import Unet

        model = Unet(encoder_name='resnet50', encoder_weights='imagenet',
                     decoder_use_batchnorm=True, decoder_attention_type='scse',
                     classes=2, activation=None)
        checkpoint = torch.load(config.model_path,
                                map_location=lambda storage, loc: storage,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        model.eval()
        print(f'Model Loaded: {config.model_path}')

        test_df = pd.read_csv(config.df_path)
        transforms = get_transforms(config.input_size, need=('val'))
        test_dataset = dsbTestDataset(config.data_dir, config.mask_dir, test_df,
                                      tfms=transforms['val'])
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                 num_workers=config.num_workers, shuffle=False,
                                 sampler=None, pin_memory=True)

        if config.save_result:
            save_dir = os.path.join(os.path.dirname(config.model_path), 'predictions')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None

        iou, mdice = inference_2d(model, test_loader, config.device, save_dir)
        print(f'IoU: {iou:.4f}, mDice: {mdice:.4f}')

    elif config.mode == '3d':
        model = build_model_3d(config.model_name).to(config.device)

        checkpoint = torch.load(config.model_path, map_location=config.device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model Loaded: {config.model_path}")
        print(f"  model: {config.model_name}, best_score = {checkpoint.get('best_score', 'N/A')}, "
              f"epoch = {checkpoint.get('epoch', 'N/A')}")

        inference_3d(model, config)
