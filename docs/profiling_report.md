# Inference Pipeline Profiling & Optimization Report

**Date:** 2026-03-23
**Author:** Todd
**Script:** `Inference_custom.py`
**Tool:** NVIDIA Nsight Systems + PyTorch NVTX

---

## 1. Objective

Profile the single-GPU inference pipeline of `Inference_custom.py` to identify CPU/GPU bottlenecks and reduce per-frame inference time for large multi-frame TIFF stacks.

## 2. Environment

| Item | Spec |
|---|---|
| GPU | 4× NVIDIA RTX A6000 (48 GB VRAM each) |
| Data | `xyvar_assemble_0.tif` — 4160 frames, 4176×4176 px, uint8 grayscale |
| Model | UNet (ResNet50 encoder + scSE attention), trained on 512×512 tiles |
| Inference config | `tile_size=512`, `overlap=64`, `batch_size=16`, single GPU |
| Profiling tool | Nsight Systems 2023.1.2, `nsys profile -t cuda,nvtx,osrt,cudnn,cublas` |

## 3. Profiling Method

NVTX markers were inserted at key stages of the inference pipeline to measure time spent in each phase. The profiling command:

```
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o <output> --cuda-memory-usage=true --force-overwrite true \
    python Inference_custom.py --input_dir <path> --tile_size 512 --overlap 64 --batch_size 16 --profile
```

Each frame goes through the following pipeline:

```
CPU: load_frames → extract_tiles → inference_tiles (loop) → stitch_tiles → save
                                        │
                                   Per batch (×7):
                                   CPU: tile_preprocess (albumentations Normalize + ToTensorV2)
                                   H2D: .to(device) — CPU RAM → GPU VRAM
                                   GPU: UNet forward pass + softmax
                                   D2H: .cpu().numpy() — GPU VRAM → CPU RAM
```

A single 4176×4176 frame produces ~100 tiles (512×512 with overlap=64), processed in ~7 batches of 16. This results in 7 H2D and 7 D2H transfers per frame.

**Important note on CUDA async behavior:** NVTX ranges for `forward_pass` only capture kernel launch time (async, returns immediately). The actual GPU compute time manifests in the `D2H_transfer` range, because `.cpu()` forces synchronization — the CPU must wait for all pending GPU operations to finish before copying results back.

## 4. Baseline Profiling Results

**Profile:** `single_gpu_profile.nsys-rep` (56 frames)
**Data source:** SQLite direct query on NVTX_EVENTS table

| Stage | Total (s) | Per Frame (s) | % of Frame | Description |
|---|---|---|---|---|
| inference_tiles | 86.55 | 1.546 | — (container) | Entire tile inference loop |
| D2H_transfer | 58.72 | 1.049 | 55.4% | GPU→CPU + implicit GPU sync wait |
| stitch_tiles | 17.52 | 0.318 | 16.8% | CPU: overlap averaging + threshold |
| tile_preprocess | 11.42 | 0.204 | 10.8% | CPU: albumentations normalize |
| H2D_transfer | 8.35 | 0.149 | 7.9% | CPU→GPU data transfer |
| forward_pass | 7.86 | 0.140 | 7.4% | GPU: UNet forward (launch time only) |
| extract_tiles | 1.98 | 0.035 | 1.9% | CPU: numpy pad + slice |
| softmax | 0.04 | 0.001 | ~0% | GPU: negligible |
| **Frame total** | — | **1.894** | **100%** | — |

**GPU Memory Transfer (actual DMA time):**

| Direction | Total (s) | Per Frame (s) | Data Volume |
|---|---|---|---|
| HtoD | 4.68 | 0.084 | 31.2 MB/frame |
| DtoH | 1.64 | 0.029 | 103.0 MB/frame |

**Key finding:** `D2H_transfer` appears to dominate at 55.4%, but this is misleading. The actual DMA time is only 0.029s/frame. The remaining ~1.0s is the CPU waiting for GPU forward pass to complete (CUDA async behavior). The true compute-bound time is hidden inside this range.

Actionable bottlenecks:
1. **stitch_tiles (0.318s)** — recomputes identical count_map every frame
2. **H2D_transfer (0.149s)** — uses pageable memory, blocking transfer
3. **tile_preprocess (0.204s)** — albumentations already optimized (C/OpenCV backend)

## 5. Optimizations Applied

### 5.1 stitch_tiles — Cache count_map with pre-computed reciprocal

**Problem:** `count_map` depends only on image size + tile layout, which is identical across all 4160 frames. It was recomputed from scratch every frame.

**Solution:** `_get_count_map()` computes the count_map once, takes its reciprocal, and caches it. Subsequent frames retrieve the cached result and use multiplication instead of division.

```python
# Before (every frame):
count_map = np.zeros(...)
for r, c in positions: count_map[r:r+tile, c:c+tile] += 1.0
result = prediction_sum / count_map

# After (first frame computes, rest use cache):
inv_count_map = _get_count_map(...)   # cached reciprocal
prediction_sum *= inv_count_map        # multiply instead of divide
```

### 5.2 H2D transfer — pin_memory() + non_blocking=True

**Problem:** Default `.to(device)` allocates pageable host memory. CUDA DMA requires pinned (page-locked) memory, so the driver performs an implicit extra copy (pageable → pinned) before the actual transfer. Additionally, the CPU blocks until the transfer completes.

**Solution:**
```python
# Before:
batch = torch.stack(tile_batch).to(device)

# After:
batch = torch.stack(tile_batch).pin_memory().to(device, non_blocking=True)
```

- `pin_memory()`: allocates directly in pinned memory, eliminating the extra copy
- `non_blocking=True`: CPU initiates DMA and continues immediately, allowing overlap with subsequent tile preprocessing

### 5.3 tile_preprocess — Manual numpy (REVERTED)

**Attempted:** Replace `albumentations.Compose([Normalize(), ToTensorV2()])` with manual numpy operations to reduce Python overhead.

**Result:** Slower (0.204s → 0.525s per frame). Albumentations uses optimized C/OpenCV internals that outperform naive numpy. Reverted to original albumentations with a module-level singleton to avoid re-creating the Compose object per call.

## 6. Optimized Profiling Results

**Profile:** `optimized_v2_profile.nsys-rep` (69 frames)

| Stage | Before (s/frame) | After (s/frame) | Speedup |
|---|---|---|---|
| stitch_tiles | 0.318 | 0.151 | **2.11×** |
| H2D_transfer | 0.149 | 0.067 | **2.27×** |
| D2H_transfer | 1.049 | 0.656 | 1.60× |
| tile_preprocess | 0.204 | 0.217 | — (unchanged) |
| forward_pass | 0.140 | 0.149 | — (unchanged) |
| extract_tiles | 0.035 | 0.036 | — (unchanged) |
| **Frame total** | **1.894** | **1.262** | **1.50× (33% faster)** |

**GPU Memory Transfer (actual DMA time):**

| Direction | Before (s/frame) | After (s/frame) | Speedup |
|---|---|---|---|
| HtoD | 0.084 | 0.027 | **3.14×** |
| DtoH | 0.029 | 0.029 | — |

**CUDA API impact:**

| API | Before | After | Change |
|---|---|---|---|
| cudaMemcpyAsync | 63.56s | 45.22s | -29% |
| cudaStreamSynchronize | 0.28s | 0.01s | **-97%** |

The D2H_transfer improvement (1.60×) is an indirect effect — `pin_memory` + `non_blocking` enables better overlap between CPU and GPU work, reducing GPU idle time between batches.

## 7. Projected Impact on Full Dataset

| Metric | Before | After | Savings |
|---|---|---|---|
| Per-frame time | 1.894s | 1.262s | -0.632s |
| 4160 frames estimated | ~131 min | ~87 min | **~44 min** |

## 8. Remaining Bottleneck

The dominant cost is now GPU compute itself (visible as the implicit wait inside `D2H_transfer`). The UNet forward pass at 4176×4176 resolution (tiled to 512×512) is inherently compute-bound.

## 9. Future Directions

1. **Eliminate tiling overhead entirely:** RTX A6000 (48 GB) can fit full 4192×4192 inference in 12 GB. A GPU-only pipeline (`Inference_custom_gpu.py`) was prototyped with two modes:
   - `--mode full`: no tiling, entire image in one forward pass (1× H2D + 1× D2H per frame)
   - `--mode tiling`: tile extraction on GPU via `torch.Tensor.unfold` (1× H2D + 1× D2H per frame)

   **Caveat:** Model was trained on 512×512 patches. Full-image inference may produce different results due to receptive field ratio changes. Quality comparison is needed before deployment.

2. **FP16 inference:** Use `torch.cuda.amp` or TensorRT to halve GPU compute time and memory usage.

3. **Multi-GPU profiling:** Current `mp.Process` spawn workers are not captured by `nsys profile`. Options include per-GPU profiling or using CUDA Profiler API within each worker.

4. **Larger training resolution:** Training at 1024×1024 or 2048×2048 (feasible on A6000) would allow larger inference tiles, reducing tile count and H2D/D2H round trips.
