# Scribble2Label 使用說明

本專案基於 [Scribble2Label (MICCAI 2020)](https://arxiv.org/abs/2006.12890)，並新增 **3D 體素資料**（neuron / microscopy stacks）的訓練與推論支援，含 sliding-window 推論、chunk streaming、多 GPU 切 z 軸平行處理。

---

## 1. 環境安裝

### 1.1 在新機器上重建 conda 環境

```bash
conda env create -f environment-nobuilds.yml
conda activate scribble2label_new
```

> `environment-nobuilds.yml` 不含 build hash，跨機器最安全。若 CUDA driver 版本不同，PyTorch / `torchvision` 可能要手動到 [pytorch.org](https://pytorch.org) 重抓對應的 `+cuXXX` wheel。

### 1.2 三份環境檔差異

| 檔案 | 用途 |
|---|---|
| `environment.yml` | 完整 build hash，僅同 OS + 同 CUDA 適用 |
| `environment-nobuilds.yml` | **跨機器推薦** |
| `environment-fromhistory.yml` | 只記 conda 明確安裝的套件（很精簡） |
| `requirements.txt` | 純 pip freeze，當 conda 環境已存在、只想補 pip 套件時用 |

### 1.3 Docker（可選）

```bash
docker build -t scribble2label .
```

---

## 2. 專案結構

```
scribble2label/
├── Train.py                  # 訓練入口（支援 --mode 2d / 3d、--model 選架構）
├── Inference.py              # 推論入口（含 3D streaming、多 GPU 切 z 軸）
├── Inference_custom.py       # 自訂資料集 2D 推論
├── Learner.py                # 2D 訓練 loop（含 pseudo-label filtering）
├── Learner3D.py              # 3D 訓練 loop
├── process_3d_tiff.py        # 從 3D mask 自動產生 scribble (skeletonize)
├── tile_dataset.py           # 將大圖切 tile / 產生 CSV split
├── generate_from_masks.py    # 2D scribble 產生工具
├── preprocess.py             # BBBC038 預處理
├── scripts/
│   ├── dataset.py            # 2D Dataset
│   ├── dataset_3d.py         # 3D Dataset（patch sampling）
│   ├── metric.py             # IoU
│   ├── metric_mdice.py       # Instance Dice
│   ├── run_multi_gpu_inference.sh   # 多 GPU 平行推論
│   ├── merge_tiff_parts.py   # 合併分段 TIFF
│   └── utils.py
├── examples/                 # 範例影像與 scribble label
├── logs/                     # checkpoint + tensorboard
└── docs/
    ├── USAGE.md              # 本文件
    ├── EXPERIMENT_TEMPLATE.md
    └── profiling_report.md
```

---

## 3. 資料準備

### 3.1 2D 資料（BBBC038 範例）

```bash
/bin/bash preprocess_dataset.sh
```

從官方下載 BBBC038v1、分模態（螢光 / 組織 / 明場）、自動產生 scribble label、切 train/test。

### 3.2 自有 2D 資料

```bash
python generate_from_masks.py --masks_dir <dir> --out_dir <dir>   # 由 mask 自動長 scribble
python tile_dataset.py                                            # 切 tile + 產生 train.csv
```

### 3.3 3D 資料

從整顆 3D mask TIFF 自動長出 scribble（用 skeletonize）：

```bash
python process_3d_tiff.py \
    --tiff_path /path/to/full_mask.tif \
    --output_dir ./examples \
    --modality my_dataset_name \
    --step 10 \
    --ratio 0.5
```

最終訓練吃的是三份體積一致的 3D TIFF（shape `(D, H, W)`）：
- `image_path`：原始影像
- `scr_path`：scribble label（含 ignore_index=250 表示未標註）
- `mask_path`：完整 GT mask（用於驗證 IoU）

---

## 4. 訓練

### 4.1 2D

```bash
python Train.py --mode 2d
```

Backbone：`segmentation_models_pytorch` 的 U-Net + ResNet50 encoder（ImageNet 預訓練）+ SCSE attention。
資料 / 模態 / scribble 細度等改 `Train.py` 內 `get_config('2d')` 區段。

### 4.2 3D

```bash
python Train.py --mode 3d --model basicunet
```

支援的 3D 架構（在 `Train.py:11-20`）：

| `--model` | 架構 | 參數量 |
|---|---|---|
| `basicunet`（預設） | MONAI BasicUNet, features=(16,32,64,128,256,16) | ~5.7M |
| `basicunet-s` | BasicUNet 小版 | ~1.4M |
| `unet` / `unet-s` | MONAI UNet + 殘差 | 1.2M / 0.3M |
| `segresnet` | SegResNet init_filters=8 | ~1.2M |
| `segresnet-l` | SegResNet init_filters=16 | ~4.7M |
| `attention` | AttentionUnet | ~1.5M |
| `highresnet` | HighResNet (dilated) | ~0.8M |

3D 訓練的關鍵設定（`get_config('3d')`）：

```python
patch_size      = (256, 256, 256)
batch_size      = 2
samples_per_epoch = 100
sw_roi          = (256, 256, 256)
sw_overlap      = 0.25
thr_epoch       = 200       # 此 epoch 開始啟用 pseudo-label
period_epoch    = 5         # 每隔幾 epoch 重新生成 pseudo-label
thr_conf        = 0.8       # pseudo-label 信心門檻
alpha           = 0.2       # EMA 權重
ignore_index    = 250
```

Checkpoint 與 TensorBoard log 寫到 `./logs/<config.name>/`。

> **斷點續訓**：如果 `logs/<name>/best_model.pth` 存在會自動載入再 fine-tune。

---

## 5. 推論

### 5.1 2D

```bash
python Inference.py --mode 2d
```

讀 `examples/labels/<name>/test.csv`，把結果存在 `logs/<name>/predictions/`。

### 5.2 3D（單 GPU streaming）

```bash
python Inference.py --mode 3d --model segresnet-l \
    --input /path/to/volume.tif \
    --out_dir ./logs/<run_name>/inference \
    --checkpoint ./logs/<run_name>/best_model.pth \
    --stream_depth 256 --stream_overlap 64 \
    --sw_batch_size 1 --sw_stitch_device cpu
```

主要旗標：

| 旗標 | 預設 | 說明 |
|---|---|---|
| `--input` | `config.input_path` | 輸入 TIFF 路徑 |
| `--out_dir` | `config.out_dir` | 輸出目錄（寫 `pred_mask.tif`、可選 `pred_prob.tif`）|
| `--checkpoint` | `logs/<name>/best_model.pth` | 模型權重 |
| `--device` | `cuda:1` | 覆寫 GPU |
| `--stream_depth` | 256 | 每個 chunk 在 z 方向的深度 |
| `--stream_overlap` | 64 | chunk 之間的 z-overlap，避免接縫 |
| `--sw_batch_size` | 1 | MONAI sliding window 的 batch size |
| `--sw_stitch_device` | `cpu` | 拼接位置（`cpu` 省 VRAM、`cuda:0` 較快） |
| `--save_prob` | 否 | 同時輸出機率體積（檔案會很大） |
| `--prob_dtype` | `float16` | 機率輸出 dtype |
| `--downscale_xy` | 1.0 | 實驗性：每個 chunk 在 XY 方向下採樣後推論再放大 |
| `--z_range LO HI` | 全圖 | 只處理 z ∈ [LO, HI)，用於多 GPU 切片 |
| `--z_guard` | 128 | z_range 的兩側多讀 N 張當 sliding-window context |

> ⚠️ **影像深度建議是 16 的倍數**（segresnet 系列特別敏感）。否則 chunk 的尾段可能不對齊 SegResNet 的 down/up 倍率而報 size mismatch（例：深度 513 切到最後一段為 129，÷8 後 encoder 為 17、decoder 上採樣到 34 但 skip 是 33 → 炸）。修法：把 ROI 永遠保持為 `(256, 256, 256)` 讓 MONAI 自動補 padding。

### 5.3 3D 多 GPU 平行推論

依 z 軸切成 N 段，每張 GPU 跑一段：

```bash
./scripts/run_multi_gpu_inference.sh 2 \
    /data/.../volume.tif \
    ./logs/<run_name>/inference_mg \
    --stream_depth 256 --stream_overlap 64
```

每張卡寫到 `inference_mg/part_<i>/pred_mask.tif`。最後合併：

```bash
python scripts/merge_tiff_parts.py \
    --parts_dir ./logs/<run_name>/inference_mg \
    --out_path  ./logs/<run_name>/inference_mg/pred_mask.tif
```

---

## 6. 輸出格式

| 檔名 | dtype | 內容 |
|---|---|---|
| `pred_mask.tif` | uint8 | 二值化 mask（0 / 255）|
| `pred_prob.tif` | float16 / float32 | 前景機率（只有 `--save_prob` 才存）|

兩個檔都用 BigTIFF 寫出，可直接餵 Fiji / napari / `tifffile.memmap`。

---

## 7. Pretrained 權重（原作者）

| DSB-Fluo 10% | 30% | 50% | 100% |
|:---:|:---:|:---:|:---:|
| [link](https://drive.google.com/file/d/11vWtzi9ippVeGnerW2X1-6tTJt9pdY_u/view?usp=sharing) | [link](https://drive.google.com/file/d/1y8EtLGaEL-tTAjVfGJgy2sRlJkIxUog6/view?usp=sharing) | [link](https://drive.google.com/file/d/1BuyOSrWC7QdlsTXoH2KAIrXL0sxVlDGS/view?usp=sharing) | [link](https://drive.google.com/file/d/1UNrl1p4Z4t05lf7q_zo6XSN9S0-LLS3-/view?usp=sharing) |

---

## 8. 常見問題

**Q1. `RuntimeError: size of tensor a (X) must match tensor b (X±1)`**
chunk 的某個維度不是模型下採樣倍率的倍數。修法：保留完整 ROI 讓 MONAI 自動 pad（移除 `Inference.py` 中對 `roi_size` 的 `min()` 縮減），或讓影像深度為 16 倍數。

**Q2. CUDA OOM**
- 減 `--sw_batch_size`
- 加 `--sw_stitch_device cpu`
- 減 `--stream_depth`（但要保持 ≥ `sw_roi[0]`）
- 試 `--downscale_xy 0.5`（會犧牲解析度）

**Q3. 拼接處有接縫**
拉大 `--stream_overlap` 或 `--z_guard`，並確認 `--sw_overlap`（預設 0.5）夠高。

**Q4. 訓練 IoU 卡住**
確認 `thr_epoch` 到了之後 pseudo-label 有開始生效；看 TensorBoard 的 `train/pseudo_ratio`。`thr_conf` 太高會幾乎沒 pseudo-label，太低會學到雜訊。

---

## 9. 開發備忘

- 訓練 / 推論的 device 預設不同（訓練 `cuda:0`、3D 推論 `cuda:1`），多卡機器要視情況用 `--device` 或 `CUDA_VISIBLE_DEVICES` 覆寫。
- `Train.py` 與 `Inference.py` 共用 `MODELS_3D` 與 `build_model_3d`，新增架構在 `Train.py:11-54`。
- 大型 TIFF 用 `tifffile.memmap` 讀取，記憶體不會爆。
