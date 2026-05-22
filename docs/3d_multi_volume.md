# 3D 訓練：多 Volume 與 Leave-One-Volume-Out 驗證

> 技術筆記 — 2026-05-18
> 對應 commit：`56f666e`
> 影響檔案：`scripts/dataset_3d.py`、`Learner3D.py`、`Train.py`、`Train_3d.py`

---

## 1. 背景與動機

### 1.1 原始 2D pipeline 的驗證機制

Scribble2Label 的 2D 訓練流程以**影像為單位**的 K-fold cross-validation 分離訓練/驗證集。`examples/labels/<dataset>/train.csv` 預先把每張影像標上 `fold ∈ {0, 1, 2, 3}`，訓練時：

```python
# Train.py:154-156
train_df = df[df.fold != config.fold].reset_index(drop=True)
valid_df = df[df.fold == config.fold].reset_index(drop=True)
```

驗證集影像**完全沒有進入訓練**，所以 val IoU 能反映模型對「未見過影像」的泛化能力。

### 1.2 原始 3D pipeline 的問題

3D 流程原本為**單一 volume** 設計：訓練端與驗證端共用同一份 image / scribble / mask：

```python
# 舊 Train_3d.py / Train.py 3D 分支
train_dataset = dsb3DDataset(image_path=A, scr_path=B, mask_path=C, mode='train')
valid_dataset = dsb3DDataset(image_path=A, scr_path=B, mask_path=C, mode='val')
```

訓練時以 random patch sampling 重複觀測整個 volume（`samples_per_epoch=100`、`n_epochs=10000`），驗證時又對**同一個 volume** 做 sliding-window inference 比對 ground-truth。這導致：

- **Val IoU 衡量的是訓練集記憶力，不是泛化能力。**
- 數字虛高、無法與其他模型公平比較、無法選 best checkpoint。
- 形式上有「驗證」但本質上是 data leakage。

### 1.3 目標

把 2D 的「fold-by-image hold-out」對應到 3D 的「leave-one-volume-out (LOVO)」，使 3D val IoU 也能反映泛化能力，並支援多 volume 訓練資料的彈性配置。

---

## 2. 改造概要

| 層級 | 改動 |
|---|---|
| `scripts/dataset_3d.py` | `dsb3DDataset` API 改為 list-based，新增 `val_index` 參數控制 hold-out |
| `Learner3D.py` | `ensemble_prediction` 改為對每個 train volume 各跑一次 sliding-window + EMA |
| `Train.py` 3D 分支 | config 改為 list，加 `--val_index` CLI flag |
| `Train_3d.py` | 同步更新成新 API（避免破壞 standalone 入口）|

`Learner3D.train_one_epoch`、`Learner3D.validation` **完全沒動** — 它們本來就是 dataloader-agnostic 的設計，新 dataset 自然接得上。

---

## 3. 關鍵設計

整套改造表面有 ~150 行 diff，但**真正改變運算行為**的只有 3 處。其他都是 plumbing。

### 3.1 載入期的物理隔離（`__init__`）

**位置**：`scripts/dataset_3d.py:43-64`

在 dataset 物件建立的瞬間，根據 `mode` + `val_index`，**只把該載的 volume 讀進 RAM**：

```python
if val_index is None:
    global_idxs = list(range(n_total))            # 全部都看（leaky 模式）
else:
    assert n_total >= 2
    if mode == 'train':
        global_idxs = [i for i in range(n_total) if i != val_index]
    else:  # 'val'
        global_idxs = [val_index]

self.images    = [tifffile.imread(image_paths[i]) for i in global_idxs]
self.scribbles = [tifffile.imread(scr_paths[i])   for i in global_idxs]
self.masks     = [(tifffile.imread(mask_paths[i]) > 0).astype(np.uint8)
                  for i in global_idxs]
```

**設計理由**：

hold-out 的執行點必須選在「最早能做的地方」。可選兩種設計：

| 設計 | 載入時機 | hold-out 機制 |
|---|---|---|
| A. 全部載入，sample 時過濾 | 兩個 dataset 都載 N 個 volume | `_train_item` 內部寫條件判斷排除 val |
| **B. 載入時就切分（採用）** | train_ds 只載 N-1 個、val_ds 只載 1 個 | 物理上 train 拿不到 val |

採用 B 的理由：

1. **記憶體**：A 設計下兩邊都重複載入，浪費。B 設計每邊只載自己要的。
2. **Bug surface**：A 設計把 hold-out 變成「runtime filter」——條件寫錯一行 val 就 leak，但 metric 看起來正常，**最難 debug**。B 設計讓 val 的 voxel **物理上不存在於 train_dataset**，要 leak 必須在初始化就走錯 branch，那種錯會立刻爆。
3. **語意清晰**：`train_dataset.images` 本身就代表「訓練集」，無需外部再檢查 `val_index`。

這是「**make wrong code look wrong**」原則的應用：把容易出錯的隱性 filter，改成顯性的「資料根本不在」。

### 3.2 兩階段隨機抽樣（`_train_item`）

**位置**：`scripts/dataset_3d.py:106-119`

訓練的每一個 step 必須回答「下一個 patch 從哪個 volume 來」。改為兩階段：

```python
def _train_item(self):
    vi = random.randrange(len(self.images))   # 階段 1：抽 volume
    z, y, x = self._random_origin(vi)          # 階段 2：在該 volume 內抽 patch
    img_p = self.images[vi][z:z+pd, y:y+ph, x:x+pw]
    scr_p = self.scribbles[vi][z:z+pd, y:y+ph, x:x+pw]
    w_p   = self.weights[vi][z:z+pd, y:y+ph, x:x+pw]
    ...
```

**設計理由**：

沒有階段 1 會出現的 silent bug 包括「永遠抽 `self.images[0]`，其他 volume 白載」——loss 會下降、TensorBoard 圖會漂亮，**但 N-1 個 volume 的訊號完全沒進 model**。這種錯不會 crash、不會在 metric 上立刻顯露。

uniform random 的選擇理由：

- **公平**：每個 volume 每 epoch 有同等機會貢獻訓練訊號。
- **等價性**：當所有 volume 同樣大小（例如 512³），uniform-by-volume 等價於 uniform-by-voxel。
- **可擴充**：未來若 volume 大小不一，可改為「依 volume size 加權抽樣」。

**直觀類比**：三副撲克牌（A/B/C）混在一個袋子裡訓練「分辨花色」——先隨機選一副（A/B/C 等機率），再從該副裡抽一張。不會「永遠抽 A」，也不能「把三副物理性疊成一副厚牌」。

### 3.3 Pseudo-label EMA 對齊（`ensemble_prediction`）

**位置**：`Learner3D.py:127-170`

Scribble2Label 的核心訊號之一是「pseudo-label EMA 累積 → 給 confidence 高的 unlabeled voxel 當補強 label」（`Learner3D.py:77-83`）。原本是 1 個 volume 1 個 weight buffer；改造後是 N 個 volume N 個 weight buffer，**必須對每個 train volume 各跑一次**：

```python
def ensemble_prediction(self):
    ds = self.train_loader.dataset
    for vi in range(len(ds.images)):                   # ← 多了這層 loop
        volume = torch.from_numpy(ds.images[vi].astype(np.float32) / 255.0)
        volume = volume.unsqueeze(0).unsqueeze(0).to(self.config.device)

        logits = sliding_window_inference(inputs=volume, ...)
        prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy()

        ds.weights[vi][...] = (
            self.config.alpha * prob
            + (1 - self.config.alpha) * ds.weights[vi]
        )
        # 每個 volume 各自存 pseudo-label TIFF，檔名 = image_id
        tifffile.imwrite(f'{ds.image_ids[vi]}.tif', vis)
```

**設計理由**：

如果這個 loop 沒包起來、只 EMA 第一個 volume，**其他 N-1 個 volume 在 unlabeled 區域永遠拿不到 pseudo-label 訊號**，只能靠 scribble 那 ~1% 的標記學。等於把 Scribble2Label 退化成純 scribble-supervised baseline，**論文方法的主要貢獻就消失了**。

**hold-out volume 完全不參與此流程** — 任何訊號都不該從 val 回流到 model。

---

## 4. API 與使用方式

### 4.1 Dataset 簽名變更

```python
# 舊
dsb3DDataset(image_path, scr_path, mask_path,
             patch_size=(128,128,128), samples_per_epoch=100, mode='train')

# 新
dsb3DDataset(image_paths, scr_paths, mask_paths,
             val_index=None,
             patch_size=(128,128,128), samples_per_epoch=100, mode='train')
```

三份 `*_paths` 為平行 list，長度 N（同 index 對應同一個 volume）。`val_index` 語意：

| 設定 | 行為 |
|---|---|
| `val_index=None` | 不 hold-out，train/val 都載全部 volume（N=1 時等同舊行為，leaky）|
| `val_index=k` | 第 k 個 volume hold out 當 val，其餘 N-1 個訓練（要求 N≥2）|

### 4.2 Config（`Train.py` 3D 分支）

```python
elif mode == '3d':
    config.image_paths = [
        '/path/to/vol_A_image.tif',
        '/path/to/vol_B_image.tif',
        # ...
    ]
    config.scr_paths = [
        '/path/to/vol_A_scribble.tif',
        '/path/to/vol_B_scribble.tif',
        # ...
    ]
    config.mask_paths = [
        '/path/to/vol_A_full.tif',
        '/path/to/vol_B_full.tif',
        # ...
    ]
    config.val_index = None   # 由 --val_index CLI 覆寫
```

> **重要**：三個 list 用 index 對齊。Shape assert 擋得了「大小不對」，**擋不了「順序錯位」**。實作時請依字母順序或一致命名規則維持對應。

### 4.3 CLI

```bash
# 單 volume sanity check（等同舊行為）
python Train.py --mode 3d

# Leave-one-volume-out
python Train.py --mode 3d --val_index 0    # 第 0 個 volume 當 val
python Train.py --mode 3d --val_index 1    # 第 1 個 volume 當 val
# ... 完整 LOVO 需跑 N 次
```

---

## 5. 計算成本估算

### 5.1 記憶體（CPU RAM）

每個 512³ uint8 volume 在 dataset 內佔的空間：

| 內容 | 型別 | 大小 |
|---|---|---|
| image | uint8 | 128 MiB |
| scribble | uint8 | 128 MiB |
| mask | uint8 | 128 MiB |
| **weight** | **float32** | **512 MiB**（最大宗）|
| **合計／volume** | | **896 MiB ≈ 0.88 GiB** |

### 5.2 不同 N 的記憶體 footprint

LOVO 模式（`val_index=k`）下，train_ds 載入 N-1 個、val_ds 載入 1 個：

| N | 模式 | 載入 volume | RAM |
|---|---|---|---|
| 1 | leaky（舊行為）| 1 + 1 | ~1.7 GiB |
| 2 | LOVO | 1 + 1 | ~1.8 GiB |
| 5 | LOVO | 4 + 1 | ~4.4 GiB |
| **8** | **LOVO** | **7 + 1** | **~7.0 GiB** |
| 10 | LOVO | 9 + 1 | ~8.8 GiB |

> Val dataset 仍會為自己的 1 個 volume 配置 weight buffer（512 MiB），雖然 pseudo-label 不流經 val。這部分可進一步優化（見 §8）。

### 5.3 GPU VRAM

**不隨 N 變動**。每個 iteration 只處理一個 patch（256³ × batch_size），與載入幾個 volume 無關。`ensemble_prediction` 雖然要對每個 volume 做 sliding-window，但**逐個依序**進 GPU，不會疊加。

### 5.4 時間

`ensemble_prediction` 的牆鐘時間隨 N 線性成長。原本 `period_epoch=5`（每 5 epoch 觸發一次）；N=8 時建議拉長至 `period_epoch=20-40`，避免訓練時間被 sliding-window 吃掉太多。

---

## 6. 驗證

### 6.1 單元測試（`test_3d_dataset.py`，僅本地使用，未進 git）

| Test | 驗證內容 |
|---|---|
| `test_holdout_isolation` | `val_index=2` 時 train_ds = {vol 0,1,3}、val_ds = {vol 2}，val 資料絕對不在 train_ds |
| `test_holdout_isolation_val_index_none` | `val_index=None` 保留舊單 volume 行為 |
| `test_sampling_uniformity` | 4000 次抽樣，4 個 volume 各 ~1000 次（誤差 < 5%）|
| `test_ensemble_prediction_per_volume` | 每個 train volume 的 weight 都被更新、per-volume pseudo-label TIFF 都生成，且不同 volume 的 weight 不相等（證明各自跑 SW，非 aliasing）|

合成 4 個 32³ 假 volume，每個用不同 uniform intensity 填充，藉此追蹤 dataloader 拿到的 patch 來自哪個 volume。

### 6.2 預期實驗趨勢（待跑）

- 舊 leaky 模式 → 新 LOVO 模式 → **val IoU 預期會下降**，這是正常的（從「測記憶」變成「測泛化」）。
- N 越多 train volume，val IoU 越穩定、越接近真實泛化能力。
- 與 2D fold-by-image 的 val IoU 量級應該可以對齊（同樣的 hold-out 邏輯）。

---

## 7. 數字結果

> ⏳ Pending：等實驗跑出來再補。

### 7.1 LOVO N=2 的 val IoU

| val_index | val volume | train volumes | val IoU | 備註 |
|---|---|---|---|---|
| 0 | volume_0 | volume_1 | — | |
| 1 | volume_1 | volume_0 | — | |
| **平均** | | | **—** | |

### 7.2 Leaky vs LOVO 對比（N=2）

| 模式 | val IoU | 訓練 voxel 是否在 val 中 |
|---|---|---|
| 舊（`val_index=None`，1 volume） | — | 是（leaky）|
| 新（`val_index=0`） | — | 否 |
| 新（`val_index=1`） | — | 否 |

### 7.3 訓練時間 / 記憶體實測

| N | RAM peak | GPU VRAM peak | 1 epoch 牆鐘時間 | `ensemble_prediction` 牆鐘時間 |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |

---

## 8. 待辦與後續優化

### 8.1 短期

- [ ] 跑 N=2 LOVO 完整 CV（兩個 fold 各跑一次）並填表 §7
- [ ] 確認多 volume intensity 分佈是否一致；若差異大，加入 per-volume normalization 選項

### 8.2 中期

- [ ] **Val dataset 跳過 weight buffer 配置**：val 端的 `self.weights[idx]` 配置了 512 MiB float32 但完全沒用到，可在 `mode='val'` 時跳過，省 ~0.5 GiB
- [ ] **三 list 對位防呆**：改為 list of dict（`[{'img': ..., 'scr': ..., 'mask': ...}]`），杜絕順序錯位的可能
- [ ] **Volume 加權抽樣**：當 volume 大小不一時，按 voxel count 加權，避免小 volume 被過度採樣

### 8.3 長期

- [ ] 多 val volume 支援：`val_index` 改為 `val_indices: list[int]`
- [ ] K-fold splits 預先寫進 CSV / YAML，對應 2D 的 `train.csv` 設計
- [ ] Lazy loading / memmap：大 N 時 RAM 壓力大，可改為按需讀檔

---

## 附錄 A：差異統計

```
 Learner3D.py          |  57 +++++++++++++++-----------
 Train.py              |  40 +++++++++++++-----
 Train_3d.py           |  30 +++++++-------
 scripts/dataset_3d.py | 111 ++++++++++++++++++++++++++++++++++----------------
 4 files changed, 153 insertions(+), 85 deletions(-)
```

## 附錄 B：相關檔案

- 程式變更：`scripts/dataset_3d.py`, `Learner3D.py`, `Train.py`, `Train_3d.py`
- 單元測試：`test_3d_dataset.py`（本地用，未 commit）
- 使用指南：`docs/USAGE.md`
- 原 paper：[Scribble2Label (MICCAI 2020)](https://arxiv.org/abs/2006.12890)
