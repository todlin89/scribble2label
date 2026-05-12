# Experiment

## 資料

- **Input**: `xy`(original image)
- **Target**:
  - `xystd` — mask, obtained by thresholding xyvar (xy after Monte Carlo) at 90 and then applying image subtraction.
  - `scribble` — scribble label, skeletonized from xystd.

## Procudure
xy $\xrightarrow{Monte Carlo}$ xyvar → $\xrightarrow{threshold=90+filling+substract}$ xystd $\xrightarrow{skeletionized}$ scribble 

Training:

Before: xyvar $\xrightarrow{Scribble2Label}$ xystd (+scribble)

Now: xy $\xrightarrow{Scribble2Label}$ xystd (+scribble)

## Parameters

| Parameter | value |
|------|-----|
| model | `segresnet-l` |
| patch_size | `128³` |
| batch_size | 2 |
| thres_epoch | 200 |
| period_epoch | 5 |
| thres_conf | 0.8 |
