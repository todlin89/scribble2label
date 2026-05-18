import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class dsb3DDataset(Dataset):
    """Multi-volume 3D dataset for Scribble2Label.

    Loads N >= 1 volumes (image + scribble + full mask) into memory. Each
    volume must have shape (D, H, W); shapes may differ between volumes.

    ``val_index`` semantics:
      * ``None`` -- no held-out volume. Both ``mode='train'`` and ``mode='val'``
        load all N volumes. With N=1 this matches the legacy single-volume
        behaviour (leaky; useful as a sanity check, not as a generalisation
        metric).
      * ``int k`` -- volume k is held out for validation. Requires N >= 2.
        ``mode='train'`` loads volumes ``[i for i != k]``; ``mode='val'``
        loads only volume k.

    Train mode serves random 3D patches drawn from a uniformly-picked train
    volume, with flip + 90 deg rotation augmentation. Val mode serves each
    loaded val volume in full, one item per volume, for sliding-window
    inference downstream.

    The per-voxel ``weights`` list (one float32 array per loaded train
    volume, exponentially-smoothed pseudo-label probability) is updated in
    place by ``Learner3D.ensemble_prediction``.
    """

    def __init__(self, image_paths, scr_paths, mask_paths,
                 val_index=None, patch_size=(128, 128, 128),
                 samples_per_epoch=100, mode='train'):
        assert len(image_paths) == len(scr_paths) == len(mask_paths), (
            f"path list length mismatch: img {len(image_paths)}, "
            f"scr {len(scr_paths)}, mask {len(mask_paths)}"
        )
        n_total = len(image_paths)
        assert n_total >= 1, "need at least one volume"

        if val_index is None:
            global_idxs = list(range(n_total))
        else:
            assert 0 <= val_index < n_total, (
                f"val_index {val_index} out of range [0, {n_total})"
            )
            assert n_total >= 2, (
                "val_index requires N>=2; use val_index=None for single-volume"
            )
            if mode == 'train':
                global_idxs = [i for i in range(n_total) if i != val_index]
            else:
                global_idxs = [val_index]

        self.images = [tifffile.imread(image_paths[i]) for i in global_idxs]
        self.scribbles = [tifffile.imread(scr_paths[i]) for i in global_idxs]
        self.masks = [
            (tifffile.imread(mask_paths[i]) > 0).astype(np.uint8)
            for i in global_idxs
        ]

        for k, (im, sc, mk) in enumerate(zip(self.images, self.scribbles, self.masks)):
            assert im.shape == sc.shape == mk.shape, (
                f"shape mismatch in volume_{global_idxs[k]}: "
                f"img {im.shape}, scr {sc.shape}, mask {mk.shape}"
            )

        self.weights = [np.zeros(im.shape, dtype=np.float32) for im in self.images]
        self.image_ids = [f'volume_{i}' for i in global_idxs]
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode

    def __len__(self):
        if self.mode == 'val':
            return len(self.images)
        return self.samples_per_epoch

    def _random_origin(self, vi):
        D, H, W = self.images[vi].shape
        pd, ph, pw = self.patch_size
        return (random.randint(0, D - pd),
                random.randint(0, H - ph),
                random.randint(0, W - pw))

    @staticmethod
    def _augment(arrays):
        for axis in range(3):
            if random.random() < 0.5:
                arrays = [np.flip(a, axis=axis) for a in arrays]
        k = random.randint(0, 3)
        if k:
            arrays = [np.rot90(a, k=k, axes=(1, 2)) for a in arrays]
        return [np.ascontiguousarray(a) for a in arrays]

    def _val_item(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(self.masks[idx].astype(np.int64))
        return self.image_ids[idx], img_t, mask_t

    def _train_item(self):
        vi = random.randrange(len(self.images))
        z, y, x = self._random_origin(vi)
        pd, ph, pw = self.patch_size

        img_p = self.images[vi][z:z+pd, y:y+ph, x:x+pw]
        scr_p = self.scribbles[vi][z:z+pd, y:y+ph, x:x+pw]
        w_p = self.weights[vi][z:z+pd, y:y+ph, x:x+pw]

        img_p, scr_p, w_p = self._augment([img_p, scr_p, w_p])

        img_t = torch.from_numpy(img_p.astype(np.float32) / 255.0).unsqueeze(0)
        scr_t = torch.from_numpy(scr_p.astype(np.int64))
        w_t = torch.from_numpy(w_p.astype(np.float32)).unsqueeze(-1)
        return img_t, scr_t, w_t

    def __getitem__(self, idx):
        if self.mode == 'val':
            return self._val_item(idx)
        return self._train_item()
