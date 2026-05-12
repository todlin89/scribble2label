import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class dsb3DDataset(Dataset):
    """3D volume dataset for Scribble2Label.

    Loads one 3D volume (image + scribble + full mask) into memory.
    Training mode: serves random 3D patches with flip/rotation augmentation.
    Validation mode: serves the full volume for sliding-window inference.

    The per-voxel `weight` volume (exponentially-smoothed pseudo-label
    probability) is shared across all training patches. It lives on the
    dataset so that Learner3D.ensemble_prediction can update it in place.
    """

    def __init__(self, image_path, scr_path, mask_path,
                 patch_size=(128, 128, 128), samples_per_epoch=100,
                 mode='train'):
        self.image = tifffile.imread(image_path)
        self.scribble = tifffile.imread(scr_path)
        self.mask = (tifffile.imread(mask_path) > 0).astype(np.uint8)

        assert self.image.shape == self.scribble.shape == self.mask.shape, (
            f"shape mismatch: img {self.image.shape}, "
            f"scr {self.scribble.shape}, mask {self.mask.shape}"
        )

        self.weight = np.zeros(self.image.shape, dtype=np.float32)
        self.image_ids = ['volume_0']
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode

    def __len__(self):
        return self.samples_per_epoch if self.mode == 'train' else 1

    def _random_origin(self):
        D, H, W = self.image.shape
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

    def _val_item(self):
        img = self.image.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, D, H, W)
        mask_t = torch.from_numpy(self.mask.astype(np.int64))  # (D, H, W)
        return 'volume_0', img_t, mask_t

    def _train_item(self):
        z, y, x = self._random_origin()
        pd, ph, pw = self.patch_size

        img_p = self.image[z:z+pd, y:y+ph, x:x+pw]
        scr_p = self.scribble[z:z+pd, y:y+ph, x:x+pw]
        w_p = self.weight[z:z+pd, y:y+ph, x:x+pw]

        img_p, scr_p, w_p = self._augment([img_p, scr_p, w_p])

        img_t = torch.from_numpy(img_p.astype(np.float32) / 255.0).unsqueeze(0)
        scr_t = torch.from_numpy(scr_p.astype(np.int64))
        w_t = torch.from_numpy(w_p.astype(np.float32)).unsqueeze(-1)
        return img_t, scr_t, w_t

    def __getitem__(self, idx):
        if self.mode == 'val':
            return self._val_item()
        return self._train_item()
