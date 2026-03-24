import os
from PIL import Image
import warnings

from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

warnings.filterwarnings("ignore")


def get_transforms(input_size=256, need=('train', 'val')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.RandomCrop(height=input_size, width=input_size, p=1.),
            A.ShiftScaleRotate(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'scr': 'mask', 'weight': 'mask'})
    if 'val' in need:
        transformations['val'] = A.Compose([
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'scr': 'mask', 'weight': 'mask'})
    return transformations


class dsbDataset(Dataset):
    def __init__(self, data_folder, scr_folder, mask_folder, df, tfms, return_id=False,
                 lazy=False):
        self.data_folder = data_folder
        self.scr_folder = scr_folder
        self.mask_folder = mask_folder
        self.tfms = tfms
        self.return_id = return_id
        self.lazy = lazy
        self.image_ids = [row.ImageID for _, row in df.iterrows()]
        self.length = len(df)

        self.images = defaultdict(dict)
        if lazy:
            # Lazy loading: only store weights in memory, read images on-the-fly
            for idx, image_id in enumerate(self.image_ids):
                img = Image.open(os.path.join(data_folder, f'{image_id}.png'))
                w_img, h_img = img.size
                h = (h_img // 32) * 32
                w = (w_img // 32) * 32
                self.images[idx]['h'] = h
                self.images[idx]['w'] = w
                self.images[idx]['weight'] = np.zeros((h, w, 1), dtype=np.float32)
        else:
            # Eager loading: keep everything in memory (original behavior)
            for idx, image_id in enumerate(self.image_ids):
                img = np.array(Image.open(os.path.join(data_folder, f'{image_id}.png')).convert('RGB'))
                scr = np.array(Image.open(os.path.join(scr_folder, f'{image_id}.png')))
                mask = (np.array(Image.open(os.path.join(mask_folder, f'{image_id}.png')).convert('L')) > 0)
                h, w = mask.shape
                h = (h // 32) * 32
                w = (w // 32) * 32
                self.images[idx]['image'] = img[:h, :w, :]
                self.images[idx]['mask'] = mask[:h, :w].astype('uint8')
                self.images[idx]['scr'] = scr[:h, :w].astype('uint8')
                self.images[idx]['weight'] = np.zeros((h, w, 1), dtype=np.float32)

    def _load(self, idx):
        image_id = self.image_ids[idx]
        h, w = self.images[idx]['h'], self.images[idx]['w']
        img = np.array(Image.open(os.path.join(self.data_folder, f'{image_id}.png')).convert('RGB'))[:h, :w, :]
        scr = np.array(Image.open(os.path.join(self.scr_folder, f'{image_id}.png')))[:h, :w].astype('uint8')
        mask = (np.array(Image.open(os.path.join(self.mask_folder, f'{image_id}.png')).convert('L')) > 0)[:h, :w].astype('uint8')
        return img, scr, mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.lazy:
            image, scribble, mask = self._load(idx)
        else:
            image = self.images[idx]['image']
            scribble = self.images[idx]['scr']
            mask = self.images[idx]['mask']
        weight = self.images[idx]['weight']

        if self.tfms:
            augmented = self.tfms(image=image,
                                  mask=mask,
                                  scr=scribble,
                                  weight=weight)
            image, scribble, weight, mask = augmented['image'], augmented['scr'],\
                                               augmented['weight'], augmented['mask']
        if self.return_id:
            return image_id, image, scribble, mask
        else:
            return image, scribble, weight

    def __len__(self):
        return self.length


class dsbTestDataset(Dataset):
    def __init__(self, data_folder, mask_folder, df, tfms):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.ImageIDs = df.ImageID.values
        self.tfms = tfms

    def __getitem__(self, idx):
        image_id = self.ImageIDs[idx]
        image = np.array(Image.open(os.path.join(self.data_folder, f'{image_id}.png')).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.mask_folder, f'{image_id}.png')).convert('L'))

        h, w = mask.shape
        h = (h // 32) * 32
        w = (w // 32) * 32
        image, mask = image[:h, :w, :], mask[:h, :w]

        if self.tfms:
            augmented = self.tfms(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image_id, image, mask

    def __len__(self):
        return len(self.ImageIDs)
