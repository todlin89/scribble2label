import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scripts.utils import seed_everything


# ─── 3D model registry ──────────────────────────────────────────────────────

MODELS_3D = {
    'basicunet':    '(default) BasicUNet features=(16,32,64,128,256,16), ~5.7M',
    'basicunet-s':  'BasicUNet features=(8,16,32,64,128,8), ~1.4M',
    'unet':         'UNet channels=(16,32,64,128) + res blocks, ~1.2M',
    'unet-s':       'UNet channels=(8,16,32,64) + res blocks, ~0.3M',
    'segresnet':    'SegResNet init_filters=8, ~1.2M',
    'segresnet-l':  'SegResNet init_filters=16, ~4.7M',
    'attention':    'AttentionUnet channels=(16,32,64,128), ~1.5M',
    'highresnet':   'HighResNet dilated conv, ~0.8M',
}


def build_model_3d(model_name):
    if model_name == 'basicunet':
        from monai.networks.nets import BasicUNet
        return BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                         features=(16, 32, 64, 128, 256, 16))
    elif model_name == 'basicunet-s':
        from monai.networks.nets import BasicUNet
        return BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                         features=(8, 16, 32, 64, 128, 8))
    elif model_name == 'unet':
        from monai.networks.nets import UNet
        return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                    channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2)
    elif model_name == 'unet-s':
        from monai.networks.nets import UNet
        return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                    channels=(8, 16, 32, 64), strides=(2, 2, 2), num_res_units=2)
    elif model_name == 'segresnet':
        from monai.networks.nets import SegResNet
        return SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=8)
    elif model_name == 'segresnet-l':
        from monai.networks.nets import SegResNet
        return SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16)
    elif model_name == 'attention':
        from monai.networks.nets import AttentionUnet
        return AttentionUnet(spatial_dims=3, in_channels=1, out_channels=2,
                             channels=(16, 32, 64, 128), strides=(2, 2, 2))
    elif model_name == 'highresnet':
        from monai.networks.nets import HighResNet
        return HighResNet(spatial_dims=3, in_channels=1, out_channels=2)
    else:
        raise ValueError(f"Unknown 3D model: {model_name}. Choose from: {list(MODELS_3D.keys())}")


# ─── Config ──────────────────────────────────────────────────────────────────

def get_config(mode, model_name='basicunet'):
    class config:
        pass

    config.mode = mode
    config.model_name = model_name
    config.seed = 42
    config.n_epochs = 10000
    config.lr = 3e-4
    config.weight_decay = 5e-5
    config.ignore_index = 250
    config.thr_epoch = 200
    config.period_epoch = 5
    config.thr_conf = 0.8
    config.alpha = 0.2

    if mode == '2d':
        config.name = 'thres_90_from_filled_mask'
        config.device = torch.device('cuda:1')
        config.data_dir = f'./examples/images/{config.name}/'
        config.scr_dir = f'./examples/labels/{config.name}/scribble100/'
        config.mask_dir = f'./examples/labels/{config.name}/full/'
        config.df_path = f'./examples/labels/{config.name}/train.csv'
        config.log_dir = f'./logs/{config.name}'
        config.fold = 0
        config.input_size = 512
        config.batch_size = 8
        config.num_workers = 8

    elif mode == '3d':
        config.name = f'thres_90_3d_{model_name}'
        config.device = torch.device('cuda:0')
        config.image_path = '/data/datahere/Todd/data/ImagesTr-1/imagesTr.tif'
        config.scr_path = '/data/datahere/Todd/data/ImagesTr-1/scribble100.tif'
        config.mask_path = '/data/datahere/Todd/data/ImagesTr-1/full.tif'
        config.log_dir = f'./logs/{config.name}'
        config.patch_size = (256, 256, 256)
        config.batch_size = 2
        config.samples_per_epoch = 100
        config.num_workers = 2
        config.use_amp = True
        config.sw_roi = (256, 256, 256)
        config.sw_overlap = 0.25
        config.sw_batch_size = 2

    return config


if __name__ == '__main__':
    model_help = '\n'.join(f'  {k:15s} {v}' for k, v in MODELS_3D.items())
    parser = argparse.ArgumentParser(
        description='Scribble2Label Training',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'],
                        help='Training mode: 2d or 3d (default: 3d)')
    parser.add_argument('--model', type=str, default='basicunet',
                        choices=list(MODELS_3D.keys()),
                        help=f'3D model architecture (ignored for 2d):\n{model_help}')
    args = parser.parse_args()

    config = get_config(args.mode, args.model)
    seed_everything(config.seed)
    os.makedirs(config.log_dir, exist_ok=True)

    if config.mode == '2d':
        from scripts.dataset import get_transforms, dsbDataset
        from segmentation_models_pytorch import Unet
        from Learner import Learner

        model = Unet(encoder_name='resnet50', encoder_weights='imagenet',
                     decoder_use_batchnorm=True, decoder_attention_type='scse',
                     classes=2, activation=None)

        df = pd.read_csv(config.df_path)
        train_df = df[df.fold != config.fold].reset_index(drop=True)
        valid_df = df[df.fold == config.fold].reset_index(drop=True)
        transforms = get_transforms(config.input_size, need=('train', 'val'))

        train_dataset = dsbDataset(config.data_dir, config.scr_dir, config.mask_dir,
                                   train_df, tfms=transforms['train'],
                                   return_id=False, lazy=True)
        valid_dataset = dsbDataset(config.data_dir, config.scr_dir, config.mask_dir,
                                   valid_df, tfms=transforms['val'],
                                   return_id=True, lazy=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1,
                                  num_workers=config.num_workers, shuffle=False)

        learner = Learner(model, train_loader, valid_loader, config)
        pretrained_path = os.path.join(config.log_dir, 'best_model.pth')
        if os.path.isfile(pretrained_path):
            learner.load(pretrained_path)
            learner.log(f"Checkpoint Loaded: {pretrained_path}")
        learner.fit(config.n_epochs)

    elif config.mode == '3d':
        from scripts.dataset_3d import dsb3DDataset
        from Learner3D import Learner3D

        model = build_model_3d(config.model_name)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {config.model_name} ({n_params/1e6:.2f}M params)")

        train_dataset = dsb3DDataset(
            image_path=config.image_path,
            scr_path=config.scr_path,
            mask_path=config.mask_path,
            patch_size=config.patch_size,
            samples_per_epoch=config.samples_per_epoch,
            mode='train',
        )
        valid_dataset = dsb3DDataset(
            image_path=config.image_path,
            scr_path=config.scr_path,
            mask_path=config.mask_path,
            patch_size=config.patch_size,
            samples_per_epoch=1,
            mode='val',
        )
        valid_dataset.weight = train_dataset.weight

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  num_workers=config.num_workers, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1,
                                  num_workers=0, shuffle=False)

        learner = Learner3D(model, train_loader, valid_loader, config)
        pretrained_path = os.path.join(config.log_dir, 'best_model.pth')
        if os.path.isfile(pretrained_path):
            learner.load(pretrained_path)
            learner.log(f"Checkpoint Loaded: {pretrained_path}")
        learner.fit(config.n_epochs)
