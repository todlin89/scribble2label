import os
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import BasicUNet

from scripts.dataset_3d import dsb3DDataset
from scripts.utils import seed_everything
from Learner3D import Learner3D


class config:
    seed = 42
    mode = '3d'
    name = 'thres_90_3d'
    device = torch.device('cuda:0')

    """ Data paths (parallel lists of 3D TIFF volumes, each shape (D, H, W)).
    Set val_index = k to hold out volume k for validation (requires N>=2);
    leave val_index = None for the legacy single-volume sanity check. """
    image_paths = ['/data/datahere/Todd/data/ImagesTr-1/imagesTr.tif']
    scr_paths = ['/data/datahere/Todd/data/ImagesTr-1/scribble100.tif']
    mask_paths = ['/data/datahere/Todd/data/ImagesTr-1/full.tif']
    val_index = None
    log_dir = f'./logs/{name}'

    """ Training """
    n_epochs = 10000
    patch_size = (128, 128, 128)
    batch_size = 2
    samples_per_epoch = 100
    lr = 3e-4
    weight_decay = 5e-5
    num_workers = 2
    ignore_index = 250
    use_amp = True

    """ Inference (sliding-window over the full 512^3 volume) """
    sw_roi = (128, 128, 128)
    sw_overlap = 0.25
    sw_batch_size = 2

    """ Scribble2Label params """
    thr_epoch = 200
    period_epoch = 5
    thr_conf = 0.8
    alpha = 0.2

    """ Model """
    features = (16, 32, 64, 128, 256, 16)


if __name__ == '__main__':
    seed_everything(config.seed)
    os.makedirs(config.log_dir, exist_ok=True)

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        features=config.features,
    )

    train_dataset = dsb3DDataset(
        image_paths=config.image_paths,
        scr_paths=config.scr_paths,
        mask_paths=config.mask_paths,
        val_index=config.val_index,
        patch_size=config.patch_size,
        samples_per_epoch=config.samples_per_epoch,
        mode='train',
    )
    valid_dataset = dsb3DDataset(
        image_paths=config.image_paths,
        scr_paths=config.scr_paths,
        mask_paths=config.mask_paths,
        val_index=config.val_index,
        patch_size=config.patch_size,
        mode='val',
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1,
                              num_workers=0, shuffle=False)

    learner = Learner3D(model, train_loader, valid_loader, config)
    pretrained_path = os.path.join(config.log_dir, 'best_model.pth')
    if os.path.isfile(pretrained_path):
        learner.load(pretrained_path)
        learner.log(f'Checkpoint Loaded: {pretrained_path}')
    learner.fit(config.n_epochs)
