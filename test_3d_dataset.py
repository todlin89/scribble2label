"""Sanity tests for the multi-volume 3D dataset and ensemble_prediction.

Usage
-----
    python test_3d_dataset.py            # plain run, prints PASS / raises on fail
    pytest test_3d_dataset.py -v         # also works (functions named test_*)

What it verifies
----------------
1. Hold-out isolation -- when val_index=k, the train dataset never loads
   volume k and the val dataset loads only volume k. The val volume's
   voxel data must not appear anywhere in the train dataset.

2. Train sampling distribution -- across many _train_item() draws, every
   train volume is picked with ~uniform probability. Catches the silent
   bug where a missing `vi = random.randrange(...)` would lock training
   onto a single volume.

3. ensemble_prediction per-volume update -- after one ensemble pass,
   every train volume's weight buffer has been updated (not just
   volume 0) and a per-volume pseudo-label TIFF is exported.
"""

import os
import sys
import tempfile
from collections import Counter

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dataset_3d import dsb3DDataset


# ---------------------------------------------------------------------------
# Synthetic data: each volume_i is uniformly filled with intensity (i+1)*30
# so a single voxel sample tells us which source volume a patch came from.
# ---------------------------------------------------------------------------

def make_fake_volumes(tmpdir, n=3, shape=(32, 32, 32)):
    image_paths, scr_paths, mask_paths = [], [], []
    for i in range(n):
        intensity = (i + 1) * 30
        img = np.full(shape, intensity, dtype=np.uint8)

        scr = np.full(shape, 250, dtype=np.uint8)
        scr[:, :2, :2] = 1
        scr[:, -2:, -2:] = 0

        mask = np.zeros(shape, dtype=np.uint8)
        mask[:, : shape[1] // 2, :] = 1

        ip = os.path.join(tmpdir, f'vol_{i}_img.tif')
        sp = os.path.join(tmpdir, f'vol_{i}_scr.tif')
        mp = os.path.join(tmpdir, f'vol_{i}_full.tif')
        tifffile.imwrite(ip, img)
        tifffile.imwrite(sp, scr)
        tifffile.imwrite(mp, mask)

        image_paths.append(ip)
        scr_paths.append(sp)
        mask_paths.append(mp)
    return image_paths, scr_paths, mask_paths


# ---------------------------------------------------------------------------
# 1. Hold-out isolation
# ---------------------------------------------------------------------------

def test_holdout_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        img_p, scr_p, mask_p = make_fake_volumes(tmp, n=4)

        train_ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=2, patch_size=(16, 16, 16), mode='train',
        )
        val_ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=2, patch_size=(16, 16, 16), mode='val',
        )

        assert train_ds.image_ids == ['volume_0', 'volume_1', 'volume_3'], \
            f"unexpected train ids: {train_ds.image_ids}"
        assert val_ds.image_ids == ['volume_2'], \
            f"unexpected val ids: {val_ds.image_ids}"

        # val volume intensity is (2+1)*30 = 90; must NOT appear in train.
        train_intensities = {int(im.flat[0]) for im in train_ds.images}
        val_intensity = int(val_ds.images[0].flat[0])
        assert val_intensity == 90, f"val intensity wrong: {val_intensity}"
        assert val_intensity not in train_intensities, (
            f"val volume leaked into train! train={train_intensities}, "
            f"val={val_intensity}"
        )

        assert len(train_ds.images) == 3
        assert len(train_ds.weights) == 3
        assert len(val_ds.images) == 1

        print('[PASS] hold-out isolation '
              f'(train={train_ds.image_ids}, val={val_ds.image_ids})')


def test_holdout_isolation_val_index_none():
    """val_index=None: both modes load every volume. Recovers the legacy
    single-volume sanity-check behaviour when N=1."""
    with tempfile.TemporaryDirectory() as tmp:
        img_p, scr_p, mask_p = make_fake_volumes(tmp, n=1)

        train_ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=None, patch_size=(16, 16, 16), mode='train',
        )
        val_ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=None, patch_size=(16, 16, 16), mode='val',
        )

        assert train_ds.image_ids == ['volume_0']
        assert val_ds.image_ids == ['volume_0']
        assert len(train_ds.images) == len(val_ds.images) == 1
        print('[PASS] val_index=None single-volume legacy mode')


# ---------------------------------------------------------------------------
# 2. Train sampling distribution
# ---------------------------------------------------------------------------

def test_sampling_uniformity():
    with tempfile.TemporaryDirectory() as tmp:
        img_p, scr_p, mask_p = make_fake_volumes(tmp, n=4)
        ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=None, patch_size=(16, 16, 16),
            samples_per_epoch=1, mode='train',
        )

        # intensity -> source volume id
        id_of = {(i + 1) * 30: i for i in range(4)}

        n_draws = 4000
        counts = Counter()
        for _ in range(n_draws):
            img_t, _, _ = ds._train_item()
            # _train_item normalises by /255; recover the source intensity.
            intensity = int(round(float(img_t.flatten()[0]) * 255))
            counts[id_of[intensity]] += 1

        expected = n_draws / 4
        tolerance = 0.20 * expected  # ±20% — far above 3-sigma multinomial noise
        for vol_id in range(4):
            c = counts.get(vol_id, 0)
            assert abs(c - expected) <= tolerance, (
                f"volume {vol_id} drawn {c} times, expected "
                f"~{expected:.0f} (+/-{tolerance:.0f})"
            )
        print(f'[PASS] train sampling uniformity counts={dict(counts)}')


# ---------------------------------------------------------------------------
# 3. ensemble_prediction touches every train volume
# ---------------------------------------------------------------------------

def test_ensemble_prediction_per_volume():
    import torch
    from torch.utils.data import DataLoader
    from monai.networks.nets import BasicUNet
    from Learner3D import Learner3D

    with tempfile.TemporaryDirectory() as tmp:
        img_p, scr_p, mask_p = make_fake_volumes(tmp, n=3, shape=(32, 32, 32))

        train_ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=None, patch_size=(32, 32, 32),
            samples_per_epoch=1, mode='train',
        )
        val_ds = dsb3DDataset(
            img_p, scr_p, mask_p,
            val_index=None, patch_size=(32, 32, 32), mode='val',
        )

        class Cfg:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            log_dir = os.path.join(tmp, 'logs')
            lr = 1e-3
            weight_decay = 0.0
            ignore_index = 250
            thr_epoch = 200
            period_epoch = 5
            thr_conf = 0.8
            alpha = 0.2
            sw_roi = (32, 32, 32)
            sw_batch_size = 1
            sw_overlap = 0.25
            use_amp = False
        os.makedirs(Cfg.log_dir, exist_ok=True)

        model = BasicUNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            features=(4, 8, 16, 32, 64, 4),
        )

        train_loader = DataLoader(train_ds, batch_size=1)
        val_loader = DataLoader(val_ds, batch_size=1)
        learner = Learner3D(model, train_loader, val_loader, Cfg)

        for vi in range(len(train_ds.images)):
            assert train_ds.weights[vi].sum() == 0, \
                f"weight buffer {vi} not zero before ensemble"

        learner.ensemble_prediction()

        # Every train volume must have a non-zero, correctly-shaped weight buffer.
        for vi in range(len(train_ds.images)):
            assert train_ds.weights[vi].sum() > 0, \
                f"volume {vi} weight was NOT updated by ensemble_prediction"
            assert train_ds.weights[vi].shape == train_ds.images[vi].shape, \
                f"volume {vi} weight shape != image shape"

        # The two volumes' weights should differ since their input intensities
        # differ -- this confirms each volume drove its own forward pass rather
        # than being silently aliased to one prediction.
        w0 = train_ds.weights[0]
        w1 = train_ds.weights[1]
        assert not np.allclose(w0, w1), (
            "weights for volumes 0 and 1 are identical -- ensemble_prediction "
            "may be running on the same volume twice"
        )

        # Per-volume pseudo-label TIFFs must exist.
        ensemble_dir = os.path.join(Cfg.log_dir, 'pseudo_labels')
        rounds = os.listdir(ensemble_dir)
        assert len(rounds) == 1, f"unexpected ensemble round dirs: {rounds}"
        round_dir = os.path.join(ensemble_dir, rounds[0])
        files = set(os.listdir(round_dir))
        expected = {f'{vid}.tif' for vid in train_ds.image_ids}
        assert expected.issubset(files), (
            f"missing pseudo-label TIFFs: expected {expected}, got {files}"
        )
        print(f'[PASS] ensemble_prediction per-volume update '
              f'(volumes={train_ds.image_ids}, files={sorted(files)})')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_holdout_isolation()
    test_holdout_isolation_val_index_none()
    test_sampling_uniformity()
    test_ensemble_prediction_per_volume()
    print('\nAll tests passed.')
