import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from monai.inferers import sliding_window_inference

from scripts.metric import AverageMeter
from scripts.optimizer import RAdam
from scripts.tb_utils import init_tb_logger
from scripts.utils import init_logger


class Learner3D:
    def __init__(self, model, train_loader, valid_loader, config):
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(self.config.device)

        self.logger = init_logger(self.config.log_dir, 'train_main.log')
        self.tb_logger = init_tb_logger(self.config.log_dir, 'train_main')
        self.log('\n'.join([f"{k} = {v}" for k, v in self.config.__dict__.items()]))

        self.summary_loss = AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        self.u_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        self.optimizer = RAdam(self.model.parameters(),
                               lr=self.config.lr,
                               weight_decay=self.config.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=2, T_mult=2, eta_min=1e-6)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(getattr(self.config, 'use_amp', False) and self.config.device.type == 'cuda')
        )

        self.n_ensemble = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.best_score = -np.inf

    def _autocast(self):
        enabled = bool(getattr(self.config, 'use_amp', False) and self.config.device.type == 'cuda')
        return torch.cuda.amp.autocast(enabled=enabled)

    @staticmethod
    def _binary_iou(targets, preds):
        targets = (targets > 0)
        preds = (preds > 0)
        intersection = np.logical_and(targets, preds).sum(dtype=np.int64)
        union = np.logical_or(targets, preds).sum(dtype=np.int64)
        return (intersection + 1e-6) / (union + 1e-6)

    def train_one_epoch(self):
        self.model.train()
        self.summary_loss.reset()
        iters = len(self.train_loader)

        for step, (images, scribbles, weights) in enumerate(self.train_loader):
            self.tb_logger.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'],
                                      iters * self.epoch + step)

            images = images.to(self.config.device, non_blocking=True)
            scribbles = scribbles.to(self.config.device, non_blocking=True).long()
            weights = weights.to(self.config.device, non_blocking=True)
            batch_size = images.shape[0]

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                outputs = self.model(images)
                if self.epoch < self.config.thr_epoch:
                    loss = self.criterion(outputs, scribbles)
                else:
                    x_loss = self.criterion(outputs, scribbles)
                    mean = weights[..., 0]
                    unlabeled = scribbles == self.config.ignore_index
                    confident = (mean < (1 - self.config.thr_conf)) | (mean > self.config.thr_conf)
                    pseudo = torch.where(confident & unlabeled,
                                         mean.round().long(),
                                         torch.full_like(scribbles, self.config.ignore_index))
                    u_loss = self.u_criterion(outputs, pseudo)
                    loss = x_loss + 0.5 * u_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.summary_loss.update(loss.detach().item(), batch_size)
            if self.scheduler.__class__.__name__ != 'ReduceLROnPlateau':
                self.scheduler.step()

        return self.summary_loss.avg

    def validation(self):
        self.model.eval()
        self.summary_loss.reset()
        scores = []

        for _, images, targets in self.valid_loader:
            images = images.to(self.config.device, non_blocking=True)
            targets = targets.to(self.config.device, non_blocking=True).long()
            batch_size = images.shape[0]

            with torch.no_grad(), self._autocast():
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=self.config.sw_roi,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.sw_overlap,
                    mode='gaussian',
                )
                loss = self.criterion(logits, targets)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            target_np = targets.cpu().numpy()
            scores.append(self._binary_iou(target_np, preds))
            self.summary_loss.update(loss.detach().item(), batch_size)

        score = float(np.mean(scores)) if scores else 0.0
        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        return self.summary_loss.avg, score

    def ensemble_prediction(self):
        ds = self.train_loader.dataset
        volume = torch.from_numpy(ds.image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        volume = volume.to(self.config.device)

        save_dir = os.path.join(self.config.log_dir, 'pseudo_labels',
                                f'ensemble_{self.n_ensemble:03d}_epoch_{self.epoch:04d}')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad(), self._autocast():
            logits = sliding_window_inference(
                inputs=volume,
                roi_size=self.config.sw_roi,
                sw_batch_size=self.config.sw_batch_size,
                predictor=self.model,
                overlap=self.config.sw_overlap,
                mode='gaussian',
            )
            prob = F.softmax(logits, dim=1)[0, 1].to(torch.float32).cpu().numpy()

        ds.weight[...] = self.config.alpha * prob + (1 - self.config.alpha) * ds.weight

        try:
            import tifffile

            thr_hi = self.config.thr_conf
            thr_lo = 1 - self.config.thr_conf
            vis = np.full(ds.weight.shape, 127, dtype=np.uint8)
            vis[ds.weight > thr_hi] = 255
            vis[ds.weight < thr_lo] = 0
            tifffile.imwrite(os.path.join(save_dir, 'volume_0.tif'), vis)
        except Exception as exc:
            self.log(f'Pseudo-label export skipped: {exc}')

        self.n_ensemble += 1

    def fit(self, epochs):
        for _ in range(epochs):
            t = time.time()
            train_loss = self.train_one_epoch()
            self.log(f'[Train] \t Epoch: {self.epoch}, loss: {train_loss:.5f}, time: {(time.time() - t):.2f}')
            self.tb_log(train_loss, None, 'Train', self.epoch)

            t = time.time()
            valid_loss, score = self.validation()
            self.log(f'[Valid] \t Epoch: {self.epoch}, loss: {valid_loss:.5f}, IoU: {score:.4f}, time: {(time.time() - t):.2f}')
            self.tb_log(valid_loss, score, 'Valid', self.epoch)
            self.post_processing(valid_loss, score)

            if (self.epoch + 1) % self.config.period_epoch == 0:
                self.log(f'[Ensemble] \t the {self.n_ensemble}th Prediction Ensemble ...')
                self.ensemble_prediction()

            self.epoch += 1

        self.log(f'best epoch: {self.best_epoch}, best loss: {self.best_loss}, best_score: {self.best_score}')

    def post_processing(self, loss, score):
        if loss < self.best_loss:
            self.best_loss = loss

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch

            self.model.eval()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_score': self.best_score,
                'epoch': self.epoch,
            }, os.path.join(self.config.log_dir, 'best_model.pth'))
            self.log(f'best model: {self.epoch} epoch - {score:.4f}')

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_score = checkpoint.get('best_score', self.best_score)
        self.epoch = checkpoint.get('epoch', -1) + 1

    def log(self, text):
        self.logger.info(text)

    def tb_log(self, loss, iou, split, step):
        if loss is not None:
            self.tb_logger.add_scalar(f'{split}/Loss', loss, step)
        if iou is not None:
            self.tb_logger.add_scalar(f'{split}/IoU', iou, step)
