"""
src/training/trainer.py
PyTorch Lightning module for training image-based and fusion models.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import pauc_score
from src.training.losses import FocalLoss


class SkinCancerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for binary skin cancer classification.

    Supports both image-only and fusion (image + metadata) models.
    Uses Focal Loss to handle class imbalance (~3.5% positive rate).
    Tracks pAUC @ TPR>=0.80 as the primary validation metric.

    Args:
        model: PyTorch model (ImageOnlyClassifier or FusionClassifier)
        lr: Learning rate (default 1e-4 for EfficientNet fine-tuning)
        weight_decay: L2 regularization
        use_metadata: True for fusion model, False for image-only
        focal_alpha: Focal loss alpha (positive class weight)
        focal_gamma: Focal loss gamma (focusing parameter)
        min_tpr: pAUC lower bound for TPR
        warmup_epochs: Linear LR warmup epochs before cosine decay
    """

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 use_metadata: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 min_tpr: float = 0.80,
                 warmup_epochs: int = 2,
                 total_epochs: int = 20):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.use_metadata = use_metadata
        self.min_tpr = min_tpr

        # Storage for OOF predictions (validation epoch)
        self.val_preds = []
        self.val_labels = []

    def forward(self, images, metadata=None):
        if self.use_metadata and metadata is not None:
            return self.model(images, metadata)
        return self.model(images)

    def _shared_step(self, batch, stage: str):
        if self.use_metadata:
            images, metadata, labels = batch
            logits = self.model(images, metadata).squeeze(1)
        else:
            images, labels = batch
            logits = self.model(images).squeeze(1)

        loss = self.criterion(logits, labels.float())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()

        self.log(f'{stage}/loss', loss, on_step=(stage == 'train'),
                 on_epoch=True, prog_bar=True)

        return loss, probs, labels_np

    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch, 'val')
        self.val_preds.extend(probs)
        self.val_labels.extend(labels)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return

        preds = np.array(self.val_preds)
        labels = np.array(self.val_labels)

        try:
            pauc = pauc_score(labels, preds, min_tpr=self.min_tpr)
            auc = roc_auc_score(labels, preds)
        except Exception:
            pauc, auc = 0.0, 0.0

        self.log('val/pauc', pauc, prog_bar=True)
        self.log('val/auc', auc, prog_bar=True)

        # Clear buffers
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing with linear warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return epoch / max(1, self.hparams.warmup_epochs)
            progress = (epoch - self.hparams.warmup_epochs) / \
                       max(1, self.hparams.total_epochs - self.hparams.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }


def get_callbacks(checkpoint_dir: str, monitor: str = 'val/pauc'):
    """Standard training callbacks."""
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, EarlyStopping, LearningRateMonitor
    )
    return [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val/pauc:.4f}',
            monitor=monitor,
            mode='max',
            save_top_k=1,
            verbose=True,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=7,
            mode='max',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
