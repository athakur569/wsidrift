"""DANN: Domain-Adversarial Neural Network for stain-shift adaptation."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from .grl import GradientReversalLayer, lambda_schedule


class DANNModel(pl.LightningModule):
    """ResNet-50 backbone with task and domain classification heads.

    Args:
        n_classes:    Number of task classes.
        n_domains:    Number of source domains (institutions / centers).
        lr:           Learning rate.
        gamma:        GRL lambda schedule steepness.
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_domains: int = 5,
        lr: float = 1e-4,
        gamma: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        backbone = resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.task_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, n_classes)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, n_domains)
        )
        self.grl = GradientReversalLayer()

    def forward(self, x: torch.Tensor):
        return self.task_head(self.backbone(x))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _step(self, batch, stage: str):
        imgs, labels, meta = batch
        p = self.global_step / max(self.trainer.max_steps, 1)
        self.grl.lam = lambda_schedule(p, self.hparams.gamma)
        feats = self.backbone(imgs)
        task_loss = F.cross_entropy(self.task_head(feats), labels)
        domain_labels = meta.get("domain_idx", torch.zeros_like(labels))
        domain_loss = F.cross_entropy(self.domain_head(self.grl(feats)), domain_labels)
        loss = task_loss + domain_loss
        self.log(f"{stage}/task_loss", task_loss, prog_bar=True)
        self.log(f"{stage}/domain_loss", domain_loss)
        self.log(f"{stage}/loss", loss)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
