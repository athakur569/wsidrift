"""CycleGAN training harness (stub — full training loop in scripts/train_cyclegan.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .generator import ResNetGenerator


@dataclass
class CycleGANConfig:
    source_dir: str = ""
    target_dir: str = ""
    checkpoint_dir: str = "checkpoints/cyclegan"
    n_epochs: int = 200
    batch_size: int = 4
    lr: float = 2e-4
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    n_residual: int = 9


class CycleGANTrainer:
    """Minimal wrapper to load/save CycleGAN generators and apply inference.

    Full adversarial training is handled by scripts/train_cyclegan.py.
    This class handles checkpoint management and inference for E4/E7/E9.
    """

    def __init__(self, config: CycleGANConfig | None = None) -> None:
        self.config = config or CycleGANConfig()
        self.G_AB = ResNetGenerator(n_residual=self.config.n_residual)
        self.G_BA = ResNetGenerator(n_residual=self.config.n_residual)

    def save(self, epoch: int) -> None:
        outdir = Path(self.config.checkpoint_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(self.G_AB.state_dict(), outdir / f"G_AB_epoch{epoch:03d}.pt")
        torch.save(self.G_BA.state_dict(), outdir / f"G_BA_epoch{epoch:03d}.pt")

    def load(self, checkpoint_dir: str, epoch: int) -> None:
        d = Path(checkpoint_dir)
        self.G_AB.load_state_dict(torch.load(d / f"G_AB_epoch{epoch:03d}.pt", map_location="cpu"))
        self.G_BA.load_state_dict(torch.load(d / f"G_BA_epoch{epoch:03d}.pt", map_location="cpu"))

    @torch.no_grad()
    def translate_AtoB(self, x: torch.Tensor) -> torch.Tensor:
        self.G_AB.eval()
        return self.G_AB(x)

    @torch.no_grad()
    def translate_BtoA(self, x: torch.Tensor) -> torch.Tensor:
        self.G_BA.eval()
        return self.G_BA(x)
