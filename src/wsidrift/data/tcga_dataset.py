"""TCGA patch dataset for wsidrift experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


_DEFAULT_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TCGAPatchDataset(Dataset):
    """Patch-level dataset loaded from a manifest CSV.

    Expected manifest columns: path, label, slide_id, tss, cohort, split.

    Args:
        manifest_path: Path to manifest CSV.
        split:         One of 'train', 'val', 'test', or None (all).
        cohort:        Filter to a specific cohort (e.g. 'BRCA').
        transform:     torchvision transform; defaults to ImageNet normalization.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        cohort: str | None = None,
        transform=None,
    ) -> None:
        df = pd.read_csv(manifest_path)
        if split is not None:
            df = df[df["split"] == split]
        if cohort is not None:
            df = df[df["cohort"] == cohort]
        self.df = df.reset_index(drop=True)
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])
        meta = {
            "slide_id": row["slide_id"],
            "tss": str(row.get("tss", "")),
            "cohort": str(row.get("cohort", "")),
        }
        return img, label, meta

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    @property
    def tss_codes(self) -> list[str]:
        return self.df["tss"].tolist()
