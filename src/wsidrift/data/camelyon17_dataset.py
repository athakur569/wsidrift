"""CAMELYON17 patch dataset for wsidrift experiments."""

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

# Map stages.csv slide-level labels to binary: negative=0, any metastasis=1
_BINARY_MAP = {"negative": 0, "itc": 1, "micro": 1, "macro": 1}


class CAMELYON17PatchDataset(Dataset):
    """Patch-level dataset for CAMELYON17 loaded from a manifest CSV.

    Expected manifest columns: path, slide_label, center, patient_id, split.
    slide_label is the raw stages.csv value (negative/itc/micro/macro).

    Args:
        manifest_path: Path to manifest CSV.
        split:         'train', 'val', 'test', or None.
        center:        Filter to a specific center (0–4).
        transform:     torchvision transform.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        center: int | None = None,
        transform=None,
    ) -> None:
        df = pd.read_csv(manifest_path)
        if split is not None:
            df = df[df["split"] == split]
        if center is not None:
            df = df[df["center"] == center]
        df = df.reset_index(drop=True)
        df["label"] = df["slide_label"].map(_BINARY_MAP).fillna(0).astype(int)
        self.df = df
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])
        meta = {
            "center": int(row["center"]),
            "patient_id": str(row.get("patient_id", "")),
            "slide_label": str(row["slide_label"]),
        }
        return img, label, meta

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    @property
    def centers(self) -> list[int]:
        return self.df["center"].tolist()
