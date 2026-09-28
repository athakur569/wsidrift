"""Reinhard stain normalization via Lab color space mean/std transfer."""

from __future__ import annotations

import numpy as np


class ReinhardNormalizer:
    """Transfer stain appearance by matching Lab channel statistics to a reference.

    Args:
        None — fit() sets reference statistics.
    """

    def __init__(self) -> None:
        self._ref_mean: np.ndarray | None = None
        self._ref_std: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> "ReinhardNormalizer":
        """Compute reference Lab statistics from image (H, W, 3) uint8."""
        lab = self._rgb_to_lab(reference)
        self._ref_mean = lab.mean(axis=(0, 1))
        self._ref_std = lab.std(axis=(0, 1)) + 1e-6
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._ref_mean is None:
            raise RuntimeError("Call fit() before transform().")
        lab = self._rgb_to_lab(image)
        mean = lab.mean(axis=(0, 1))
        std = lab.std(axis=(0, 1)) + 1e-6
        lab_norm = (lab - mean) / std * self._ref_std + self._ref_mean
        return self._lab_to_rgb(lab_norm)

    @staticmethod
    def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
        rgb = image.astype(float) / 255.0
        mask = rgb > 0.04045
        rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
        rgb[~mask] /= 12.92
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        xyz = rgb @ M.T
        xyz /= np.array([0.95047, 1.0, 1.08883])
        eps, kappa = 0.008856, 903.3
        f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        eps = 0.008856
        xyz = np.stack([
            np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / 903.3),
            np.where(L > kappa := 903.3 * eps, ((L + 16.0) / 116.0) ** 3, L / kappa),
            np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / 903.3),
        ], axis=-1)
        xyz *= np.array([0.95047, 1.0, 1.08883])
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ])
        rgb = xyz @ M_inv.T
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * rgb[mask] ** (1 / 2.4) - 0.055
        rgb[~mask] *= 12.92
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
