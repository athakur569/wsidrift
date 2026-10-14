"""Macenko stain normalization via SVD stain vector estimation."""

from __future__ import annotations

import numpy as np


class MacenkoNormalizer:
    """Normalize H&E patches to a reference stain using Macenko's SVD method.

    Args:
        beta:    OD threshold for background removal.
        alpha:   Percentile for robust min/max scaling.
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0) -> None:
        self.beta = beta
        self.alpha = alpha
        self._he_ref: np.ndarray | None = None
        self._max_ref: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> "MacenkoNormalizer":
        """Estimate stain vectors from a reference image (H, W, 3) uint8."""
        he = self._get_stain_matrix(reference)
        self._he_ref = he
        self._max_ref = np.percentile(self._get_concentrations(reference, he), 99, axis=0)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._he_ref is None:
            raise RuntimeError("Call fit() before transform().")
        he = self._get_stain_matrix(image)
        conc = self._get_concentrations(image, he)
        conc_norm = conc / (np.percentile(conc, 99, axis=0) + 1e-6) * self._max_ref
        od_norm = np.dot(conc_norm, self._he_ref)
        rgb = np.exp(-od_norm).reshape(image.shape)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(float) + 1e-6
        return -np.log(img / 255.0)

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        od = self._rgb_to_od(image)
        od = od[np.all(od > self.beta, axis=1)]
        _, _, vt = np.linalg.svd(od, full_matrices=False)
        plane = vt[:2].T
        plane *= np.where(plane.sum(axis=0) < 0, -1.0, 1.0)  # SVD sign is arbitrary
        # Angles of the tissue pixels within the plane; their extremes give the stain vectors.
        proj = od @ plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        phi_min = np.percentile(angles, self.alpha)
        phi_max = np.percentile(angles, 100 - self.alpha)
        v1 = np.array([np.cos(phi_min), np.sin(phi_min)])
        v2 = np.array([np.cos(phi_max), np.sin(phi_max)])
        he = np.array([plane @ v1, plane @ v2])
        he /= np.linalg.norm(he, axis=1, keepdims=True) + 1e-6
        if he[0, 0] < he[1, 0]:
            he = he[[1, 0]]
        return he

    def _get_concentrations(self, image: np.ndarray, he: np.ndarray) -> np.ndarray:
        od = self._rgb_to_od(image).T
        return np.linalg.lstsq(he.T, od, rcond=None)[0].T
