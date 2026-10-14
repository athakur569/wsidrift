"""Tests for stain normalizers and package-wide importability."""

import importlib
import pkgutil

import numpy as np
import pytest

import wsidrift
from wsidrift.norm.macenko import MacenkoNormalizer
from wsidrift.norm.reinhard import ReinhardNormalizer


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


H, E = _unit([0.65, 0.70, 0.29]), _unit([0.07, 0.99, 0.11])  # Ruifrok & Johnston H&E vectors


def _he_tile(seed, h=H, e=E, scale=1.0):
    """Synthetic two-stain tile that follows the Beer-Lambert model."""
    r = np.random.default_rng(seed)
    n = 128 * 128
    C = np.column_stack([r.gamma(2, 0.25, n) * (r.random(n) < 0.4), r.gamma(3, 0.15, n)]) * scale
    od = C @ np.vstack([h, e])
    return np.clip(255 * np.exp(-od), 1, 255).reshape(128, 128, 3).astype(np.uint8)


@pytest.mark.parametrize("name", [m.name for m in pkgutil.walk_packages(wsidrift.__path__, "wsidrift.")])
def test_every_module_imports(name):
    importlib.import_module(name)


class TestMacenko:
    def test_recovers_stain_vectors(self):
        m = MacenkoNormalizer().fit(_he_tile(1))
        assert abs(m._he_ref[0] @ H) > 0.98
        assert abs(m._he_ref[1] @ E) > 0.98

    def test_self_normalisation_is_near_identity(self):
        ref = _he_tile(1)
        out = MacenkoNormalizer().fit(ref).transform(ref)
        assert np.abs(out.astype(int) - ref).mean() < 2.0

    def test_moves_other_stain_toward_reference(self):
        ref = _he_tile(1)
        src = _he_tile(2, _unit([0.55, 0.80, 0.25]), _unit([0.15, 0.95, 0.25]), scale=1.3)
        out = MacenkoNormalizer().fit(ref).transform(src)
        mean = lambda a: a.reshape(-1, 3).mean(0)
        assert np.abs(mean(out) - mean(ref)).sum() < np.abs(mean(src) - mean(ref)).sum()


class TestReinhard:
    def test_lab_round_trip(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        rt = ReinhardNormalizer._lab_to_rgb(ReinhardNormalizer._rgb_to_lab(img))
        assert np.abs(rt.astype(int) - img).max() <= 1

    def test_matches_reference_lab_statistics(self):
        ref, src = _he_tile(1), _he_tile(2, scale=1.4)
        out = ReinhardNormalizer().fit(ref).transform(src)
        lab = ReinhardNormalizer._rgb_to_lab
        assert np.abs(lab(out).mean((0, 1)) - lab(ref).mean((0, 1))).max() < 1.0
