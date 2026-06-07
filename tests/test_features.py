"""test_features.py

Autor: Kevin Cifuentes Quintero (cifuentesclud@gmail.com)
Universidad ICESI - Algoritmos y Programacion III, 2026-1.
"""
"""Tests basicos para src/utils/features.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.features import (  # noqa: E402
    FEATURE_DIM,
    HSV_BINS_PER_CHANNEL,
    HU_DIMS,
    LBP_BINS,
    CHANNEL_STAT_DIMS,
    channel_stats,
    extract_features,
    feature_names,
    hsv_histogram,
    hu_moments,
    lbp_histogram,
)

IMG_SIZE = (224, 224, 3)


@pytest.fixture
def random_image():
    rng = np.random.default_rng(0)
    return rng.random(IMG_SIZE, dtype=np.float32)


@pytest.fixture
def black_image():
    return np.zeros(IMG_SIZE, dtype=np.float32)


# ---------------------------------------------------------------------------
# Bloques individuales
# ---------------------------------------------------------------------------
def test_hsv_histogram_shape_and_norm(random_image):
    h = hsv_histogram(random_image)
    assert h.shape == (HSV_BINS_PER_CHANNEL * 3,)
    # Cada canal (32 bins) debe sumar 1.0
    for ch in range(3):
        s = h[ch * HSV_BINS_PER_CHANNEL : (ch + 1) * HSV_BINS_PER_CHANNEL].sum()
        assert abs(s - 1.0) < 1e-5


def test_hu_moments_shape(random_image):
    hu = hu_moments(random_image)
    assert hu.shape == (HU_DIMS,)


def test_lbp_histogram_shape_and_norm(random_image):
    h = lbp_histogram(random_image)
    assert h.shape == (LBP_BINS,)
    assert abs(h.sum() - 1.0) < 1e-5


def test_channel_stats_shape_and_range(random_image):
    cs = channel_stats(random_image)
    assert cs.shape == (CHANNEL_STAT_DIMS,)
    assert (cs >= 0).all() and (cs <= 1).all()


# ---------------------------------------------------------------------------
# Vector completo
# ---------------------------------------------------------------------------
def test_extract_features_dimensions(random_image):
    v = extract_features(random_image)
    assert v.shape == (FEATURE_DIM,)
    assert v.shape == (141,)
    assert v.dtype == np.float32


def test_extract_features_deterministic(random_image):
    v1 = extract_features(random_image)
    v2 = extract_features(random_image)
    np.testing.assert_array_almost_equal(v1, v2, decimal=5)


def test_extract_features_finite(random_image, black_image):
    for img in (random_image, black_image):
        v = extract_features(img)
        assert np.isfinite(v).all(), "El vector tiene NaN o Inf"


def test_extract_features_rejects_wrong_shape():
    with pytest.raises(ValueError):
        extract_features(np.zeros((224, 224), dtype=np.float32))
    with pytest.raises(ValueError):
        extract_features(np.zeros((224, 224, 4), dtype=np.float32))


def test_extract_features_rejects_non_normalized():
    img_uint = (np.random.rand(*IMG_SIZE) * 255).astype(np.float32)
    with pytest.raises(ValueError):
        extract_features(img_uint)


# ---------------------------------------------------------------------------
# Nombres de features (para feature importance de RF)
# ---------------------------------------------------------------------------
def test_feature_names_length():
    names = feature_names()
    assert len(names) == FEATURE_DIM


def test_feature_names_unique():
    names = feature_names()
    assert len(set(names)) == len(names), "Hay nombres duplicados"
