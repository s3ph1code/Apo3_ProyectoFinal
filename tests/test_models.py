"""
test_models.py
--------------
Pruebas unitarias básicas para verificar que los módulos funcionan correctamente.

Uso:
    pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocess import load_image, split_dataset, get_augmentation
from src.utils.features import (
    color_histogram_hsv, hu_moments, lbp_histogram,
    channel_stats, extract_features,
)
from src.models.cnn_model import build_cnn


# ---------------------------------------------------------------------------
# Fixture: imagen sintética 224x224 RGB
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_image():
    """Imagen sintética float32 [0,1] de forma (224, 224, 3)."""
    rng = np.random.default_rng(42)
    return rng.random((224, 224, 3)).astype(np.float32)


@pytest.fixture
def dummy_dataset():
    """Dataset sintético: 30 imágenes, 3 clases."""
    rng = np.random.default_rng(42)
    X = rng.random((30, 224, 224, 3)).astype(np.float32)
    y = np.repeat([0, 1, 2], 10)
    return X, y


# ---------------------------------------------------------------------------
# Tests de preprocesamiento
# ---------------------------------------------------------------------------

def test_image_shape(dummy_image):
    assert dummy_image.shape == (224, 224, 3)
    assert dummy_image.dtype == np.float32


def test_image_range(dummy_image):
    assert dummy_image.min() >= 0.0
    assert dummy_image.max() <= 1.0


def test_split_dataset_sizes(dummy_dataset):
    X, y = dummy_dataset
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_dataset(X, y)
    assert len(X_tr) + len(X_va) + len(X_te) == len(X)


def test_augmentation_output_shape(dummy_image):
    aug = get_augmentation(training=True)
    img8 = (dummy_image * 255).astype(np.uint8)
    result = aug(image=img8)["image"]
    assert result.shape == (224, 224, 3)


# ---------------------------------------------------------------------------
# Tests de características
# ---------------------------------------------------------------------------

def test_hsv_histogram_length(dummy_image):
    feat = color_histogram_hsv(dummy_image, bins=32)
    assert feat.shape == (96,)


def test_hsv_histogram_normalized(dummy_image):
    feat = color_histogram_hsv(dummy_image, bins=32)
    assert abs(feat.sum() - 1.0) < 1e-5


def test_hu_moments_length(dummy_image):
    feat = hu_moments(dummy_image)
    assert feat.shape == (7,)


def test_lbp_histogram_length(dummy_image):
    feat = lbp_histogram(dummy_image, bins=32)
    assert feat.shape == (32,)


def test_channel_stats_length(dummy_image):
    feat = channel_stats(dummy_image)
    assert feat.shape == (6,)


def test_extract_features_total_length(dummy_image):
    feat = extract_features(dummy_image)
    assert feat.shape == (141,), f"Esperado 141, obtenido {feat.shape[0]}"


# ---------------------------------------------------------------------------
# Tests del modelo CNN
# ---------------------------------------------------------------------------

def test_cnn_output_shape():
    model = build_cnn(num_classes=3, input_shape=(224, 224, 3))
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 3)


def test_cnn_softmax_sum():
    model = build_cnn(num_classes=3, input_shape=(224, 224, 3))
    dummy_input = np.random.rand(4, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    sums = output.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)
