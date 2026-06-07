"""
Tests basicos para src/data/preprocess.py

Cubre:
- Esquema del manifest cargado.
- Filtrado de imagenes pequenas.
- Reproducibilidad y propiedades del split estratificado (sum, presencia
  de las 3 clases y las 6 frutas en cada particion, proporciones 70/15/15).
- Augmentation no rompe el shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to sys.path so we can import src.*
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (  # noqa: E402
    CLASS_NAMES,
    IMG_SIZE,
    MIN_SIDE,
    SPLIT_SIZES,
    filter_small_images,
    get_train_augmentation,
    get_val_augmentation,
    load_manifest,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def manifest_df():
    """Carga el manifest real del proyecto una sola vez por modulo."""
    return load_manifest()


@pytest.fixture(scope="module")
def clean_df(manifest_df):
    """Manifest filtrado por tamano minimo."""
    return filter_small_images(manifest_df, min_side=MIN_SIDE)


@pytest.fixture(scope="module")
def splits(clean_df):
    """Splits estratificados con seed fija."""
    return stratified_split(clean_df)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def test_manifest_has_required_columns(manifest_df):
    expected = {"path", "fruit", "quality", "class", "source", "width", "height"}
    assert expected.issubset(set(manifest_df.columns))


def test_manifest_has_three_qualities(manifest_df):
    assert set(manifest_df["quality"].unique()) == set(CLASS_NAMES)


def test_manifest_has_six_fruits(manifest_df):
    assert manifest_df["fruit"].nunique() == 6


def test_manifest_has_18_class_combinations(manifest_df):
    assert manifest_df["class"].nunique() == 18


# ---------------------------------------------------------------------------
# Filtrado
# ---------------------------------------------------------------------------
def test_filter_drops_small_images(clean_df):
    assert (clean_df["width"] >= MIN_SIDE).all()
    assert (clean_df["height"] >= MIN_SIDE).all()


def test_filter_drops_at_most_25_percent(manifest_df, clean_df):
    # El EDA revelo que ~18% de las imagenes son patches/thumbnails con lado < 64 px
    # (otros grupos del curso cortaron imagenes en pedacitos antes de subir).
    # Descartar mas del 25% indicaria un dataset degradado o threshold mal elegido.
    dropped_frac = 1 - len(clean_df) / len(manifest_df)
    assert dropped_frac <= 0.25, f'Filtrado descarto {100*dropped_frac:.1f}% del dataset'
    assert dropped_frac >= 0.0, 'Filtrado no puede agregar filas'




# ---------------------------------------------------------------------------
# Split estratificado
# ---------------------------------------------------------------------------
def test_split_sums_to_total(clean_df, splits):
    tr, va, te = splits
    assert len(tr) + len(va) + len(te) == len(clean_df)


def test_split_no_overlap(splits):
    tr, va, te = splits
    paths_tr = set(tr["path"])
    paths_va = set(va["path"])
    paths_te = set(te["path"])
    assert paths_tr.isdisjoint(paths_va)
    assert paths_tr.isdisjoint(paths_te)
    assert paths_va.isdisjoint(paths_te)


@pytest.mark.parametrize("split_idx,name", [(0, "train"), (1, "val"), (2, "test")])
def test_split_has_all_quality_classes(splits, split_idx, name):
    split = splits[split_idx]
    assert set(split["quality"].unique()) == set(CLASS_NAMES), (
        f"En split '{name}' faltan clases de quality"
    )


@pytest.mark.parametrize("split_idx,name", [(0, "train"), (1, "val"), (2, "test")])
def test_split_has_all_fruits(splits, split_idx, name):
    split = splits[split_idx]
    assert split["fruit"].nunique() == 6, f"En split '{name}' faltan frutas"


def test_split_proportions(clean_df, splits):
    tr, va, te = splits
    n = len(clean_df)
    train_size, val_size, test_size = SPLIT_SIZES
    # Tolerancia de +-1% por redondeos del estratificador.
    assert abs(len(tr) / n - train_size) < 0.01
    assert abs(len(va) / n - val_size) < 0.01
    assert abs(len(te) / n - test_size) < 0.01


def test_split_reproducible(clean_df):
    tr1, va1, te1 = stratified_split(clean_df)
    tr2, va2, te2 = stratified_split(clean_df)
    assert tr1["path"].tolist() == tr2["path"].tolist()
    assert va1["path"].tolist() == va2["path"].tolist()
    assert te1["path"].tolist() == te2["path"].tolist()


def test_split_preserves_quality_distribution(clean_df, splits):
    """Stratificacion conserva proporciones de quality en cada split."""
    global_dist = clean_df["quality"].value_counts(normalize=True)
    for split in splits:
        split_dist = split["quality"].value_counts(normalize=True)
        for cls in CLASS_NAMES:
            # Tolerancia de +-1% (estratificacion es por las 18 combinaciones)
            assert abs(split_dist[cls] - global_dist[cls]) < 0.01


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def test_train_augmentation_preserves_shape():
    aug = get_train_augmentation()
    img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
    out = aug(image=img)["image"]
    assert out.shape == (*IMG_SIZE, 3)


def test_val_augmentation_is_identity():
    aug = get_val_augmentation()
    img = (np.random.rand(*IMG_SIZE, 3) * 255).astype(np.float32)
    out = aug(image=img)["image"]
    # Con Compose([]) la imagen debe quedar igual.
    np.testing.assert_array_equal(out, img)
