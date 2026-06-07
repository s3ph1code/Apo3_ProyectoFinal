"""
features.py
-----------
Extraccion de features handcrafted para los modelos de ML tradicional
(SVM y Random Forest) del proyecto APO3 ICESI 2026-1.

Vector resultante de 141 dimensiones por imagen:

    | Bloque              | Dims | Razon de eleccion                                  |
    |---------------------|-----:|----------------------------------------------------|
    | HSV histogram       |   96 | 32 bins x 3 canales (H, S, V). Captura color y     |
    |                     |      | distribucion de luminosidad sin depender de pose.  |
    | Momentos de Hu      |    7 | Invariantes a traslacion, rotacion y escala.       |
    |                     |      | Capturan la forma del contorno frutal.             |
    | LBP histogram       |   32 | Local Binary Patterns: textura local (manchas,     |
    |                     |      | suavidad de piel, podredumbre).                    |
    | Channel statistics  |    6 | mu y sigma de R, G, B en [0, 1].                   |
    |                     |      | Caracteriza tono dominante y dispersion.           |
    |                     |      |                                                    |
    | TOTAL               |  141 |                                                    |

Estas features se calculan sobre la imagen YA preprocesada (224x224, RGB en [0,1]).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuracion del vector
# ---------------------------------------------------------------------------
HSV_BINS_PER_CHANNEL: int = 32          # 32 + 32 + 32 = 96
HU_DIMS: int = 7
LBP_BINS: int = 32
CHANNEL_STAT_DIMS: int = 6              # mu(R), mu(G), mu(B), sigma(R), sigma(G), sigma(B)

FEATURE_DIM: int = (
    HSV_BINS_PER_CHANNEL * 3 + HU_DIMS + LBP_BINS + CHANNEL_STAT_DIMS
)
assert FEATURE_DIM == 141, f"FEATURE_DIM debe ser 141, es {FEATURE_DIM}"

# Parametros LBP
LBP_RADIUS: int = 1
LBP_N_POINTS: int = 8


# ---------------------------------------------------------------------------
# Bloques individuales
# ---------------------------------------------------------------------------
def hsv_histogram(img_rgb: np.ndarray, bins_per_channel: int = HSV_BINS_PER_CHANNEL) -> np.ndarray:
    """
    Histograma HSV concatenado y normalizado.

    Parametros
    ----------
    img_rgb : (H, W, 3) float32 en [0, 1] (salida de preprocess.load_and_preprocess_image)
    bins_per_channel : numero de bins por canal H, S, V.

    Retorna
    -------
    ndarray de shape (bins_per_channel * 3,) sumando 1.0 por canal (cada bloque
    es una densidad de probabilidad sobre su canal).
    """
    import cv2  # importacion perezosa

    # Convertir a uint8 BGR para cv2.cvtColor
    img_bgr = (img_rgb[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    ranges = [(0, 180), (0, 256), (0, 256)]
    hists = []
    for ch in range(3):
        h, _ = np.histogram(hsv[..., ch], bins=bins_per_channel, range=ranges[ch])
        h = h.astype(np.float32)
        s = h.sum()
        if s > 0:
            h = h / s
        hists.append(h)
    return np.concatenate(hists)


def hu_moments(img_rgb: np.ndarray) -> np.ndarray:
    """
    7 momentos de Hu invariantes en escala log-firmada.

    Se aplica `sign(h) * log10(|h| + eps)` para que numericamente sean comparables
    (los Hu crudos varian en muchos ordenes de magnitud).

    Parametros
    ----------
    img_rgb : (H, W, 3) float32 en [0, 1].
    """
    import cv2

    img_gray = (
        (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]) * 255
    ).astype(np.uint8)
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    eps = 1e-12
    return -np.sign(hu) * np.log10(np.abs(hu) + eps)


def lbp_histogram(img_rgb: np.ndarray, n_bins: int = LBP_BINS) -> np.ndarray:
    """
    Histograma de Local Binary Patterns sobre la luminancia.

    LBP captura textura local: el patron de cada pixel comparado con sus vecinos.
    Sensible a manchas, podredumbre, irregularidades de superficie.

    Implementacion: scikit-image `local_binary_pattern` con metodo 'uniform'.
    """
    from skimage.feature import local_binary_pattern

    img_gray = (
        (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]) * 255
    ).astype(np.uint8)
    lbp = local_binary_pattern(img_gray, P=LBP_N_POINTS, R=LBP_RADIUS, method="uniform")
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    h = h.astype(np.float32)
    s = h.sum()
    if s > 0:
        h = h / s
    return h


def channel_stats(img_rgb: np.ndarray) -> np.ndarray:
    """
    Estadisticas mu y sigma de cada canal RGB en [0, 1].

    Retorna ndarray de shape (6,): [mu_R, mu_G, mu_B, sigma_R, sigma_G, sigma_B].
    """
    means = img_rgb.reshape(-1, 3).mean(axis=0)
    stds = img_rgb.reshape(-1, 3).std(axis=0)
    return np.concatenate([means, stds]).astype(np.float32)


# ---------------------------------------------------------------------------
# Vector completo
# ---------------------------------------------------------------------------
def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae el vector 141-D handcrafted desde una imagen preprocesada.

    Parametros
    ----------
    img_rgb : (H, W, 3) float32 en [0, 1].

    Retorna
    -------
    ndarray shape (141,) float32.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"Se esperaba (H, W, 3); recibido shape={img_rgb.shape}"
        )
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32)
    if img_rgb.max() > 1.5:
        # Permitimos un poco de margen pero 0-255 indica falta de normalizacion.
        raise ValueError(
            "extract_features espera imagen normalizada en [0, 1]; "
            f"se recibio max={img_rgb.max():.2f}"
        )

    blocks = [
        hsv_histogram(img_rgb),       # 96
        hu_moments(img_rgb),          # 7
        lbp_histogram(img_rgb),       # 32
        channel_stats(img_rgb),       # 6
    ]
    vec = np.concatenate(blocks).astype(np.float32)
    if vec.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"Bug: vector resultante {vec.shape} != ({FEATURE_DIM},)"
        )
    return vec


def feature_names() -> list[str]:
    """Lista de nombres de las 141 features (utiles para feature importance del RF)."""
    names = []
    for ch in ("H", "S", "V"):
        for i in range(HSV_BINS_PER_CHANNEL):
            names.append(f"hsv_{ch}_{i:02d}")
    for i in range(HU_DIMS):
        names.append(f"hu_{i+1}")
    for i in range(LBP_BINS):
        names.append(f"lbp_{i:02d}")
    for ch in ("R", "G", "B"):
        names.append(f"mu_{ch}")
    for ch in ("R", "G", "B"):
        names.append(f"sigma_{ch}")
    assert len(names) == FEATURE_DIM
    return names
