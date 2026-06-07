"""
app.py
------
Interfaz Streamlit para el proyecto APO3 ICESI 2026-1.

Funcionalidad:
- Selector de modelo: SVM (RBF), Random Forest, CNN.
- Entrada: upload de imagen o captura desde camara.
- Salida: prediccion de calidad (3 clases) + confianza + estimacion de tamano.
- Visualizacion: probabilidades por clase, imagen original y preprocesada.

Uso
---
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Permitir imports relativos al repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import load_and_preprocess_image, CLASS_NAMES  # noqa: E402
from src.utils.features import extract_features  # noqa: E402
from src.models.ml_models import SVMClassifier, RFClassifier  # noqa: E402
from src.utils.size_estimator import (  # noqa: E402
    compute_fruit_area,
    fit_size_thresholds,
    estimate_size,
    FRUITS,
)

# ---------------------------------------------------------------------------
# Config y constantes
# ---------------------------------------------------------------------------
CKPT_DIR = REPO_ROOT / "experiments" / "checkpoints"
THRESH_PATH = REPO_ROOT / "experiments" / "size_thresholds.json"

QUALITY_ES = {"Good": "Alta", "Regular": "Media", "Bad": "Baja"}
QUALITY_COLOR = {"Good": "🟢", "Regular": "🟡", "Bad": "🔴"}
FRUIT_ES = {
    "Apple": "Manzana", "Banana": "Banano", "Guava": "Guayaba",
    "Lime": "Lima", "Orange": "Naranja", "Pomegranate": "Granada",
}

st.set_page_config(
    page_title="Clasificador de Calidad de Frutas - ICESI",
    page_icon="🍎",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache de modelos y thresholds
# ---------------------------------------------------------------------------
@st.cache_resource
def load_svm():
    path = CKPT_DIR / "svm_rbf.joblib"
    if not path.exists():
        return None
    return SVMClassifier.load(path)


@st.cache_resource
def load_rf():
    path = CKPT_DIR / "random_forest.joblib"
    if not path.exists():
        return None
    return RFClassifier.load(path)


@st.cache_resource
def load_cnn():
    path = CKPT_DIR / "cnn_best.keras"
    if not path.exists():
        path = CKPT_DIR / "cnn_final.keras"
    if not path.exists():
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"No se pudo cargar la CNN: {e}")
        return None


@st.cache_resource
def load_size_thresholds(sample_per_fruit: int = 60):
    """
    Carga umbrales de tamano desde JSON o los computa la primera vez
    a partir del train manifest (muestra pequena por fruta para velocidad).
    """
    if THRESH_PATH.exists():
        with open(THRESH_PATH) as f:
            return {k: tuple(v) for k, v in json.load(f).items()}

    train_csv = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
    if not train_csv.exists():
        st.warning(
            "No se encontro data/processed/train_manifest.csv. "
            "Tamano usara umbrales por defecto."
        )
        return {f: (0.20, 0.45) for f in FRUITS}

    df = pd.read_csv(train_csv)
    paths, fruits = [], []
    rng = np.random.default_rng(42)
    for f in FRUITS:
        sub = df[df["fruit"] == f]
        if len(sub) == 0:
            continue
        sample = sub.sample(min(sample_per_fruit, len(sub)), random_state=42)
        for _, row in sample.iterrows():
            paths.append(REPO_ROOT / row["path"])
            fruits.append(f)

    thresholds = fit_size_thresholds(paths, fruits)
    THRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESH_PATH, "w") as f:
        json.dump({k: list(v) for k, v in thresholds.items()}, f, indent=2)
    return thresholds


# ---------------------------------------------------------------------------
# Helpers de inferencia
# ---------------------------------------------------------------------------
def pil_to_rgb_array(img_pil: Image.Image) -> np.ndarray:
    """Convierte PIL Image a ndarray RGB float32 [0,1] de 224x224."""
    img = np.array(img_pil.convert("RGB"))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def predict_with_svm_or_rf(model, img_rgb: np.ndarray):
    """Inferencia con modelo ML tradicional. Retorna (label, proba_dict)."""
    vec = extract_features(img_rgb).reshape(1, -1)
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.pipeline.classes_
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    return label, proba_dict


def predict_with_cnn(model, img_rgb: np.ndarray):
    """Inferencia con CNN. Retorna (label, proba_dict)."""
    proba = model.predict(img_rgb[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(proba))
    label = CLASS_NAMES[idx]
    proba_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, proba)}
    return label, proba_dict


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.title("🍎 Clasificador de Calidad de Frutas")
    st.markdown(
        "**Proyecto APO3 ICESI 2026-1** — Sistema universal de clasificacion "
        "de calidad (Alta / Media / Baja) para 6 frutas: manzana, banano, "
        "guayaba, lima, naranja y granada."
    )

    # --- Sidebar: configuracion ---
    st.sidebar.header("Configuracion")
    model_choice = st.sidebar.selectbox(
        "Modelo de clasificacion",
        ["SVM (RBF)", "Random Forest", "CNN"],
        index=0,
        help="SVM gano en test (F1-macro=0.982); CNN es la opcion de deep learning."
    )
    fruit_choice = st.sidebar.selectbox(
        "Tipo de fruta (para estimar tamano)",
        FRUITS,
        format_func=lambda f: f"{FRUIT_ES[f]} ({f})",
        help="El tamano se calibra por especie con percentiles del area frutal.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Metricas en test (Fase 4):**")
    st.sidebar.markdown("- SVM (RBF): F1-macro **0.982**")
    st.sidebar.markdown("- Random Forest: F1-macro **0.969**")
    st.sidebar.markdown("- CNN (3 conv): F1-macro **0.955**")

    # --- Tabs: upload vs camara ---
    tab_upload, tab_cam = st.tabs(["📁 Subir imagen", "📷 Usar camara"])

    img_pil = None
    with tab_upload:
        f = st.file_uploader(
            "Selecciona una imagen (JPG / PNG)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if f is not None:
            img_pil = Image.open(f)

    with tab_cam:
        cam = st.camera_input("Toma una foto de la fruta")
        if cam is not None:
            img_pil = Image.open(cam)

    if img_pil is None:
        st.info(
            "👈 Sube una imagen o toma una foto. Asegurate de que la fruta "
            "este sobre un fondo simple (idealmente blanco)."
        )
        st.stop()

    # --- Preprocesar imagen ---
    img_rgb = pil_to_rgb_array(img_pil)

    # --- Cargar modelo elegido ---
    with st.spinner(f"Cargando modelo {model_choice}..."):
        if model_choice == "SVM (RBF)":
            model = load_svm()
        elif model_choice == "Random Forest":
            model = load_rf()
        else:
            model = load_cnn()

    if model is None:
        st.error(
            f"No se encontro el modelo {model_choice} en {CKPT_DIR}. "
            "Asegurate de haber ejecutado los notebooks 03 y 04."
        )
        st.stop()

    # --- Inferencia ---
    with st.spinner("Clasificando..."):
        if model_choice == "CNN":
            label, proba = predict_with_cnn(model, img_rgb)
        else:
            label, proba = predict_with_svm_or_rf(model, img_rgb)

    # --- Estimacion de tamano ---
    thresholds = load_size_thresholds()
    try:
        size_label = estimate_size(img_rgb, fruit_choice, thresholds)
        area = compute_fruit_area(img_rgb)
    except Exception:
        size_label = "n/a"
        area = None

    # --- Layout de resultados ---
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(img_pil, caption="Imagen analizada", use_container_width=True)

    with col_pred:
        st.markdown("### Resultado")
        st.markdown(
            f"#### {QUALITY_COLOR[label]} Calidad: **{QUALITY_ES[label]}** "
            f"(`{label}`)"
        )
        confidence = proba[label] * 100
        st.metric("Confianza", f"{confidence:.1f}%")

        st.markdown("#### Probabilidades por clase")
        proba_df = pd.DataFrame({
            "Clase": [f"{QUALITY_COLOR[c]} {QUALITY_ES[c]} ({c})" for c in CLASS_NAMES],
            "Probabilidad (%)": [proba[c] * 100 for c in CLASS_NAMES],
        })
        st.dataframe(proba_df, hide_index=True, use_container_width=True)
        st.bar_chart(proba_df.set_index("Clase"), height=200)

        st.markdown("#### Tamano estimado")
        size_es = {"pequeno": "Pequeno", "mediano": "Mediano",
                   "grande": "Grande"}.get(size_label, size_label)
        if area is not None:
            st.markdown(f"📏 **{size_es}** (area frutal: {area * 100:.1f}% del frame)")
        else:
            st.markdown("📏 No disponible")

        st.markdown("---")
        st.caption(
            f"Modelo: {model_choice} | Fruta declarada: {FRUIT_ES[fruit_choice]}"
        )

    # --- Detalle expandible ---
    with st.expander("Detalles tecnicos"):
        st.markdown(
            f"- **Modelo:** `{model_choice}`\n"
            f"- **Etiqueta predicha:** `{label}` -> *{QUALITY_ES[label]}*\n"
            f"- **Tamano:** `{size_label}` (umbrales {fruit_choice}: q33={thresholds[fruit_choice][0]:.3f}, q66={thresholds[fruit_choice][1]:.3f})\n"
            f"- **Pipeline:** imagen -> resize 224x224 -> normalizar [0,1] -> "
            f"{'extract_features (141-D)' if model_choice != 'CNN' else 'CNN forward'} -> softmax\n"
        )


if __name__ == "__main__":
    main()
