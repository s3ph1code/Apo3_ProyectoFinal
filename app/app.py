"""
app.py
------
Interfaz Streamlit para clasificación de calidad de manzanas en tiempo real.

Uso:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Asegurar que src/ esté en el path
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing import load_image
from src.models import load_sklearn_model

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Calidad de Manzanas",
    page_icon="🍎",
    layout="centered",
)

st.title("🍎 Clasificador de Calidad de Manzanas")
st.markdown(
    "Sube una foto de una manzana (sobre fondo simple) y el sistema "
    "determinará su **clase de calidad** y **tamaño estimado**."
)

# ---------------------------------------------------------------------------
# Clases y modelos disponibles
# ---------------------------------------------------------------------------
CLASS_NAMES = ["alta", "baja", "media"]   # orden alfabético = orden de etiquetas
SIZE_LABELS = ["pequeño", "mediano", "grande"]

MODEL_PATHS = {
    "SVM": "models/svm_model.pkl",
    "Random Forest": "models/rf_model.pkl",
}

# ---------------------------------------------------------------------------
# Sidebar – selección de modelo
# ---------------------------------------------------------------------------
st.sidebar.header("Configuración")
model_choice = st.sidebar.selectbox("Modelo de clasificación", list(MODEL_PATHS.keys()))

@st.cache_resource
def load_model(path: str):
    """Carga el modelo una sola vez y lo mantiene en caché."""
    try:
        return load_sklearn_model(path)
    except FileNotFoundError:
        return None

model = load_model(MODEL_PATHS[model_choice])

if model is None:
    st.warning(
        f"⚠️ El modelo **{model_choice}** no está entrenado aún. "
        "Ejecuta primero el notebook `03_modelado_ml.ipynb`."
    )

# ---------------------------------------------------------------------------
# Carga de imagen
# ---------------------------------------------------------------------------
st.subheader("Cargar imagen")
upload_tab, camera_tab = st.tabs(["📁 Subir archivo", "📷 Cámara"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Selecciona una imagen", type=["jpg", "jpeg", "png"]
    )

with camera_tab:
    camera_image = st.camera_input("Captura con tu cámara")

# Prioridad: cámara > archivo subido
raw_image = camera_image or uploaded_file

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
if raw_image is not None:
    pil_img = Image.open(raw_image).convert("RGB")
    st.image(pil_img, caption="Imagen cargada", use_container_width=True)

    if model is not None:
        # Preprocesamiento
        img_array = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

        # Extracción de características para ML
        from src.features import extract_features
        features = extract_features(img_array).reshape(1, -1)

        # Predicción
        pred_label = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        quality_class = CLASS_NAMES[pred_label]
        confidence = pred_proba[pred_label] * 100

        # Estimación de tamaño (heurística por área de píxeles brillantes)
        gray = img_array.mean(axis=2)
        bright_ratio = (gray > 0.5).mean()
        if bright_ratio < 0.25:
            size_label = SIZE_LABELS[0]
        elif bright_ratio < 0.50:
            size_label = SIZE_LABELS[1]
        else:
            size_label = SIZE_LABELS[2]

        # Resultados
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Clase de calidad", quality_class.upper())
        col2.metric("Confianza", f"{confidence:.1f}%")
        col3.metric("Tamaño estimado", size_label)

        # Probabilidades por clase
        st.subheader("Probabilidades por clase")
        for name, prob in zip(CLASS_NAMES, pred_proba):
            st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

        # Advertencia ética
        st.info(
            "ℹ️ **Nota:** Esta predicción es orientativa. "
            "El sistema fue entrenado con un conjunto limitado de imágenes "
            "y puede cometer errores. No reemplaza la inspección humana."
        )
