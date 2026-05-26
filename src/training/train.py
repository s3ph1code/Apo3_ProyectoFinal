"""
train.py
--------
Scripts de entrenamiento para todos los modelos del proyecto.

Uso desde la raíz del repo:
    python src/training/train.py --model svm
    python src/training/train.py --model rf
    python src/training/train.py --model cnn
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Asegurar que src/ esté en el path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocess import load_dataset, split_dataset
from src.utils.features import extract_features
from src.models.ml_models import train_svm, train_random_forest, save_model
from src.models.cnn_model import build_cnn

DATA_DIR = "data/raw"
CHECKPOINTS_DIR = "experiments/checkpoints"
LOGS_DIR = "experiments/logs"


def train_ml_model(model_name: str):
    """Entrena SVM o Random Forest sobre características manuales."""
    print(f"\n{'='*50}")
    print(f" Entrenando {model_name.upper()}")
    print(f"{'='*50}\n")

    # 1. Cargar datos
    X_img, y, class_names = load_dataset(DATA_DIR)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_dataset(X_img, y)

    # 2. Extraer características
    print("[INFO] Extrayendo características...")
    X_tr_feat = np.array([extract_features(img) for img in X_tr])
    X_te_feat = np.array([extract_features(img) for img in X_te])

    # 3. Entrenar
    if model_name == "svm":
        model = train_svm(X_tr_feat, y_tr)
        save_path = f"{CHECKPOINTS_DIR}/svm_model.pkl"
    else:
        model = train_random_forest(X_tr_feat, y_tr)
        save_path = f"{CHECKPOINTS_DIR}/rf_model.pkl"

    # 4. Guardar
    save_model(model, save_path)

    # 5. Evaluación rápida en test
    from sklearn.metrics import f1_score
    y_pred = model.predict(X_te_feat)
    f1 = f1_score(y_te, y_pred, average="macro")
    print(f"\n[RESULT] F1-macro en test: {f1:.4f}")


def train_cnn_model():
    """Entrena la CNN sobre píxeles normalizados."""
    try:
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
    except ImportError:
        raise ImportError("Instala TensorFlow: pip install tensorflow")

    print(f"\n{'='*50}")
    print(f" Entrenando CNN")
    print(f"{'='*50}\n")

    # 1. Cargar datos
    X_img, y, class_names = load_dataset(DATA_DIR)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_dataset(X_img, y)

    # 2. Construir modelo
    model = build_cnn(num_classes=len(class_names))

    # 3. Callbacks
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(f"{CHECKPOINTS_DIR}/cnn_best.h5",
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        CSVLogger(f"{LOGS_DIR}/cnn_training.csv"),
    ]

    # 4. Entrenar
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Guardar curvas
    from src.utils.helpers import save_learning_curves
    save_learning_curves(history)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos de clasificación")
    parser.add_argument("--model", choices=["svm", "rf", "cnn"], required=True,
                        help="Modelo a entrenar: svm | rf | cnn")
    args = parser.parse_args()

    if args.model in ("svm", "rf"):
        train_ml_model(args.model)
    else:
        train_cnn_model()
