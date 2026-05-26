"""
evaluate.py
-----------
Scripts de validación y generación de reportes para todos los modelos.

Uso:
    python src/evaluation/evaluate.py --model svm
    python src/evaluation/evaluate.py --model rf
    python src/evaluation/evaluate.py --model cnn
    python src/evaluation/evaluate.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, classification_report

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocess import load_dataset, split_dataset
from src.utils.features import extract_features
from src.utils.helpers import (
    save_confusion_matrix,
    save_model_comparison,
    print_metrics,
)

CHECKPOINTS_DIR = "experiments/checkpoints"
CLASS_NAMES = ["alta", "baja", "media"]


def evaluate_sklearn(model_name: str):
    """Evalúa SVM o RF en el conjunto de test."""
    from src.models.ml_models import load_model

    model_path = f"{CHECKPOINTS_DIR}/{model_name}_model.pkl"
    if not Path(model_path).exists():
        print(f"[ERROR] No se encontró el modelo en {model_path}")
        print("        Ejecuta primero: python src/training/train.py --model", model_name)
        return None

    X_img, y, class_names = load_dataset("data/raw")
    _, _, X_te, _, _, y_te = split_dataset(X_img, y)
    X_te_feat = np.array([extract_features(img) for img in X_te])

    model = load_model(model_path)
    y_pred = model.predict(X_te_feat)

    print(f"\n{'='*50}")
    print(f" Evaluación: {model_name.upper()}")
    print(f"{'='*50}")
    print_metrics(y_te, y_pred, class_names)
    save_confusion_matrix(y_te, y_pred, class_names,
                          title=f"Confusion Matrix – {model_name.upper()}",
                          filename=f"cm_{model_name}.svg")

    return f1_score(y_te, y_pred, average="macro")


def evaluate_cnn():
    """Evalúa la CNN en el conjunto de test."""
    try:
        from tensorflow.keras.models import load_model as load_keras
    except ImportError:
        raise ImportError("Instala TensorFlow: pip install tensorflow")

    model_path = f"{CHECKPOINTS_DIR}/cnn_best.h5"
    if not Path(model_path).exists():
        print(f"[ERROR] No se encontró el modelo en {model_path}")
        print("        Ejecuta primero: python src/training/train.py --model cnn")
        return None

    X_img, y, class_names = load_dataset("data/raw")
    _, _, X_te, _, _, y_te = split_dataset(X_img, y)

    model = load_keras(model_path)
    y_prob = model.predict(X_te)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"\n{'='*50}")
    print(f" Evaluación: CNN")
    print(f"{'='*50}")
    print_metrics(y_te, y_pred, class_names)
    save_confusion_matrix(y_te, y_pred, class_names,
                          title="Confusion Matrix – CNN",
                          filename="cm_cnn.svg")

    return f1_score(y_te, y_pred, average="macro")


def compare_all():
    """Genera la comparativa final de todos los modelos."""
    results = {}
    for name in ("svm", "rf"):
        score = evaluate_sklearn(name)
        if score is not None:
            results[name.upper()] = score
    cnn_score = evaluate_cnn()
    if cnn_score is not None:
        results["CNN"] = cnn_score

    if results:
        save_model_comparison(list(results.keys()), list(results.values()))
        print("\n[RESUMEN] F1-macro por modelo:")
        for name, score in results.items():
            print(f"  {name:10s}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svm", "rf", "cnn"], default=None)
    parser.add_argument("--all", action="store_true", help="Evaluar todos los modelos")
    args = parser.parse_args()

    if args.all:
        compare_all()
    elif args.model in ("svm", "rf"):
        evaluate_sklearn(args.model)
    elif args.model == "cnn":
        evaluate_cnn()
    else:
        parser.print_help()
