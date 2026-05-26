"""
helpers.py
----------
Funciones auxiliares: métricas, visualizaciones vectoriales (SVG).

Todas las gráficas se guardan en SVG como exige la asignatura.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

RESULTS_DIR = Path("experiments/results")


def print_metrics(y_true, y_pred, class_names):
    """Imprime classification report completo (precision, recall, F1 por clase)."""
    print(classification_report(y_true, y_pred, target_names=class_names))


def save_confusion_matrix(y_true, y_pred, class_names,
                          title="Confusion Matrix", filename="confusion_matrix.svg"):
    """Genera y guarda la matriz de confusión en SVG."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, format="svg")
    plt.close()
    print(f"[FIG] {RESULTS_DIR / filename}")


def save_learning_curves(history, filename="learning_curves.svg"):
    """Grafica accuracy y loss train vs val de la CNN."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(h["accuracy"], label="Train"); ax1.plot(h["val_accuracy"], label="Val")
    ax1.set(xlabel="Épocas", ylabel="Accuracy", title="Curva de accuracy"); ax1.legend()
    ax2.plot(h["loss"], label="Train"); ax2.plot(h["val_loss"], label="Val")
    ax2.set(xlabel="Épocas", ylabel="Loss", title="Curva de pérdida"); ax2.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, format="svg")
    plt.close()
    print(f"[FIG] {RESULTS_DIR / filename}")


def save_model_comparison(model_names, f1_scores, filename="model_comparison.svg"):
    """Gráfico de barras comparando F1-macro de cada modelo."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(model_names, f1_scores, color=colors[:len(model_names)])
    ax.set_ylim(0, 1.05)
    ax.set(ylabel="F1-score (macro)", title="Comparativa de modelos – Test set")
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, format="svg")
    plt.close()
    print(f"[FIG] {RESULTS_DIR / filename}")


def save_class_distribution(y, class_names, filename="class_distribution.svg"):
    """Grafica conteo de imágenes por clase (EDA)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = [int((y == i).sum()) for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, counts, color="#4C72B0")
    ax.set(xlabel="Clase", ylabel="# Imágenes", title="Distribución de clases")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, format="svg")
    plt.close()
    print(f"[FIG] {RESULTS_DIR / filename}")
