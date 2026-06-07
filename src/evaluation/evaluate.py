"""
evaluate.py
-----------
Utilidades de evaluacion y analisis de sesgo para el clasificador de calidad
de 3 clases. Funciones:

- classification_report_dict: metricas por clase y macro.
- plot_confusion_matrix: matriz 3x3 del target.
- plot_bias_matrix: matriz 6x6 (fruta x prediccion correcta?) para analisis de sesgo.
- accuracy_per_fruit: accuracy condicional por especie (para responder
  "el modelo es uniformemente bueno entre frutas?").
- compare_models: tabla comparativa SVM vs RF vs CNN en F1-macro.

Todas las figuras se guardan como SVG vectorial (requisito IEEE).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASS_NAMES = ["Good", "Regular", "Bad"]
FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Diccionario con accuracy, precision/recall/F1 por clase y macro."""
    rep = classification_report(
        y_true, y_pred, labels=CLASS_NAMES, output_dict=True, zero_division=0
    )
    rep["_accuracy"] = accuracy_score(y_true, y_pred)
    rep["_f1_macro"] = f1_score(y_true, y_pred, average="macro", labels=CLASS_NAMES)
    return rep


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de confusion (target = 3 clases)",
    out_path: str | Path | None = None,
):
    """Matriz 3x3 normalizada por fila, con anotaciones."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES, normalize="true")
    counts = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    annot = np.array([[f"{cm[i,j]*100:.1f}%\n({counts[i,j]})"
                       for j in range(len(CLASS_NAMES))]
                      for i in range(len(CLASS_NAMES))])
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        vmin=0, vmax=1, cbar_kws={"label": "Recall por clase"},
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdadero")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
    return fig


def accuracy_per_fruit(
    y_true: np.ndarray, y_pred: np.ndarray, fruits: np.ndarray
) -> pd.DataFrame:
    """
    Accuracy condicional por especie: Acc(f) = #aciertos|f / #total|f.

    Una disparidad alta indica que el modelo aprende calidad real con sesgo
    por especie (por ejemplo, mejor en Pomegranate por su mayor representacion).
    """
    df = pd.DataFrame({"fruit": fruits, "y_true": y_true, "y_pred": y_pred})
    rows = []
    for f in FRUITS:
        sub = df[df["fruit"] == f]
        if len(sub) == 0:
            continue
        acc = (sub["y_true"] == sub["y_pred"]).mean()
        f1 = f1_score(sub["y_true"], sub["y_pred"],
                      average="macro", labels=CLASS_NAMES, zero_division=0)
        rows.append({"fruit": f, "n": len(sub), "accuracy": acc, "f1_macro": f1})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def plot_bias_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, fruits: np.ndarray,
    title: str = "Aciertos por especie x calidad real",
    out_path: str | Path | None = None,
):
    """
    Heatmap 6 frutas x 3 calidades reales con accuracy condicional.
    Verde uniforme = sin sesgo por especie. Filas dispares = el modelo
    funciona mejor en unas frutas que en otras.
    """
    df = pd.DataFrame({"fruit": fruits, "y_true": y_true, "y_pred": y_pred})
    df["hit"] = (df["y_true"] == df["y_pred"]).astype(int)
    pivot = (
        df.groupby(["fruit", "y_true"])["hit"].mean()
        .unstack().reindex(index=FRUITS, columns=CLASS_NAMES).fillna(np.nan)
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        ax=ax, cbar_kws={"label": "Accuracy condicional"},
    )
    ax.set_xlabel("Calidad real")
    ax.set_ylabel("Fruta")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
    return fig


def compare_models(
    results: dict[str, dict],
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Construye tabla comparativa de modelos.

    `results` es un dict como:
        {"SVM": {"y_true": ..., "y_pred": ...},
         "Random Forest": {...},
         "CNN": {...}}

    Retorna DataFrame con accuracy, F1-macro, F1 por clase. Tambien guarda
    una figura comparativa de F1-macro.
    """
    rows = []
    for name, blob in results.items():
        rep = classification_report_dict(blob["y_true"], blob["y_pred"])
        row = {
            "model": name,
            "accuracy": rep["_accuracy"],
            "f1_macro": rep["_f1_macro"],
        }
        for cls in CLASS_NAMES:
            row[f"f1_{cls}"] = rep[cls]["f1-score"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    if out_path:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(df))
        width = 0.2
        ax.bar(x - 1.5*width, df["accuracy"], width, label="Accuracy")
        ax.bar(x - 0.5*width, df["f1_macro"], width, label="F1-macro")
        ax.bar(x + 0.5*width, df["f1_Good"], width, label="F1-Good")
        ax.bar(x + 1.5*width, df["f1_Bad"],  width, label="F1-Bad")
        ax.set_xticks(x); ax.set_xticklabels(df["model"])
        ax.set_ylabel("Metrica")
        ax.set_title("Comparativa de modelos (test set)")
        ax.set_ylim(0, 1)
        ax.legend(ncol=2)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
    return df
