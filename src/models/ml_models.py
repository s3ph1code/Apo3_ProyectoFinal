"""
ml_models.py
------------
Modelos de Machine Learning tradicional: SVM y Random Forest.

Ambos usan GridSearchCV con validación cruzada estratificada (5-fold)
y son evaluados con F1-macro como métrica de selección.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    """
    Entrena SVM con kernel RBF mediante GridSearchCV (5-fold estratificado).

    Justificación matemática:
        SVM maximiza el margen entre clases:
            min_{w,b}  ½‖w‖² + C Σᵢ ξᵢ
        sujeto a:  yᵢ(w·φ(xᵢ) + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0

        El kernel RBF  k(x,x') = exp(−γ‖x−x'‖²)  proyecta los datos
        a un espacio de Hilbert de dimensión infinita, permitiendo
        clasificar distribuciones no linealmente separables.

    Espacio de búsqueda:
        C     ∈ {0.1, 1, 10, 100}
        gamma ∈ {0.001, 0.01, 0.1, 'scale'}
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": [0.001, 0.01, 0.1, "scale"],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5,
                        scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"[SVM] Mejor config : {grid.best_params_}")
    print(f"[SVM] F1-macro CV  : {grid.best_score_:.4f}")
    return grid


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    """
    Entrena Random Forest con GridSearchCV (5-fold estratificado).

    Justificación matemática:
        Ensemble de T árboles entrenados con bootstrap (bagging) y
        selección aleatoria de m características por nodo:
            ŷ = argmax_k  Σ_{t=1}^{T} 𝟙[hₜ(x) = k]

        La varianza del ensemble escala como:
            Var(ȳ) = ρσ² + (1−ρ)σ²/T
        donde ρ es la correlación entre árboles, reducida por la
        aleatoriedad en la selección de características.

    Espacio de búsqueda:
        n_estimators ∈ {100, 200, 300}
        max_depth    ∈ {None, 10, 20}
        max_features ∈ {'sqrt', 'log2'}
    """
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 10, 20],
        "max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(rf, param_grid, cv=5,
                        scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"[RF] Mejor config : {grid.best_params_}")
    print(f"[RF] F1-macro CV  : {grid.best_score_:.4f}")
    return grid


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    """Guarda modelo scikit-learn como .pkl en experiments/checkpoints/."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVE] {path}")


def load_model(path: str):
    """Carga modelo scikit-learn desde .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)
