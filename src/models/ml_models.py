"""
ml_models.py
------------
Wrappers de SVM (RBF) y Random Forest para clasificacion de 3 clases
(Good / Regular / Bad) sobre el vector handcrafted 141-D.

Cada wrapper:
- Encapsula un Pipeline(StandardScaler, classifier).
- Soporta tuning con GridSearchCV 5-fold sobre F1-macro.
- Tiene save/load via joblib.
- Expone predict/predict_proba.

Uso desde un notebook
---------------------
    from src.models.ml_models import SVMClassifier, RFClassifier

    svm = SVMClassifier()
    svm.fit(X_train, y_train, do_grid_search=True)
    print(svm.best_params_)
    y_pred = svm.predict(X_test)
    svm.save("experiments/checkpoints/svm.joblib")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_NAMES = ["Good", "Regular", "Bad"]
RANDOM_STATE = 42


class _BaseQualityClassifier:
    """Clase base con interfaz comun para SVM y RF."""

    name: str = "base"

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.best_params_: dict[str, Any] | None = None
        self.cv_results_: dict[str, Any] | None = None

    # -- API publica -------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        do_grid_search: bool = True,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> "_BaseQualityClassifier":
        if do_grid_search:
            self._fit_with_grid_search(X, y, cv=cv, n_jobs=n_jobs, verbose=verbose)
        else:
            self.pipeline = self._default_pipeline()
            self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(X)

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "best_params": self.best_params_,
            "name": self.name,
            "classes": CLASS_NAMES,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "_BaseQualityClassifier":
        blob = joblib.load(path)
        obj = cls()
        obj.pipeline = blob["pipeline"]
        obj.best_params_ = blob.get("best_params")
        return obj

    # -- Implementaciones subclase ----------------------------------------
    def _default_pipeline(self) -> Pipeline:
        raise NotImplementedError

    def _param_grid(self) -> dict[str, list]:
        raise NotImplementedError

    # -- Privado -----------------------------------------------------------
    def _fit_with_grid_search(self, X, y, cv, n_jobs, verbose) -> None:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            self._default_pipeline(),
            self._param_grid(),
            cv=cv_splitter,
            scoring="f1_macro",
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
        gs.fit(X, y)
        self.pipeline = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.cv_results_ = gs.cv_results_

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} no esta entrenado; llama a .fit primero.")


# ---------------------------------------------------------------------------
# SVM con kernel RBF
# ---------------------------------------------------------------------------
class SVMClassifier(_BaseQualityClassifier):
    name = "svm_rbf"

    def _default_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ])

    def _param_grid(self) -> dict[str, list]:
        # Grid moderado: 3 valores de C x 3 de gamma = 9 combinaciones x 5 folds = 45 fits.
        return {
            "clf__C": [1.0, 10.0, 100.0],
            "clf__gamma": ["scale", 0.01, 0.001],
        }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
class RFClassifier(_BaseQualityClassifier):
    name = "random_forest"

    def _default_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),  # RF no lo necesita, pero da consistencia con SVM
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    def _param_grid(self) -> dict[str, list]:
        # 3 x 3 x 2 = 18 combinaciones x 5 folds = 90 fits (rapidos en RF).
        return {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 20, 40],
            "clf__max_features": ["sqrt", "log2"],
        }

    def feature_importances(self) -> np.ndarray | None:
        if self.pipeline is None:
            return None
        return self.pipeline.named_steps["clf"].feature_importances_
