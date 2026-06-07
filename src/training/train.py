"""
train.py
--------
CLI de entrenamiento para los modelos del proyecto. Wrapper conveniente
de los notebooks 03/04 para correr en linea de comandos.

Uso
---
    python src/training/train.py --model svm
    python src/training/train.py --model rf
    python src/training/train.py --model cnn --epochs 20 --batch-size 32

Los modelos entrenados se guardan en `experiments/checkpoints/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svm", "rf", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-grid-search", action="store_true",
                        help="Skip GridSearchCV (use default hyperparams)")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "experiments" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("svm", "rf"):
        _train_ml(args, out_dir)
    else:
        _train_cnn(args, out_dir)


def _train_ml(args, out_dir):
    """Entrena SVM o RF sobre el vector 141-D extraido del manifest."""
    import numpy as np
    import pandas as pd
    from src.data.preprocess import load_and_preprocess_image
    from src.utils.features import extract_features
    from src.models.ml_models import SVMClassifier, RFClassifier
    from tqdm import tqdm

    print("[train] Cargando splits ...")
    train_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "train_manifest.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "val_manifest.csv")

    cache = REPO_ROOT / "data" / "processed" / "features_train.npz"
    if cache.exists():
        print(f"[train] Usando cache de features: {cache}")
        z = np.load(cache, allow_pickle=True)
        X_train, y_train = z["X"], z["y"]
    else:
        print("[train] Extrayendo features 141-D para train (esto toma varios minutos)")
        X_train = np.stack([
            extract_features(load_and_preprocess_image(REPO_ROOT / p))
            for p in tqdm(train_df["path"])
        ])
        y_train = train_df["quality"].values
        np.savez(cache, X=X_train, y=y_train)

    if args.model == "svm":
        model = SVMClassifier()
        name = "svm_rbf"
    else:
        model = RFClassifier()
        name = "random_forest"

    print(f"[train] Entrenando {name} (GridSearchCV={'no' if args.no_grid_search else 'si'}) ...")
    model.fit(X_train, y_train, do_grid_search=not args.no_grid_search)
    print(f"[train] best_params: {model.best_params_}")
    path = model.save(out_dir / f"{name}.joblib")
    print(f"[train] Guardado: {path}")


def _train_cnn(args, out_dir):
    """Entrena la CNN simple. Usa tf.data para streaming de imagenes."""
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from src.data.preprocess import (
        load_and_preprocess_image, get_train_augmentation, get_val_augmentation,
        CLASS_NAMES,
    )
    from src.models.cnn_model import build_cnn, get_training_callbacks

    print("[train] Cargando splits ...")
    train_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "train_manifest.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "val_manifest.csv")

    label2idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def make_dataset(df, augment, shuffle):
        train_aug = get_train_augmentation() if augment else None

        def gen():
            idx = np.arange(len(df))
            if shuffle:
                np.random.shuffle(idx)
            for i in idx:
                row = df.iloc[i]
                img = load_and_preprocess_image(REPO_ROOT / row["path"])
                if train_aug is not None:
                    img = train_aug(image=img)["image"]
                y = np.zeros(3, dtype=np.float32)
                y[label2idx[row["quality"]]] = 1.0
                yield img.astype(np.float32), y

        sig = (tf.TensorSpec((224, 224, 3), tf.float32),
               tf.TensorSpec((3,), tf.float32))
        ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_df, augment=True, shuffle=True)
    val_ds = make_dataset(val_df, augment=False, shuffle=False)

    model = build_cnn()
    model.summary()

    cw = train_df["quality"].value_counts()
    N = cw.sum(); K = 3
    class_weight = {label2idx[c]: N / (K * cw[c]) for c in CLASS_NAMES}

    cb = get_training_callbacks(str(out_dir / "cnn_best.h5"))
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, class_weight=class_weight, callbacks=cb, verbose=1,
    )
    final_path = out_dir / "cnn_final.h5"
    model.save(final_path)
    print(f"[train] Guardado: {final_path}")


if __name__ == "__main__":
    main()
