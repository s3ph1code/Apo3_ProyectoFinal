"""
main.py
-------
Punto de entrada principal del proyecto.

Uso:
    python src/main.py --action train   --model svm
    python src/main.py --action train   --model rf
    python src/main.py --action train   --model cnn
    python src/main.py --action eval    --model svm
    python src/main.py --action eval    --all
    python src/main.py --action predict --image ruta/imagen.jpg --model svm
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Clasificación de calidad de manzanas – APO3 ICESI 2026-1"
    )
    parser.add_argument("--action", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--model", choices=["svm", "rf", "cnn"], default="svm")
    parser.add_argument("--all", action="store_true", help="Evaluar todos los modelos")
    parser.add_argument("--image", type=str, help="Ruta imagen para predicción")
    args = parser.parse_args()

    if args.action == "train":
        from src.training.train import train_ml_model, train_cnn_model
        if args.model == "cnn":
            train_cnn_model()
        else:
            train_ml_model(args.model)

    elif args.action == "eval":
        from src.evaluation.evaluate import evaluate_sklearn, evaluate_cnn, compare_all
        if args.all:
            compare_all()
        elif args.model == "cnn":
            evaluate_cnn()
        else:
            evaluate_sklearn(args.model)

    elif args.action == "predict":
        if not args.image:
            print("[ERROR] Especifica --image <ruta>")
            sys.exit(1)

        import numpy as np
        from src.data.preprocess import load_image
        from src.models.ml_models import load_model
        from src.utils.features import extract_features

        CLASS_NAMES = ["alta", "baja", "media"]
        model_path = f"experiments/checkpoints/{args.model}_model.pkl"
        model = load_model(model_path)

        img = load_image(args.image)
        feat = extract_features(img).reshape(1, -1)
        pred = model.predict(feat)[0]
        prob = model.predict_proba(feat)[0]

        print(f"\n[PREDICCIÓN]")
        print(f"  Clase de calidad : {CLASS_NAMES[pred].upper()}")
        print(f"  Confianza        : {prob[pred]*100:.1f}%")
        for name, p in zip(CLASS_NAMES, prob):
            print(f"  {name:8s}: {p*100:.1f}%")


if __name__ == "__main__":
    main()
