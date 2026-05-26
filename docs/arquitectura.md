# Arquitectura del sistema

## Diagrama de flujo (CRISP-DM)

```
[Imágenes raw]
      │
      ▼
[src/data/preprocess.py]
  - Carga y resize a 224×224
  - Normalización [0, 1]
  - Split 70/15/15 estratificado
  - Data augmentation (train only)
      │
      ├──────────────────────────────┐
      ▼                              ▼
[src/utils/features.py]     [Píxeles directos]
  - HSV histogram (96)
  - Momentos de Hu (7)
  - LBP histogram (32)
  - Channel stats (6)
  → vector de 141 features
      │                              │
      ▼                              ▼
[src/models/ml_models.py]   [src/models/cnn_model.py]
  - SVM (kernel RBF)          - 3 bloques Conv2D+BN+Pool
  - Random Forest             - Dense(256) + Dropout(0.5)
  - GridSearchCV 5-fold       - Softmax(3 clases)
      │                              │
      └──────────────┬───────────────┘
                     ▼
         [src/evaluation/evaluate.py]
           - Classification report
           - Matriz de confusión (SVG)
           - Curvas de aprendizaje (SVG)
           - Comparativa F1-macro (SVG)
                     │
                     ▼
              [app/app.py]
           Interfaz Streamlit
```

## Modelos implementados

### SVM con kernel RBF

Optimiza el margen entre clases en el espacio de características:

```
min_{w,b}  ½‖w‖² + C Σᵢ ξᵢ
sujeto a:  yᵢ(w·φ(xᵢ) + b) ≥ 1 − ξᵢ
```

Kernel: `k(x,x') = exp(−γ‖x−x'‖²)`

### Random Forest

Ensemble de T árboles de decisión con bagging y selección aleatoria de características:

```
ŷ = argmax_k  Σ_{t=1}^{T} 𝟙[hₜ(x) = k]
```

### CNN (3 bloques convolucionales)

```
Input (224×224×3)
→ Conv2D(32) + BN + MaxPool  →  112×112×32
→ Conv2D(64) + BN + MaxPool  →   56×56×64
→ Conv2D(128) + BN + MaxPool →   28×28×128
→ Flatten → Dense(256) → Dropout(0.5)
→ Dense(3) + Softmax
```

## Clases de calidad

| Etiqueta | Descripción |
|----------|-------------|
| `alta`   | Color uniforme, sin manchas ni deformaciones |
| `media`  | Manchas leves o pequeñas deformaciones |
| `baja`   | Daños visibles, podredumbre o deformación severa |

## Estimación de tamaño

Heurística basada en área de región brillante de la imagen:
- `pequeño` → < 25% de píxeles brillantes
- `mediano` → 25–50%
- `grande`  → > 50%
