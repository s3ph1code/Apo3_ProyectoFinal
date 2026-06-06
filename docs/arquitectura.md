# Arquitectura del sistema

## Alcance

Clasificación de **calidad (alta / media / baja) cruzada con especie** para seis frutas: manzana, banano, guayaba, lima, naranja, granada. El espacio de salida tiene 18 etiquetas combinadas `Fruta_Calidad`.

## Pipeline general (CRISP-DM)

```
[Fruits/   (recolección del curso, 9515 imgs)]
[data/raw/kaggle/   (descarga reproducible con kagglehub)]
                  │
                  ▼
   [src/data/build_manifest.py]
   → data/annotations/manifest.csv
     (path, fruit, quality, class_id, source, w, h)
                  │
                  ▼
   [src/data/preprocess.py]
     - Resize 224×224
     - Normalización [0, 1]
     - Split estratificado 70/15/15
     - Data augmentation (train only)
                  │
        ┌─────────┴──────────────────┐
        ▼                            ▼
[src/utils/features.py]    [Píxeles directos (CNN)]
  - HSV histogram (96)
  - Momentos de Hu (7)
  - LBP histogram (32)
  - Channel stats (6)
  → vector 141-D
        │                            │
        ▼                            ▼
[src/models/ml_models.py]   [src/models/cnn_model.py]
  - SVM kernel RBF            - 3 bloques Conv2D+BN+Pool
  - Random Forest             - Dense(256) + Dropout(0.5)
  - GridSearchCV 5-fold       - Softmax(18 clases)
        │                            │
        └────────────┬───────────────┘
                     ▼
         [src/evaluation/evaluate.py]
           - Classification report (18 clases)
           - Matriz de confusión 18×18 (SVG)
           - Marginalización a 3 clases de calidad
           - Marginalización a 6 clases de fruta
           - Curvas de aprendizaje (SVG)
           - Comparativa F1-macro (SVG)
                     │
                     ▼
              [app/app.py]
           Interfaz Streamlit
```

## Modelos implementados

### SVM con kernel RBF (multiclase one-vs-rest)

Optimización del margen suave en el espacio de características:

```
min_{w,b}  ½‖w‖² + C Σᵢ ξᵢ
sujeto a:  yᵢ(w·φ(xᵢ) + b) ≥ 1 − ξᵢ,    ξᵢ ≥ 0
```

Kernel gaussiano: `k(x, x') = exp(−γ ‖x − x'‖²)`

Para 18 clases se aplica estrategia *one-vs-rest*: se entrenan 18 clasificadores binarios y se asigna la clase con mayor score:

```
ŷ = argmax_{k ∈ {1,...,18}}  fₖ(x)
```

### Random Forest

Ensemble de T árboles de decisión entrenados con bagging y selección aleatoria de características por nodo:

```
ŷ = argmax_{k ∈ {1,...,18}}  Σ_{t=1}^{T} 𝟙[hₜ(x) = k]
```

La impureza de Gini por nodo se calcula sobre las 18 clases:

```
Gini(S) = 1 − Σ_{k=1}^{18} pₖ²
```

### CNN (3 bloques convolucionales)

```
Input (224×224×3)
→ Conv2D(32, 3×3) + BN + ReLU + MaxPool(2×2)  →  112×112×32
→ Conv2D(64, 3×3) + BN + ReLU + MaxPool(2×2)  →   56×56×64
→ Conv2D(128, 3×3) + BN + ReLU + MaxPool(2×2) →   28×28×128
→ Flatten → Dense(256) + ReLU → Dropout(0.5)
→ Dense(18) + Softmax
```

Pérdida: entropía cruzada categórica

```
ℒ = − (1/N) Σᵢ Σ_{k=1}^{18} yᵢₖ log(ŷᵢₖ)
```

Si el desbalanceo lo justifica (decisión tras EDA), se aplica `class_weight` inverso a la frecuencia:

```
wₖ = N / (K · nₖ)        K = 18, nₖ = #ejemplos clase k
```

## Mapeo de etiquetas

Identificadores en código = nombres del filesystem (single source of truth):

```
CLASS_NAMES = sorted([
    "Apple_Good",  "Apple_Regular",  "Apple_Bad",
    "Banana_Good", "Banana_Regular", "Banana_Bad",
    "Guava_Good",  "Guava_Regular",  "Guava_Bad",
    "Lime_Good",   "Lime_Regular",   "Lime_Bad",
    "Orange_Good", "Orange_Regular", "Orange_Bad",
    "Pomegranate_Good", "Pomegranate_Regular", "Pomegranate_Bad",
])
```

Traducción a español (solo en UI / informe IEEE):

| Inglés | Español |
|---|---|
| Good / Regular / Bad | alta / media / baja |
| Apple / Banana / Guava / Lime / Orange / Pomegranate | manzana / banano / guayaba / lima / naranja / granada |

## Marginalización para análisis

A partir del clasificador de 18 clases se puede derivar:

- **Calidad universal (3 clases):**  `P(calidad = c) = Σ_{f} P(fruta = f, calidad = c)`
- **Especie (6 clases):**            `P(fruta  = f) = Σ_{c} P(fruta = f, calidad = c)`

Esto permite reportar en el informe IEEE tres matrices de confusión complementarias (18×18, 3×3, 6×6).

## Estimación de tamaño

Heurística basada en el área de la región frutal segmentada (umbral en canal V de HSV + cierre morfológico). El umbral de tamaño se calibra **por especie** porque "grande" no es comparable entre granada y lima:

```
tamaño(f) = {
  pequeño   si área < q₃₃(f)
  mediano   si q₃₃(f) ≤ área < q₆₆(f)
  grande    si área ≥ q₆₆(f)
}
```

donde `q₃₃(f)` y `q₆₆(f)` son los percentiles 33 y 66 del área para la fruta `f` en el conjunto de entrenamiento.

## Métricas reportadas

- Accuracy global y por clase.
- Precision, Recall, F1 (por clase y macro).
- Matrices de confusión: 18×18, 3×3 (calidad), 6×6 (fruta).
- Curvas de aprendizaje (loss y accuracy de train vs validation).
- Comparación tabular SVM vs Random Forest vs CNN en F1-macro.

Todas las figuras se guardan en formato **SVG** (requisito vectorial del informe IEEE) en `outputs/figures/`.
