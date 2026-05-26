"""
cnn_model.py
------------
Red Neuronal Convolucional (CNN) desde cero con Keras/TensorFlow.

Arquitectura: 3 bloques Conv2D + BatchNorm + MaxPool → Dense → Softmax

Justificación matemática:
    Cada capa convolucional calcula:
        z_l = σ(W_l * x_{l-1} + b_l)
    donde * es convolución 2D y σ = ReLU(x) = max(0, x).

    MaxPooling reduce la resolución espacial por un factor de 2,
    aportando invarianza local a pequeñas traslaciones.

    BatchNorm normaliza las activaciones de cada mini-batch:
        x̂ = (x − μ_B) / √(σ²_B + ε)
    acelerando la convergencia y reduciendo la sensibilidad al lr.

    Dropout(0.5) desactiva neuronas aleatoriamente durante entrenamiento,
    aproximando el promedio de 2^N submodelos (Srivastava et al., 2014).
"""


def build_cnn(num_classes: int, input_shape=(224, 224, 3)):
    """
    Construye y compila la CNN.

    Parámetros
    ----------
    num_classes  : int   – número de categorías de calidad
    input_shape  : tuple – (H, W, C)

    Retorna
    -------
    model : keras.Sequential compilado con Adam + sparse_categorical_crossentropy
    """
    try:
        from tensorflow.keras import layers, models
    except ImportError:
        raise ImportError("Instala TensorFlow: pip install tensorflow")

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # ── Bloque 1 ─────────────────────────────────────────────
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # ── Bloque 2 ─────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # ── Bloque 3 ─────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # ── Clasificador ─────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),

    ], name="apple_quality_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model
