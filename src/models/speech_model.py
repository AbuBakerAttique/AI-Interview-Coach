"""
Speech Confidence Model — CNN + LSTM Architecture (From Scratch)
Input: MFCC features (200, 40)
Output: 3 classes — nervous, neutral, confident
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_speech_cnn_lstm(
    input_shape: tuple = (200, 40),
    num_classes: int = 3,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Custom CNN + LSTM architecture for speech confidence classification.
    Built 100% from scratch — no pretrained weights.

    Architecture:
        Input → Conv1D (x3) → LSTM → Dense → Output

    Args:
        input_shape: (time_steps, n_mfcc) = (200, 40)
        num_classes: 3 (nervous, neutral, confident)
        dropout_rate: Dropout probability

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape, name="mfcc_input")

    # --- CNN Block 1 ---
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- CNN Block 2 ---
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- CNN Block 3 ---
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- LSTM Block ---
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Dense Head ---
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)

    # --- Output ---
    outputs = layers.Dense(num_classes, activation="softmax", name="confidence_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SpeechConfidenceCNN_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_speech_pure_cnn(
    input_shape: tuple = (200, 40),
    num_classes: int = 3,
) -> tf.keras.Model:
    """
    Alternative: Pure CNN (faster, good baseline to compare with CNN+LSTM)
    """
    inputs = tf.keras.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SpeechConfidenceCNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = build_speech_cnn_lstm()
    model.summary()
