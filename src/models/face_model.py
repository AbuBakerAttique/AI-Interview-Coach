"""
Facial Expression Model — Custom CNN Architecture (From Scratch)
Input: 48x48 grayscale face images
Output: 3 classes — nervous, neutral, confident
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_face_cnn(
    input_shape: tuple = (48, 48, 1),
    num_classes: int = 3,
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    """
    Custom CNN for facial expression → interview state classification.
    Architecture built from scratch. No VGG, no ResNet, no transfer learning.

    Architecture:
        Input → Conv2D Block (x4) → Flatten → Dense → Output

    Args:
        input_shape: (height, width, channels) = (48, 48, 1)
        num_classes: 3 (nervous, neutral, confident)
        dropout_rate: Dropout probability

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape, name="face_input")

    # --- Conv Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Conv Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Conv Block 3 ---
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Conv Block 4 ---
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # --- Dense Head ---
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Output ---
    outputs = layers.Dense(num_classes, activation="softmax", name="expression_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FacialExpressionCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_face_cnn()
    model.summary()
