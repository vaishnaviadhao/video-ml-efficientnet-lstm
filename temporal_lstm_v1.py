import tensorflow as tf
from tensorflow.keras import layers, models

SEQ_LEN = 64
FEAT_DIM = 1536

def build_lstm_model():
    inputs = layers.Input(shape=(SEQ_LEN, FEAT_DIM))

    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(128)
    )(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="B3_LSTM_V1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model
