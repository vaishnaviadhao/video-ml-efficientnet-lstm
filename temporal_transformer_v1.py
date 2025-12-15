import tensorflow as tf
from tensorflow.keras import layers, models

SEQ_LEN = 64
FEAT_DIM = 1536

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn = self.att(inputs, inputs)
        x = self.norm1(inputs + self.drop1(attn, training=training))
        ffn = self.ffn(x)
        return self.norm2(x + self.drop2(ffn, training=training))

def build_transformer_model():
    inputs = layers.Input(shape=(SEQ_LEN, FEAT_DIM))
    x = layers.Masking(mask_value=0.0)(inputs)

    x = TransformerBlock(FEAT_DIM, num_heads=8, ff_dim=1024)(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="B3_Transformer_V1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model
