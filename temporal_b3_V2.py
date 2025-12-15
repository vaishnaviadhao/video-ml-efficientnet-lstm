"""
temporal_b3_V2.py

Temporal training (Transformer-style) using precomputed B3 feature sequences.
- Avoids RNN/cuDNN by using attention + feedforward blocks.
- Keeps augmentation path compatible with existing pipeline.
- Performs a quick sanity batch check before calling model.fit.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Paths (adjust if needed)
SEQ_META = "./Sequences_V1/sequences_metadata_v1.csv"
TRAIN_CSV = "/home/jovyan/Metadata/train_frame_metadata.csv"
VAL_CSV = "/home/jovyan/Metadata/val_frame_metadata.csv"
MODEL_SAVE = "./models/b3_temporal_model_v2.keras"

# Hyperparams
SEQ_LEN = 64
FEATURE_DIM = 1536
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation: safe, won't drop entire sequence
def temporal_augment_array(arr: np.ndarray) -> np.ndarray:
    # arr: (N, FEATURE_DIM)
    n = arr.shape[0]
    if n == 0:
        return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

    # randomly drop a few frames (max 5% of SEQ_LEN or less)
    drop_prob = 0.05
    keep_mask = np.random.rand(n) > drop_prob
    if not keep_mask.any():
        keep_mask[np.random.randint(0, n)] = True
    arr = arr[keep_mask]

    # pad or trim to SEQ_LEN
    if arr.shape[0] < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > SEQ_LEN:
        # uniform sampling to reduce to SEQ_LEN
        idx = np.linspace(0, arr.shape[0] - 1, SEQ_LEN).astype(int)
        arr = arr[idx]

    # small gaussian noise
    arr = arr.astype(np.float32) + np.random.normal(0, 2e-3, arr.shape).astype(np.float32)
    return arr.astype(np.float32)


def load_sequence_np(path_bytes: bytes, augment: bool = False) -> np.ndarray:
    # path_bytes comes from tf.numpy_function -> bytes
    path = path_bytes.decode() if isinstance(path_bytes, (bytes, bytearray)) else str(path_bytes)
    # guard: if path is relative, allow it
    try:
        arr = np.load(path)
    except Exception as e:
        # in case of any load error, return padded zeros
        print(f"Warning: failed to load {path}: {e}")
        return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

    # if original features were shorter/longer, handle it similar to pad_or_trim
    if augment:
        return temporal_augment_array(arr)
    # otherwise pad/trim deterministically
    n = arr.shape[0]
    if n >= SEQ_LEN:
        idx = np.linspace(0, n - 1, SEQ_LEN).astype(int)
        seq = arr[idx]
    else:
        pad = np.zeros((SEQ_LEN - n, arr.shape[1]), dtype=np.float32)
        seq = np.concatenate([arr, pad], axis=0)
    return seq.astype(np.float32)


def make_tf_dataset(df: pd.DataFrame, batch_size=BATCH_SIZE, shuffle=False, augment=False):
    paths = df["feature_path"].values.astype(str)
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_fn(p, y):
        seq = tf.numpy_function(func=lambda x: load_sequence_np(x, augment=augment), inp=[p], Tout=tf.float32)
        seq = tf.ensure_shape(seq, (SEQ_LEN, FEATURE_DIM))
        return seq, y

    ds = ds.map(load_fn, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# Transformer block
def transformer_block(x, num_heads=4, ff_dim=512, dropout=0.2, name=None):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim // num_heads, name=f"{name}_mha" if name else None)(x, x)
    attn = layers.Dropout(dropout)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation="relu")(out1)
    ff = layers.Dense(out1.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff)
    return out2


def build_temporal_model():
    inp = layers.Input(shape=(SEQ_LEN, FEATURE_DIM), name="seq_input")

    x = layers.LayerNormalization(epsilon=1e-6)(inp)
    # project down to a smaller dim for attention efficiency
    proj_dim = 512
    x = layers.Dense(proj_dim, activation="relu", name="proj_dense")(x)

    # learnable positional embeddings
    pos_embed = tf.Variable(initial_value=tf.random.normal([SEQ_LEN, proj_dim], stddev=0.02), trainable=True, name="pos_embed")
    x = x + pos_embed[None, :, :]

    # stack of transformer blocks (depth 2-3)
    for i in range(3):
        x = transformer_block(x, num_heads=8, ff_dim=1024, dropout=0.25, name=f"block{i}")

    # pooling and head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = models.Model(inputs=inp, outputs=out, name="temporal_b3_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def safe_label_map(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if s in ("real", "0", "r"):
        return 0
    if s in ("fake", "1", "f"):
        return 1
    # fallback: try integer coercion
    try:
        v = int(s)
        return 1 if v == 1 else 0
    except Exception:
        return np.nan


def main(args):
    print("\n=== Training Temporal B3 V2 Model ===")

    if not os.path.exists(SEQ_META):
        raise SystemExit("Missing sequences_metadata_v1.csv — run build_sequences_b3_V1.py first")

    seq_df = pd.read_csv(SEQ_META)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_ids = set(train_df["video_id"].unique())
    val_ids = set(val_df["video_id"].unique())

    train_split = seq_df[seq_df["video_id"].isin(train_ids)].copy()
    val_split = seq_df[seq_df["video_id"].isin(val_ids)].copy()

    # Build label map from frame-level metadata (handles different casings/formats)
    full_label_map = pd.concat([train_df[["video_id", "label"]], val_df[["video_id", "label"]]]).drop_duplicates(subset=["video_id"])
    full_label_map["label_clean"] = full_label_map["label"].apply(safe_label_map)

    # Merge
    train_split = train_split.merge(full_label_map[["video_id", "label_clean"]], on="video_id", how="left")
    val_split = val_split.merge(full_label_map[["video_id", "label_clean"]], on="video_id", how="left")

    train_split = train_split.rename(columns={"label_clean": "label"})
    val_split = val_split.rename(columns={"label_clean": "label"})

    # Drop any rows missing labels (should be few)
    missing_train = train_split["label"].isna().sum()
    missing_val = val_split["label"].isna().sum()
    if missing_train or missing_val:
        print(f"Warning: missing labels - train:{missing_train}, val:{missing_val}. Dropping them.")
        train_split = train_split.dropna(subset=["label"]).copy()
        val_split = val_split.dropna(subset=["label"]).copy()

    train_split["label"] = train_split["label"].astype(int)
    val_split["label"] = val_split["label"].astype(int)

    print(f"Train videos: {len(train_split)}, Val videos: {len(val_split)}")

    # datasets
    train_ds = make_tf_dataset(train_split, batch_size=args.batch, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_split, batch_size=args.batch, shuffle=False, augment=False)

    # quick sanity check: fetch one batch and print stats
    print("Running sanity batch check...")
    try:
        for bx, by in train_ds.take(1):
            print("SANITY: batch_x shape:", bx.shape, "batch_y shape:", by.shape)
            bx_min = tf.reduce_min(bx).numpy()
            bx_max = tf.reduce_max(bx).numpy()
            bx_nan = tf.math.reduce_any(tf.math.is_nan(bx)).numpy()
            bx_inf = tf.math.reduce_any(tf.math.is_inf(bx)).numpy()
            print("SANITY: min,max,nan,inf:", float(bx_min), float(bx_max), bool(bx_nan), bool(bx_inf))
    except Exception as e:
        print("Sanity check failed:", e)
        raise

    model = build_temporal_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        
    except Exception as e:
        print("=== ERROR during model.fit() ===")
        raise

    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
    model.save(MODEL_SAVE)
    print("\nSaved model:", MODEL_SAVE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args)
