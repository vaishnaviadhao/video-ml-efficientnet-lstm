"""
temporal_b3_V1.py
Train temporal model (LSTM + Attention) using precomputed B3 features.
Compatible with V1 pipeline.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
METADATA_DIR = "/home/jovyan/Metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train_frame_metadata.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val_frame_metadata.csv")

SEQUENCES_DIR = "./Sequences_V1"
SEQ_META = os.path.join(SEQUENCES_DIR, "sequences_metadata_v1.csv")

MODEL_SAVE = "./models/b3_temporal_model.keras"

# Hyperparameters
SEQ_LEN = 64
FEATURE_DIM = 1536
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------
# Utilities
# --------------------------------------------------------
def pad_or_trim(arr, target_len=SEQ_LEN):
    """Pad or uniformly sample sequence to fixed length."""
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n < target_len:
        pad = np.zeros((target_len - n, arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    idxs = np.linspace(0, n - 1, target_len).astype(int)
    return arr[idxs]


def load_sequence(path):
    arr = np.load(path)
    return pad_or_trim(arr)


# inside file: replace the existing make_tf_dataset load_fn and build_temporal_model

def make_tf_dataset(df, batch_size=BATCH_SIZE, shuffle=False):
    paths = df["feature_path"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_fn(p, y):
        seq = tf.numpy_function(load_sequence, [p], tf.float32)
        # ensure static shape for the graph / cuDNN
        seq.set_shape((SEQ_LEN, FEATURE_DIM))
        return seq, y

    ds = ds.map(load_fn, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(df))
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_temporal_model():
    inp = layers.Input(shape=(SEQ_LEN, FEATURE_DIM))

    # Transformer Encoder
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(inp, inp)
    attn = layers.Dropout(0.1)(attn)
    x = layers.LayerNormalization()(inp + attn)

    ff = layers.Dense(256, activation="relu")(x)
    ff = layers.Dense(FEATURE_DIM)(ff)
    ff = layers.Dropout(0.1)(ff)
    x = layers.LayerNormalization()(x + ff)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )
    return model

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main(args):
    print("\n=== Training Temporal B3 V1 Model ===")

    # --- Load sequence metadata ---
    if not os.path.exists(SEQ_META):
        raise SystemExit("Missing sequences_metadata_v1.csv — run build_sequences_b3_V1.py first")

    seq_df = pd.read_csv(SEQ_META)

    # --- Load frame metadata for labels ---
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_ids = set(train_df["video_id"].unique())
    val_ids = set(val_df["video_id"].unique())

    train_split = seq_df[seq_df["video_id"].isin(train_ids)].copy()
    val_split = seq_df[seq_df["video_id"].isin(val_ids)].copy()

    # ----------------------------------------------------
    # Assign labels using original metadata
    # ----------------------------------------------------
    full_label_map = (
        pd.concat([train_df, val_df])[["video_id", "label"]]
        .drop_duplicates(subset=["video_id"])
    )

    # Merge labels into splits
    train_split = train_split.merge(full_label_map, on="video_id", how="left")
    val_split = val_split.merge(full_label_map, on="video_id", how="left")

    # Safety check
    if train_split["label"].isna().any() or val_split["label"].isna().any():
        missing = pd.concat([
            train_split[train_split["label"].isna()],
            val_split[val_split["label"].isna()]
        ])
        print("WARNING: Missing labels for videos:", missing["video_id"].tolist())

    # Encode labels
    label_map_dict = {"real": 0, "Real": 0, "fake": 1, "Fake": 1}
    train_split["label"] = train_split["label"].map(label_map_dict)
    val_split["label"] = val_split["label"].map(label_map_dict)

    print(f"Train videos: {len(train_split)}, Val videos: {len(val_split)}")

    # Dataset creation
    train_ds = make_tf_dataset(train_split, shuffle=True)
    val_ds = make_tf_dataset(val_split, shuffle=False)

    # Build temporal model
    model = build_temporal_model()
    model.summary()

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True)
    ]

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    model.save(MODEL_SAVE)
    print("\nSaved temporal model to:", MODEL_SAVE)


# --------------------------------------------------------
# CLI
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args)
