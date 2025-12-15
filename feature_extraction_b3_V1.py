"""
feature_extraction_b3_V1.py
Extract EfficientNet-B3 frame features per video. Produces:
- Features_V1/<video_id>.npy
- features_metadata_v1.csv
Works with train_frame_metadata.csv and val_frame_metadata.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

# ---------------------- Config ----------------------

METADATA_DIR = "./Metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train_frame_metadata.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val_frame_metadata.csv")

FEATURES_DIR = "./Features_V1"
OUTPUT_METADATA = os.path.join(FEATURES_DIR, "features_metadata_v1.csv")

IMG_SIZE = (240, 240)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------- Preprocessing ----------------------

def preprocess(path):
    """Load, decode, resize and normalize a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std

    return img


# ---------------------- Backbone ----------------------

def load_b3_backbone():
    """Load EfficientNet-B3 feature extractor without classification head."""
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base = EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")
    outputs = layers.GlobalAveragePooling2D(dtype="float32")(base.output)
    return models.Model(inputs, outputs, name="B3_backbone")


def extract_features(model, frame_paths):
    """Extract features for a list of frames."""
    ds = tf.data.Dataset.from_tensor_slices(frame_paths)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    feats = model.predict(ds, verbose=0)
    return feats.astype(np.float32)


# ---------------------- Main ----------------------

def main():
    print("\n=== Starting EfficientNet-B3 Feature Extraction (V1) ===\n")

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Load metadata
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV)):
        raise SystemExit("train_frame_metadata.csv or val_frame_metadata.csv not found.")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    df = pd.concat([train_df, val_df], ignore_index=True)

    if "video_id" not in df.columns or "frame_path" not in df.columns:
        raise SystemExit("Metadata must contain 'video_id' and 'frame_path' columns.")

    grouped = df.groupby("video_id")["frame_path"].apply(list).to_dict()
    print(f"Found {len(grouped)} videos.\n")

    backbone = load_b3_backbone()
    print("Backbone loaded.\n")

    rows = []

    for i, (vid, frames) in enumerate(grouped.items(), start=1):
        try:
            feats = extract_features(backbone, frames)
            out_path = os.path.join(FEATURES_DIR, f"{vid}.npy")
            np.save(out_path, feats)

            rows.append({
                "video_id": vid,
                "n_frames": feats.shape[0],
                "feature_path": out_path,
            })

            if i % 25 == 0:
                print(f"[{i}/{len(grouped)}] Processed {vid}")

        except Exception as e:
            print(f"Error processing {vid}: {e}")

    pd.DataFrame(rows).to_csv(OUTPUT_METADATA, index=False)
    print("\nFeature extraction complete.")
    print("Saved metadata:", OUTPUT_METADATA)


if __name__ == "__main__":
    main()
