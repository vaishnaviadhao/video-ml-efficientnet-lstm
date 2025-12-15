"""
Diagnostic toolkit for EfficientNet-B3 V1 pipeline.
Checks:
- class distribution
- sample frame loading
- augmentation preview
- feature statistics
- corrupted image detection
- single-frame inference
- GPU memory stats
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

METADATA_DIR = '/home/jovyan/Metadata'
MODEL_PATH = "./models/b3_img_model.keras"
SAMPLE_SAVE_DIR = "./graphs/b3_v1_diagnostics"

IMG_SIZE = (240, 240)

os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)

def preview_samples(csv_path):
    df = pd.read_csv(csv_path)
    print("\nClass Distribution:")
    print(df['label'].value_counts())

    sample = df.sample(4)
    plt.figure(figsize=(8, 8))
    for i, (_, row) in enumerate(sample.iterrows()):
        p = row['frame_path']
        img = plt.imread(p)
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.title(f"{row['label']}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLE_SAVE_DIR, "sample_frames.png"))
    print("Saved: sample_frames.png")

def check_corrupted(csv_path):
    df = pd.read_csv(csv_path)
    bad = []
    for p in df['frame_path'].values:
        try:
            _ = plt.imread(p)
        except Exception:
            bad.append(p)
    print(f"\nCorrupted Images Found: {len(bad)}")
    if bad:
        with open(os.path.join(SAMPLE_SAVE_DIR, "corrupted.txt"), "w") as f:
            for b in bad:
                f.write(b + "\n")

def check_augmentations(frame_path):
    img = tf.io.read_file(frame_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    def aug(x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.1)
        return x

    plt.figure(figsize=(10,4))
    for i in range(5):
        a = aug(img)
        plt.subplot(1,5,i+1)
        plt.imshow(a.numpy())
        plt.axis("off")
    plt.savefig(os.path.join(SAMPLE_SAVE_DIR, "augment_preview.png"))
    print("Saved: augment_preview.png")

def check_model_prediction(model_path, frame_path):
    model = load_model(model_path)
    img = image.load_img(frame_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = arr / 255.0
    pred = model.predict(arr)[0][0]
    print(f"\nPrediction for {frame_path}: {pred:.4f}")

def feature_stats_from_b3(model_path, frame_path):
    model = load_model(model_path)
    b3 = tf.keras.Model(model.input, model.layers[-3].output)

    img = image.load_img(frame_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = arr / 255.0

    feat = b3(arr)[0].numpy()

    print("\nFeature Stats:")
    print("Dim:", feat.shape)
    print("Mean:", feat.mean())
    print("Std :", feat.std())

def gpu_stats():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print("\nGPUs:", gpus)
        for i, g in enumerate(gpus):
            details = tf.config.experimental.get_memory_info(f"GPU:{i}")
            print(f"GPU {i} memory:", details)
    except:
        print("Unable to read GPU memory info.")

def main():
    train_csv = os.path.join(METADATA_DIR, "train_frame_metadata.csv")

    print("=== Diagnostic: Class Distribution & Sample Frames ===")
    preview_samples(train_csv)

    print("\n=== Diagnostic: Corrupted Images ===")
    check_corrupted(train_csv)

    print("\n=== Diagnostic: Augmentation Preview ===")
    df = pd.read_csv(train_csv)
    example = df.sample(1).iloc[0]['frame_path']
    check_augmentations(example)

    print("\n=== Diagnostic: Model Prediction Check ===")
    check_model_prediction(MODEL_PATH, example)

    print("\n=== Diagnostic: Feature Vector Statistics ===")
    feature_stats_from_b3(MODEL_PATH, example)

    print("\n=== Diagnostic: GPU Stats ===")
    gpu_stats()

    print("\nDiagnostics Complete.")

if __name__ == "__main__":
    main()
