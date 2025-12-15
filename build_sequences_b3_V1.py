"""
build_sequences_b3_V1.py
Build fixed-length temporal sequences from B3 feature .npy files.
Output: sequences_metadata_v1.csv
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURE_DIR = "./Features_V1"
SEQ_DIR = "./Sequences_V1"
OUT_CSV = os.path.join(SEQ_DIR, "sequences_metadata_v1.csv")

SEQ_LEN = 64   # final sequence length


def pad_or_trim(arr, target=SEQ_LEN):
    """Pad or uniformly sample a feature sequence."""
    n = arr.shape[0]

    if n == target:
        return arr

    if n < target:
        pad = np.zeros((target - n, arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    # too many frames → uniformly sample
    idx = np.linspace(0, n - 1, target).astype(int)
    return arr[idx]


def main():
    os.makedirs(SEQ_DIR, exist_ok=True)

    rows = []
    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")])

    for f in tqdm(feature_files, desc="Building sequences"):
        video_id = f.replace(".npy", "")

        feat_path = os.path.join(FEATURE_DIR, f)
        arr = np.load(feat_path)

        seq = pad_or_trim(arr)
        out_path = os.path.join(SEQ_DIR, f"{video_id}.npy")
        np.save(out_path, seq)

        rows.append({
            "video_id": video_id,
            "n_frames": arr.shape[0],
            "feature_path": out_path
        })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Done. Sequences saved to:", OUT_CSV)


if __name__ == "__main__":
    main()
