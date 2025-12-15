import os
import pandas as pd

SEQ_DIR = "./Sequences_V1"

rows = []

for fname in os.listdir(SEQ_DIR):
    if not fname.endswith(".npy"):
        continue

    video_id = fname.replace(".npy", "")

    # label logic based on name / origin
    # hotshot = Fake, msrvtt = Real (based on your dataset)
    if video_id.startswith("hotshot"):
        label = 1   # Fake
    else:
        label = 0   # Real

    rows.append({
        "video_id": video_id,
        "label": label
    })

df = pd.DataFrame(rows)
df.to_csv("video_labels.csv", index=False)

print("✅ video_labels.csv created")
print(df["label"].value_counts())
