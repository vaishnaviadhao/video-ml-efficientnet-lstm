import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = "./Frames"
OUTPUT_DIR = "./Metadata"

rows = []

# Mapping folders to labels
label_map = {
    "Fake": 1,
    "Real": 0
}

for class_name, label in label_map.items():
    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_dir):
        print(f"Skipping missing folder: {class_dir}")
        continue

    for video_id in os.listdir(class_dir):
        video_path = os.path.join(class_dir, video_id)

        if not os.path.isdir(video_path):
            continue

        for frame in os.listdir(video_path):
            if frame.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "video_id": video_id,
                    "frame_path": os.path.join(video_path, frame),
                    "label": label
                })

df = pd.DataFrame(rows)

print("Total frames found:", len(df))
print("Unique videos:", df["video_id"].nunique())

# Train / Val split (stratified by label)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train_frame_metadata.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_frame_metadata.csv"), index=False)

print("✅ Metadata CSVs created successfully!")
print("Train frames:", len(train_df))
print("Val frames:", len(val_df))
