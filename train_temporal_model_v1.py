from temporal_lstm_v1 import build_lstm_model
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

SEQ_DIR = "./Sequences_V1"
CSV_PATH = "./video_labels.csv"  # video_id, label

df = pd.read_csv(CSV_PATH)

X, y = [], []

for _, row in df.iterrows():
    seq = np.load(os.path.join(SEQ_DIR, f"{row.video_id}.npy"))
    X.append(seq)
    y.append(row.label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = build_lstm_model()  # or build_transformer_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_temporal_model.keras", save_best_only=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=callbacks
)

preds = model.predict(X_val).ravel()
auc = roc_auc_score(y_val, preds)

print("Validation AUC:", auc)
print(classification_report(y_val, (preds > 0.5).astype(int)))
