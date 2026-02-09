import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "elevator_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "kone_production_lstm.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "production_config.joblib")
CM_PATH = os.path.join(MODEL_DIR, "production_safety_cm.png")

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

print("Loading data...")
data = pd.read_csv(CSV_PATH)

X = data.drop('entrapment_risk', axis=1)
y = data['entrapment_risk']

print("Original imbalance:")
print(y.value_counts(normalize=True))

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

sm = SMOTE(random_state=42)
Xb, yb = sm.fit_resample(X_scaled, y)

Xb = pd.DataFrame(Xb, columns=X.columns)
yb = pd.Series(yb)

print("After SMOTE balance:")
print(yb.value_counts())

X_seq, y_seq = create_sequences(Xb, yb)
print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq,
    test_size=0.2,
    random_state=42,
    stratify=y_seq
)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return focal_loss_fixed

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.4),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(0.0007),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

callbacks = [
    EarlyStopping(monitor='val_recall', patience=10, restore_best_weights=True, mode='max'),
    ModelCheckpoint(MODEL_PATH, monitor='val_recall', save_best_only=True, mode='max')
]

print("Training safety LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("Evaluating safety performance...")
y_proba = model.predict(X_test).flatten()

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

best_f1 = 0
best_thresh = 0.5

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if rec >= 0.95 and f1 > best_f1:
        best_f1 = f1
        best_thresh = t

safety_thresh = best_thresh
y_pred_safety = (y_proba > safety_thresh).astype(int)

print(f"\n🎯 SAFETY RESULTS (Threshold={safety_thresh:.3f})")
print(classification_report(y_test, y_pred_safety, target_names=['Safe', 'Entrapment Risk']))

cm = confusion_matrix(y_test, y_pred_safety)
print(f"\nFalse Negatives: {cm[1,0]}")
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Safe', 'Risk'], yticklabels=['Safe', 'Risk'])
plt.title("KONE Safety LSTM – False Negatives Minimized")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig(CM_PATH, dpi=300)
plt.show()

joblib.dump({
    "scaler": scaler,
    "threshold": float(safety_thresh),
    "features": list(X.columns)
}, CONFIG_PATH)

print("\n✅ PRODUCTION READY FILES CREATED:")
print(f"1) Model: {MODEL_PATH}")
print(f"2) Config: {CONFIG_PATH}")
print(f"3) Confusion Matrix: {CM_PATH}")