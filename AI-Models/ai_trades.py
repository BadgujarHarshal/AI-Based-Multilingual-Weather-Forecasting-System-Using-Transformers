# ai_trade_model.py — AI-Trade Model (14-Feature Aligned, Colab-Ready)

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib, zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras import regularizers

# ─── Config ───
SEQLEN = 20
zones_file = "ai_training_data.parquet"
ohlcv_file = "1m_ohlcv.csv"

# ─── Load Zone Dataset ───
print("📥 Loading dataset ...")
zones = pd.read_parquet(zones_file)
if "timestamp" not in zones.columns:
    zones.rename(columns={"time": "timestamp"}, inplace=True)
zones["timestamp"] = pd.to_datetime(zones["timestamp"])
zones = zones.dropna(subset=["result"])
zones["result"] = zones["result"].astype(str)

# Encode target
label_encoder = LabelEncoder()
zones["result_encoded"] = label_encoder.fit_transform(zones["result"])
joblib.dump(label_encoder, "Label_Encoder.pkl")

zones["is_tp2"] = (zones["result"] == "tp2").astype(int)

# ─── Load OHLCV Dataset ───
print("📥 Loading OHLCV data ...")
ohlcv = pd.read_csv(ohlcv_file)
if "timestamp" not in ohlcv.columns:
    ohlcv.rename(columns={"time": "timestamp"}, inplace=True)
ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
ohlcv = ohlcv.set_index("timestamp").sort_index()
ohlcv = ohlcv[["open", "high", "low", "close", "volume"]].dropna()

# ─── Feature Engineering ───
extra_cols = [
    "rsi", "atr", "rr", "zone_distance", "momentum_strength", "adx",
    "pinbar", "engulfing", "choch_angle", "volume_slope"
]
for col in extra_cols:
    if col not in zones.columns:
        zones[col] = 0.0

# Dummy encode zone_type with both demand and supply columns
zones = pd.get_dummies(zones, columns=["zone_type"])
if "zone_type_demand" not in zones.columns:
    zones["zone_type_demand"] = 0
if "zone_type_supply" not in zones.columns:
    zones["zone_type_supply"] = 0

# Trend encoding
zones["trend_encoded"] = zones["trend"].map({"bullish": 1, "bearish": -1, "sideways": 0}).fillna(0)

# Ensure in_zone exists
if "in_zone" not in zones.columns:
    zones["in_zone"] = 0

# Meta feature order — exactly 14 features
meta_features = [
    "rsi", "atr", "rr", "zone_distance", "momentum_strength", "adx",
    "pinbar", "engulfing", "choch_angle", "volume_slope",
    "in_zone", "zone_type_demand", "zone_type_supply", "trend_encoded"
]

# Scale meta features
scaler = StandardScaler()
zones[meta_features] = scaler.fit_transform(zones[meta_features].fillna(0))
joblib.dump(scaler, "Scaler.pkl")

# ─── Build sequences ───
features, targets, extras = [], [], []
for _, row in zones.iterrows():
    ts = row["timestamp"]
    window = ohlcv.loc[:ts].tail(SEQLEN)
    if len(window) == SEQLEN:
        features.append(window.values)
        targets.append(row["result_encoded"])
        extras.append([row[feat] for feat in meta_features])

X_seq = np.array(features, dtype=np.float32)
X_meta = np.array(extras, dtype=np.float32)
y_multi = np.array(targets)
y_binary = (y_multi == label_encoder.transform(["tp2"])[0]).astype(int)

# ─── SMOTE Oversampling for Multi-class ───
print("📈 Applying SMOTE for multi-class oversampling...")

# --- NEW: Remove any rare classes (< 2 samples) before SMOTE ---
class_counts = pd.Series(y_multi).value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
if rare_classes:
    print(f"⚠ Removing rare classes with < 2 samples: {rare_classes}")
    mask = ~np.isin(y_multi, rare_classes)
    X_seq, X_meta, y_multi = X_seq[mask], X_meta[mask], y_multi[mask]

# Re-encode target labels after removing rare classes
unique_labels = np.unique(y_multi)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
y_multi = np.array([label_mapping[label] for label in y_multi])
# ---------------------------------------------------------------

X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
X_all = np.hstack([X_seq_flat, X_meta])
imputer = SimpleImputer(strategy="mean")
X_all = imputer.fit_transform(X_all)

# NEW: Set k_neighbors=1 to ensure SMOTE can run on any class with >= 2 samples
sm = SMOTE(k_neighbors=1, random_state=42)
X_all_res, y_multi_res = sm.fit_resample(X_all, y_multi)

X_seq_res = X_all_res[:, :SEQLEN*5].reshape(-1, SEQLEN, 5)
X_meta_res = X_all_res[:, SEQLEN*5:]

# ─── Train/Validation Split ───
X_seq_train, X_seq_val, X_meta_train, X_meta_val, y_train_multi, y_val_multi = train_test_split(
    X_seq_res, X_meta_res, y_multi_res, test_size=0.2, stratify=y_multi_res, random_state=42
)
# Re-split for binary model using the same data, but with binary labels
y_binary_res = (y_multi_res == label_encoder.transform(["tp2"])[0]).astype(int)
_, _, _, _, y_train_bin, y_val_bin = train_test_split(
    X_seq_res, X_meta_res, y_binary_res, test_size=0.2, stratify=y_binary_res, random_state=42
)

# ─── Train/Validation Split ───
X_seq_train, X_seq_val, X_meta_train, X_meta_val, y_train_multi, y_val_multi = train_test_split(
    X_seq_res, X_meta_res, y_multi_res, test_size=0.2, stratify=y_multi_res, random_state=42
)
# Re-split for binary model using the same data, but with binary labels
y_binary_res = (y_multi_res == label_encoder.transform(["tp2"])[0]).astype(int)
_, _, _, _, y_train_bin, y_val_bin = train_test_split(
    X_seq_res, X_meta_res, y_binary_res, test_size=0.2, stratify=y_binary_res, random_state=42
)


# ─── Class Weights ───
weights = compute_class_weight('balanced', classes=np.unique(y_train_multi), y=y_train_multi)
class_weights = dict(zip(np.unique(y_train_multi), weights))
print(f"ℹ️ Class weights: {class_weights}")

# ─── Multi-class Model (Updated) ───
seq_input = layers.Input(shape=(SEQLEN, 5), name="sequence_input")
x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(seq_input)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
x = layers.LayerNormalization()(x)
query = layers.Dense(128)(x)
attention_layer = layers.Attention()
attn_out = attention_layer([query, x])
x = layers.GlobalAveragePooling1D()(attn_out)

meta_input = layers.Input(shape=(X_meta.shape[1],), name="meta_input")
y = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(meta_input)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.2)(y)

combined = layers.concatenate([x, y])
combined = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.Dropout(0.3)(combined)

output = layers.Dense(len(np.unique(y_train_multi)), activation="softmax")(combined)

model = models.Model([seq_input, meta_input], output, name="AI-Trade-GRU-Multi")
model.compile(optimizer=tf.keras.optimizers.Adam(0.0003),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ─── Callbacks ───
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)

# ─── Training ───
model.fit([X_seq_train, X_meta_train], y_train_multi,
          validation_data=([X_seq_val, X_meta_val], y_val_multi),
          epochs=500, batch_size=128, class_weight=class_weights,
          callbacks=[es, rlr], verbose=1)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import numpy as np
import joblib

# ─── Evaluation ───
y_probs = model.predict([X_seq_val, X_meta_val])
y_pred = np.argmax(y_probs, axis=1)

labels = sorted(np.unique(y_val_multi))
class_names = label_encoder.inverse_transform(labels)

print("\n📊 Classification Report (Raw):")
print(classification_report(y_val_multi, y_pred, labels=labels, target_names=class_names))

# ─── Threshold tuning for a target class ───
target_class = 'tp1'  # change this to any class name from class_names
if target_class in class_names:
    class_idx = label_encoder.transform([target_class])[0]
    precision, recall, thresholds = precision_recall_curve(
        (y_val_multi == class_idx), y_probs[:, class_idx]
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"\n📊 Auto-tuned {target_class} Threshold = {best_thresh:.2f}")
    joblib.dump(best_thresh, f"{target_class}_Threshold.pkl")
else:
    print(f"\n⚠️ Target class '{target_class}' not found in current model outputs. Available classes: {class_names}")

# ─── Confusion Matrix & ROC AUC ───
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_val_multi, y_pred))

try:
    auc_score = roc_auc_score(y_val_multi, y_probs, multi_class='ovr')
    print(f"🧪 ROC AUC Score: {auc_score:.4f}")
except ValueError as e:
    print(f"⚠️ ROC AUC calculation skipped: {e}")

# ─── Binary TP2 Model (Unchanged) ───
seq_in = layers.Input(shape=(SEQLEN, 5))
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(seq_in)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.LayerNormalization()(x)
meta_in = layers.Input(shape=(X_meta.shape[1],))
x = layers.concatenate([x, meta_in])
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="sigmoid")(x)

binary_model = models.Model([seq_in, meta_in], out, name="TP2-LSTM-Binary")
binary_model.compile(optimizer=tf.keras.optimizers.Adam(0.0003),
                    loss="binary_crossentropy", metrics=["accuracy"])
binary_model.fit([X_seq_train, X_meta_train], y_train_bin,
                validation_data=([X_seq_val, X_meta_val], y_val_bin),
                epochs=500, batch_size=128, callbacks=[es, rlr], verbose=1)

# ─── Binary Evaluation ───
pred_scores = binary_model.predict([X_seq_val, X_meta_val]).flatten()
precision, recall, thresholds = precision_recall_curve(y_val_bin, pred_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_binary_thresh = thresholds[np.argmax(f1_scores)]
print(f"\n📊 TP2 Binary Threshold (F1-Optimized): {best_binary_thresh:.2f}")
joblib.dump(best_binary_thresh, "TP2_Binary_Threshold.pkl")

bin_preds = (pred_scores > best_binary_thresh).astype(int)
print("\n📊 TP2 Binary Classification Report:")
print(classification_report(y_val_bin, bin_preds))
print(f"🧪 ROC AUC Score: {roc_auc_score(y_val_bin, pred_scores):.4f}")

# ─── Save Everything ───
model.save("AI_Trade_LSTM.h5")
binary_model.save("TP2_LSTM_Binary.h5")

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
with open("AI_Trade_LSTM.tflite", "wb") as f: f.write(tflite_model)

# Zip artifacts
with zipfile.ZipFile("AI_Trade_Model.zip", "w") as zipf:
    for file in [
        "AI_Trade_LSTM.h5", "TP2_LSTM_Binary.h5", "AI_Trade_LSTM.tflite",
        "Label_Encoder.pkl", "Scaler.pkl",
        "TP2_Threshold.pkl", "TP2_Binary_Threshold.pkl"
    ]:
        if os.path.exists(file):
            zipf.write(file)

print("\n✅ All model artifacts saved in: AI_Trade_Model.zip")

from google.colab import files
files.download("AI_Trade_Model.zip")