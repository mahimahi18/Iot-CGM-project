#Removing leakage and not loading 3gb
# ==========================================
# CLEANED-UP AUTOENCODER + GBT PIPELINE (NO LEAKAGE, LIGHTWEIGHT OUTPUT)
# ==========================================
import os, numpy as np, pandas as pd, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# ---- CONFIG ----
file_path = '/content/drive/MyDrive/CGM_iot_Project/Processed/master_cgm_clean_no_faults.csv'
chunk_size = 50000
autoencoder_epochs = 30
autoencoder_batch = 64
random_state = 42

# ---- Load small sample to train autoencoder ----
print("📥 Sampling 100k rows to fit autoencoder...")
sample_data = pd.read_csv(file_path, usecols=["glucose"], nrows=100000).dropna()
scaler = StandardScaler()
sample_data["glucose_scaled"] = scaler.fit_transform(sample_data[["glucose"]])
X = sample_data[["glucose_scaled"]].values.astype(np.float32)

X_train, X_val = train_test_split(X, test_size=0.2, random_state=random_state)

# ---- Build autoencoder ----
input_dim = X.shape[1]
encoding_dim = 1
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("Training autoencoder...")
autoencoder.fit(X_train, X_train,
                epochs=autoencoder_epochs,
                batch_size=autoencoder_batch,
                validation_data=(X_val, X_val),
                verbose=1)

# ---- Compute anomaly threshold ----
val_recon = autoencoder.predict(X_val, verbose=0)
val_err = np.abs(X_val - val_recon).flatten()
threshold = float(np.mean(val_err) + 3 * np.std(val_err))
print(f" Threshold set at {threshold:.6f}")

# ---- Detect anomalies in chunks ----
reader = pd.read_csv(file_path, chunksize=chunk_size)
results = []
chunk_id = 0
for chunk in reader:
    chunk_id += 1
    print(f" Processing chunk {chunk_id}")
    if "glucose" not in chunk.columns:
        continue
    chunk = chunk.dropna(subset=["glucose"])
    chunk["glucose_scaled"] = scaler.transform(chunk[["glucose"]])
    X_chunk = chunk[["glucose_scaled"]].values.astype(np.float32)
    recon = autoencoder.predict(X_chunk, verbose=0)
    err = np.abs(X_chunk - recon).flatten()
    chunk["recon_error"] = err
    chunk["is_anomaly"] = err > threshold
    results.append(chunk[["glucose", "recon_error", "is_anomaly"]])
    
df_all = pd.concat(results, ignore_index=True)
anomalies = df_all[df_all["is_anomaly"]].copy()
print(f" Total anomalies detected: {len(anomalies)}")

# ---- Inject synthetic sensor faults for classification ----
if len(anomalies) > 0:
    np.random.seed(random_state)
    anomalies["label"] = np.where(
        np.random.rand(len(anomalies)) < 0.5, "sensor_fault", "emergency"
    )

    # ---- Sample a manageable subset ----
    sample = anomalies.sample(n=min(50000, len(anomalies)), random_state=random_state)
    X_all = sample[["glucose", "recon_error"]]
    y_all = sample["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    print(" Training Gradient Boosted Trees classifier...")
    gbt = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state
    )
    gbt.fit(X_train, y_train)
    y_pred = gbt.predict(X_test)

    print("\n=== Gradient Boosted Trees Results ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # ---- Save only anomaly summary (not full 3GB data) ----
    output_summary = sample.copy()
    output_summary["predicted_label"] = gbt.predict(X_all)
    output_summary.head(5000).to_csv("/content/anomaly_summary.csv", index=False)
    print(" Saved summary of 5000 anomaly samples → /content/anomaly_summary.csv")

else:
    print("No anomalies found. Skipping classification.")
