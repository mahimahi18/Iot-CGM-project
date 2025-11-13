"""
CGM DATA PREPROCESSING 
This script:
1. Loads 6 CGM datasets
2. Renames columns to a standard format
3. Cleans and validates timestamps and glucose values
4. Resamples each patient’s CGM signal to 5-minute intervals
5. Engineers dynamic and temporal features
6. Labels data using *clinical thresholds only*:
      - normal
      - emergency_hypo (< 70 mg/dL)
      - emergency_hyper (> 180 mg/dL)
7. Combines all datasets into one master clean file

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime

DATA_PATH = '/content/drive/MyDrive/CGM_iot_Project/Raw_data/Datasets'
OUTPUT_PATH = '/content/drive/MyDrive/CGM_iot_Project/Processed'
OUTPUT_FILE = 'master_cgm_clean_no_faults.csv'

DATASET_FILES = [
    'aleppo.csv',
    'hall.csv',
    'shah.csv',
    'colas.csv',
    'wadwa.csv',
    'brown.csv'
]

# Clinical thresholds (mg/dL)
HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180

# Column standardization
RENAME_MAP = {
    'gl': 'glucose',
    'Time': 'timestamp',
    'Timestamp': 'timestamp',
    'time': 'timestamp',
    'ID': 'id',
    'SubjectID': 'id',
    'Patient': 'id'
}


def assign_labels(df):
    """
    Assign clinical labels based only on physiological thresholds.
    Labels:
      - emergency_hypo: glucose < 70
      - emergency_hyper: glucose > 180
      - normal: 70 <= glucose <= 180
    """
    df['label'] = 'normal'
    df.loc[df['glucose'] < HYPO_THRESHOLD, 'label'] = 'emergency_hypo'
    df.loc[df['glucose'] > HYPER_THRESHOLD, 'label'] = 'emergency_hyper'
    return df


def preprocess_dataset(file_path, dataset_name):
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")

    # Step 1: Load CSV
    print("  [1/7] Loading CSV...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"        Loaded: {len(df):,} rows")

    # Step 2: Rename columns
    print("  [2/7] Renaming columns...")
    df = df.rename(columns=RENAME_MAP)

    if 'glucose' not in df.columns or 'timestamp' not in df.columns:
        print(f"   ERROR: Missing required columns!")
        return pd.DataFrame()

    # Step 3: Parse timestamps
    print("  [3/7] Parsing timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Step 4: Convert glucose to numeric
    print("  [4/7] Converting glucose to numeric...")
    df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')

    # Step 5: Remove missing values
    print("  [5/7] Cleaning missing values...")
    df = df.dropna(subset=['timestamp', 'glucose'])
    if 'id' not in df.columns:
        df['id'] = 0

    df = df[['id', 'timestamp', 'glucose']].sort_values(['id', 'timestamp'])

    # Step 6: Resample and interpolate
    print("  [6/7] Resampling to 5-min intervals...")
    resampled = []
    for pid in df['id'].unique():
        patient = df[df['id'] == pid].set_index('timestamp')
        r = patient.resample('5min').mean()
        r['glucose'] = r['glucose'].interpolate(method='linear', limit_direction='both')
        r['id'] = pid
        resampled.append(r.reset_index())

    df = pd.concat(resampled, ignore_index=True)
    df['dataset'] = dataset_name

    # Step 7: Feature engineering
    print("  [7/7] Engineering features + labeling...")
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    df['delta_glucose'] = df.groupby('id')['glucose'].diff()
    df['roll_mean'] = df.groupby('id')['glucose'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['roll_std'] = df.groupby('id')['glucose'].transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Assign labels
    df = assign_labels(df)

    print(f"   Done: {len(df):,} rows")
    return df


def main():
    print("\n" + "="*70)
    print("CGM DATA PREPROCESSING - ACCURATE CLINICAL LABELING")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    all_datasets = []
    for i, fname in enumerate(DATASET_FILES, 1):
        path = os.path.join(DATA_PATH, fname)
        if not os.path.exists(path):
            print(f"   File not found: {fname}")
            continue

        df = preprocess_dataset(path, fname.split('.')[0])
        if len(df) > 0:
            all_datasets.append(df)
            print(f"   Processed: {fname}")
        gc.collect()

    if not all_datasets:
        print(" No datasets processed successfully.")
        return

    master = pd.concat(all_datasets, ignore_index=True)
    print(f"\n Combined all datasets — {len(master):,} rows total")

    # Summary stats
    print("\nLabel distribution:")
    print(master['label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    # Save
    output_file = os.path.join(OUTPUT_PATH, OUTPUT_FILE)
    master.to_csv(output_file, index=False)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n Saved master dataset: {output_file} ({size_mb:.1f} MB)")

    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n STEP 1 COMPLETE — Accurate clinical labeling only.\n")
    return master


if __name__ == "__main__":
    master_df = main()
