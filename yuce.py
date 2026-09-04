# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

CONFIG = {
    "model_path": r"G:\model2date\pmic_rf_model.pkl",
    "scaler_path": r"C:\Users\lyh\Desktop\ML-date\training_dataset\raw_pmic_scaler.pkl",
    "train_data_path": r"C:\Users\lyh\Desktop\ML-date\training_dataset\RAW.csv",
    "predict_data_path": r"C:\Users\lyh\Desktop\prediction.csv",
    "output_dir": r"G:\prediction_output",
    "mic_clip_min": 0.125,
    "mic_clip_max": 128.0,
    "perturbation_std": 0.2,
    "deterministic_perturbation": True
}

def calculate_pmic(mic_vals):
    return -np.log10(mic_vals)

def get_train_feature_info():
    train_df = pd.read_csv(CONFIG["train_data_path"])
    target_col = [col for col in train_df.columns if 'pmic' in col.lower()][0]
    train_feature_cols = [col for col in train_df.columns if col not in ['smiles', target_col]]
    train_medians = train_df[train_feature_cols].median()
    return train_feature_cols, train_medians

def get_deterministic_perturbation(smiles_list, std, seed_offset=0):
    perturbations = []
    for sm in smiles_list:
        hash_obj = hashlib.md5(sm.encode('utf-8'))
        hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
        seed = (hash_int + seed_offset) % (2**32)
        rng = np.random.RandomState(seed)
        perturbations.append(rng.normal(0, std))
    return np.array(perturbations)

def predict_and_rank():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train_feature_cols, train_medians = get_train_feature_info()
    model = joblib.load(CONFIG["model_path"])
    scaler = joblib.load(CONFIG["scaler_path"])
    pred_df = pd.read_csv(CONFIG["predict_data_path"])
    if pred_df.empty or 'smiles' not in pred_df.columns:
        raise ValueError("Prediction set is empty or missing 'smiles' column")
    pred_df = pred_df.dropna(subset=['smiles'])
    if pred_df.empty:
        raise ValueError("All 'smiles' entries are empty")
    non_numeric_cols = [col for col in pred_df.columns
                        if col != 'smiles' and not pd.api.types.is_numeric_dtype(pred_df[col])]
    if non_numeric_cols:
        pred_df = pred_df.drop(columns=non_numeric_cols)
    X_pred = pd.DataFrame(index=pred_df.index, columns=train_feature_cols)
    for col in train_feature_cols:
        if col in pred_df.columns:
            X_pred[col] = pred_df[col].fillna(train_medians[col])
        else:
            X_pred[col] = train_medians[col]
    if X_pred.shape[1] != model.n_features_in_:
        raise ValueError(f"Feature dimension mismatch: {X_pred.shape[1]} vs {model.n_features_in_}")
    X_pred_scaled = scaler.transform(X_pred)
    y_pred_pmic = model.predict(X_pred_scaled)
    pmic_min = calculate_pmic(CONFIG["mic_clip_max"])
    pmic_max = calculate_pmic(CONFIG["mic_clip_min"])
    y_pred_pmic = np.clip(y_pred_pmic, pmic_min - 0.5, pmic_max + 0.5)
    if CONFIG["deterministic_perturbation"]:
        perturbation = get_deterministic_perturbation(pred_df["smiles"].tolist(),
                                                      CONFIG["perturbation_std"])
    else:
        perturbation = np.random.normal(0, CONFIG["perturbation_std"], size=len(y_pred_pmic))
    y_pred_pmic = y_pred_pmic + perturbation
    y_pred_pmic = np.clip(y_pred_pmic, pmic_min, pmic_max)
    y_pred_mic = 10 ** (-y_pred_pmic)
    y_pred_mic = np.clip(y_pred_mic, CONFIG["mic_clip_min"], CONFIG["mic_clip_max"])
    rank_series = pd.Series(y_pred_mic).rank(method='min', ascending=True).astype(int)
    output_df = pd.DataFrame({
        "smiles": pred_df["smiles"].values,
        "rank": rank_series
    })
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(CONFIG["output_dir"], f"smiles_rank_{timestamp_str}.csv")
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    try:
        predict_and_rank()
    except Exception as e:
        print(f"Error: {e}")
        raise
