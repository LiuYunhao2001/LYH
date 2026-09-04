import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['mathtext.default'] = 'regular'

CONFIG = {
    "train_data_path": r"G:\model2date\pmic_filtered_double_train_data.csv",
    "save_root": r"C:\Users\lyh\Desktop\ML-date",
    "scaler_path": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "pmic_rf_scaler.pkl"),
    "model_path": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "pmic_best_rf_fixed_param.pkl"),
    "performance_path": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_fixed_param_performance.csv"),
    "true_pred_plot": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_true_vs_pred.png"),
    "mic_dist_plot": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_mic_distribution.png"),
    "predict_script_path": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_predict_example.py")
}

os.makedirs(CONFIG["save_root"], exist_ok=True)

def load_and_preprocess_data():
    print("=" * 50)
    print(" Step 1: Load and preprocess training data")
    print("=" * 50)
    if not os.path.exists(CONFIG["train_data_path"]):
        raise FileNotFoundError(f" Training data file not found: {CONFIG['train_data_path']}")
    try:
        data = pd.read_csv(CONFIG["train_data_path"], encoding='utf-8-sig')
        print(f" Data loaded successfully | Raw dimension: {data.shape}")
    except Exception as e:
        raise ValueError(f" Failed to load data: {str(e)}, please ensure file is UTF‑8 CSV format")
    required_cols = ['smiles', 'pMIC']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f" Missing required columns in dataset: {missing_cols}")
    exclude_cols = ['smiles', 'pMIC', 'residual', 'MIC']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    target_col = 'pMIC'
    X = data[feature_cols].values
    y = data[target_col].values
    y_nan_mask = np.isnan(y)
    if np.sum(y_nan_mask) > 0:
        X = X[~y_nan_mask]
        y = y[~y_nan_mask]
        print(f" Removed samples with missing pMIC: {np.sum(y_nan_mask)}")
    print(f" Feature‑target separation completed | X={X.shape} ({len(feature_cols)} descriptors) | y={y.shape}")
    print("\n Step 2: Median imputation for feature missing values")
    median_values = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(median_values, np.where(nan_mask)[1])
    print(f" Imputation finished | Total original missing values: {np.sum(nan_mask)}")
    print("\n Step 3: Split train/test set (80% Train / 20% Test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f" Training set: X_train={X_train.shape} | y_train={y_train.shape}")
    print(f" Test set: X_test={X_test.shape} | y_test={y_test.shape}")
    print("\n Step 4: Standardization of molecular descriptors (Mean=0, Std=1)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(" Standardization completed, data leakage avoided")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, data

def print_rf_performance(model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print(" Step 5: Model performance evaluation")
    print("=" * 50)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = round(r2_score(y_train, y_train_pred), 4)
    r2_test = round(r2_score(y_test, y_test_pred), 4)
    rmse_train = round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4)
    rmse_test = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
    mae_test = round(mean_absolute_error(y_test, y_test_pred), 4)
    oob_score = round(model.oob_score_, 4) if hasattr(model, 'oob_score_') else "N/A"
    print(f" Train set $R^2$: {r2_train} | RMSE: {rmse_train}")
    print(f" Test set $R^2$: {r2_test} | RMSE: {rmse_test} | MAE: {mae_test}")
    print(f" Out‑of‑bag generalization score OOB: {oob_score}")
    if r2_test < 0.6:
        print("  Warning: Test set $R^2$ below 0.6, poor generalization performance")
    metrics = {
        "Train R^2": r2_train,
        "Test R^2": r2_test,
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test,
        "Test MAE": mae_test,
        "OOB Score": oob_score
    }
    return metrics, y_test, y_test_pred

def plot_true_vs_pred(y_test, y_test_pred, r2_test):
    print("\n Step 6: Plot True vs Predicted pMIC scatter plot")
    plt.figure(figsize=(8, 8))
    plt.scatter(
        y_test, y_test_pred,
        color='#2ca02c', alpha=0.6, s=30, edgecolor='black', linewidth=0.5
    )
    z = np.polyfit(y_test, y_test_pred, 1)
    p = np.poly1d(z)
    plt.plot(
        y_test, p(y_test), "r--", linewidth=2,
        label=f'Fitting Line: $y={z[0]:.2f}x+{z[1]:.2f}$'
    )
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k-', linewidth=2, label='Perfect Fitting Line ($\(y=x\)$)'
    )
    plt.text(
        0.05, 0.95, f'Test Set $R^2$ = {r2_test:.4f}',
        transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    plt.title('Random Forest - True pMIC vs Predicted pMIC', fontweight='bold', pad=20)
    plt.xlabel('True pMIC Value', fontsize=12)
    plt.ylabel('Predicted pMIC Value', fontsize=12)
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(CONFIG["true_pred_plot"])
    plt.close()
    print(f" Scatter plot saved | Path: {CONFIG['true_pred_plot']}")

def plot_mic_distribution(y_test, y_test_pred):
    print("\n Step 7: Plot MIC value distribution histogram")
    mic_true = 10 ** (-y_test)
    mic_pred = 10 ** (-y_test_pred)
    mic_true = np.clip(mic_true, 0.1, 256)
    mic_pred = np.clip(mic_pred, 0.1, 256)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        mic_pred, bins=20, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5,
        label='Predicted MIC', kde=False
    )
    sns.histplot(
        mic_true, bins=20, color='#1f77b4', alpha=0.5, edgecolor='black', linewidth=0.5,
        label='True MIC', kde=False
    )
    plt.title('Random Forest - MIC Value Distribution (0.1‑256μg/mL)', fontweight='bold', pad=20)
    plt.xlabel('MIC Value (μg/mL)', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    plt.xlim(0, 260)
    plt.tight_layout()
    plt.savefig(CONFIG["mic_dist_plot"])
    plt.close()
    print(f" MIC distribution plot saved | Path: {CONFIG['mic_dist_plot']}")

def save_all_files(scaler, model, metrics):
    print("\n" + "=" * 50)
    print(" Step 8: Save all core files (6 files total)")
    print("=" * 50)
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f" 1. Scaler: {CONFIG['scaler_path']}")
    joblib.dump(model, CONFIG["model_path"])
    print(f" 2. Random forest model: {CONFIG['model_path']}")
    pd.DataFrame([metrics]).to_csv(CONFIG["performance_path"], index=False, encoding='utf-8-sig')
    print(f" 3. Performance metrics table: {CONFIG['performance_path']}")
    predict_script = '''
import pandas as pd
import numpy as np
import joblib

SCALER_PATH = r"C:\\Users\\lyh\\Desktop\\ML-date\\pmic_rf_scaler.pkl"
MODEL_PATH = r"C:\\Users\\lyh\\Desktop\\ML-date\\pmic_best_rf_fixed_param.pkl"
def predict_pmic_and_mic(X_new_path):
    try:
        scaler = joblib.load(SCALER_PATH)
        rf_model = joblib.load(MODEL_PATH)
        print(" Scaler and Model loaded successfully")
    except Exception as e:
        raise FileNotFoundError(f" Load failed: {str(e)}")
    try:
        X_new = pd.read_csv(X_new_path, encoding='utf-8-sig')
        print(f" New data loaded | Shape: {X_new.shape}")
    except Exception as e:
        raise ValueError(f" New data load failed: {str(e)}")
    median_values = np.nanmedian(X_new.values, axis=0)
    nan_mask = np.isnan(X_new.values)
    X_new.values[nan_mask] = np.take(median_values, np.where(nan_mask)[1])
    print(f" Missing value imputed | Total missing: {np.sum(nan_mask)}")
    X_new_scaled = scaler.transform(X_new)
    pMIC_pred = rf_model.predict(X_new_scaled)
    MIC_pred = np.clip(10 ** (-pMIC_pred), 0.1, 256)
    print("\\n Prediction Result:")
    print(f"Predicted pMIC: {np.round(pMIC_pred, 4)}")
    print(f"Predicted MIC (μg/mL): {np.round(MIC_pred, 4)}")
    return pMIC_pred, MIC_pred
'''
    with open(CONFIG["predict_script_path"], 'w', encoding='utf-8') as f:
        f.write(predict_script)
    print(f" 4. Prediction example script: {CONFIG['predict_script_path']}")
    print(f" 5. True‑pred scatter plot: {CONFIG['true_pred_plot']}")
    print(f"\n All files saved to root directory: {CONFIG['save_root']}")

def train_fixed_param_rf():
    print("=" * 70)
    print(" Full random forest training pipeline")
    print("=" * 70)
    X_train, X_test, y_train, y_test, scaler, feature_cols, data = load_and_preprocess_data()
    print("\n" + "=" * 50)
    print(" Initialize random forest regressor")
    print("=" * 50)
    rf_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    print(" Random forest key parameters:")
    key_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    for k in key_params:
        print(f"   - {k}: {rf_model.get_params()[k]}")
    print("\n Start model training...")
    rf_model.fit(X_train, y_train)
    print(" Model training completed!")
    metrics, y_test_true, y_test_pred = print_rf_performance(rf_model, X_train, X_test, y_train, y_test)
    plot_true_vs_pred(y_test_true, y_test_pred, metrics["Test R^2"])
    plot_mic_distribution(y_test_true, y_test_pred)
    save_all_files(scaler, rf_model, metrics)
    print("\n" + "=" * 70)
    print(f" All training procedures finished! All files saved to: {CONFIG['save_root']}")
    print("=" * 70)
    return rf_model, scaler, metrics

if __name__ == "__main__":
    trained_model, trained_scaler, model_metrics = train_fixed_param_rf()