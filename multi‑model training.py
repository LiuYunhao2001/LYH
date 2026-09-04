from sklearn.model_selection import RepeatedKFold
import os
import datetime
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

def load_filtered_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Filtered training set does not exist: {file_path}, please check the path!")
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        data = pd.read_excel(file_path)
    print(f"Load filtered training set successfully | Shape: {data.shape}")
    print(f"Column preview: {data.columns.tolist()[:5]}...{data.columns.tolist()[-5:]}")
    required_cols = ['smiles', 'pMIC']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col} in filtered data!")
    exclude_cols = ['smiles', 'pMIC', 'residual', 'MIC']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    target_col = 'pMIC'
    print(f"Feature columns count: {len(feature_cols)} (molecular descriptors)")
    print(f"Target column: {target_col} (antibacterial activity index)")
    data = data.dropna(subset=[target_col])
    print(f"Data count after removing NaN in target: {len(data)}")
    return data, feature_cols, target_col

if __name__ == '__main__':
    random_state = 42
    np.random.seed(random_state)
    print("="*60)
    print("6‑Models Training Start (Ultra Fast + No Normalization)")
    print("="*60)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = r"C:\Users\lyh\Desktop\ML-date"
    os.makedirs(root_dir, exist_ok=True)
    output_dir = os.path.join(root_dir, f"6models_ultrafast_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    FILTERED_DATA_PATH = r"G:\model2date\pmic_filtered_double_train_data.csv"
    try:
        data, feature_cols, target_col = load_filtered_data(FILTERED_DATA_PATH)
    except Exception as e:
        print(f"Data loading failed: {e}")
        exit()
    X = data[feature_cols].values
    y = data[target_col].values
    print(f"Data shape: X={X.shape} | y={y.shape} (original data, no normalization)")
    print("\nData Preprocessing (Only missing value imputation, no normalization)")
    median_values = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(median_values, np.where(nan_mask)[1])
    print("Missing value imputation completed (median fill)")
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nDataset Split (80% Train / 20% Test):")
    print(f"   Train set: X_train={X_train.shape} | y_train={y_train.shape}")
    print(f"   Test set: X_test={X_test.shape} | y_test={y_test.shape}")
    cv_n_splits = min(5, len(X_train))
    cv = RepeatedKFold(n_splits=cv_n_splits, n_repeats=3, random_state=random_state)
    print(f"\nCross‑validation setup: {cv_n_splits}‑fold × 3 repeats")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=random_state),
        'XGBoost': XGBRegressor(
            objective='reg:squarederror', random_state=random_state,
            max_depth=3, learning_rate=0.05, subsample=0.8
        ),
        'LightGBM': LGBMRegressor(random_state=random_state, verbose=-1),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10,
            random_state=random_state
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=250,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=random_state,
            n_jobs=-1
        )
    }
    print(f"\nNumber of models to train: {len(models)} (all ultra fast, no stuck)")
    for idx, model_name in enumerate(models.keys(), 1):
        print(f"   {idx}. {model_name}")
    print("\n" + "="*50)
    print("Start Ultra Fast Models Training and Evaluation")
    print("="*50)
    model_metrics = {}
    y_pred_dict = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...", end=' ')
        try:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train):
                fold_model = clone(model)
                fold_model.fit(X_train[train_idx], y_train[train_idx])
                val_pred = fold_model.predict(X_train[val_idx])
                cv_scores.append(r2_score(y_train[val_idx], val_pred))
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            model_metrics[model_name] = {
                'Train R2': train_r2, 'Test R2': test_r2,
                'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                'CV R2 Mean': cv_mean, 'CV R2 Std': cv_std
            }
            y_pred_dict[model_name] = {
                'y_test_true': y_test, 'y_test_pred': y_test_pred
            }
            print(f"Done")
            print(f"   Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f} | CV R2: {cv_mean:.4f}±{cv_std:.4f}")
        except Exception as e:
            print(f"Failed: {str(e)[:50]}...")
            continue
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.round(4)
        metrics_df.to_csv(f"{output_dir}/6models_ultrafast_performance.csv", encoding='utf-8-sig')
        print(f"\nModel performance table saved: {output_dir}/6models_ultrafast_performance.csv")
        best_model_name = metrics_df['Test R2'].idxmax()
        best_model = models[best_model_name]
        print(f"Best model: {best_model_name} (Test R2: {metrics_df.loc[best_model_name, 'Test R2']:.4f})")
        model_save_name = best_model_name.replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(best_model, f"{output_dir}/best_model_{model_save_name}.pkl")
        print(f"Best model saved: {output_dir}/best_model_{model_save_name}.pkl")
        print(f"\nPlotting 6‑models performance comparison (R2 + RMSE)...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sorted_model = metrics_df.sort_values('Test R2', ascending=False)
        model_names = sorted_model.index.tolist()
        bars1 = ax1.bar(model_names, sorted_model['Test R2'], color='#1f77b4', alpha=0.8, edgecolor='black')
        ax1.set_title('6‑Models Test Set R2 Score Comparison', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Model Name', fontsize=12)
        ax1.set_ylabel('R2 Score (Closer to 1 is better)', fontsize=12)
        ax1.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(sorted_model['Test R2'])*1.1)
        add_value_labels(bars1)
        bars2 = ax2.bar(model_names, sorted_model['RMSE'], color='#ff7f0e', alpha=0.8, edgecolor='black')
        ax2.set_title('6‑Models Test Set RMSE Comparison', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Model Name', fontsize=12)
        ax2.set_ylabel('RMSE (Smaller is better)', fontsize=12)
        ax2.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        add_value_labels(bars2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/6models_R2_RMSE_comparison.png")
        plt.close()
        print(f"Plotting {best_model_name} True vs Predicted pMIC...")
        best_pred = y_pred_dict[best_model_name]
        y_test_true = best_pred['y_test_true']
        y_test_pred = best_pred['y_test_pred']
        z = np.polyfit(y_test_true, y_test_pred, 1)
        p = np.poly1d(z)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_true, y_test_pred, color='#2ca02c', alpha=0.6, s=30, edgecolor='black')
        plt.plot(y_test_true, p(y_test_true), "r--", linewidth=2, label=f'Fitting Line: y={z[0]:.2f}x+{z[1]:.2f}')
        plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'k-', linewidth=2, label='Perfect Fitting Line')
        plt.title(f'{best_model_name} - True pMIC vs Predicted pMIC', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('True pMIC Value', fontsize=12)
        plt.ylabel('Predicted pMIC Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.text(0.05, 0.95, f'R2 = {r2_score(y_test_true, y_test_pred):.4f}',
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plot_save_name = best_model_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f"{output_dir}/{plot_save_name}_true_vs_pred.png")
        plt.close()
        print(f"Plotting {best_model_name} MIC Distribution (0.1‑256μg/mL)...")
        mic_test_true = 10 ** (-y_test_true)
        mic_test_pred = 10 ** (-y_test_pred)
        mic_test_true = np.clip(mic_test_true, 0.1, 256)
        mic_test_pred = np.clip(mic_test_pred, 0.1, 256)
        plt.figure(figsize=(10, 6))
        sns.histplot(mic_test_pred, bins=20, color='#d62728', alpha=0.7, edgecolor='black', label='Predicted MIC')
        sns.histplot(mic_test_true, bins=20, color='#1f77b4', alpha=0.5, edgecolor='black', label='True MIC')
        plt.title(f'{best_model_name} - MIC Value Distribution (0.1‑256μg/mL)', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('MIC Value (μg/mL)', fontsize=12)
        plt.ylabel('Sample Count', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xlim(0, 260)
        plt.savefig(f"{output_dir}/{plot_save_name}_MIC_distribution.png")
        plt.close()
    print("\n" + "="*60)
    print(f"6‑Ultra‑Fast Models Training Completed! All results saved to: {output_dir}")
    print("="*60)
