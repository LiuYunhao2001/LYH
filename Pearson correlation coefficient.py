import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

CONFIG = {
    "train_data_path": r"G:\model2date\pmic_filtered_double_train_data.csv",
    "save_root": r"C:\Users\lyh\Desktop\ML-date",
    "corr_heatmap_path": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_feature_top30_corr_heatmap.png"),
    "corr_matrix_excel": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_feature_top30_corr_matrix.xlsx"),
    "feature_importance_excel": os.path.join(r"C:\Users\lyh\Desktop\ML-date", "rf_feature_importance_top30.xlsx")
}

EXCLUDE_COLS = ['smiles', 'pMIC', 'residual', 'MIC']

HEATMAP_CONFIG = {
    "figsize": (12, 10),
    "tick_fontsize": 8,
    "save_dpi": 600,
    "corr_threshold": 0.8,
    "linewidths": 0.2,
    "linecolor": "#F5F5F5"
}

COLORS = ["#3D50C4", "#7497f4", "#FFFFFF", "#E27557", "#B7132A"]
cmap = LinearSegmentedColormap.from_list("rf_corr_cmap", COLORS)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = HEATMAP_CONFIG["tick_fontsize"]
plt.rcParams['ytick.labelsize'] = HEATMAP_CONFIG["tick_fontsize"]

os.makedirs(CONFIG["save_root"], exist_ok=True)

def load_rf_train_features():
    print("Step 1: Loading training data and features")
    if not os.path.exists(CONFIG["train_data_path"]):
        raise FileNotFoundError(f"Training data not found: {CONFIG['train_data_path']}")
    data = pd.read_csv(CONFIG["train_data_path"], encoding='utf-8-sig')
    required_cols = ['smiles', 'pMIC']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    feature_cols = [col for col in data.columns if col not in EXCLUDE_COLS]
    X = data[feature_cols].copy()
    y = data['pMIC'].values
    median_values = np.nanmedian(X.values, axis=0)
    nan_mask = np.isnan(X.values)
    X.values[nan_mask] = np.take(median_values, np.where(nan_mask)[1])
    print(f"Data loaded. Features: {len(feature_cols)}, Samples: {X.shape[0]}")
    return X, feature_cols, y

def get_rf_top30_important_features(X, y):
    print("Step 2: Training random forest and extracting top 30 features")
    rf = RandomForestRegressor(
        n_estimators=250, max_depth=20, min_samples_split=3, min_samples_leaf=2,
        max_features='sqrt', bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top30 = importance.head(30).index.tolist()
    X_top30 = X[top30].copy()
    importance.reset_index().rename(columns={"index": "Feature_Name"}).to_excel(
        CONFIG["feature_importance_excel"], index=False, sheet_name="RF_Feature_Importance"
    )
    print(f"Top 30 features saved to {CONFIG['feature_importance_excel']}")
    return X_top30, top30, importance

def plot_corr_heatmap(X_top30, top30_feats):
    print("Step 3: Computing correlation matrix and plotting heatmap")
    corr = X_top30.corr(method="pearson")
    plt.figure(figsize=HEATMAP_CONFIG["figsize"])
    ax = sns.heatmap(
        corr, cmap=cmap, center=0, vmin=-1, vmax=1, annot=False,
        linewidths=HEATMAP_CONFIG["linewidths"], linecolor=HEATMAP_CONFIG["linecolor"],
        square=True, cbar_kws={"shrink": 0.9, "label": "Pearson Correlation"}
    )
    ax.set_xticklabels(top30_feats, rotation=45, ha="right")
    ax.set_yticklabels(top30_feats, rotation=0)
    plt.title("Random Forest - Top 30 Features Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(CONFIG["corr_heatmap_path"], dpi=HEATMAP_CONFIG["save_dpi"], facecolor="white")
    plt.close()
    corr.to_excel(CONFIG["corr_matrix_excel"], index=True, sheet_name="Corr_Matrix")
    print(f"Heatmap saved to {CONFIG['corr_heatmap_path']}")
    print(f"Correlation matrix saved to {CONFIG['corr_matrix_excel']}")
    return corr

def stat_high_corr_pairs(corr_matrix, threshold=0.8):
    print(f"Step 4: Identifying highly correlated pairs (|r| >= {threshold})")
    corr_no_diag = corr_matrix.mask(np.eye(len(corr_matrix)) == 1)
    pairs = corr_no_diag.unstack()
    pairs = pairs[np.abs(pairs) >= threshold]
    pairs = pairs.sort_values(ascending=False)
    seen = set()
    high_pairs = []
    for (f1, f2), r in pairs.items():
        if (f2, f1) not in seen and f1 != f2:
            seen.add((f1, f2))
            high_pairs.append((f1, f2, round(r, 4)))
    if high_pairs:
        print(f"Found {len(high_pairs)} highly correlated pairs:")
        for i, (f1, f2, r) in enumerate(high_pairs, 1):
            print(f"  {i}. {f1} <-> {f2} : r = {r}")
    else:
        print("No highly correlated pairs found.")
    return high_pairs

def main():
    try:
        X, all_feats, y = load_rf_train_features()
        X_top30, top30_feats, imp = get_rf_top30_important_features(X, y)
        corr_mat = plot_corr_heatmap(X_top30, top30_feats)
        high_pairs = stat_high_corr_pairs(corr_mat, HEATMAP_CONFIG["corr_threshold"])
        print("\nAll tasks completed successfully.")
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    main()
