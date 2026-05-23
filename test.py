import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams
import matplotlib.patches as mpatches

# 科研风格参数设置
#plt.style.use('seaborn-v0_8-darkgrid')  # 使用 seaborn 配色
sns.set_palette("husl")

# 配置 matplotlib，用于科研风格的可视化
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
})


# 数据加载与预处理
def load_data(file_path="processed/data_seq14.npz"):
    data = np.load(file_path, allow_pickle=True)
    return data


def load_best_model(params_path="outputs/optuna_xgb_study.pkl"):
    study = joblib.load(params_path)
    return study.best_params


def train_xgb_model(X_train, y_train, X_val, y_val, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, 'validation')],
                    early_stopping_rounds=30, verbose_eval=False)
    return bst


# 计算模型评估指标
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}


# SHAP分析
def shap_analysis(model, X_val, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # SHAP总结图（条形）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig("outputs/05_shap_summary_bar.eps", bbox_inches='tight', facecolor='white')
    plt.savefig("outputs/05_shap_summary_bar.pdf", bbox_inches='tight', facecolor='white')
    plt.close()

    # SHAP总结图（点）
    plt.figure(figsize=(11, 7))
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    shap.summary_plot(shap_values, X_val_df, feature_names=feature_names, plot_type='dot', show=False, max_display=15)
    plt.tight_layout()
    plt.savefig("outputs/06_shap_summary_dot.eps", bbox_inches='tight', facecolor='white')
    plt.savefig("outputs/06_shap_summary_dot.pdf", bbox_inches='tight', facecolor='white')
    plt.close()


# 环状热力图
def plot_circular_heatmap(corr_matrix, feature_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5,
                cbar_kws={"shrink": .8}, xticklabels=feature_names, yticklabels=feature_names)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/07_circular_heatmap.eps", bbox_inches='tight', facecolor='white')
    plt.savefig("outputs/07_circular_heatmap.pdf", bbox_inches='tight', facecolor='white')
    plt.close()


# 主函数：加载数据、训练模型、计算性能指标和SHAP分析
def main():
    # 1. 数据加载
    data = load_data("processed/data_seq14.npz")
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    feature_names = data['tab_features'].tolist()

    # 2. 加载XGBoost最优参数
    best_params = load_best_model("outputs/optuna_xgb_study.pkl")

    # 3. 训练XGBoost模型
    bst = train_xgb_model(X_train, y_train, X_val, y_val, best_params)

    # 4. 计算并打印评估指标
    y_pred_val = bst.predict(xgb.DMatrix(X_val))
    y_pred_test = bst.predict(xgb.DMatrix(X_test))
    metrics_val = calculate_metrics(y_val, y_pred_val)
    metrics_test = calculate_metrics(y_test, y_pred_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Validation Set:")
    for k, v in metrics_val.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTest Set:")
    for k, v in metrics_test.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)

    # 5. SHAP分析与可视化
    shap_analysis(bst, X_val, feature_names)

    # 6. 计算特征相关性并生成环状热力图
    corr_matrix = np.corrcoef(X_train.T)  # 转置，得到特征之间的相关性矩阵
    plot_circular_heatmap(corr_matrix, feature_names)


if __name__ == "__main__":
    main()
