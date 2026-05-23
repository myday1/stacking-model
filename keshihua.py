import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
import xgboost as xgb
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.patches as mpatches

# ======================
# 更新后的科研风格参数设置
# ======================
# 使用 Seaborn 风格，或者您可以选择其他样式
plt.style.use('seaborn-darkgrid')  # 改为使用 seaborn-darkgrid
sns.set_palette("husl")

# Configure matplotlib for publication quality
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

COLOR_PALETTE = {
    'main': '#1f77b4',
    'positive': '#2ca02c',
    'negative': '#d62728',
    'ci': '#aec7e8',
    'background': '#f8f9fa',
    'text': '#333333',
    'grid': '#cccccc',
}

# ======================
# 数据加载并转换列名为英文
# ======================

# ======================
# 数据加载
# ======================

# 列名映射：中文列名转换为英文
column_mapping = {
    'PM2.5': 'PM2_5',
    'PM10': 'PM10',
    'SO2': 'SO2',
    'CO': 'CO',
    'NO2': 'NO2',
    'O3_8h': 'O3_8h',
    '2米平均气温(°C)': 'Temp_2m',
    '降雨量(mm)': 'Rain_mm',
    '降雪量(cm)': 'Snow_cm',
    '总降水量(雨+雪)(mm)': 'Precip_mm',
    '10米平均风速(m/s)': 'WindSpeed_10m',
    '10米最大阵风(m/s)': 'WindGust_10m',
    '10米主导风向(°)': 'WindDir_10m',
    'FAO参考蒸发量(mm)': 'FAO_RefEvap_mm',
    '2米平均相对湿度(%)': 'RH_2m',
    '海平面气压平均值(hPa)': 'SeaPressure_hPa',
    '表面气压平均值(hPa)': 'SurfacePressure_hPa',
    '2米平均露点温度(°C)': 'DewPoint_2m'
}

# 假设已经加载了数据 `X_train`， `X_val`，`X_test` 是 numpy 数组
# 如果是 DataFrame 格式，可以使用：
# X_train_df = pd.DataFrame(X_train, columns=[column_mapping.get(c, c) for c in feature_names])

# ======================
# 计算相关性矩阵并绘制环状热力图
# ======================
# 计算相关性矩阵
corr_matrix = np.corrcoef(X_train, rowvar=False)

# 环状热力图
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True,
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_xticks(np.arange(len(column_mapping)))
ax.set_yticks(np.arange(len(column_mapping)))
ax.set_xticklabels(list(column_mapping.values()), rotation=45, ha='right')
ax.set_yticklabels(list(column_mapping.values()), rotation=0)
plt.title("Correlation Heatmap (Circular)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/circular_heatmap.eps", bbox_inches='tight')
plt.savefig("outputs/circular_heatmap.pdf", bbox_inches='tight')
plt.close()

# ======================
# SHAP 可解释性分析
# ======================
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(final_model)
shap_values_val = explainer(X_val)

# 获取特征列名
feature_names = [
    'PM2_5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h', 'Temp_2m', 'Rain_mm', 'Snow_cm',
    'Precip_mm', 'WindSpeed_10m', 'WindGust_10m', 'WindDir_10m', 'FAO_RefEvap_mm',
    'RH_2m', 'SeaPressure_hPa', 'SurfacePressure_hPa', 'DewPoint_2m', 'Month', 'Weekday',
    'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos', 'AQI_lag1', 'AQI_lag2',
    'AQI_lag3', 'AQI_lag7', 'AQI_roll3', 'AQI_roll7'
]

# ======================
# 图1：SHAP Summary Plot (Bar)
# ======================
print("Generating Figure 1: SHAP Summary Plot (Bar)...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_val, X_val, feature_names=feature_names,
                  plot_type='bar', show=False)
ax = plt.gca()
ax.set_xlabel("Mean |SHAP value| (Average impact on output)", fontsize=11, fontweight='bold')
ax.set_ylabel("Features", fontsize=11, fontweight='bold')
ax.set_title("SHAP Feature Importance (Mean Absolute SHAP)", fontsize=12, fontweight='bold')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/05_shap_summary_bar.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/05_shap_summary_bar.pdf", bbox_inches='tight', facecolor='white')
plt.close()

# ======================
# 图2：SHAP Summary Plot (Dot)
# ======================
print("Generating Figure 2: SHAP Summary Plot (Dot)...")
plt.figure(figsize=(11, 7))
X_val_df = pd.DataFrame(X_val, columns=feature_names)
shap.summary_plot(shap_values_val.values, X_val_df, feature_names=feature_names,
                  plot_type='dot', show=False, max_display=15)
ax = plt.gca()
ax.set_xlabel("SHAP value (impact on model output)", fontsize=11, fontweight='bold')
ax.set_title("SHAP Summary Plot: Feature Values and Impact", fontsize=12, fontweight='bold')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_shap_summary_dot.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/06_shap_summary_dot.pdf", bbox_inches='tight', facecolor='white')
plt.close()

# ======================
# 图3：SHAP Dependence Plots for Top 4 Features
# ======================
print("Generating Figure 3: SHAP Dependence Plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

top_features = [feature_names.index(f) for f in feature_names[:4]]  # 获取前四个特征的索引

for idx, feature_idx in enumerate(top_features):
    ax = axes[idx]

    # Create scatter plot
    feature_vals = X_val[:, feature_idx]
    shap_vals = shap_values_val.values[:, feature_idx]

    scatter = ax.scatter(feature_vals, shap_vals, c=feature_vals,
                         cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(f"{feature_names[feature_idx]} Value", fontsize=10, fontweight='bold')
    ax.set_ylabel("SHAP value", fontsize=10, fontweight='bold')
    ax.set_title(
        f"({'a' if idx == 0 else 'b' if idx == 1 else 'c' if idx == 2 else 'd'}) {feature_names[feature_idx]} Dependence",
        fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor(COLOR_PALETTE['background'])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Feature Value", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/07_shap_dependence.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/07_shap_dependence.pdf", bbox_inches='tight', facecolor='white')
plt.close()

# ======================
# 打印总结
# ======================
print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print("Generated files:")
print("  05_shap_summary_bar.eps, 05_shap_summary_bar.pdf - SHAP feature importance (Bar plot)")
print("  06_shap_summary_dot.eps, 06_shap_summary_dot.pdf - SHAP summary dot plot")
print("  07_shap_dependence.eps, 07_shap_dependence.pdf - SHAP dependence plots for top features")
print("=" * 60)
