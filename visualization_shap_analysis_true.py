import joblib
import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from matplotlib import rcParams
import matplotlib.patches as mpatches
from scipy import stats
import pickle
# ======================
# 科研风格参数设置
# ======================
plt.style.use('default')
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.7,
    'lines.markersize': 5,
    'savefig.dpi': 1200,
    'savefig.format': 'eps',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
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
# 数据加载与特征列名定义（使用真实列名）
# ======================
print("Loading data...")
data = np.load("outputs/stack_data.npz", allow_pickle=True)
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

# 加载最优参数
# ======================
# 自定义 XGBoost 参数
# ======================
best_params = {
    'objective': 'reg:squarederror',  # 回归任务
    'eta': 0.15637598896156554,       # 学习率
    'max_depth': 3,                   # 树深度
    'subsample': 0.5066097413529567,  # 行采样比例
    'colsample_bytree': 0.7308999142823026,  # 列采样比例
    'lambda': 0.0027469781159004457,  # L2 正则化
    'alpha': 4.025804732341239e-06,   # L1 正则化
    'seed': 42                        # 随机种子，保证可复现
}

print("✅ Using custom fixed parameters instead of Optuna:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

#study = joblib.load("outputs/optuna_xgb_study.pkl")
#best_params = study.best_params
#print(f"✓ Best hyperparameters loaded: {best_params}")

# 定义真实特征列名（覆盖pickle加载，确保一致性）
feature_names = [
    'Date', 'PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h',
    'Temp_2m', 'Rain_mm', 'Snow_cm', 'Precip_mm',
    'WindSpeed_10m', 'WindGust_10m', 'WindDir_10m',
    'FAO_RefEvap_mm', 'RH_2m', 'SeaPressure_hPa', 'SurfacePressure_hPa', 'DewPoint_2m',
    'Month', 'Weekday', 'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos',
    'AQI_lag1', 'AQI_lag2', 'AQI_lag3', 'AQI_lag7', 'AQI_roll3', 'AQI_roll7'
]
print("特征列名：", feature_names)
# 确保特征列数与数据维度匹配
assert len(feature_names) == X_train.shape[1], f"特征列数({len(feature_names)})与数据维度({X_train.shape[1]})不匹配"

# ======================
# 训练最终模型
# ======================
print("\nTraining final XGBoost model...")
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, 'validation'), (dtest, 'test')],
    early_stopping_rounds=30,
    verbose_eval=False
)

y_pred_val = final_model.predict(dval)
y_pred_test = final_model.predict(dtest)

# ======================
# 模型评估指标
# ======================
def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # 避免除以零
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    r2 = r2_score(y_true, y_pred)
    # 相关系数
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2, 'Correlation': corr}

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

# ======================
# 图1：Observed vs Predicted (Validation & Test)
# ======================
print("\nGenerating Figure 1: Observed vs Predicted...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Validation set
axes[0].plot(range(len(y_val)), y_val, 'o-', label='Observed',
             color=COLOR_PALETTE['main'], linewidth=2, markersize=4, alpha=0.7)
axes[0].plot(range(len(y_val)), y_pred_val, 's-', label='Predicted',
             color=COLOR_PALETTE['positive'], linewidth=2, markersize=4, alpha=0.7)
axes[0].fill_between(range(len(y_val)), y_pred_val - 5, y_pred_val + 5,
                     color=COLOR_PALETTE['ci'], alpha=0.2, label='±5 margin')
axes[0].set_xlabel("Sample Index", fontsize=11, fontweight='bold')
axes[0].set_ylabel("AQI", fontsize=11, fontweight='bold')
axes[0].set_title(f"(a) Validation Set\nR² = {metrics_val['R²']:.3f}, RMSE = {metrics_val['RMSE']:.3f}",
                  fontsize=11, fontweight='bold')
axes[0].legend(loc='best', framealpha=0.95)
axes[0].grid(True, linestyle='--', alpha=0.4, color=COLOR_PALETTE['grid'])
axes[0].set_facecolor(COLOR_PALETTE['background'])

# Test set
axes[1].plot(range(len(y_test)), y_test, 'o-', label='Observed',
             color=COLOR_PALETTE['main'], linewidth=2, markersize=4, alpha=0.7)
axes[1].plot(range(len(y_test)), y_pred_test, 's-', label='Predicted',
             color=COLOR_PALETTE['positive'], linewidth=2, markersize=4, alpha=0.7)
axes[1].fill_between(range(len(y_test)), y_pred_test - 5, y_pred_test + 5,
                     color=COLOR_PALETTE['ci'], alpha=0.2, label='±5 margin')
axes[1].set_xlabel("Sample Index", fontsize=11, fontweight='bold')
axes[1].set_ylabel("AQI", fontsize=11, fontweight='bold')
axes[1].set_title(f"(b) Test Set\nR² = {metrics_test['R²']:.3f}, RMSE = {metrics_test['RMSE']:.3f}",
                  fontsize=11, fontweight='bold')
axes[1].legend(loc='best', framealpha=0.95)
axes[1].grid(True, linestyle='--', alpha=0.4, color=COLOR_PALETTE['grid'])
axes[1].set_facecolor(COLOR_PALETTE['background'])

plt.tight_layout()
# 同时保存EPS和PDF格式
plt.savefig("outputs/01_pred_vs_true.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/01_pred_vs_true.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: outputs/01_pred_vs_true.eps & 01_pred_vs_true.pdf")

# ======================
# 图2：Scatter Plot with Regression Line
# ======================
print("Generating Figure 2: Scatter Plot with Regression...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (ax, y_true, y_pred, title_suffix) in enumerate([
    (axes[0], y_val, y_pred_val, "Validation"),
    (axes[1], y_test, y_pred_test, "Test")
]):
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color=COLOR_PALETTE['main'], edgecolors='black', linewidth=0.5)

    # 完美预测线
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

    # 回归线
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), color=COLOR_PALETTE['positive'], linewidth=2.5, label='Fitted Line')

    metrics_dict = metrics_val if idx == 0 else metrics_test
    ax.set_xlabel("Observed AQI", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted AQI", fontsize=11, fontweight='bold')
    ax.set_title(
        f"({'a' if idx == 0 else 'b'}) {title_suffix} Set\nR² = {metrics_dict['R²']:.3f}, MAE = {metrics_dict['MAE']:.3f}",
        fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.4, color=COLOR_PALETTE['grid'])
    ax.set_facecolor(COLOR_PALETTE['background'])

plt.tight_layout()
plt.savefig("outputs/02_scatter_regression.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/02_scatter_regression.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: outputs/02_scatter_regression.eps & 02_scatter_regression.pdf")

# ======================
# 图3：Residuals Distribution
# ======================
print("Generating Figure 3: Residuals Analysis...")
residuals_val = y_val - y_pred_val
residuals_test = y_test - y_pred_test

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Residuals vs Predicted (Validation)
axes[0, 0].scatter(y_pred_val, residuals_val, alpha=0.6, s=30,
                   color=COLOR_PALETTE['main'], edgecolors='black', linewidth=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel("Predicted AQI", fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel("Residuals", fontsize=10, fontweight='bold')
axes[0, 0].set_title("(a) Residuals vs Predicted (Validation)", fontsize=10, fontweight='bold')
axes[0, 0].grid(True, linestyle='--', alpha=0.4)
axes[0, 0].set_facecolor(COLOR_PALETTE['background'])

# Residuals Distribution (Validation)
axes[0, 1].hist(residuals_val, bins=30, color=COLOR_PALETTE['main'],
                alpha=0.7, edgecolor='black', linewidth=1)
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel("Residuals", fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel("Frequency", fontsize=10, fontweight='bold')
axes[0, 1].set_title(
    f"(b) Residuals Distribution (Validation)\nMean = {np.mean(residuals_val):.3f}, Std = {np.std(residuals_val):.3f}",
    fontsize=10, fontweight='bold')
axes[0, 1].grid(True, linestyle='--', alpha=0.4, axis='y')
axes[0, 1].set_facecolor(COLOR_PALETTE['background'])

# Residuals vs Predicted (Test)
axes[1, 0].scatter(y_pred_test, residuals_test, alpha=0.6, s=30,
                   color=COLOR_PALETTE['positive'], edgecolors='black', linewidth=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel("Predicted AQI", fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel("Residuals", fontsize=10, fontweight='bold')
axes[1, 0].set_title("(c) Residuals vs Predicted (Test)", fontsize=10, fontweight='bold')
axes[1, 0].grid(True, linestyle='--', alpha=0.4)
axes[1, 0].set_facecolor(COLOR_PALETTE['background'])

# Residuals Distribution (Test)
axes[1, 1].hist(residuals_test, bins=30, color=COLOR_PALETTE['positive'],
                alpha=0.7, edgecolor='black', linewidth=1)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel("Residuals", fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel("Frequency", fontsize=10, fontweight='bold')
axes[1, 1].set_title(
    f"(d) Residuals Distribution (Test)\nMean = {np.mean(residuals_test):.3f}, Std = {np.std(residuals_test):.3f}",
    fontsize=10, fontweight='bold')
axes[1, 1].grid(True, linestyle='--', alpha=0.4, axis='y')
axes[1, 1].set_facecolor(COLOR_PALETTE['background'])

plt.tight_layout()
plt.savefig("outputs/03_residuals_analysis.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/03_residuals_analysis.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: outputs/03_residuals_analysis.eps & 03_residuals_analysis.pdf")

# ======================
# 图4：Feature Importance（真实列名）
# ======================
print("Generating Figure 4: Feature Importance...")
feature_importance = final_model.get_score(importance_type='weight')
# 过滤掉不存在的特征（避免键错误）
feature_importance = {k: v for k, v in feature_importance.items() if k in feature_names}
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]

fig, ax = plt.subplots(figsize=(10, 6))
features, importances = zip(*sorted_importance)
y_pos = np.arange(len(features))

bars = ax.barh(y_pos, importances, color=COLOR_PALETTE['main'], alpha=0.8, edgecolor='black', linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=10)  # 显示真实特征名
ax.set_xlabel("Feature Importance (Weight)", fontsize=11, fontweight='bold')
ax.set_title("Top 15 Features by Importance", fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4, axis='x', color=COLOR_PALETTE['grid'])
ax.set_facecolor(COLOR_PALETTE['background'])

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, importances)):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f'{int(val)}',
            va='center', ha='left', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/04_feature_importance.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/04_feature_importance.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: outputs/04_feature_importance.eps & 04_feature_importance.pdf")

# ======================
# SHAP 可解释性分析（使用真实列名）
# ======================
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(final_model)
shap_values_val = explainer(X_val)

# 确保SHAP使用真实特征名
X_val_df = pd.DataFrame(X_val, columns=feature_names)

# ======================
# 图5：SHAP Summary Plot (Bar)（真实列名）
# ======================
print("Generating Figure 5: SHAP Summary Plot (Bar)...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_val, X_val_df, feature_names=feature_names,
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
print("✓ Saved: outputs/05_shap_summary_bar.eps & 05_shap_summary_bar.pdf")

# ======================
# 图6：SHAP Summary Plot (Dot)（真实列名）
# ======================
print("Generating Figure 6: SHAP Summary Plot (Dot)...")
plt.figure(figsize=(11, 7))
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
print("✓ Saved: outputs/06_shap_summary_dot.eps & 06_shap_summary_dot.pdf")

# ======================
# 图7：SHAP Dependence Plots（真实列名，Top4重要特征）
# ======================
print("Generating Figure 7: SHAP Dependence Plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 获取SHAP Top4重要特征（基于mean absolute SHAP）
shap_importance = np.abs(shap_values_val.values).mean(axis=0)
top4_idx = np.argsort(shap_importance)[-4:][::-1]  # 降序排列
top4_features = [feature_names[i] for i in top4_idx]

for idx, (feature_idx, feature_name) in enumerate(zip(top4_idx, top4_features)):
    ax = axes[idx]

    # 绘制依赖图
    feature_vals = X_val[:, feature_idx]
    shap_vals = shap_values_val.values[:, feature_idx]

    scatter = ax.scatter(feature_vals, shap_vals, c=feature_vals,
                         cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(f"{feature_name} Value", fontsize=10, fontweight='bold')  # 显示真实特征名
    ax.set_ylabel("SHAP value", fontsize=10, fontweight='bold')
    ax.set_title(
        f"({'a' if idx == 0 else 'b' if idx == 1 else 'c' if idx == 2 else 'd'}) {feature_name} Dependence",
        fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor(COLOR_PALETTE['background'])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"{feature_name} Value", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/07_shap_dependence.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/07_shap_dependence.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: outputs/07_shap_dependence.eps & 07_shap_dependence.pdf")

# ======================
# 图8：SHAP Force Plot (多个样本)
# ======================
print("Generating Figure 8: SHAP Force Plot...")
try:
    # Force plot保存为PDF（EPS对复杂图形兼容性差）
    shap_html = shap.plots.force(explainer.expected_value, shap_values_val.values[:100],
                                 X_val_df.iloc[:100], show=False)
    # 先保存为HTML，再转换为PDF（推荐方式）
    shap.save_html("outputs/08_shap_force.html", shap_html)
    # 同时尝试直接保存为PDF（视环境兼容性）
    plt.figure(figsize=(15, 5))
    shap.plots.force(explainer.expected_value, shap_values_val.values[:100],
                     X_val_df.iloc[:100], show=False, matplotlib=True)
    plt.tight_layout()
    plt.savefig("outputs/08_shap_force.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: outputs/08_shap_force.html & 08_shap_force.pdf")
except Exception as e:
    print(f"⚠ SHAP Force Plot skipped: {str(e)}")

# ======================
# 图9：Performance Metrics Summary Table
# ======================
print("Generating Figure 9: Metrics Summary Table...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

metrics_data = [
    ['Metric', 'Validation', 'Test'],
    ['RMSE', f"{metrics_val['RMSE']:.4f}", f"{metrics_test['RMSE']:.4f}"],
    ['MAE', f"{metrics_val['MAE']:.4f}", f"{metrics_test['MAE']:.4f}"],
    ['MAPE (%)', f"{metrics_val['MAPE']:.4f}", f"{metrics_test['MAPE']:.4f}"],
    ['R²', f"{metrics_val['R²']:.4f}", f"{metrics_test['R²']:.4f}"],
    ['Correlation', f"{metrics_val['Correlation']:.4f}", f"{metrics_test['Correlation']:.4f}"],
]

table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                 colWidths=[0.35, 0.3, 0.3])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 美化表头
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_edgecolor('black')
    table[(0, i)].set_linewidth(1.5)

# 美化数据行
for i in range(1, len(metrics_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')
        table[(i, j)].set_edgecolor('black')
        table[(i, j)].set_linewidth(1)

plt.title("Model Performance Metrics Summary", fontsize=13, fontweight='bold', pad=20)
plt.savefig("outputs/09_metrics_summary.eps", bbox_inches='tight', facecolor='white')
plt.savefig("outputs/09_metrics_summary.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: outputs/09_metrics_summary.eps & 09_metrics_summary.pdf")

# ======================
# 打印总结
# ======================
print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print("Generated files (each figure has EPS and PDF format):")
print("  01_pred_vs_true - Time series comparison")
print("  02_scatter_regression - Scatter plots with regression lines")
print("  03_residuals_analysis - Residuals distribution & patterns")
print("  04_feature_importance - Feature importance ranking (real names)")
print("  05_shap_summary_bar - SHAP mean importance (real names)")
print("  06_shap_summary_dot - SHAP detailed impact plot (real names)")
print("  07_shap_dependence - Top4 features dependence analysis (real names)")
print("  08_shap_force - Individual prediction explanations")
print("  09_metrics_summary - Performance metrics table")
print("=" * 60)