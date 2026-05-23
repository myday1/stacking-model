import joblib
import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

# ======================
# 完全原版 无全局字体设置
# ======================
plt.style.use('default')
COLOR_PALETTE = {
    'main': '#3b5b92',
    'ci': '#8395b1',
    'positive': '#36a168',
    'negative': '#e05263',
    'data': '#e0e0e0',
    'zero_line': '#666666',
    'background': '#f9f9f9'
}

# ======================
# 数据模型 完全不动
# ======================
data = np.load("outputs/stack_data.npz", allow_pickle=True)
X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']

best_params = {
    'objective': 'reg:squarederror',
    'eta': 0.021000254758558302,
    'max_depth': 9,
    'subsample': 0.8781114273540895,
    'colsample_bytree':  0.9560255641008711,
    'lambda': 0.6769388565525195,
    'alpha':  0.0061977531306365834,
    'seed': 42
}

with open("processed/tab_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

feature_names = [
    'date', 'PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h',
    'Temp_2m', 'Rain_mm', 'Snow_cm', 'Precip_mm',
    'WindSpeed_10m', 'WindGust_10m', 'WindDir_10m',
    'FAO_RefEvap_mm', 'RH_2m', 'SeaPressure_hPa', 'SurfacePressure_hPa', 'DewPoint_2m',
    'Month', 'Weekday', 'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos',
    'AQI_lag1', 'AQI_lag2', 'AQI_lag3', 'AQI_lag7', 'AQI_roll3', 'AQI_roll7'
]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

final_model = xgb.train(best_params, dtrain, num_boost_round=500, evals=[(dval, 'validation')],
                        early_stopping_rounds=30, verbose_eval=False)

y_pred = final_model.predict(dval)
rmse, mae, r2 = np.sqrt(mean_squared_error(y_val, y_pred)), mean_absolute_error(y_val, y_pred), r2_score(y_val, y_pred)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_val)
X_val_df = pd.DataFrame(X_val, columns=feature_names)

os.makedirs("outputs1/figures_eps_pdf", exist_ok=True)

# ======================
# 图1
# ======================
plt.figure(figsize=(5,4))
plt.scatter(y_val, y_pred, color=COLOR_PALETTE['main'], alpha=0.6)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'k--', lw=1)
plt.xlabel("Observed AQI", fontsize=12, fontfamily="Times New Roman")
plt.ylabel("Predicted AQI", fontsize=12, fontfamily="Times New Roman")
plt.text(0.05, 0.9, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes, fontsize=11, fontfamily="Times New Roman")
plt.xticks(fontsize=11, fontfamily="Times New Roman")
plt.yticks(fontsize=11, fontfamily="Times New Roman")
plt.tight_layout()
plt.savefig("outputs1/figures_eps_pdf/Figure1_pred_vs_obs.eps", bbox_inches='tight')
plt.savefig("outputs1/figures_eps_pdf/Figure1_pred_vs_obs.pdf", bbox_inches='tight')
plt.close()

# ======================
# 图2
# ======================
shap.summary_plot(shap_values.values, X_val_df, show=False)
plt.xlabel("SHAP value", fontsize=12, fontfamily="Times New Roman")
for t in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    t.set_fontsize(11)
    t.set_fontfamily("Times New Roman")
plt.tight_layout()
plt.savefig("outputs1/figures_eps_pdf/Figure2_SHAP_summary.eps", bbox_inches='tight')
plt.savefig("outputs1/figures_eps_pdf/Figure2_SHAP_summary.pdf", bbox_inches='tight')
plt.close()

# ======================
# 图3 完全不改动你的代码！只加字体！
# ======================
# 图3：SHAP Heatmap
# ======================

# 设置Times New Roman字体（无例外）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 12    # 坐标轴标签：12pt
plt.rcParams['axes.titlesize'] = 13     # 标题：13pt
plt.rcParams['xtick.labelsize'] = 11    # X轴刻度：11pt
plt.rcParams['ytick.labelsize'] = 11    # Y轴刻度：11pt
plt.rcParams['legend.fontsize'] = 11    # 图例：11pt

# 生成热图
shap.plots.heatmap(shap.Explanation(values=shap_values.values, data=X_val_df.values, feature_names=feature_names), show=False)

ax = plt.gca()

# 设置标题
title = ax.get_title()
if not title:
    title = "SHAP Heatmap"
ax.set_title(title, fontsize=13, fontname='Times New Roman')

# 确保所有刻度标签使用Times New Roman
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(11)
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(11)

# 色条(colorbar)字体设置
if ax.get_children():
    for child in ax.get_children():
        if hasattr(child, 'set_fontname'):
            child.set_fontname('Times New Roman')
        if hasattr(child, 'set_fontsize'):
            child.set_fontsize(11)

plt.tight_layout()
plt.savefig("outputs1/Figure3_SHAP_heatmap.eps", bbox_inches='tight')
plt.savefig("outputs1/Figure3_SHAP_heatmap.pdf", bbox_inches='tight')
plt.close()
# ======================
# 图4-10
# ======================
importance = np.mean(np.abs(shap_values.values), axis=0)
top_idx = np.argsort(importance)[-7:][::-1]
top_features = [feature_names[i] for i in top_idx]

for i, f in enumerate(top_features, 4):
    shap.dependence_plot(f, shap_values.values, X_val_df, show=False)
    plt.xlabel(f, fontsize=12, fontfamily="Times New Roman")
    plt.ylabel(f"SHAP value for {f}", fontsize=12, fontfamily="Times New Roman")
    for t in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        t.set_fontsize(11)
        t.set_fontfamily("Times New Roman")
    plt.tight_layout()
    plt.savefig(f"outputs1/figures_eps_pdf/Figure{i}_SHAP_{f}.eps", bbox_inches='tight')
    plt.savefig(f"outputs1/figures_eps_pdf/Figure{i}_SHAP_{f}.pdf", bbox_inches='tight')
    plt.close()

print("✅ 完成！全部保存到 outputs1/figures_eps_pdf/")