import joblib
import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams
import os
import pickle
# ======================
# 图像与科研参数设置
# ======================
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.7,
    'lines.markersize': 5,
    'savefig.dpi': 600,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
})
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
# 数据与模型加载
# ======================
data = np.load("outputs/stack_data.npz", allow_pickle=True)
X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']


# ======================
# 使用自定义参数（替代 Optuna）
# ======================
best_params = {
#'eta': 0.021000254758558302, 'max_depth': 9, 'subsample': 0.8781114273540895, 'colsample_bytree': 0.9560255641008711, 'lambda': 0.6769388565525195, 'alpha': 0.0061977531306365834
    'objective': 'reg:squarederror',
    'eta': 0.021000254758558302,
    'max_depth': 9,
    'subsample': 0.8781114273540895,
    'colsample_bytree':  0.9560255641008711,
    'lambda': 0.6769388565525195,
    'alpha':  0.0061977531306365834,
    'seed': 42
}

print("✅ Using manually defined XGBoost parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
#study = joblib.load("outputs/optuna_xgb_study.pkl")
#best_params = study.best_params
#print("✅ Loaded best params:", best_params)


with open("processed/tab_features.pkl", "rb") as f:
    feature_names = pickle.load(f)  # 英文列名列表

print("特征列名：", feature_names)
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
print(f"📊 Model Performance: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

# ======================
# SHAP计算
# ======================
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_val)
X_val_df = pd.DataFrame(X_val, columns=feature_names)

# ======================
# 输出目录
# ======================
os.makedirs("outputs/figures_eps_pdf", exist_ok=True)

# ======================
# 图1：实测 vs 预测
# ======================
plt.figure(figsize=(5,4))
plt.scatter(y_val, y_pred, color=COLOR_PALETTE['main'], alpha=0.6)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'k--', lw=1)
plt.xlabel("Observed AQI")
plt.ylabel("Predicted AQI")
plt.text(0.05, 0.9, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig("outputs/figures_eps_pdf/Figure1_pred_vs_obs.eps", bbox_inches='tight')
plt.savefig("outputs/figures_eps_pdf/Figure1_pred_vs_obs.pdf", bbox_inches='tight')
plt.close()

# ======================
# 图2：SHAP Summary Plot
# ======================
shap.summary_plot(shap_values.values, X_val_df, show=False)
plt.tight_layout()
plt.savefig("outputs/figures_eps_pdf/Figure2_SHAP_summary.eps", bbox_inches='tight')
plt.savefig("outputs/figures_eps_pdf/Figure2_SHAP_summary.pdf", bbox_inches='tight')
plt.close()

# ======================
# 图3：SHAP Heatmap
# ======================
shap.plots.heatmap(shap.Explanation(values=shap_values.values, data=X_val_df.values, feature_names=feature_names), show=False)
plt.tight_layout()
plt.savefig("outputs/figures_eps_pdf/Figure3_SHAP_heatmap.eps", bbox_inches='tight')
plt.savefig("outputs/figures_eps_pdf/Figure3_SHAP_heatmap.pdf", bbox_inches='tight')
plt.close()

# ======================
# 图4–10：特征SHAP Dependence（Top 7重要特征）
# ======================
importance = np.mean(np.abs(shap_values.values), axis=0)
top_idx = np.argsort(importance)[-7:][::-1]
top_features = [feature_names[i] for i in top_idx]

for i, f in enumerate(top_features, 4):
    shap.dependence_plot(f, shap_values.values, X_val_df, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/figures_eps_pdf/Figure{i}_SHAP_{f}.eps", bbox_inches='tight')
    plt.savefig(f"outputs/figures_eps_pdf/Figure{i}_SHAP_{f}.pdf", bbox_inches='tight')
    plt.close()

print("✅ 已生成全部 10 张高分辨率科研级图 (EPS + PDF) 位于 outputs/figures_eps_pdf/")