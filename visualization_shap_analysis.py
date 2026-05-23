import joblib
import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams

# ======================
# 科研风格参数设置
# ======================
plt.style.use('default')
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.linewidth': 2,
    'grid.linewidth': 0.7,
    'lines.markersize': 6,
    'savefig.dpi': 1200,
    'savefig.format': 'eps',
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
    'tipping_point': '#f0746e',
    'background': '#f9f9f9'
}

# ======================
# 数据加载
# ======================
data = np.load("outputs/stack_data.npz", allow_pickle=True)
X_train = data['X_train']
X_val = data['X_val']
y_train = data['y_train']
y_val = data['y_val']

# 载入Optuna最优参数
study = joblib.load("outputs/optuna_xgb_study.pkl")
best_params = study.best_params
print("✅ Loaded best params:", best_params)

# ======================
# 训练最终模型
# ======================
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, 'validation')],
    early_stopping_rounds=30,
    verbose_eval=False
)

y_pred = final_model.predict(dval)

# ======================
# 模型评估
# ======================
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"📊 XGBoost Final Model - RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

# ======================
# 图1：实测 vs 预测
# ======================
plt.figure(figsize=(6, 5))
plt.plot(y_val, label='Observed', color=COLOR_PALETTE['main'])
plt.plot(y_pred, label='Predicted', color=COLOR_PALETTE['positive'])
plt.fill_between(range(len(y_val)), y_pred - 5, y_pred + 5,
                 color=COLOR_PALETTE['ci'], alpha=0.3, label='±5 margin')
plt.xlabel("Samples")
plt.ylabel("AQI")
plt.legend()
plt.title("Observed vs Predicted AQI")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("outputs/pred_vs_true.eps", bbox_inches='tight')

# ======================
# SHAP 可解释性分析
# ======================
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_val)
feature_names = [f"F{i}" for i in range(X_val.shape[1])]
X_val_df = pd.DataFrame(X_val, columns=feature_names)

# 图2：SHAP Summary
shap.summary_plot(shap_values.values, X_val_df, plot_type="dot", show=False)
plt.title("SHAP Summary Plot", fontsize=14)
plt.savefig("outputs/shap_summary.eps", bbox_inches='tight')

# 图3：SHAP Heatmap
shap.plots.heatmap(shap.Explanation(values=shap_values.values,
                                                    data=X_val_df.values,
                                                    feature_names=feature_names),
                           show=False)
plt.title("SHAP Heatmap", fontsize=14)
plt.savefig("outputs/shap_heatmap.eps", bbox_inches='tight')

# 图4：SHAP Dependence Plot（以第1特征为例）
shap.dependence_plot(0, shap_values.values, X_val_df,
                             interaction_index=None, show=False)
plt.title(f"SHAP Dependence Plot: {feature_names[0]}", fontsize=14)
plt.savefig("outputs/shap_dependence_feature1.eps", bbox_inches='tight')

print("✅ 已生成科研级EPS图：")
print(" - outputs/pred_vs_true.eps")
print(" - outputs/shap_summary.eps")
print(" - outputs/shap_heatmap.eps")
print(" - outputs/shap_dependence_feature1.eps")
