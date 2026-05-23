import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle


def load_and_merge(aq_path, met_path, column_mapping=None):
    """
    加载和合并数据，可选地将列名转换为英文

    Args:
        aq_path: 空气质量数据路径
        met_path: 气象数据路径
        column_mapping: 列名映射字典 (中文 -> 英文)
    """
    aq = pd.read_excel(aq_path)
    met = pd.read_excel(met_path)
    aq['日期'] = pd.to_datetime(aq['日期'])
    met['日期'] = pd.to_datetime(met['日期'])
    df = pd.merge(aq, met, on='日期', how='inner').sort_values('日期').reset_index(drop=True)

    # drop common irrelevant columns
    for c in ['城市', '地名', '质量等级', 'rank', '经度', '纬度']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # 重命名列为英文
    if column_mapping is not None:
        df.rename(columns=column_mapping, inplace=True)

    return df


def feature_engineer(df, lags=[1, 2, 3, 7], rolls=[3, 7], date_col='date', target_col='aqi'):
    """
    特征工程

    Args:
        df: 数据框
        lags: 滞后步数列表
        rolls: 滚动窗口大小列表
        date_col: 日期列名（默认英文'date'）
        target_col: 目标列名（默认英文'aqi'）
    """
    df = df.copy()
    df['month'] = df[date_col].dt.month
    df['weekday'] = df[date_col].dt.weekday
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    for w in rolls:
        df[f"{target_col}_roll{w}"] = df[target_col].rolling(w).mean()
    df = df.dropna().reset_index(drop=True)
    return df


def build_sequences(df, seq_len=14, target_col='aqi', date_col='date', scaler=None):
    """
    构建序列数据

    Args:
        df: 数据框
        seq_len: 序列长度
        target_col: 目标列名（英文，默认'aqi'）
        date_col: 日期列名（英文，默认'date'）
        scaler: StandardScaler对象，如果为None则创建新的
    """
    exclude = [date_col, target_col]
    tab_features = [c for c in df.columns if c not in exclude]
    X_tab = df[tab_features].copy()
    y = df[target_col].values

    if scaler is None:
        scaler = StandardScaler()
        X_tab_scaled = pd.DataFrame(scaler.fit_transform(X_tab), columns=tab_features)
    else:
        X_tab_scaled = pd.DataFrame(scaler.transform(X_tab), columns=tab_features)

    seq_X, seq_y, seq_meta, dates = [], [], [], []
    n = len(df) - seq_len

    for i in range(n):
        seq_X.append(X_tab_scaled.values[i:i + seq_len])
        seq_y.append(y[i + seq_len])
        seq_meta.append(X_tab_scaled.values[i + seq_len])
        dates.append(df[date_col].iloc[i + seq_len])

    seq_X = np.stack(seq_X)
    seq_y = np.array(seq_y)
    seq_meta = np.stack(seq_meta)
    dates = np.array(dates)

    # 验证序列长度
    assert len(seq_X) == len(seq_y) == len(seq_meta), "序列长度不匹配"
    assert len(seq_X) > 0, "未生成任何序列"

    return seq_X, seq_y, seq_meta, tab_features, scaler, dates


def save_preprocessing_artifacts(out_dir, seq_X, seq_y, seq_meta, tab_features, scaler, dates, feature_names=None):
    """
    保存预处理后的数据和元信息

    Args:
        out_dir: 输出目录
        seq_X, seq_y, seq_meta: 序列数据
        tab_features: 特征名列表（英文）
        scaler: StandardScaler对象
        dates: 日期数组
        feature_names: 特征的中文名称映射字典（可选）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. 保存序列数据
    np.savez_compressed(
        os.path.join(out_dir, "data_seq14.npz"),
        seq_X=seq_X,
        seq_y=seq_y,
        seq_meta=seq_meta,
        dates=dates
    )
    print(f"✓ 保存数据: {out_dir}/data_seq14.npz")

    # 2. 保存特征列名（英文）
    with open(os.path.join(out_dir, "tab_features.pkl"), "wb") as f:
        pickle.dump(tab_features, f)
    print(f"✓ 保存特征列名: {out_dir}/tab_features.pkl")
    print(f"  特征列表: {tab_features}")

    # 3. 保存特征中文名称映射（可选）
    if feature_names is not None:
        with open(os.path.join(out_dir, "feature_names_cn.pkl"), "wb") as f:
            pickle.dump(feature_names, f)
        print(f"✓ 保存特征中文名称: {out_dir}/feature_names_cn.pkl")

    # 4. 保存scaler（用于推理时的数据缩放）
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ 保存缩放器: {out_dir}/scaler.pkl")

    # 5. 保存元信息（JSON格式，便于查看）
    import json
    meta_info = {
        "seq_len": 14,
        "num_features": len(tab_features),
        "num_samples": len(seq_X),
        "feature_names_en": tab_features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    print(f"✓ 保存元信息: {out_dir}/metadata.json")


def load_preprocessing_artifacts(out_dir):
    """加载预处理的数据和元信息"""
    # 加载数据
    data = np.load(os.path.join(out_dir, "data_seq14.npz"), allow_pickle=True)

    # 加载列名（英文）
    with open(os.path.join(out_dir, "tab_features.pkl"), "rb") as f:
        tab_features = pickle.load(f)

    # 加载scaler
    with open(os.path.join(out_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return data, tab_features, scaler


if __name__ == "__main__":
    # 使用示例
    aq_path = "E:/空气质量/数据//临汾市_空气质量数据.xlsx"
    met_path = "E:/空气质量/数据//临汾市, 山西省, 气候数据.xlsx"

    # 定义列名映射（中文 -> 英文）
    column_mapping = {
        '日期': 'date',
        'AQI': 'aqi',
        '温度': 'temperature',
        '湿度': 'humidity',
        '风速': 'wind_speed',
        '风向': 'wind_direction',
        '气压': 'pressure',
        '降水': 'precipitation',
        # 添加其他列名映射
    }

    # 定义特征的中文名称（用于说明文档）
    feature_names_cn = {
        'month': '月份',
        'weekday': '周几',
        'month_sin': '月份_sin',
        'month_cos': '月份_cos',
        'weekday_sin': '周几_sin',
        'weekday_cos': '周几_cos',
        'aqi_lag1': 'AQI_滞后1天',
        'aqi_lag2': 'AQI_滞后2天',
        'aqi_lag3': 'AQI_滞后3天',
        'aqi_lag7': 'AQI_滞后7天',
        'aqi_roll3': 'AQI_滚动平均3天',
        'aqi_roll7': 'AQI_滚动平均7天',
    }

    df = load_and_merge(aq_path, met_path, column_mapping=column_mapping)
    # 在 main 函数中，load_and_merge 之后添加
    df = load_and_merge(aq_path, met_path, column_mapping=column_mapping)
    print("=== 合并后的数据列（原始列+重命名后） ===")
    print("列数：", len(df.columns))
    print("列名：", df.columns.tolist())  # 查看是否有未被映射的中文列或额外列
    df = feature_engineer(df, date_col='date', target_col='aqi')
    # 在 main 函数中，feature_engineer 之后添加
    df = feature_engineer(df, date_col='date', target_col='aqi')
    print("\n=== 特征工程后的数据列（原始列+新增特征） ===")
    print("列数：", len(df.columns))
    print("列名：", df.columns.tolist())  # 对比是否有未预期的新增列
    seq_X, seq_y, seq_meta, tab_features, scaler, dates = build_sequences(df, seq_len=14, target_col='aqi')
    # 在 main 函数中，build_sequences 之后添加
    seq_X, seq_y, seq_meta, tab_features, scaler, dates = build_sequences(df, seq_len=14, target_col='aqi')
    print("\n=== 最终特征列（tab_features） ===")
    print("特征数：", len(tab_features))
    print("特征名：", tab_features)  # 直接查看多出来的列名
    # 保存所有预处理产物
    save_preprocessing_artifacts(
        out_dir="processed",
        seq_X=seq_X,
        seq_y=seq_y,
        seq_meta=seq_meta,
        tab_features=tab_features,
        scaler=scaler,
        dates=dates,
        feature_names=feature_names_cn
    )

    # 演示如何加载
    print("\n--- 加载预处理数据 ---")
    data, loaded_features, loaded_scaler = load_preprocessing_artifacts("processed")
    print(f"特征列表（英文）: {loaded_features}")