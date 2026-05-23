# utils_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_merge(aq_path, met_path):
    aq = pd.read_excel(aq_path)
    met = pd.read_excel(met_path)
    aq['日期'] = pd.to_datetime(aq['日期'])
    met['日期'] = pd.to_datetime(met['日期'])
    df = pd.merge(aq, met, on='日期', how='inner').sort_values('日期').reset_index(drop=True)
    # drop common irrelevant columns
    for c in ['城市','地名','质量等级','rank','经度','纬度']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    return df

def feature_engineer(df, lags=[1,2,3,7], rolls=[3,7]):
    df = df.copy()
    df['month'] = df['日期'].dt.month
    df['weekday'] = df['日期'].dt.weekday
    df['month_sin'] = np.sin(2*np.pi*df['month']/12); df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7); df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)
    for lag in lags:
        df[f"AQI_lag{lag}"] = df['AQI'].shift(lag)
    for w in rolls:
        df[f"AQI_roll{w}"] = df['AQI'].rolling(w).mean()
    # You can add wind direction sin/cos, interaction terms etc. here
    df = df.dropna().reset_index(drop=True)
    return df

def build_sequences(df, seq_len=14, target_col='AQI', scaler=None):
    exclude = ['日期', target_col]
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
        seq_X.append(X_tab_scaled.values[i:i+seq_len])
        seq_y.append(y[i+seq_len])
        seq_meta.append(X_tab_scaled.values[i+seq_len])  # meta features at prediction time
        dates.append(df['日期'].iloc[i+seq_len])
    seq_X = np.stack(seq_X)
    seq_y = np.array(seq_y)
    seq_meta = np.stack(seq_meta)
    dates = np.array(dates)
    return seq_X, seq_y, seq_meta, tab_features, scaler, dates

def save_npz(out_path, **arrays):
    np.savez_compressed(out_path, **arrays)
    print("Saved:", out_path)

if __name__ == "__main__":
    # usage example
    aq_path = "E:/空气质量/数据//临汾市_空气质量数据.xlsx"
    met_path = "E:/空气质量/数据//临汾市, 山西省, 气候数据.xlsx"
    df = load_and_merge(aq_path, met_path)
    df = feature_engineer(df)
    seq_X, seq_y, seq_meta, tab_features, scaler, dates = build_sequences(df, seq_len=14)
    os.makedirs("processed", exist_ok=True)
    save_npz("processed/data_seq14.npz",
             seq_X=seq_X, seq_y=seq_y, seq_meta=seq_meta,
             tab_features=tab_features, dates=dates)
