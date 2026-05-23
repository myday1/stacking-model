# train_oof_stacking.py
import numpy as np, os, argparse, joblib
from utils_preprocess import load_and_merge, feature_engineer, build_sequences, save_npz
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import matplotlib.pyplot as plt
import shap

# ---------- Model definitions (same as earlier but modular) ----------
class SeqDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

class TransformerModel(nn.Module):
    def __init__(self,n_feats, seq_len=14, d_model=64, n_heads=4, n_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(n_feats,d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2,1))
    def forward(self,x):
        x = self.input(x) + self.pos[:,:x.size(1),:]
        x = x.permute(1,0,2)
        x = self.encoder(x)
        x = x.permute(1,2,0)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)

# ---------- helpers ----------
def metrics(y_true,y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true,y_pred)),
        'MAE': mean_absolute_error(y_true,y_pred),
        'R2': r2_score(y_true,y_pred)
    }

def train_one_transformer(X_seq_train,y_train,X_seq_val,y_val,params,device='cpu'):
    # params: dict with d_model,n_heads,n_layers,dim_ff,dropout,lr,epochs,batch_size
    model = TransformerModel(n_feats=X_seq_train.shape[2], seq_len=X_seq_train.shape[1],
                             d_model=params['d_model'], n_heads=params['n_heads'],
                             n_layers=params['n_layers'], dim_ff=params['dim_ff'],
                             dropout=params['dropout'])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(SeqDataset(X_seq_train,y_train), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(SeqDataset(X_seq_val,y_val), batch_size=params['batch_size'], shuffle=False)
    best_state=None; best_val=1e9
    for ep in range(params['epochs']):
        model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        # val
        model.eval()
        vals=[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                vals.append(loss_fn(model(xb), yb).item())
        mean_val = np.mean(vals) if len(vals)>0 else 1e9
        if mean_val < best_val:
            best_val=mean_val
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_torch(model, X_seq, device='cpu', batch_size=256):
    model.eval()
    loader = DataLoader(SeqDataset(X_seq, np.zeros(len(X_seq))), batch_size=batch_size, shuffle=False)
    preds=[]
    with torch.no_grad():
        for xb,_ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)

# ---------- OOF expanding window ----------
def expanding_window_oof(seq_X, seq_y, seq_meta, n_splits=5, transformer_params=None, device='cpu'):
    n = len(seq_X)
    fold_size = (n // (n_splits+1))  # conservative split; we'll create n_splits folds
    oof_preds = np.zeros_like(seq_y, dtype=float)
    test_preds_list = []  # if you want to average test preds from each fold
    models = []
    for i in range(n_splits):
        train_end = fold_size * (i+1)
        val_end = train_end + fold_size
        if val_end >= n:
            val_end = n - 1
        X_tr = seq_X[:train_end]; y_tr = seq_y[:train_end]
        X_val = seq_X[train_end:val_end]; y_val = seq_y[train_end:val_end]
        if len(X_val)==0: break
        print(f"Fold {i+1}: train {len(X_tr)} -> val {len(X_val)}")
        model = train_one_transformer(X_tr,y_tr, X_val,y_val, transformer_params, device=device)
        models.append(model)
        # predict validation part
        oof_preds[train_end:val_end] = predict_torch(model, X_val, device=device)
    return oof_preds, models

# ---------- main pipeline ----------
def main(args):
    # load processed npz
    data = np.load(args.npz_path, allow_pickle=True)
    seq_X = data['seq_X']; seq_y = data['seq_y']; seq_meta = data['seq_meta']
    tab_features = list(data['tab_features'])
    dates = data['dates']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    # OOF
    transformer_params = {
        'd_model': args.d_model, 'n_heads': args.n_heads, 'n_layers': args.n_layers,
        'dim_ff': args.dim_ff, 'dropout': args.dropout, 'lr': args.lr,
        'epochs': args.epochs, 'batch_size': args.batch_size
    }
    print("Running expanding-window OOF...")
    oof_preds, models = expanding_window_oof(seq_X, seq_y, seq_meta,
                                             n_splits=args.n_folds, transformer_params=transformer_params, device=device)
    # Now build training matrix for stacking
    # use oof_preds as meta feature aligned with seq (same length)
    X_stack = np.hstack([seq_meta, oof_preds.reshape(-1,1)])
    y = seq_y
    # split into train/val/test (same scheme as earlier: 70/15/15)
    total = len(X_stack)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    X_train, X_val, X_test = X_stack[:train_end], X_stack[train_end:val_end], X_stack[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    # XGBoost train (simple default; you can replace with optuna search below)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {'objective':'reg:squarederror','eta':args.xgb_lr,'max_depth':args.xgb_max_depth,
              'subsample':args.xgb_subsample,'colsample_bytree':args.xgb_colsample,'seed':42}
    bst = xgb.train(params, dtrain, num_boost_round=args.xgb_rounds, evals=[(dval,'val')],
                    early_stopping_rounds=20, verbose_eval=10)
    # metrics
    pred_val = bst.predict(dval)
    pred_test = bst.predict(xgb.DMatrix(X_test))
    print("Stack val:", metrics(y_val, pred_val))
    print("Stack test:", metrics(y_test, pred_test))
    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(bst, os.path.join(args.out_dir, 'xgb_stack.model'))
    # Save transformer models
    for idx, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(args.out_dir, f"transformer_fold{idx+1}.pth"))
    np.savez_compressed(os.path.join(args.out_dir, "stack_data.npz"),
                        X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test,
                        dates=dates)
    # SHAP on final XGBoost
    explainer = shap.TreeExplainer(bst)
    shap_vals = explainer.shap_values(X_val)
    # quick summary plot save
    try:
        plt.figure(figsize=(8,6)); shap.summary_plot(shap_vals, X_val, feature_names=tab_features + ['transformer_oof'], show=False)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir,"shap_summary_val.png"), dpi=200)
        print("Saved SHAP summary:", os.path.join(args.out_dir,"shap_summary_val.png"))
    except Exception as e:
        print("SHAP plot failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", default="processed/data_seq14.npz")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_folds", type=int, default=5)
    # transformer hyperparams
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    # xgb
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample", type=float, default=0.8)
    parser.add_argument("--xgb_rounds", type=int, default=500)
    args = parser.parse_args()
    main(args)
