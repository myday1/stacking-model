# optuna_search.py
import optuna, joblib, numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import argparse

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective':'reg:squarederror',
        'eta': trial.suggest_loguniform('eta', 1e-3, 1e-0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'seed': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(param, dtrain, num_boost_round=1000, evals=[(dval,'val')], early_stopping_rounds=30, verbose_eval=False)
    pred = bst.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack_npz", default="outputs/stack_data.npz")
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()
    data = np.load(args.stack_npz, allow_pickle=True)
    X_train = data['X_train']; X_val = data['X_val']; y_train = data['y_train']; y_val = data['y_val']
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val)
    study.optimize(func, n_trials=args.n_trials)
    print("Best:", study.best_params, "best_value:", study.best_value)
    joblib.dump(study, "outputs/optuna_xgb_study.pkl")

