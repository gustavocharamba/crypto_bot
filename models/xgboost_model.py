import warnings
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore', message='.*DMatrix.*', category=UserWarning, module='xgboost')


class PlattScaler:
    """
    Calibração manual de probabilidades via Platt Scaling (regressão logística
    sobre as probabilidades brutas do modelo base).
    Substitui CalibratedClassifierCV(cv='prefit') que foi removido no sklearn>=1.4.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self._lr = LogisticRegression()

    def fit(self, X, y):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        self._lr.fit(raw, y)
        return self

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self._lr.predict_proba(raw)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



def xgboost_model(X_train, y_train):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    # ── 1. Busca de hiperparâmetros (CPU para evitar mismatch no CV) ──
    base_xgb = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        device='cpu',
        n_jobs=-1,
        scale_pos_weight=spw
    )

    param_dist_xgb = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.05, 0.1, 0.5],
        'reg_lambda': [1, 2, 5, 10],
        'gamma': [0, 0.05, 0.1, 0.5],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        base_xgb,
        param_dist_xgb,
        n_iter=30,
        scoring='roc_auc',
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        refit=False
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_

    # ── 2. Refit final com GPU + early stopping ───────────────────────
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    final_model = XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
        n_jobs=1,
        early_stopping_rounds=30,
        scale_pos_weight=spw,
        random_state=42
    )
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    return final_model


