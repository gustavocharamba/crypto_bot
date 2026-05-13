import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.triple_barrier import (
    load_backtest_base,
    run_triple_barrier_backtest,
    sweep_multi_timeframe_thresholds,
    sweep_triple_barrier_thresholds,
)
from indicators.loader_statistical import get_statistical_indicators
from indicators.loader_technical import get_technical_indicators
from models.xgboost_model import get_xgboost_model
from processing.get_target import get_target
from processing.pre_processing import get_preprocessing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

NON_FEATURE_COLUMNS = [
    "Target",
    "Raw_Target",
    "Return",
    "RealizedReturn",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Date",
]

TARGET_HORIZON_DAYS = 2
BASE_TIMEFRAME = "30m"
BASE_CANDLES_PER_DAY = 48
BASE_MAX_HOLDING_BARS = TARGET_HORIZON_DAYS * BASE_CANDLES_PER_DAY

DEFAULT_RUNS = [
    {"name": "30M", "timeframe": "30m", "filename": "btc_30m.csv", "horizon": 96},
    {"name": "1H", "timeframe": "1h", "filename": "btc_1h.csv", "horizon": 48},
    {"name": "2H", "timeframe": "2h", "filename": "btc_2h.csv", "horizon": 24},
    {"name": "4H", "timeframe": "4h", "filename": "btc_4h.csv", "horizon": 12},
]

BACKTEST_FILTERS = {
    "Volume_Z": {"min": 0.0},
    "Volume_Rel": {"min": 1.0},
    "range_expansion": {"min": 0.6},
    "vol_regime": {"min": 0.7, "max": 2.5},
}

SCORE_WEIGHT_PRESETS = {
    "balanced_4tf": {
        "pred_30m": 0.25,
        "pred_1h": 0.35,
        "pred_2h": 0.25,
        "pred_4h": 0.15,
    },
    "directional_1h": {
        "pred_30m": 0.25,
        "pred_1h": 0.45,
        "pred_2h": 0.20,
        "pred_4h": 0.10,
    },
    "execution_30m": {
        "pred_30m": 0.40,
        "pred_1h": 0.35,
        "pred_2h": 0.15,
        "pred_4h": 0.10,
    },
    "daytrade_1h_30m": {
        "pred_30m": 0.35,
        "pred_1h": 0.40,
        "pred_2h": 0.20,
        "pred_4h": 0.05,
    },
}

BACKTEST_FILTER_PRESETS = {
    "sem_filtro": {},
    "volume_muito_leve": {
        "Volume_Rel": {"min": 0.8},
        "range_expansion": {"min": 0.4},
        "vol_regime": {"min": 0.5, "max": 3.0},
    },
    "volume_basico": BACKTEST_FILTERS,
    "volume_fluxo": {
        "Volume_Z": {"min": 0.3},
        "Volume_Rel": {"min": 1.1},
        "range_expansion": {"min": 0.8},
        "vol_regime": {"min": 0.8, "max": 2.2},
        "OBV_Slope": {"min": 0.0},
        "MFI_14_norm": {"min": 0.0},
    },
    "breakout_vwap": {
        "Volume_Z": {"min": 0.5},
        "Volume_Rel": {"min": 1.2},
        "range_expansion": {"min": 1.0},
        "vol_regime": {"min": 0.8, "max": 2.5},
        "VWAP_dist_20": {"min": 0.0},
        "OBV_Slope": {"min": 0.0},
        "MFI_14_norm": {"min": 0.0},
    },
}

BACKTEST_CONFIRM_PRESETS = {
    "sem_confirmacao": {},
    "mtf_leve": {
        "pred_1h": 0.04,
        "pred_2h": 0.04,
        "pred_4h": 0.04,
    },
    "mtf_medio": {
        "pred_1h": 0.08,
        "pred_2h": 0.08,
        "pred_4h": 0.06,
    },
    "mtf_forte": {
        "pred_1h": 0.12,
        "pred_2h": 0.10,
        "pred_4h": 0.08,
    },
    "30m_2h_4h_medio": {
        "pred_30m": 0.08,
        "pred_2h": 0.08,
        "pred_4h": 0.06,
    },
    "4tf_min_leve": {
        "pred_30m": 0.04,
        "pred_1h": 0.06,
        "pred_2h": 0.05,
        "pred_4h": 0.04,
    },
    "4tf_min_medio": {
        "pred_30m": 0.08,
        "pred_1h": 0.10,
        "pred_2h": 0.08,
        "pred_4h": 0.06,
    },
    "4tf_min_forte": {
        "pred_30m": 0.12,
        "pred_1h": 0.14,
        "pred_2h": 0.10,
        "pred_4h": 0.08,
    },
    "exec_30m_dir_1h_leve": {
        "pred_30m": 0.03,
        "pred_1h": 0.05,
        "pred_2h": 0.03,
        "pred_4h": 0.02,
    },
    "exec_30m_dir_1h_minimo": {
        "pred_30m": 0.01,
        "pred_1h": 0.02,
    },
    "exec_30m_dir_1h_medio": {
        "pred_30m": 0.04,
        "pred_1h": 0.07,
        "pred_2h": 0.04,
        "pred_4h": 0.03,
    },
    "exec_30m_dir_1h_forte": {
        "pred_30m": 0.06,
        "pred_1h": 0.10,
        "pred_2h": 0.06,
        "pred_4h": 0.04,
    },
}

BACKTEST_BARRIER_CONFIGS = [
    {"barrier": "tp1.3_sl1.0_48", "take_profit_mult": 1.3, "stop_loss_mult": 1.0, "max_holding_bars": 48},
    {"barrier": "tp1.5_sl1.2_48", "take_profit_mult": 1.5, "stop_loss_mult": 1.2, "max_holding_bars": 48},
    {"barrier": "tp1.6_sl1.0_96", "take_profit_mult": 1.6, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp1.8_sl1.0_96", "take_profit_mult": 1.8, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp2.0_sl1.0_96", "take_profit_mult": 2.0, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp2.5_sl1.0_96", "take_profit_mult": 2.5, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
]

BACKTEST_TARGET_TRADE_RANGE = (400, 600)
BACKTEST_TARGET_WIN_RATE_RANGE = (0.55, 0.60)
BACKTEST_TARGET_MONTHLY_RETURN = 0.02
BACKTEST_MIN_TRADES_FOR_SELECTION = 100
BACKTEST_MAX_DRAWDOWN_LIMIT = -0.20
FINAL_MIN_MONTHLY_RETURN = 0.012
FINAL_MIN_PROFIT_FACTOR = 1.20
FINAL_MIN_POSITIVE_MONTH_RATE = 0.60
FINAL_MIN_TRADES = 50
FINAL_MAX_DRAWDOWN_LIMIT = -0.06
FINAL_MAX_WORST_MONTH_LIMIT = -0.035
BACKTEST_FEE = 0.0004
BACKTEST_SLIPPAGE = 0.0002
BACKTEST_RISK_PROFILE = "moderado_btc"
BACKTEST_LEVERAGE = 2.0
BACKTEST_CAPITAL_FRACTION = 0.5
BACKTEST_MAINTENANCE_MARGIN = 0.005
RUN_GENERAL_BACKTEST_SWEEP = False
RUN_RESEARCH_BACKTESTS = False
RUN_FINAL_BACKTEST = True

BACKTEST_4TF_THRESHOLD_GRID = {
    "pred_30m": [0.0, 0.04, 0.08, 0.12],
    "pred_1h": [0.0, 0.02, 0.04, 0.08],
    "pred_2h": [0.0, 0.02, 0.04],
    "pred_4h": [0.0, 0.005, 0.02, 0.04],
}

BACKTEST_AUTO_SIGNAL_COUNTS = (
    400,
    500,
    600,
    800,
    1000,
    1500,
    2000,
    3000,
    5000,
)

BACKTEST_4TF_TARGET_CONFIGS = [
    {
        "strategy": "4tf_thresholds_sem_filtro",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "4tf_thresholds_volume_leve",
        "filter_preset": "volume_muito_leve",
    },
    {
        "strategy": "4tf_thresholds_volume_basico",
        "filter_preset": "volume_basico",
    },
]

BACKTEST_4TF_TARGET_BARRIERS = [
    {"barrier": "tp2.2_sl1.0_96", "take_profit_mult": 2.2, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
]

FINAL_BACKTEST_VALIDATION_SIZE = 0.5
FINAL_BACKTEST_CONFIGS = [
    {
        "strategy": "final_directional_1h",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
        "barrier": "tp2.2_sl1.0_96",
        "take_profit_mult": 2.2,
        "stop_loss_mult": 1.0,
        "max_holding_bars": BASE_MAX_HOLDING_BARS,
    },
    {
        "strategy": "final_directional_1h_volume",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "volume_muito_leve",
        "barrier": "tp2.2_sl1.0_96",
        "take_profit_mult": 2.2,
        "stop_loss_mult": 1.0,
        "max_holding_bars": BASE_MAX_HOLDING_BARS,
    },
    {
        "strategy": "final_directional_1h_confirmado",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "exec_30m_dir_1h_minimo",
        "filter_preset": "sem_filtro",
        "barrier": "tp2.2_sl1.0_96",
        "take_profit_mult": 2.2,
        "stop_loss_mult": 1.0,
        "max_holding_bars": BASE_MAX_HOLDING_BARS,
    },
    {
        "strategy": "final_balanced_4tf",
        "primary_col": "pred_score_balanced_4tf",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
        "barrier": "tp2.2_sl1.0_96",
        "take_profit_mult": 2.2,
        "stop_loss_mult": 1.0,
        "max_holding_bars": BASE_MAX_HOLDING_BARS,
    },
    {
        "strategy": "final_balanced_4tf_volume",
        "primary_col": "pred_score_balanced_4tf",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "volume_muito_leve",
        "barrier": "tp2.2_sl1.0_96",
        "take_profit_mult": 2.2,
        "stop_loss_mult": 1.0,
        "max_holding_bars": BASE_MAX_HOLDING_BARS,
    },
]

BACKTEST_WEIGHTED_SCORE_GRID = np.round(np.arange(0.01, 0.36, 0.01), 4)

BACKTEST_WEIGHTED_TARGET_CONFIGS = [
    {
        "strategy": "score_balanced_4tf",
        "primary_col": "pred_score_balanced_4tf",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "score_directional_1h",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "score_directional_1h_confirmado",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "exec_30m_dir_1h_minimo",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "score_execution_30m",
        "primary_col": "pred_score_execution_30m",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "score_daytrade_1h_30m",
        "primary_col": "pred_score_daytrade_1h_30m",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "score_directional_1h_volume",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "exec_30m_dir_1h_minimo",
        "filter_preset": "volume_muito_leve",
    },
    {
        "strategy": "score_balanced_4tf_volume",
        "primary_col": "pred_score_balanced_4tf",
        "confirm_preset": "exec_30m_dir_1h_minimo",
        "filter_preset": "volume_muito_leve",
    },
]

BACKTEST_STRATEGY_CONFIGS = [
    {
        "strategy": "modelo_30m",
        "primary_col": "pred_30m",
        "confirm_preset": "sem_confirmacao",
        "filter_preset": "sem_filtro",
    },
    {
        "strategy": "mtf_volume_leve",
        "primary_col": "pred_30m",
        "confirm_preset": "mtf_leve",
        "filter_preset": "volume_basico",
    },
    {
        "strategy": "mtf_volume_fluxo",
        "primary_col": "pred_30m",
        "confirm_preset": "mtf_medio",
        "filter_preset": "volume_fluxo",
    },
    {
        "strategy": "mtf_breakout_vwap",
        "primary_col": "pred_30m",
        "confirm_preset": "mtf_medio",
        "filter_preset": "breakout_vwap",
    },
    {
        "strategy": "4tf_score_volume",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "4tf_min_medio",
        "filter_preset": "volume_fluxo",
    },
    {
        "strategy": "4tf_score_breakout",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "4tf_min_medio",
        "filter_preset": "breakout_vwap",
    },
    {
        "strategy": "4tf_score_forte",
        "primary_col": "pred_score_directional_1h",
        "confirm_preset": "4tf_min_forte",
        "filter_preset": "volume_fluxo",
    },
    {
        "strategy": "score_balanced_4tf_volume",
        "primary_col": "pred_score_balanced_4tf",
        "confirm_preset": "exec_30m_dir_1h_medio",
        "filter_preset": "volume_muito_leve",
    },
    {
        "strategy": "score_execution_30m_volume",
        "primary_col": "pred_score_execution_30m",
        "confirm_preset": "exec_30m_dir_1h_medio",
        "filter_preset": "volume_muito_leve",
    },
]


def build_xgb_dataset(data_path, horizon, target_kwargs=None):
    target_kwargs = target_kwargs or {}

    df = pd.read_csv(data_path)
    df = get_preprocessing(df)
    df = get_technical_indicators(df)
    df = get_statistical_indicators(df)
    df = get_target(df, horizon=horizon, **target_kwargs)
    df["RealizedReturn"] = df["Close"].shift(-horizon) / df["Close"] - 1

    X = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool"])
    X = X.replace([np.inf, -np.inf], np.nan)

    all_nan_features = X.columns[X.isna().all()]
    if len(all_nan_features) > 0:
        print(
            "Removendo features sem nenhum valor valido: "
            f"{', '.join(all_nan_features)}"
        )
        X = X.drop(columns=all_nan_features)

    dataset = X.join(df[["Target", "RealizedReturn"]]).dropna()
    if dataset.empty:
        raise ValueError("Dataset vazio depois do preprocessing, indicadores e target.")

    X = dataset.drop(columns=["Target", "RealizedReturn"])
    y = dataset["Target"].astype(int)
    y_real = dataset["RealizedReturn"]

    return X, y, y_real


def split_time_series(X, y, y_real, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Split invalido para o tamanho do dataset.")

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    y_real_test = y_real.iloc[split_idx:]

    if y_train.nunique() < 2:
        raise ValueError("O treino precisa ter as duas classes em Target para treinar o XGBoost.")

    return X_train, X_test, y_train, y_test, y_real_test


def evaluate_predictions(preds, y_true, threshold=0.5):
    labels = (preds >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, labels),
        "precision": precision_score(y_true, labels, zero_division=0),
        "recall": recall_score(y_true, labels, zero_division=0),
    }

    metrics["roc_auc"] = roc_auc_score(y_true, preds) if y_true.nunique() == 2 else np.nan
    return metrics


def run_pipeline_xgb(
    data_path,
    horizon,
    name=None,
    timeframe=None,
    test_size=0.2,
    target_kwargs=None,
    return_model=False,
):
    X, y, y_real = build_xgb_dataset(data_path, horizon=horizon, target_kwargs=target_kwargs)
    X_train, X_test, y_train, y_test, y_real_test = split_time_series(
        X, y, y_real, test_size=test_size
    )

    model = get_xgboost_model(X_train, y_train)
    preds = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name=f"pred_{timeframe or name or 'xgb'}",
    )
    y_real_test = y_real_test.rename(f"realized_return_{timeframe or name or 'xgb'}")

    metrics = evaluate_predictions(preds, y_test)
    label = f"{name or timeframe or Path(data_path).stem}"
    print(
        f"{label}: linhas={len(X)}, treino={len(X_train)}, teste={len(X_test)}, "
        f"roc_auc={metrics['roc_auc']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}"
    )

    if return_model:
        return preds, y_real_test, model, metrics

    return preds, y_real_test


def run_all_timeframes(data_dir=DATA_DIR, runs=None, test_size=0.2, target_kwargs=None):
    results = {}
    for config in runs or DEFAULT_RUNS:
        data_path = Path(data_dir) / config["filename"]
        if not data_path.exists():
            print(f"Pulando {config['timeframe']}: arquivo nao encontrado em {data_path}")
            continue

        preds, y_real, model, metrics = run_pipeline_xgb(
            data_path=data_path,
            horizon=config["horizon"],
            name=config["name"],
            timeframe=config["timeframe"],
            test_size=test_size,
            target_kwargs=target_kwargs,
            return_model=True,
        )
        results[config["timeframe"]] = {
            "preds": preds,
            "y_real": y_real,
            "model": model,
            "metrics": metrics,
        }

    return results


def align_predictions_to_base(results, base_timeframe="30m"):
    if base_timeframe not in results:
        raise ValueError(f"Timeframe base ausente nos resultados: {base_timeframe}")

    base_preds = results[base_timeframe]["preds"]
    aligned = pd.DataFrame(
        {
            f"pred_{base_timeframe}": base_preds,
            f"y_real_{base_timeframe}": results[base_timeframe]["y_real"],
        }
    )

    for timeframe, result in results.items():
        if timeframe == base_timeframe:
            continue
        aligned[f"pred_{timeframe}"] = result["preds"].reindex(base_preds.index, method="ffill")

    return aligned.dropna()


def add_four_timeframe_scores(predictions):
    scored = predictions.copy()
    required_cols = {
        col
        for weights in SCORE_WEIGHT_PRESETS.values()
        for col in weights
    }
    missing_cols = [col for col in required_cols if col not in scored.columns]
    if missing_cols:
        raise ValueError(f"Predicoes ausentes para score 4TF: {missing_cols}")

    for preset_name, weights in SCORE_WEIGHT_PRESETS.items():
        score_col = f"pred_score_{preset_name}"
        scored[score_col] = 0.0
        for col, weight in weights.items():
            scored[score_col] += scored[col] * weight

    scored["pred_4tf_score"] = scored["pred_score_directional_1h"]

    return scored


def build_auto_threshold_grid(series, fixed_thresholds=None, signal_counts=None, max_thresholds=45):
    values = pd.Series(series).dropna()
    if values.empty:
        return np.array([])

    thresholds = []
    if fixed_thresholds is not None:
        thresholds.extend(fixed_thresholds)

    signal_counts = signal_counts or BACKTEST_AUTO_SIGNAL_COUNTS
    for count in signal_counts:
        if 0 < count < len(values):
            thresholds.append(values.nlargest(count).iloc[-1])

    for quantile in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99):
        thresholds.append(values.quantile(quantile))

    thresholds = [
        float(threshold)
        for threshold in thresholds
        if pd.notna(threshold) and np.isfinite(threshold) and threshold >= 0
    ]
    thresholds = np.array(sorted(np.unique(np.round(thresholds, 6))))
    if len(thresholds) > max_thresholds:
        keep_idx = np.linspace(0, len(thresholds) - 1, max_thresholds).round().astype(int)
        thresholds = thresholds[keep_idx]

    return thresholds


def run_strategy_backtests(aligned_predictions, data_dir=DATA_DIR):
    base_frame = load_backtest_base(Path(data_dir) / "btc_30m.csv")
    predictions = add_four_timeframe_scores(
        aligned_predictions.drop(columns=["y_real_30m"], errors="ignore")
    )

    results = []
    for config in BACKTEST_STRATEGY_CONFIGS:
        threshold_grid = build_auto_threshold_grid(
            predictions[config["primary_col"]],
            fixed_thresholds=np.round(np.arange(0.01, 0.62, 0.01), 4),
        )
        for barrier_config in BACKTEST_BARRIER_CONFIGS:
            sweep = sweep_triple_barrier_thresholds(
                predictions=predictions,
                base_frame=base_frame,
                primary_col=config["primary_col"],
                primary_thresholds=threshold_grid,
                confirm_thresholds=BACKTEST_CONFIRM_PRESETS[config["confirm_preset"]],
                filters=BACKTEST_FILTER_PRESETS[config["filter_preset"]],
                max_holding_bars=barrier_config["max_holding_bars"],
                volatility_window=5,
                take_profit_mult=barrier_config["take_profit_mult"],
                stop_loss_mult=barrier_config["stop_loss_mult"],
                fee=BACKTEST_FEE,
                slippage=BACKTEST_SLIPPAGE,
                leverage=BACKTEST_LEVERAGE,
                capital_fraction=BACKTEST_CAPITAL_FRACTION,
                maintenance_margin=BACKTEST_MAINTENANCE_MARGIN,
                min_trades=20,
            )
            sweep.insert(0, "barrier", barrier_config["barrier"])
            sweep.insert(0, "capital_fraction", BACKTEST_CAPITAL_FRACTION)
            sweep.insert(0, "leverage", BACKTEST_LEVERAGE)
            sweep.insert(0, "filter_preset", config["filter_preset"])
            sweep.insert(0, "confirm_preset", config["confirm_preset"])
            sweep.insert(0, "primary_col", config["primary_col"])
            sweep.insert(0, "strategy", config["strategy"])
            results.append(sweep)

    backtest_results = pd.concat(results, ignore_index=True)
    backtest_results["profitable"] = (
        (backtest_results["total_return"] > 0)
        & (backtest_results["profit_factor"] > 1)
        & backtest_results["enough_trades"]
    )
    return backtest_results.sort_values(
        by=["profitable", "profit_factor", "total_return", "win_rate"],
        ascending=[False, False, False, False],
    )


def run_targeted_4tf_backtests(aligned_predictions, data_dir=DATA_DIR):
    base_frame = load_backtest_base(Path(data_dir) / "btc_30m.csv")
    predictions = aligned_predictions.drop(columns=["y_real_30m"], errors="ignore")
    results = []

    for config in BACKTEST_4TF_TARGET_CONFIGS:
        for barrier_config in BACKTEST_4TF_TARGET_BARRIERS:
            sweep = sweep_multi_timeframe_thresholds(
                predictions=predictions,
                base_frame=base_frame,
                threshold_grid=BACKTEST_4TF_THRESHOLD_GRID,
                filters=BACKTEST_FILTER_PRESETS[config["filter_preset"]],
                max_holding_bars=barrier_config["max_holding_bars"],
                volatility_window=5,
                take_profit_mult=barrier_config["take_profit_mult"],
                stop_loss_mult=barrier_config["stop_loss_mult"],
                fee=BACKTEST_FEE,
                slippage=BACKTEST_SLIPPAGE,
                leverage=BACKTEST_LEVERAGE,
                capital_fraction=BACKTEST_CAPITAL_FRACTION,
                maintenance_margin=BACKTEST_MAINTENANCE_MARGIN,
                trade_target=BACKTEST_TARGET_TRADE_RANGE,
                win_rate_target=BACKTEST_TARGET_WIN_RATE_RANGE,
            )
            sweep.insert(0, "barrier", barrier_config["barrier"])
            sweep.insert(0, "capital_fraction", BACKTEST_CAPITAL_FRACTION)
            sweep.insert(0, "leverage", BACKTEST_LEVERAGE)
            sweep.insert(0, "filter_preset", config["filter_preset"])
            sweep.insert(0, "strategy", config["strategy"])
            results.append(sweep)

    target_results = add_target_distances(pd.concat(results, ignore_index=True))
    return target_results.sort_values(
        by=[
            "monthly_target_hit",
            "sample_size_ok",
            "drawdown_ok",
            "profitable",
            "monthly_target_gap",
            "avg_monthly_return",
            "profit_factor",
        ],
        ascending=[False, False, False, False, True, False, False],
    )


def add_target_distances(results):
    target_low, target_high = BACKTEST_TARGET_TRADE_RANGE
    target_mid = (target_low + target_high) / 2
    win_low, win_high = BACKTEST_TARGET_WIN_RATE_RANGE
    enriched = results.copy()

    below_target = enriched["trades"] < target_low
    above_target = enriched["trades"] > target_high
    in_target = ~(below_target | above_target)
    enriched["trade_target_distance"] = 0.0
    enriched.loc[below_target, "trade_target_distance"] = (
        target_low - enriched.loc[below_target, "trades"]
    )
    enriched.loc[above_target, "trade_target_distance"] = (
        enriched.loc[above_target, "trades"] - target_high
    )
    enriched.loc[in_target, "trade_target_distance"] = (
        (enriched.loc[in_target, "trades"] - target_mid).abs() / target_mid
    )

    below_win = enriched["win_rate"] < win_low
    above_win = enriched["win_rate"] > win_high
    enriched["win_rate_target_distance"] = 0.0
    enriched.loc[below_win, "win_rate_target_distance"] = win_low - enriched.loc[below_win, "win_rate"]
    enriched.loc[above_win, "win_rate_target_distance"] = enriched.loc[above_win, "win_rate"] - win_high

    enriched["in_trade_target"] = enriched["trades"].between(target_low, target_high)
    enriched["in_win_rate_target"] = enriched["win_rate"].between(win_low, win_high)
    enriched["profitable"] = (enriched["total_return"] > 0) & (enriched["profit_factor"] > 1)
    enriched["monthly_target_gap"] = (
        BACKTEST_TARGET_MONTHLY_RETURN - enriched["avg_monthly_return"]
    ).clip(lower=0)
    enriched["monthly_target_hit"] = (
        enriched["avg_monthly_return"] >= BACKTEST_TARGET_MONTHLY_RETURN
    )
    enriched["sample_size_ok"] = enriched["trades"] >= BACKTEST_MIN_TRADES_FOR_SELECTION
    enriched["drawdown_ok"] = enriched["max_drawdown"] >= BACKTEST_MAX_DRAWDOWN_LIMIT
    enriched["final_monthly_return_ok"] = (
        enriched["avg_monthly_return"] >= FINAL_MIN_MONTHLY_RETURN
    )
    enriched["final_profit_factor_ok"] = (
        enriched["profit_factor"] >= FINAL_MIN_PROFIT_FACTOR
    )
    enriched["final_positive_month_ok"] = (
        enriched["positive_month_rate"] >= FINAL_MIN_POSITIVE_MONTH_RATE
    )
    enriched["final_sample_size_ok"] = enriched["trades"] >= FINAL_MIN_TRADES
    enriched["final_drawdown_ok"] = (
        enriched["max_drawdown"] >= FINAL_MAX_DRAWDOWN_LIMIT
    )
    enriched["final_worst_month_ok"] = (
        enriched["worst_monthly_return"] >= FINAL_MAX_WORST_MONTH_LIMIT
    )
    enriched["final_risk_ok"] = (
        enriched["profitable"]
        & enriched["final_monthly_return_ok"]
        & enriched["final_profit_factor_ok"]
        & enriched["final_positive_month_ok"]
        & enriched["final_sample_size_ok"]
        & enriched["final_drawdown_ok"]
        & enriched["final_worst_month_ok"]
    )
    drawdown_abs = enriched["max_drawdown"].abs().replace(0, np.nan)
    worst_month_abs = enriched["worst_monthly_return"].abs().replace(0, np.nan)
    enriched["return_to_drawdown"] = (
        enriched["avg_monthly_return"] / drawdown_abs
    ).replace([np.inf, -np.inf], 0).fillna(0)
    enriched["return_to_worst_month"] = (
        enriched["avg_monthly_return"] / worst_month_abs
    ).replace([np.inf, -np.inf], 0).fillna(0)
    return enriched


def auto_select_strategy(results, prefer_win_rate_target=True):
    candidates = results.copy()
    viable = candidates[
        candidates["profitable"]
        & candidates["monthly_target_hit"]
        & candidates["sample_size_ok"]
        & candidates["drawdown_ok"]
    ]
    if not viable.empty:
        candidates = viable
    else:
        profitable = candidates[candidates["profitable"]].copy()
        if not profitable.empty:
            candidates = profitable

    candidates = candidates.sort_values(
        by=[
            "monthly_target_hit",
            "sample_size_ok",
            "drawdown_ok",
            "monthly_target_gap",
            "avg_monthly_return",
            "profit_factor",
            "monthly_sharpe",
            "max_drawdown",
            "trade_target_distance",
            "win_rate_target_distance",
        ],
        ascending=[False, False, False, True, False, False, False, False, True, True],
    )

    return candidates.iloc[0]


def select_final_calibration_candidate(results):
    candidates = results.copy()
    viable = candidates[candidates["final_risk_ok"]].copy()
    if viable.empty:
        viable = candidates[candidates["profitable"] & candidates["sample_size_ok"]].copy()
    if viable.empty:
        viable = candidates[candidates["profitable"]].copy()
    if viable.empty:
        viable = candidates

    viable = viable.sort_values(
        by=[
            "final_risk_ok",
            "monthly_target_hit",
            "return_to_worst_month",
            "return_to_drawdown",
            "profit_factor",
            "positive_month_rate",
            "avg_monthly_return",
            "worst_monthly_return",
            "max_drawdown",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
    )
    return viable.iloc[0]


def run_targeted_weighted_score_backtests(aligned_predictions, data_dir=DATA_DIR):
    base_frame = load_backtest_base(Path(data_dir) / "btc_30m.csv")
    predictions = add_four_timeframe_scores(
        aligned_predictions.drop(columns=["y_real_30m"], errors="ignore")
    )
    results = []

    for config in BACKTEST_WEIGHTED_TARGET_CONFIGS:
        threshold_grid = build_auto_threshold_grid(
            predictions[config["primary_col"]],
            fixed_thresholds=BACKTEST_WEIGHTED_SCORE_GRID,
        )
        for barrier_config in BACKTEST_4TF_TARGET_BARRIERS:
            sweep = sweep_triple_barrier_thresholds(
                predictions=predictions,
                base_frame=base_frame,
                primary_col=config["primary_col"],
                primary_thresholds=threshold_grid,
                confirm_thresholds=BACKTEST_CONFIRM_PRESETS[config["confirm_preset"]],
                filters=BACKTEST_FILTER_PRESETS[config["filter_preset"]],
                max_holding_bars=barrier_config["max_holding_bars"],
                volatility_window=5,
                take_profit_mult=barrier_config["take_profit_mult"],
                stop_loss_mult=barrier_config["stop_loss_mult"],
                fee=BACKTEST_FEE,
                slippage=BACKTEST_SLIPPAGE,
                leverage=BACKTEST_LEVERAGE,
                capital_fraction=BACKTEST_CAPITAL_FRACTION,
                maintenance_margin=BACKTEST_MAINTENANCE_MARGIN,
                min_trades=BACKTEST_TARGET_TRADE_RANGE[0],
            )
            sweep.insert(0, "barrier", barrier_config["barrier"])
            sweep.insert(0, "capital_fraction", BACKTEST_CAPITAL_FRACTION)
            sweep.insert(0, "leverage", BACKTEST_LEVERAGE)
            sweep.insert(0, "filter_preset", config["filter_preset"])
            sweep.insert(0, "confirm_preset", config["confirm_preset"])
            sweep.insert(0, "primary_col", config["primary_col"])
            sweep.insert(0, "strategy", config["strategy"])
            results.append(sweep)

    target_results = add_target_distances(pd.concat(results, ignore_index=True))
    return target_results.sort_values(
        by=[
            "monthly_target_hit",
            "sample_size_ok",
            "drawdown_ok",
            "profitable",
            "monthly_target_gap",
            "avg_monthly_return",
            "profit_factor",
        ],
        ascending=[False, False, False, False, True, False, False],
    )


def split_final_backtest_windows(predictions, validation_size=FINAL_BACKTEST_VALIDATION_SIZE):
    if not 0 < validation_size < 1:
        raise ValueError("FINAL_BACKTEST_VALIDATION_SIZE deve estar entre 0 e 1.")

    split_idx = int(len(predictions) * (1 - validation_size))
    if split_idx <= 0 or split_idx >= len(predictions):
        raise ValueError("Split invalido para o final backtest.")

    calibration = predictions.iloc[:split_idx]
    validation = predictions.iloc[split_idx:]
    return calibration, validation


def add_backtest_metadata(results, config, window_name):
    enriched = results.copy()
    enriched.insert(0, "window", window_name)
    enriched.insert(0, "barrier", config["barrier"])
    enriched.insert(0, "capital_fraction", BACKTEST_CAPITAL_FRACTION)
    enriched.insert(0, "leverage", BACKTEST_LEVERAGE)
    enriched.insert(0, "filter_preset", config["filter_preset"])
    enriched.insert(0, "confirm_preset", config["confirm_preset"])
    enriched.insert(0, "primary_col", config["primary_col"])
    enriched.insert(0, "strategy", config["strategy"])
    return enriched


def run_final_backtest(aligned_predictions, data_dir=DATA_DIR):
    base_frame = load_backtest_base(Path(data_dir) / "btc_30m.csv")
    predictions = add_four_timeframe_scores(
        aligned_predictions.drop(columns=["y_real_30m"], errors="ignore")
    )
    calibration_predictions, validation_predictions = split_final_backtest_windows(
        predictions
    )
    min_calibration_trades = max(
        30,
        int(BACKTEST_MIN_TRADES_FOR_SELECTION * (1 - FINAL_BACKTEST_VALIDATION_SIZE)),
    )
    calibration_rows = []
    validation_rows = []

    for config in FINAL_BACKTEST_CONFIGS:
        primary_col = config["primary_col"]
        threshold_grid = build_auto_threshold_grid(
            calibration_predictions[primary_col],
            fixed_thresholds=BACKTEST_WEIGHTED_SCORE_GRID,
        )

        calibration_results = sweep_triple_barrier_thresholds(
            predictions=calibration_predictions,
            base_frame=base_frame,
            primary_col=primary_col,
            primary_thresholds=threshold_grid,
            confirm_thresholds=BACKTEST_CONFIRM_PRESETS[config["confirm_preset"]],
            filters=BACKTEST_FILTER_PRESETS[config["filter_preset"]],
            max_holding_bars=config["max_holding_bars"],
            volatility_window=5,
            take_profit_mult=config["take_profit_mult"],
            stop_loss_mult=config["stop_loss_mult"],
            fee=BACKTEST_FEE,
            slippage=BACKTEST_SLIPPAGE,
            leverage=BACKTEST_LEVERAGE,
            capital_fraction=BACKTEST_CAPITAL_FRACTION,
            maintenance_margin=BACKTEST_MAINTENANCE_MARGIN,
            min_trades=min_calibration_trades,
        )
        calibration_results = add_backtest_metadata(
            add_target_distances(calibration_results),
            config,
            "calibracao_threshold",
        )
        selected = select_final_calibration_candidate(calibration_results)
        calibration_rows.append(selected)

        thresholds = {
            **BACKTEST_CONFIRM_PRESETS[config["confirm_preset"]],
            primary_col: float(selected["threshold"]),
        }
        _, validation_summary = run_triple_barrier_backtest(
            predictions=validation_predictions,
            base_frame=base_frame,
            thresholds=thresholds,
            filters=BACKTEST_FILTER_PRESETS[config["filter_preset"]],
            max_holding_bars=config["max_holding_bars"],
            volatility_window=5,
            take_profit_mult=config["take_profit_mult"],
            stop_loss_mult=config["stop_loss_mult"],
            fee=BACKTEST_FEE,
            slippage=BACKTEST_SLIPPAGE,
            leverage=BACKTEST_LEVERAGE,
            capital_fraction=BACKTEST_CAPITAL_FRACTION,
            maintenance_margin=BACKTEST_MAINTENANCE_MARGIN,
        )
        validation_rows.append(
            add_backtest_metadata(
                add_target_distances(
                    pd.DataFrame(
                        [
                            {
                                "threshold": float(selected["threshold"]),
                                **validation_summary,
                            }
                        ]
                    )
                ),
                config,
                "validacao_fora_da_calibracao",
            ).iloc[0]
        )

    calibration_results = pd.DataFrame(calibration_rows)
    calibration_results = calibration_results.sort_values(
        by=[
            "final_risk_ok",
            "monthly_target_hit",
            "return_to_worst_month",
            "return_to_drawdown",
            "profit_factor",
            "positive_month_rate",
            "avg_monthly_return",
            "worst_monthly_return",
            "max_drawdown",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
    )
    chosen_strategy = calibration_results.iloc[0]["strategy"]
    calibration_results["chosen_by_calibration"] = (
        calibration_results["strategy"] == chosen_strategy
    )

    validation_results = pd.DataFrame(validation_rows)
    validation_results["chosen_by_calibration"] = (
        validation_results["strategy"] == chosen_strategy
    )
    validation_results = validation_results.sort_values(
        by=[
            "chosen_by_calibration",
            "final_risk_ok",
            "monthly_target_hit",
            "return_to_worst_month",
            "return_to_drawdown",
            "profit_factor",
        ],
        ascending=[False, False, False, False, False, False],
    )

    window_info = {
        "calibration_start": calibration_predictions.index.min(),
        "calibration_end": calibration_predictions.index.max(),
        "validation_start": validation_predictions.index.min(),
        "validation_end": validation_predictions.index.max(),
    }
    return calibration_results, validation_results, window_info


def print_backtest_results(backtest_results, top_n=10):
    cols = [
        "strategy",
        "primary_col",
        "confirm_preset",
        "filter_preset",
        "barrier",
        "leverage",
        "capital_fraction",
        "threshold",
        "candles",
        "signals",
        "trades",
        "trades_per_1000_candles",
        "avg_trades_per_month",
        "avg_monthly_return",
        "worst_monthly_return",
        "positive_month_rate",
        "total_return",
        "profit_factor",
        "win_rate",
        "breakeven_win_rate",
        "payoff_ratio",
        "avg_return",
        "max_drawdown",
        "avg_holding_bars",
        "take_profit_rate",
        "stop_loss_rate",
        "time_exit_rate",
    ]
    profitable = backtest_results[backtest_results["profitable"]]

    print("\n===== Backtest Triple Barrier - melhores por profit factor =====")
    print(backtest_results[cols].head(top_n).to_string(index=False))

    if not profitable.empty:
        print("\n===== Backtest Triple Barrier - maior taxa de acerto com PF > 1 =====")
        print(
            profitable.sort_values(
                by=["win_rate", "profit_factor", "total_return"],
                ascending=[False, False, False],
            )[cols]
            .head(top_n)
            .to_string(index=False)
        )


def print_targeted_4tf_results(target_results, top_n=15):
    cols = [
        "strategy",
        "filter_preset",
        "barrier",
        "th_30m",
        "th_1h",
        "th_2h",
        "th_4h",
        "leverage",
        "capital_fraction",
        "candles",
        "signals",
        "trades",
        "trades_per_1000_candles",
        "avg_trades_per_month",
        "avg_monthly_return",
        "worst_monthly_return",
        "positive_month_rate",
        "monthly_target_gap",
        "win_rate",
        "total_return",
        "profit_factor",
        "payoff_ratio",
        "breakeven_win_rate",
        "avg_return",
        "max_drawdown",
        "avg_holding_bars",
        "trade_target_distance",
        "win_rate_target_distance",
    ]

    print(
        "\n===== Backtest 4TF - alvo "
        f">= {BACKTEST_TARGET_MONTHLY_RETURN:.2%} ao mes ====="
    )
    if not target_results["monthly_target_hit"].any():
        print("Nenhuma combinacao bateu o alvo mensal; exibindo as mais proximas.")
    print(target_results[cols].head(top_n).to_string(index=False))

    selected = auto_select_strategy(target_results)
    print("\n===== Selecao automatica - thresholds 4TF =====")
    print(selected[cols].to_frame().T.to_string(index=False))


def print_targeted_weighted_results(target_results, top_n=15):
    cols = [
        "strategy",
        "primary_col",
        "confirm_preset",
        "filter_preset",
        "barrier",
        "leverage",
        "capital_fraction",
        "threshold",
        "candles",
        "signals",
        "trades",
        "trades_per_1000_candles",
        "avg_trades_per_month",
        "avg_monthly_return",
        "worst_monthly_return",
        "positive_month_rate",
        "monthly_target_gap",
        "win_rate",
        "total_return",
        "profit_factor",
        "payoff_ratio",
        "breakeven_win_rate",
        "avg_return",
        "max_drawdown",
        "avg_holding_bars",
        "trade_target_distance",
        "win_rate_target_distance",
    ]

    print(
        "\n===== Backtest score ponderado - alvo "
        f">= {BACKTEST_TARGET_MONTHLY_RETURN:.2%} ao mes ====="
    )
    if not target_results["monthly_target_hit"].any():
        print("Nenhuma combinacao bateu o alvo mensal; exibindo as mais proximas.")
    print(target_results[cols].head(top_n).to_string(index=False))

    selected = auto_select_strategy(target_results)
    print("\n===== Selecao automatica - score ponderado =====")
    print(selected[cols].to_frame().T.to_string(index=False))


def print_final_backtest_results(calibration_results, validation_results, window_info):
    cols = [
        "window",
        "chosen_by_calibration",
        "strategy",
        "primary_col",
        "confirm_preset",
        "filter_preset",
        "barrier",
        "leverage",
        "capital_fraction",
        "threshold",
        "candles",
        "signals",
        "trades",
        "trades_per_1000_candles",
        "avg_trades_per_month",
        "avg_monthly_return",
        "worst_monthly_return",
        "positive_month_rate",
        "monthly_target_gap",
        "win_rate",
        "total_return",
        "profit_factor",
        "payoff_ratio",
        "breakeven_win_rate",
        "return_to_worst_month",
        "return_to_drawdown",
        "final_risk_ok",
        "final_sample_size_ok",
        "avg_return",
        "max_drawdown",
        "avg_holding_bars",
    ]

    print("\n===== Final Backtest - calibracao e validacao separadas =====")
    print(
        "Calibracao: "
        f"{window_info['calibration_start']} -> {window_info['calibration_end']}"
    )
    print(
        "Validacao:  "
        f"{window_info['validation_start']} -> {window_info['validation_end']}"
    )

    print("\nCandidatos escolhidos apenas pela calibracao:")
    print(calibration_results[cols].to_string(index=False))

    print("\nResultado fora da janela de calibracao:")
    print(validation_results[cols].to_string(index=False))


if __name__ == "__main__":
    all_results = run_all_timeframes()
    if all_results:
        aligned_predictions = align_predictions_to_base(all_results)
        print("\nPredicoes alinhadas:")
        print(aligned_predictions.tail())

        if RUN_GENERAL_BACKTEST_SWEEP:
            backtest_results = run_strategy_backtests(aligned_predictions)
            print_backtest_results(backtest_results)

        if RUN_RESEARCH_BACKTESTS:
            target_results = run_targeted_4tf_backtests(aligned_predictions)
            print_targeted_4tf_results(target_results)

            weighted_results = run_targeted_weighted_score_backtests(aligned_predictions)
            print_targeted_weighted_results(weighted_results)

        if RUN_FINAL_BACKTEST:
            final_calibration, final_validation, final_window_info = run_final_backtest(
                aligned_predictions
            )
            print_final_backtest_results(
                final_calibration,
                final_validation,
                final_window_info,
            )
