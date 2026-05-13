import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.triple_barrier import (
    load_backtest_base,
    run_triple_barrier_backtest,
)
from xgboost_pipeline import (
    BACKTEST_CAPITAL_FRACTION,
    BACKTEST_CONFIRM_PRESETS,
    BACKTEST_FEE,
    BACKTEST_FILTER_PRESETS,
    BACKTEST_LEVERAGE,
    BACKTEST_MAINTENANCE_MARGIN,
    BACKTEST_SLIPPAGE,
    BASE_CANDLES_PER_DAY,
    BASE_MAX_HOLDING_BARS,
    DATA_DIR,
    PROJECT_ROOT,
    add_four_timeframe_scores,
    align_predictions_to_base,
    run_all_timeframes,
)


REPORT_DIR = PROJECT_ROOT / "reports" / "backtests"
WFO_CPU_THREADS = os.cpu_count() or 1
WFO_N_JOBS = int(os.getenv("WFO_N_JOBS", max(1, WFO_CPU_THREADS - 1)))
WFO_PARALLEL = True
WFO_PROGRESS_EVERY = 25
WFO_RANK_WINDOW_DAYS = 30
WFO_RANK_MIN_PERIODS = 240

WFO_CALIBRATION_DAYS = 90
WFO_VALIDATION_DAYS = 30
WFO_STEP_DAYS = 30
WFO_PURGE_BARS = BASE_MAX_HOLDING_BARS
WFO_HOLDOUT_DAYS = 45

WFO_MIN_AVG_MONTHLY_RETURN = 0.012
WFO_MIN_PROFIT_FACTOR = 1.20
WFO_MIN_POSITIVE_MONTH_RATE = 0.60
WFO_MIN_TRADES_PER_MONTH = 10
WFO_MAX_DRAWDOWN = -0.06
WFO_MAX_WORST_MONTH = -0.035
WFO_MAX_WORST_MONTH_TO_AVG = 2.5
WFO_MIN_APPROVED_CONSTRAINT_RATE = 0.60
WFO_MIN_APPROVED_BALANCE_RATE = 0.80

WFO_THRESHOLD_QUANTILES = (
    0.35,
    0.45,
    0.55,
    0.65,
    0.75,
    0.82,
    0.88,
    0.92,
    0.95,
    0.975,
    0.99,
)
WFO_SIGNAL_COUNTS = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000)
WFO_MAX_THRESHOLDS = 28

OPTIMIZER_SCORE_WEIGHTS = {
    "pred_score_wfo_no4h": {
        "pred_30m": 0.35,
        "pred_1h": 0.45,
        "pred_2h": 0.20,
        "pred_4h": 0.00,
    },
    "pred_score_wfo_1h_heavy": {
        "pred_30m": 0.25,
        "pred_1h": 0.55,
        "pred_2h": 0.15,
        "pred_4h": 0.05,
    },
    "pred_score_wfo_exec_heavy": {
        "pred_30m": 0.50,
        "pred_1h": 0.35,
        "pred_2h": 0.10,
        "pred_4h": 0.05,
    },
}

PRIMARY_COLUMNS = [
    ("directional_1h", "pred_score_directional_1h"),
    ("balanced_4tf", "pred_score_balanced_4tf"),
    ("daytrade_1h_30m", "pred_score_daytrade_1h_30m"),
    ("execution_30m", "pred_score_execution_30m"),
    ("wfo_no4h", "pred_score_wfo_no4h"),
    ("wfo_1h_heavy", "pred_score_wfo_1h_heavy"),
    ("wfo_exec_heavy", "pred_score_wfo_exec_heavy"),
]

RANK_PRIMARY_COLUMNS = [
    ("rank_directional_1h", "pred_rank_score_directional_1h"),
    ("rank_balanced_4tf", "pred_rank_score_balanced_4tf"),
    ("rank_daytrade_1h_30m", "pred_rank_score_daytrade_1h_30m"),
    ("rank_wfo_no4h", "pred_rank_score_wfo_no4h"),
]

CONFIRM_PRESETS = [
    "sem_confirmacao",
    "exec_30m_dir_1h_minimo",
]

FILTER_PRESETS = [
    "sem_filtro",
    "volume_muito_leve",
]

BARRIER_CONFIGS = [
    {"barrier": "tp1.2_sl1.0_24", "take_profit_mult": 1.2, "stop_loss_mult": 1.0, "max_holding_bars": 24},
    {"barrier": "tp1.3_sl1.0_24", "take_profit_mult": 1.3, "stop_loss_mult": 1.0, "max_holding_bars": 24},
    {"barrier": "tp1.3_sl1.0_48", "take_profit_mult": 1.3, "stop_loss_mult": 1.0, "max_holding_bars": 48},
    {"barrier": "tp1.5_sl1.0_24", "take_profit_mult": 1.5, "stop_loss_mult": 1.0, "max_holding_bars": 24},
    {"barrier": "tp1.5_sl1.0_48", "take_profit_mult": 1.5, "stop_loss_mult": 1.0, "max_holding_bars": 48},
    {"barrier": "tp1.6_sl1.0_48", "take_profit_mult": 1.6, "stop_loss_mult": 1.0, "max_holding_bars": 48},
    {"barrier": "tp1.8_sl1.0_48", "take_profit_mult": 1.8, "stop_loss_mult": 1.0, "max_holding_bars": 48},
    {"barrier": "tp1.8_sl1.0_72", "take_profit_mult": 1.8, "stop_loss_mult": 1.0, "max_holding_bars": 72},
    {"barrier": "tp2.0_sl1.0_72", "take_profit_mult": 2.0, "stop_loss_mult": 1.0, "max_holding_bars": 72},
    {"barrier": "tp2.0_sl1.0_96", "take_profit_mult": 2.0, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp2.2_sl1.0_96", "take_profit_mult": 2.2, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp2.5_sl1.0_96", "take_profit_mult": 2.5, "stop_loss_mult": 1.0, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp2.0_sl1.2_96", "take_profit_mult": 2.0, "stop_loss_mult": 1.2, "max_holding_bars": BASE_MAX_HOLDING_BARS},
    {"barrier": "tp2.2_sl1.2_96", "take_profit_mult": 2.2, "stop_loss_mult": 1.2, "max_holding_bars": BASE_MAX_HOLDING_BARS},
]

_WORKER_CALIBRATION_PREDICTIONS = None
_WORKER_VALIDATION_PREDICTIONS = None
_WORKER_BASE_FRAME = None
_WORKER_FOLD = None


def rolling_percentile_rank(series, window, min_periods):
    values = series.to_numpy(dtype=float)
    ranks = np.full(len(values), np.nan)

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue

        start = max(0, idx - window + 1)
        history = values[start : idx + 1]
        history = history[np.isfinite(history)]
        if len(history) < min_periods:
            continue

        ranks[idx] = (history <= value).mean()

    return pd.Series(ranks, index=series.index, name=series.name)


def add_optimizer_scores(predictions):
    scored = add_four_timeframe_scores(predictions)
    for score_col, weights in OPTIMIZER_SCORE_WEIGHTS.items():
        scored[score_col] = 0.0
        for col, weight in weights.items():
            scored[score_col] += scored[col] * weight

    rank_window = WFO_RANK_WINDOW_DAYS * BASE_CANDLES_PER_DAY
    rank_sources = {
        "pred_score_directional_1h": "pred_rank_score_directional_1h",
        "pred_score_balanced_4tf": "pred_rank_score_balanced_4tf",
        "pred_score_daytrade_1h_30m": "pred_rank_score_daytrade_1h_30m",
        "pred_score_wfo_no4h": "pred_rank_score_wfo_no4h",
    }
    for source_col, rank_col in rank_sources.items():
        scored[rank_col] = rolling_percentile_rank(
            scored[source_col],
            window=rank_window,
            min_periods=WFO_RANK_MIN_PERIODS,
        )

    return scored


def build_optimizer_candidates():
    candidates = []
    for (score_name, primary_col), confirm_preset, filter_preset, barrier in product(
        PRIMARY_COLUMNS + RANK_PRIMARY_COLUMNS,
        CONFIRM_PRESETS,
        FILTER_PRESETS,
        BARRIER_CONFIGS,
    ):
        candidate = {
            "score_name": score_name,
            "primary_col": primary_col,
            "confirm_preset": confirm_preset,
            "filter_preset": filter_preset,
            **barrier,
        }
        candidate["candidate_key"] = (
            f"{score_name}|{confirm_preset}|{filter_preset}|{barrier['barrier']}"
        )
        candidates.append(candidate)
    return candidates


def build_threshold_grid(series):
    values = pd.Series(series).dropna()
    thresholds = []

    if str(values.name).startswith("pred_rank_"):
        thresholds.extend([0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.925, 0.95, 0.975])

    for quantile in WFO_THRESHOLD_QUANTILES:
        thresholds.append(values.quantile(quantile))

    for count in WFO_SIGNAL_COUNTS:
        if 0 < count < len(values):
            thresholds.append(values.nlargest(count).iloc[-1])

    thresholds = [
        float(threshold)
        for threshold in thresholds
        if pd.notna(threshold) and np.isfinite(threshold) and threshold >= 0
    ]
    thresholds = np.array(sorted(np.unique(np.round(thresholds, 6))))
    if len(thresholds) > WFO_MAX_THRESHOLDS:
        keep_idx = np.linspace(0, len(thresholds) - 1, WFO_MAX_THRESHOLDS).round().astype(int)
        thresholds = thresholds[keep_idx]
    return thresholds


def iter_walk_forward_windows(length):
    calibration_bars = WFO_CALIBRATION_DAYS * BASE_CANDLES_PER_DAY
    validation_bars = WFO_VALIDATION_DAYS * BASE_CANDLES_PER_DAY
    step_bars = WFO_STEP_DAYS * BASE_CANDLES_PER_DAY

    fold = 1
    start = 0
    while start + calibration_bars + WFO_PURGE_BARS + validation_bars <= length:
        calibration_start = start
        calibration_end = start + calibration_bars
        validation_start = calibration_end + WFO_PURGE_BARS
        validation_end = validation_start + validation_bars
        yield fold, slice(calibration_start, calibration_end), slice(validation_start, validation_end)
        fold += 1
        start += step_bars


def split_optimizer_holdout(predictions):
    holdout_bars = WFO_HOLDOUT_DAYS * BASE_CANDLES_PER_DAY
    if len(predictions) <= holdout_bars:
        raise ValueError("Dados insuficientes para separar holdout final.")
    optimizer = predictions.iloc[:-holdout_bars]
    holdout = predictions.iloc[-holdout_bars:]
    return optimizer, holdout


def add_candidate_metadata(row, candidate, window_name, fold=None):
    enriched = {
        "fold": fold,
        "window": window_name,
        "candidate_key": candidate["candidate_key"],
        "score_name": candidate["score_name"],
        "primary_col": candidate["primary_col"],
        "confirm_preset": candidate["confirm_preset"],
        "filter_preset": candidate["filter_preset"],
        "barrier": candidate["barrier"],
        "take_profit_mult": candidate["take_profit_mult"],
        "stop_loss_mult": candidate["stop_loss_mult"],
        "max_holding_bars": candidate["max_holding_bars"],
        "leverage": BACKTEST_LEVERAGE,
        "capital_fraction": BACKTEST_CAPITAL_FRACTION,
    }
    enriched.update(row)
    return enriched


def run_candidate_backtest(predictions, base_frame, candidate, threshold):
    thresholds = {
        **BACKTEST_CONFIRM_PRESETS[candidate["confirm_preset"]],
        candidate["primary_col"]: threshold,
    }
    trades, summary = run_triple_barrier_backtest(
        predictions=predictions,
        base_frame=base_frame,
        thresholds=thresholds,
        filters=BACKTEST_FILTER_PRESETS[candidate["filter_preset"]],
        max_holding_bars=candidate["max_holding_bars"],
        volatility_window=5,
        take_profit_mult=candidate["take_profit_mult"],
        stop_loss_mult=candidate["stop_loss_mult"],
        fee=BACKTEST_FEE,
        slippage=BACKTEST_SLIPPAGE,
        leverage=BACKTEST_LEVERAGE,
        capital_fraction=BACKTEST_CAPITAL_FRACTION,
        maintenance_margin=BACKTEST_MAINTENANCE_MARGIN,
    )
    return trades, {"threshold": threshold, **summary}


def add_objective_metrics(results):
    if results.empty:
        return results

    enriched = results.copy()
    enriched["profitable"] = (
        (enriched["total_return"] > 0) & (enriched["profit_factor"] > 1)
    )
    enriched["enough_trades_per_month"] = (
        enriched["avg_trades_per_month"] >= WFO_MIN_TRADES_PER_MONTH
    )
    enriched["drawdown_ok"] = enriched["max_drawdown"] >= WFO_MAX_DRAWDOWN
    enriched["worst_month_ok"] = enriched["worst_monthly_return"] >= WFO_MAX_WORST_MONTH
    enriched["monthly_return_ok"] = (
        enriched["avg_monthly_return"] >= WFO_MIN_AVG_MONTHLY_RETURN
    )
    enriched["profit_factor_ok"] = enriched["profit_factor"] >= WFO_MIN_PROFIT_FACTOR
    enriched["positive_month_ok"] = (
        enriched["positive_month_rate"] >= WFO_MIN_POSITIVE_MONTH_RATE
    )

    avg_month = enriched["avg_monthly_return"].replace(0, np.nan)
    worst_abs = enriched["worst_monthly_return"].clip(upper=0).abs()
    drawdown_abs = enriched["max_drawdown"].clip(upper=0).abs()
    monthly_std = enriched["monthly_return_std"].fillna(0)

    enriched["worst_month_to_avg"] = (worst_abs / avg_month).replace(
        [np.inf, -np.inf], np.inf
    )
    enriched["monthly_balance_ok"] = (
        (enriched["worst_monthly_return"] >= 0)
        | (
            (enriched["avg_monthly_return"] > 0)
            & (enriched["worst_month_to_avg"] <= WFO_MAX_WORST_MONTH_TO_AVG)
        )
    )
    enriched["return_to_worst_month"] = (
        enriched["avg_monthly_return"] / worst_abs.replace(0, np.nan)
    ).replace([np.inf, -np.inf], 4).fillna(4)
    enriched["return_to_drawdown"] = (
        enriched["avg_monthly_return"] / drawdown_abs.replace(0, np.nan)
    ).replace([np.inf, -np.inf], 4).fillna(4)

    enriched["constraints_ok"] = (
        enriched["profitable"]
        & enriched["monthly_return_ok"]
        & enriched["profit_factor_ok"]
        & enriched["positive_month_ok"]
        & enriched["enough_trades_per_month"]
        & enriched["drawdown_ok"]
        & enriched["worst_month_ok"]
        & enriched["monthly_balance_ok"]
    )

    profit_factor_for_score = enriched["profit_factor"].replace(np.inf, 5).clip(upper=5)
    trade_bonus = np.minimum(
        enriched["avg_trades_per_month"] / WFO_MIN_TRADES_PER_MONTH,
        2,
    )
    enriched["robust_score"] = (
        enriched["avg_monthly_return"]
        + 0.004 * (profit_factor_for_score - 1).clip(lower=0)
        + 0.003 * enriched["positive_month_rate"]
        + 0.003 * trade_bonus
        + 0.004 * enriched["return_to_worst_month"].clip(upper=3)
        + 0.003 * enriched["return_to_drawdown"].clip(upper=3)
        - 0.80 * drawdown_abs
        - 1.00 * worst_abs
        - 0.25 * monthly_std
    )
    return enriched


def select_best_row(results):
    ranked = add_objective_metrics(results)
    ranked = ranked.sort_values(
        by=[
            "constraints_ok",
            "robust_score",
            "return_to_worst_month",
            "return_to_drawdown",
            "profit_factor",
            "avg_monthly_return",
            "worst_monthly_return",
            "max_drawdown",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )
    return ranked.iloc[0]


def optimize_candidate_thresholds(calibration_predictions, base_frame, candidate, fold):
    rows = []
    for threshold in build_threshold_grid(calibration_predictions[candidate["primary_col"]]):
        _, summary = run_candidate_backtest(
            calibration_predictions,
            base_frame,
            candidate,
            threshold,
        )
        rows.append(
            add_candidate_metadata(
                summary,
                candidate,
                "calibration",
                fold=fold,
            )
        )

    if not rows:
        return None
    return select_best_row(pd.DataFrame(rows))


def _init_candidate_worker(calibration_predictions, validation_predictions, base_frame, fold):
    global _WORKER_CALIBRATION_PREDICTIONS
    global _WORKER_VALIDATION_PREDICTIONS
    global _WORKER_BASE_FRAME
    global _WORKER_FOLD

    _WORKER_CALIBRATION_PREDICTIONS = calibration_predictions
    _WORKER_VALIDATION_PREDICTIONS = validation_predictions
    _WORKER_BASE_FRAME = base_frame
    _WORKER_FOLD = fold


def evaluate_candidate_worker(candidate):
    selected = optimize_candidate_thresholds(
        _WORKER_CALIBRATION_PREDICTIONS,
        _WORKER_BASE_FRAME,
        candidate,
        _WORKER_FOLD,
    )
    if selected is None:
        return None

    _, validation_summary = run_candidate_backtest(
        _WORKER_VALIDATION_PREDICTIONS,
        _WORKER_BASE_FRAME,
        candidate,
        float(selected["threshold"]),
    )
    validation_row = add_candidate_metadata(
        validation_summary,
        candidate,
        "validation",
        fold=_WORKER_FOLD,
    )
    validation_row["chosen_in_fold"] = False
    selected = selected.copy()
    selected["chosen_in_fold"] = False
    return selected.to_dict(), validation_row


def evaluate_candidates_for_fold(
    calibration_predictions,
    validation_predictions,
    base_frame,
    candidates,
    fold,
):
    if not WFO_PARALLEL or WFO_N_JOBS <= 1:
        results = []
        _init_candidate_worker(
            calibration_predictions,
            validation_predictions,
            base_frame,
            fold,
        )
        for idx, candidate in enumerate(candidates, start=1):
            result = evaluate_candidate_worker(candidate)
            if result is not None:
                results.append(result)
            if idx % WFO_PROGRESS_EVERY == 0:
                print(f"  candidatos avaliados: {idx}/{len(candidates)}")
        return results

    results = []
    with ProcessPoolExecutor(
        max_workers=WFO_N_JOBS,
        initializer=_init_candidate_worker,
        initargs=(calibration_predictions, validation_predictions, base_frame, fold),
    ) as executor:
        futures = [executor.submit(evaluate_candidate_worker, candidate) for candidate in candidates]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                results.append(result)
            if idx % WFO_PROGRESS_EVERY == 0 or idx == len(futures):
                print(f"  candidatos avaliados: {idx}/{len(futures)}")
    return results


def run_walk_forward(predictions, base_frame, candidates):
    optimizer_predictions, holdout_predictions = split_optimizer_holdout(predictions)
    calibration_rows = []
    validation_rows = []

    windows = list(iter_walk_forward_windows(len(optimizer_predictions)))
    for fold, calibration_slice, validation_slice in windows:
        calibration_predictions = optimizer_predictions.iloc[calibration_slice]
        validation_predictions = optimizer_predictions.iloc[validation_slice]
        fold_selected_rows = []

        print(
            f"Fold {fold}: calibracao {calibration_predictions.index.min()} -> "
            f"{calibration_predictions.index.max()}, validacao "
            f"{validation_predictions.index.min()} -> {validation_predictions.index.max()}"
        )

        candidate_results = evaluate_candidates_for_fold(
            calibration_predictions,
            validation_predictions,
            base_frame,
            candidates,
            fold,
        )
        for selected, validation_row in candidate_results:
            validation_rows.append(validation_row)
            fold_selected_rows.append(selected)

        if not fold_selected_rows:
            continue

        fold_calibration = add_objective_metrics(pd.DataFrame(fold_selected_rows))
        fold_best = select_best_row(fold_calibration)
        best_key = fold_best["candidate_key"]
        best_threshold = float(fold_best["threshold"])

        fold_calibration["chosen_in_fold"] = (
            (fold_calibration["candidate_key"] == best_key)
            & (fold_calibration["threshold"].astype(float) == best_threshold)
        )
        calibration_rows.extend(fold_calibration.to_dict("records"))

        for row in validation_rows:
            if row["fold"] == fold and row["candidate_key"] == best_key:
                row["chosen_in_fold"] = True

    calibration_results = add_objective_metrics(pd.DataFrame(calibration_rows))
    validation_results = add_objective_metrics(pd.DataFrame(validation_rows))
    ranking = rank_candidates(validation_results)
    holdout_result = run_holdout_validation(
        optimizer_predictions,
        holdout_predictions,
        base_frame,
        candidates,
        ranking,
    )
    return calibration_results, validation_results, ranking, holdout_result


def rank_candidates(validation_results):
    if validation_results.empty:
        return validation_results

    grouped = validation_results.groupby("candidate_key", as_index=False).agg(
        folds=("fold", "nunique"),
        chosen_folds=("chosen_in_fold", "sum"),
        score_name=("score_name", "first"),
        primary_col=("primary_col", "first"),
        confirm_preset=("confirm_preset", "first"),
        filter_preset=("filter_preset", "first"),
        barrier=("barrier", "first"),
        take_profit_mult=("take_profit_mult", "first"),
        stop_loss_mult=("stop_loss_mult", "first"),
        max_holding_bars=("max_holding_bars", "first"),
        avg_monthly_return_mean=("avg_monthly_return", "mean"),
        avg_monthly_return_median=("avg_monthly_return", "median"),
        worst_monthly_return_min=("worst_monthly_return", "min"),
        max_drawdown_min=("max_drawdown", "min"),
        profit_factor_mean=("profit_factor", "mean"),
        positive_month_rate_mean=("positive_month_rate", "mean"),
        avg_trades_per_month_mean=("avg_trades_per_month", "mean"),
        robust_score_mean=("robust_score", "mean"),
        constraints_ok_rate=("constraints_ok", "mean"),
        monthly_balance_ok_rate=("monthly_balance_ok", "mean"),
    )
    grouped["approved_candidate"] = (
        (grouped["constraints_ok_rate"] >= WFO_MIN_APPROVED_CONSTRAINT_RATE)
        & (grouped["monthly_balance_ok_rate"] >= WFO_MIN_APPROVED_BALANCE_RATE)
        & (grouped["avg_monthly_return_mean"] >= WFO_MIN_AVG_MONTHLY_RETURN)
        & (grouped["profit_factor_mean"] >= WFO_MIN_PROFIT_FACTOR)
        & (grouped["avg_trades_per_month_mean"] >= WFO_MIN_TRADES_PER_MONTH)
        & (grouped["worst_monthly_return_min"] >= WFO_MAX_WORST_MONTH)
        & (grouped["max_drawdown_min"] >= WFO_MAX_DRAWDOWN)
    )
    grouped = grouped.sort_values(
        by=[
            "approved_candidate",
            "constraints_ok_rate",
            "monthly_balance_ok_rate",
            "robust_score_mean",
            "avg_monthly_return_mean",
            "profit_factor_mean",
            "worst_monthly_return_min",
            "max_drawdown_min",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )
    return grouped


def run_holdout_validation(
    optimizer_predictions,
    holdout_predictions,
    base_frame,
    candidates,
    ranking,
):
    if ranking.empty:
        return pd.DataFrame()

    approved = ranking[ranking["approved_candidate"]]
    if approved.empty:
        return pd.DataFrame()

    candidate_by_key = {candidate["candidate_key"]: candidate for candidate in candidates}
    champion_key = approved.iloc[0]["candidate_key"]
    champion = candidate_by_key[champion_key]
    selected = optimize_candidate_thresholds(
        optimizer_predictions,
        base_frame,
        champion,
        fold="holdout_calibration",
    )
    _, holdout_summary = run_candidate_backtest(
        holdout_predictions,
        base_frame,
        champion,
        float(selected["threshold"]),
    )
    holdout_row = add_candidate_metadata(
        holdout_summary,
        champion,
        "holdout_final",
        fold="holdout",
    )
    holdout_row["threshold"] = float(selected["threshold"])
    holdout_row["chosen_in_fold"] = True
    return add_objective_metrics(pd.DataFrame([holdout_row]))


def save_reports(calibration_results, validation_results, ranking, holdout_result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_results.to_csv(REPORT_DIR / "btc_wfo_calibration_rows.csv", index=False)
    validation_results.to_csv(REPORT_DIR / "btc_wfo_validation_rows.csv", index=False)
    ranking.to_csv(REPORT_DIR / "btc_wfo_candidate_ranking.csv", index=False)
    holdout_result.to_csv(REPORT_DIR / "btc_wfo_holdout_final.csv", index=False)


def print_report(validation_results, ranking, holdout_result):
    if ranking.empty:
        print("\nNenhum candidato valido foi gerado no walk-forward.")
        return

    ranking_cols = [
        "candidate_key",
        "approved_candidate",
        "folds",
        "chosen_folds",
        "constraints_ok_rate",
        "monthly_balance_ok_rate",
        "avg_monthly_return_mean",
        "worst_monthly_return_min",
        "max_drawdown_min",
        "profit_factor_mean",
        "avg_trades_per_month_mean",
        "robust_score_mean",
    ]
    holdout_cols = [
        "candidate_key",
        "threshold",
        "trades",
        "avg_trades_per_month",
        "avg_monthly_return",
        "worst_monthly_return",
        "positive_month_rate",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "return_to_worst_month",
        "return_to_drawdown",
        "constraints_ok",
    ]

    print("\n===== Walk-Forward Ranking - melhores candidatos =====")
    print(ranking[ranking_cols].head(20).to_string(index=False))
    if not ranking["approved_candidate"].any():
        print(
            "\nNenhum candidato passou nos criterios profissionais minimos. "
            "Holdout final preservado para a proxima iteracao."
        )

    chosen = validation_results[validation_results["chosen_in_fold"]]
    if not chosen.empty:
        chosen_cols = [
            "fold",
            "candidate_key",
            "threshold",
            "trades",
            "avg_trades_per_month",
            "avg_monthly_return",
            "worst_monthly_return",
            "profit_factor",
            "max_drawdown",
            "constraints_ok",
        ]
        print("\n===== Walk-Forward - escolhido em cada fold =====")
        print(chosen[chosen_cols].to_string(index=False))

    print("\n===== Holdout final intocado =====")
    if holdout_result.empty:
        print("Holdout final nao foi executado porque nao houve candidato aprovado.")
    else:
        print(holdout_result[holdout_cols].to_string(index=False))

    print(f"\nRelatorios salvos em: {REPORT_DIR}")


def main():
    all_results = run_all_timeframes()
    if not all_results:
        raise RuntimeError("Nenhuma predicao gerada para o walk-forward.")

    aligned_predictions = align_predictions_to_base(all_results)
    predictions = add_optimizer_scores(
        aligned_predictions.drop(columns=["y_real_30m"], errors="ignore")
    )
    predictions = predictions.dropna()
    base_frame = load_backtest_base(Path(DATA_DIR) / "btc_30m.csv")
    candidates = build_optimizer_candidates()

    print(
        f"Walk-forward: {len(candidates)} candidatos, "
        f"{WFO_CALIBRATION_DAYS}d calibracao, "
        f"{WFO_VALIDATION_DAYS}d validacao, "
        f"{WFO_HOLDOUT_DAYS}d holdout final, "
        f"{WFO_N_JOBS}/{WFO_CPU_THREADS} workers CPU."
    )
    calibration_results, validation_results, ranking, holdout_result = run_walk_forward(
        predictions,
        base_frame,
        candidates,
    )
    save_reports(calibration_results, validation_results, ranking, holdout_result)
    print_report(validation_results, ranking, holdout_result)


if __name__ == "__main__":
    main()
