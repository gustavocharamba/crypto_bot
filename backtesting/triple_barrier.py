from itertools import product

import numpy as np
import pandas as pd

from indicators.loader_statistical import get_statistical_indicators
from indicators.loader_technical import get_technical_indicators
from processing.pre_processing import get_preprocessing


def load_backtest_base(data_path):
    df = pd.read_csv(data_path)
    df = get_preprocessing(df)
    df = get_technical_indicators(df)
    df = get_statistical_indicators(df)
    return df.replace([np.inf, -np.inf], np.nan)


def build_signal_mask(frame, thresholds=None, filters=None):
    thresholds = thresholds or {}
    filters = filters or {}
    mask = pd.Series(True, index=frame.index)

    for col, threshold in thresholds.items():
        if col not in frame.columns:
            raise ValueError(f"Coluna ausente no backtest: {col}")
        mask &= frame[col] >= threshold

    for col, condition in filters.items():
        if col not in frame.columns:
            raise ValueError(f"Coluna ausente no backtest: {col}")

        min_value = condition.get("min")
        max_value = condition.get("max")
        if min_value is not None:
            mask &= frame[col] >= min_value
        if max_value is not None:
            mask &= frame[col] <= max_value

    return mask.fillna(False)


def _max_drawdown(equity):
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def _monthly_trade_stats(trades):
    empty_stats = {
        "months": 0,
        "avg_trades_per_month": 0.0,
        "avg_monthly_return": 0.0,
        "median_monthly_return": 0.0,
        "best_monthly_return": 0.0,
        "worst_monthly_return": 0.0,
        "positive_month_rate": 0.0,
        "monthly_return_std": 0.0,
        "monthly_sharpe": 0.0,
    }
    if trades.empty or "entry_time" not in trades.columns:
        return empty_stats

    monthly_frame = trades.copy()
    monthly_frame["entry_month"] = pd.to_datetime(monthly_frame["entry_time"]).dt.to_period("M")
    monthly_returns = monthly_frame.groupby("entry_month")["net_return"].apply(
        lambda returns: (1 + returns).prod() - 1
    )
    monthly_trades = monthly_frame.groupby("entry_month").size()
    monthly_std = monthly_returns.std(ddof=0)

    return {
        "months": len(monthly_returns),
        "avg_trades_per_month": monthly_trades.mean(),
        "avg_monthly_return": monthly_returns.mean(),
        "median_monthly_return": monthly_returns.median(),
        "best_monthly_return": monthly_returns.max(),
        "worst_monthly_return": monthly_returns.min(),
        "positive_month_rate": (monthly_returns > 0).mean(),
        "monthly_return_std": monthly_std,
        "monthly_sharpe": (
            monthly_returns.mean() / monthly_std if monthly_std and monthly_std > 0 else 0.0
        ),
    }


def summarize_trades(trades):
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "breakeven_win_rate": 0.0,
            "total_return": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_holding_bars": 0.0,
            "take_profit_rate": 0.0,
            "stop_loss_rate": 0.0,
            "time_exit_rate": 0.0,
            "liquidation_rate": 0.0,
            **_monthly_trade_stats(trades),
        }

    returns = trades["net_return"]
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    equity = (1 + returns).cumprod()
    gross_profit = wins.sum()
    gross_loss = losses.abs().sum()
    exit_counts = trades["exit_reason"].value_counts(normalize=True)
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.abs().mean() if len(losses) > 0 else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
    breakeven_win_rate = 1 / (1 + payoff_ratio) if np.isfinite(payoff_ratio) else 0.0

    return {
        "trades": len(trades),
        "win_rate": (returns > 0).mean(),
        "avg_return": returns.mean(),
        "median_return": returns.median(),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "breakeven_win_rate": breakeven_win_rate,
        "total_return": equity.iloc[-1] - 1,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "max_drawdown": _max_drawdown(equity),
        "avg_holding_bars": trades["holding_bars"].mean(),
        "take_profit_rate": exit_counts.get("take_profit", 0.0),
        "stop_loss_rate": exit_counts.get("stop_loss", 0.0),
        "time_exit_rate": exit_counts.get("time", 0.0),
        "liquidation_rate": exit_counts.get("liquidation", 0.0),
        **_monthly_trade_stats(trades),
    }


def run_triple_barrier_backtest(
    predictions,
    base_frame,
    thresholds,
    filters=None,
    max_holding_bars=96,
    volatility_window=5,
    take_profit_mult=1.5,
    stop_loss_mult=0.5,
    fee=0.0004,
    slippage=0.0002,
    leverage=1.0,
    capital_fraction=1.0,
    maintenance_margin=0.005,
    entry_delay_bars=1,
    allow_overlap=False,
    side="long",
):
    if side not in {"long", "short"}:
        raise ValueError("side deve ser 'long' ou 'short'.")

    frame = base_frame.join(predictions, how="inner").copy()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    signal = build_signal_mask(frame, thresholds=thresholds, filters=filters)

    log_ret = np.log(frame["Close"] / frame["Close"].shift(1))
    volatility = log_ret.rolling(volatility_window).std()

    trades = []
    next_allowed_idx = 0
    signal_positions = np.flatnonzero(signal.to_numpy())

    for signal_idx in signal_positions:
        signal_time = frame.index[signal_idx]
        if signal_idx < next_allowed_idx:
            continue

        entry_idx = signal_idx + entry_delay_bars
        if entry_idx >= len(frame) - 1:
            break

        vol = volatility.iloc[signal_idx]
        if pd.isna(vol) or vol <= 0:
            continue

        entry_time = frame.index[entry_idx]
        entry_price = frame["Open"].iloc[entry_idx]
        if side == "long":
            take_profit = entry_price * np.exp(vol * take_profit_mult)
            stop_loss = entry_price * np.exp(-vol * stop_loss_mult)
        else:
            take_profit = entry_price * np.exp(-vol * take_profit_mult)
            stop_loss = entry_price * np.exp(vol * stop_loss_mult)

        liquidation_price = None
        if leverage > 1:
            if side == "long":
                liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
            else:
                liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)

        last_exit_idx = min(entry_idx + max_holding_bars, len(frame) - 1)
        exit_idx = last_exit_idx
        exit_price = frame["Close"].iloc[last_exit_idx]
        exit_reason = "time"

        for idx in range(entry_idx, last_exit_idx + 1):
            if side == "long":
                hit_tp = frame["High"].iloc[idx] >= take_profit
                hit_sl = frame["Low"].iloc[idx] <= stop_loss
                hit_liq = (
                    liquidation_price is not None
                    and frame["Low"].iloc[idx] <= liquidation_price
                )
            else:
                hit_tp = frame["Low"].iloc[idx] <= take_profit
                hit_sl = frame["High"].iloc[idx] >= stop_loss
                hit_liq = (
                    liquidation_price is not None
                    and frame["High"].iloc[idx] >= liquidation_price
                )

            if hit_liq:
                exit_idx = idx
                exit_price = liquidation_price
                exit_reason = "liquidation"
                break

            if hit_sl:
                exit_idx = idx
                exit_price = stop_loss
                exit_reason = "stop_loss"
                break
            if hit_tp:
                exit_idx = idx
                exit_price = take_profit
                exit_reason = "take_profit"
                break

        if side == "long":
            gross_return = exit_price / entry_price - 1
        else:
            gross_return = 1 - exit_price / entry_price

        round_trip_cost = 2 * (fee + slippage)
        net_price_return = gross_return - round_trip_cost
        net_return = capital_fraction * leverage * net_price_return
        if exit_reason == "liquidation":
            net_return = -capital_fraction

        trade = {
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": frame.index[exit_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "liquidation_price": liquidation_price,
            "gross_return": gross_return,
            "net_price_return": net_price_return,
            "net_return": net_return,
            "leverage": leverage,
            "capital_fraction": capital_fraction,
            "fee": fee,
            "slippage": slippage,
            "side": side,
            "exit_reason": exit_reason,
            "holding_bars": exit_idx - entry_idx + 1,
        }

        for col in predictions.columns:
            trade[col] = frame[col].iloc[signal_idx]

        trades.append(trade)

        if not allow_overlap:
            next_allowed_idx = exit_idx + 1

    trades = pd.DataFrame(trades)
    summary = summarize_trades(trades)
    candles = len(frame)
    signals = int(signal.sum())
    summary.update(
        {
            "candles": candles,
            "signals": signals,
            "signal_rate": signals / candles if candles else 0.0,
            "trades_per_1000_candles": (
                summary["trades"] / candles * 1000 if candles else 0.0
            ),
            "signal_to_trade_rate": summary["trades"] / signals if signals else 0.0,
        }
    )
    return trades, summary


def sweep_triple_barrier_thresholds(
    predictions,
    base_frame,
    primary_col="pred_30m",
    primary_thresholds=None,
    confirm_thresholds=None,
    filters=None,
    max_holding_bars=96,
    volatility_window=5,
    take_profit_mult=1.5,
    stop_loss_mult=0.5,
    fee=0.0004,
    slippage=0.0002,
    leverage=1.0,
    capital_fraction=1.0,
    maintenance_margin=0.005,
    min_trades=20,
    side="long",
):
    if primary_thresholds is None:
        primary_thresholds = np.round(np.arange(0.02, 0.42, 0.02), 4)

    confirm_thresholds = confirm_thresholds or {}
    rows = []

    for threshold in primary_thresholds:
        thresholds = {**confirm_thresholds, primary_col: threshold}
        _, summary = run_triple_barrier_backtest(
            predictions=predictions,
            base_frame=base_frame,
            thresholds=thresholds,
            filters=filters,
            max_holding_bars=max_holding_bars,
            volatility_window=volatility_window,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            fee=fee,
            slippage=slippage,
            leverage=leverage,
            capital_fraction=capital_fraction,
            maintenance_margin=maintenance_margin,
            side=side,
        )

        rows.append(
            {
                "threshold": threshold,
                **summary,
                "enough_trades": summary["trades"] >= min_trades,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    return results.sort_values(
        by=["enough_trades", "total_return", "profit_factor"],
        ascending=[False, False, False],
    )


def sweep_multi_timeframe_thresholds(
    predictions,
    base_frame,
    threshold_grid,
    filters=None,
    max_holding_bars=48,
    volatility_window=5,
    take_profit_mult=1.0,
    stop_loss_mult=1.0,
    fee=0.0004,
    slippage=0.0002,
    leverage=1.0,
    capital_fraction=1.0,
    maintenance_margin=0.005,
    trade_target=(400, 600),
    win_rate_target=(0.55, 0.60),
    side="long",
):
    keys = list(threshold_grid)
    rows = []
    target_low, target_high = trade_target
    target_mid = (target_low + target_high) / 2
    win_low, win_high = win_rate_target

    for values in product(*(threshold_grid[key] for key in keys)):
        thresholds = dict(zip(keys, values))
        _, summary = run_triple_barrier_backtest(
            predictions=predictions,
            base_frame=base_frame,
            thresholds=thresholds,
            filters=filters,
            max_holding_bars=max_holding_bars,
            volatility_window=volatility_window,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            fee=fee,
            slippage=slippage,
            leverage=leverage,
            capital_fraction=capital_fraction,
            maintenance_margin=maintenance_margin,
            side=side,
        )

        trades = summary["trades"]
        win_rate = summary["win_rate"]
        if trades < target_low:
            trade_distance = target_low - trades
        elif trades > target_high:
            trade_distance = trades - target_high
        else:
            trade_distance = abs(trades - target_mid) / target_mid

        if win_rate < win_low:
            win_rate_distance = win_low - win_rate
        elif win_rate > win_high:
            win_rate_distance = win_rate - win_high
        else:
            win_rate_distance = 0.0

        row = {
            **{f"th_{key.replace('pred_', '')}": value for key, value in thresholds.items()},
            **summary,
            "in_trade_target": target_low <= trades <= target_high,
            "in_win_rate_target": win_low <= win_rate <= win_high,
            "trade_target_distance": trade_distance,
            "win_rate_target_distance": win_rate_distance,
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["profitable"] = (results["total_return"] > 0) & (results["profit_factor"] > 1)
    return results.sort_values(
        by=[
            "in_trade_target",
            "in_win_rate_target",
            "profitable",
            "trade_target_distance",
            "win_rate_target_distance",
            "profit_factor",
        ],
        ascending=[False, False, False, True, True, False],
    )
