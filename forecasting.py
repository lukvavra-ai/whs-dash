from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DAILY_LAGS = (1, 2, 5, 7, 14, 28)
DAILY_WINDOWS = (5, 10, 20)
WEEKLY_LAGS = (1, 2, 4, 8, 13, 26, 52)
WEEKLY_WINDOWS = (4, 8, 13)


@dataclass
class RidgeModel:
    beta: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    columns: List[str]
    resid_std: float
    fitted: np.ndarray


@dataclass
class ForecastResult:
    fitted: pd.Series
    forecast: pd.Series
    lower80: pd.Series
    upper80: pd.Series
    lower95: pd.Series
    upper95: pd.Series
    model: Optional[RidgeModel] = None


@dataclass
class BacktestMetrics:
    model: str
    wape: float
    bias: float
    mae: float
    n_windows: int


def _calendar_features(date: pd.Timestamp, position: int, mode: str) -> Dict[str, float]:
    row: Dict[str, float] = {"trend": float(position)}
    if mode == "daily":
        row["weekday"] = float(date.weekday())
        for idx in range(7):
            row[f"dow_{idx}"] = 1.0 if date.weekday() == idx else 0.0
        row["month"] = float(date.month)
        row["quarter"] = float(date.quarter)
        row["is_month_start"] = float(date.is_month_start)
        row["is_month_end"] = float(date.is_month_end)
        row["dom"] = float(date.day)
        row["doy_sin"] = float(np.sin(2 * np.pi * date.dayofyear / 365.25))
        row["doy_cos"] = float(np.cos(2 * np.pi * date.dayofyear / 365.25))
    else:
        week = int(date.isocalendar().week)
        row["week"] = float(week)
        row["month"] = float(date.month)
        row["quarter"] = float(date.quarter)
        row["woy_sin"] = float(np.sin(2 * np.pi * week / 52.18))
        row["woy_cos"] = float(np.cos(2 * np.pi * week / 52.18))
    return row


def _feature_row(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    i: int,
    mode: str,
    lags: Tuple[int, ...],
    windows: Tuple[int, ...],
    exog: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, float]]:
    min_hist = max(max(lags, default=1), max(windows, default=1))
    if i < min_hist:
        return None

    row = _calendar_features(dates[i], i, mode)
    for lag in lags:
        row[f"lag_{lag}"] = float(values[i - lag])
    for window in windows:
        hist = values[i - window : i]
        row[f"roll_mean_{window}"] = float(np.nanmean(hist))
        row[f"roll_std_{window}"] = float(np.nanstd(hist))
    if mode == "daily":
        row["same_weekday_last_week"] = float(values[i - 7]) if i >= 7 else float(values[i - 1])
        row["same_weekday_last_4_weeks"] = float(np.nanmean([values[i - lag] for lag in (7, 14, 21, 28) if i >= lag]))
    else:
        row["same_week_last_year"] = float(values[i - 52]) if i >= 52 else float(values[i - 1])

    if exog is not None and not exog.empty:
        current_date = dates[i]
        ex_row = exog.reindex([current_date], method="ffill")
        if not ex_row.empty:
            for col, val in ex_row.iloc[0].items():
                row[col] = float(val) if pd.notna(val) else 0.0
    return row


def _design_matrix(
    series: pd.Series,
    mode: str,
    exog: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    s = series.astype(float).copy()
    dates = pd.DatetimeIndex(s.index)
    values = s.values.astype(float)
    lags = DAILY_LAGS if mode == "daily" else WEEKLY_LAGS
    windows = DAILY_WINDOWS if mode == "daily" else WEEKLY_WINDOWS
    ex = exog.reindex(dates).ffill().bfill() if exog is not None and not exog.empty else None

    rows: List[Dict[str, float]] = []
    y: List[float] = []
    idx: List[pd.Timestamp] = []
    for i in range(len(values)):
        row = _feature_row(values, dates, i, mode, lags, windows, ex)
        if row is None:
            continue
        rows.append(row)
        y.append(float(values[i]))
        idx.append(dates[i])
    X = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return X, np.asarray(y, dtype=float)


def _fit_ridge(X: pd.DataFrame, y: np.ndarray, alpha: float = 12.0) -> RidgeModel:
    xv = X.astype(float).values
    mu = xv.mean(axis=0)
    sigma = xv.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    xs = (xv - mu) / sigma
    xa = np.c_[np.ones(len(xs)), xs]
    ident = np.eye(xa.shape[1])
    ident[0, 0] = 0.0
    try:
        beta = np.linalg.solve(xa.T @ xa + alpha * ident, xa.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xa.T @ xa + alpha * ident) @ xa.T @ y
    fitted = xa @ beta
    resid_std = float(np.std(y - fitted, ddof=1)) if len(y) > 2 else 0.0
    return RidgeModel(beta=beta, mu=mu, sigma=sigma, columns=list(X.columns), resid_std=resid_std, fitted=fitted)


def _predict_ridge(model: RidgeModel, X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for col in model.columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[model.columns].astype(float)
    xv = X.values
    xs = (xv - model.mu) / model.sigma
    xa = np.c_[np.ones(len(xs)), xs]
    return xa @ model.beta


def seasonal_baseline(series: pd.Series, horizon: int, mode: str) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float)
    values = list(s.values)
    if mode == "daily":
        future_idx = [s.index.max() + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
        seasonal_lags = (5, 7, 10, 14, 21, 28)
        recent_span = 10
    else:
        future_idx = [s.index.max() + pd.Timedelta(weeks=i) for i in range(1, horizon + 1)]
        seasonal_lags = (1, 2, 4, 8, 13, 26, 52)
        recent_span = 6
    preds = []
    for _future_date in future_idx:
        seasonal_values = [values[-lag] for lag in seasonal_lags if len(values) >= lag]
        recent = float(np.nanmean(values[-recent_span:])) if values else 0.0
        if seasonal_values:
            pred = 0.75 * float(np.nanmean(seasonal_values)) + 0.25 * recent
        else:
            pred = recent
        pred = max(0.0, pred)
        values.append(pred)
        preds.append(pred)
    return pd.Series(preds, index=pd.DatetimeIndex(future_idx), name="baseline")


def ridge_forecast(
    series: pd.Series,
    horizon: int,
    mode: str,
    exog_hist: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    alpha: float = 12.0,
) -> ForecastResult:
    s = series.dropna().astype(float)
    if s.empty:
        empty = pd.Series(dtype=float)
        return ForecastResult(empty, empty, empty, empty, empty, empty, None)

    X, y = _design_matrix(s, mode, exog_hist)
    min_points = 28 if mode == "weekly" else 90
    if len(X) < min_points:
        baseline = seasonal_baseline(s, horizon, mode)
        return ForecastResult(baseline * np.nan, baseline, baseline, baseline, baseline, baseline, None)

    model = _fit_ridge(X, y, alpha=alpha)
    fitted = pd.Series(model.fitted, index=X.index, name="fitted")

    full_values = list(s.values)
    full_dates = list(pd.DatetimeIndex(s.index))
    all_exog = pd.DataFrame()
    if exog_hist is not None and not exog_hist.empty:
        all_exog = exog_hist.copy()
    if exog_future is not None and not exog_future.empty:
        all_exog = pd.concat([all_exog, exog_future], axis=0) if not all_exog.empty else exog_future.copy()
    if not all_exog.empty:
        all_exog = all_exog.sort_index().ffill().bfill()

    future_idx = [s.index.max() + (pd.Timedelta(days=i) if mode == "daily" else pd.Timedelta(weeks=i)) for i in range(1, horizon + 1)]
    lags = DAILY_LAGS if mode == "daily" else WEEKLY_LAGS
    windows = DAILY_WINDOWS if mode == "daily" else WEEKLY_WINDOWS
    preds: List[float] = []
    low80: List[float] = []
    high80: List[float] = []
    low95: List[float] = []
    high95: List[float] = []
    for step, future_date in enumerate(future_idx, start=1):
        full_dates.append(future_date)
        full_values.append(np.nan)
        row = _feature_row(np.asarray(full_values, dtype=float), pd.DatetimeIndex(full_dates), len(full_values) - 1, mode, lags, windows, all_exog)
        xrow = pd.DataFrame([row], index=[future_date]).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        pred = max(0.0, float(_predict_ridge(model, xrow)[0]))
        full_values[-1] = pred
        preds.append(pred)
        widen = np.sqrt(1 + 0.3 * step)
        std = model.resid_std * widen
        low80.append(max(0.0, pred - 1.2816 * std))
        high80.append(pred + 1.2816 * std)
        low95.append(max(0.0, pred - 1.96 * std))
        high95.append(pred + 1.96 * std)
    idx = pd.DatetimeIndex(future_idx)
    return ForecastResult(
        fitted=fitted,
        forecast=pd.Series(preds, index=idx, name="forecast"),
        lower80=pd.Series(low80, index=idx, name="lower80"),
        upper80=pd.Series(high80, index=idx, name="upper80"),
        lower95=pd.Series(low95, index=idx, name="lower95"),
        upper95=pd.Series(high95, index=idx, name="upper95"),
        model=model,
    )


def align_known_exog(features: pd.DataFrame, future_idx: pd.DatetimeIndex, lag_periods: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if features is None or features.empty:
        return pd.DataFrame(), pd.DataFrame()
    shifted = features.sort_index().ffill().bfill().shift(lag_periods).ffill().bfill()
    hist = shifted.copy()
    future = shifted.reindex(future_idx).ffill()
    if future.isna().all(axis=None) and not shifted.empty:
        future = pd.concat([shifted.tail(1)] * len(future_idx), axis=0)
        future.index = future_idx
    future = future.ffill().bfill()
    return hist, future


def _metric_summary(actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    err = pred - actual
    abs_err = np.abs(err)
    actual_abs_sum = np.abs(actual).sum()
    wape = float(abs_err.sum() / actual_abs_sum) if actual_abs_sum > 0 else np.nan
    bias = float(err.sum() / actual_abs_sum) if actual_abs_sum > 0 else np.nan
    mae = float(abs_err.mean()) if len(abs_err) else np.nan
    return wape, bias, mae


def rolling_backtest(
    series: pd.Series,
    mode: str,
    horizon: int,
    model_kind: str,
    exog_aligned: Optional[pd.DataFrame] = None,
    n_origins: int = 8,
) -> BacktestMetrics:
    s = series.dropna().astype(float)
    min_train = 28 if mode == "weekly" else 90
    if len(s) < min_train + horizon + 2:
        return BacktestMetrics(model_kind, np.nan, np.nan, np.nan, 0)

    origins = list(range(min_train, len(s) - horizon + 1))[-n_origins:]
    all_actual: List[float] = []
    all_pred: List[float] = []
    windows = 0
    for origin in origins:
        train = s.iloc[:origin]
        actual = s.iloc[origin : origin + horizon]
        if actual.empty:
            continue
        if model_kind == "baseline":
            pred = seasonal_baseline(train, horizon, mode).iloc[: len(actual)]
        elif model_kind == "internal":
            pred = ridge_forecast(train, horizon, mode).forecast.iloc[: len(actual)]
        elif model_kind == "global":
            if exog_aligned is None or exog_aligned.empty:
                continue
            hist = exog_aligned.reindex(train.index).ffill().bfill()
            fut = exog_aligned.reindex(actual.index).ffill().bfill()
            pred = ridge_forecast(train, horizon, mode, exog_hist=hist, exog_future=fut).forecast.iloc[: len(actual)]
        else:
            continue
        all_actual.extend(actual.values.tolist())
        all_pred.extend(pred.values.tolist())
        windows += 1
    wape, bias, mae = _metric_summary(np.asarray(all_actual, dtype=float), np.asarray(all_pred, dtype=float)) if windows else (np.nan, np.nan, np.nan)
    return BacktestMetrics(model_kind, wape, bias, mae, windows)


def compute_blend_weights(metric_map: Dict[str, BacktestMetrics]) -> Dict[str, float]:
    valid = {name: metrics.wape for name, metrics in metric_map.items() if pd.notna(metrics.wape) and metrics.wape > 0}
    if not valid:
        count = len(metric_map)
        return {name: 1.0 / count for name in metric_map} if count else {}
    inv = {name: 1.0 / score for name, score in valid.items()}
    total = sum(inv.values())
    return {name: value / total for name, value in inv.items()}


def blend_forecasts(results: Dict[str, ForecastResult], weights: Dict[str, float]) -> ForecastResult:
    if not results:
        empty = pd.Series(dtype=float)
        return ForecastResult(empty, empty, empty, empty, empty, empty, None)
    first = next(iter(results.values()))
    idx = first.forecast.index
    forecast = pd.Series(0.0, index=idx)
    lower80 = pd.Series(0.0, index=idx)
    upper80 = pd.Series(0.0, index=idx)
    lower95 = pd.Series(0.0, index=idx)
    upper95 = pd.Series(0.0, index=idx)
    for name, result in results.items():
        weight = weights.get(name, 0.0)
        forecast = forecast.add(result.forecast * weight, fill_value=0.0)
        lower80 = lower80.add(result.lower80 * weight, fill_value=0.0)
        upper80 = upper80.add(result.upper80 * weight, fill_value=0.0)
        lower95 = lower95.add(result.lower95 * weight, fill_value=0.0)
        upper95 = upper95.add(result.upper95 * weight, fill_value=0.0)
    return ForecastResult(pd.Series(dtype=float), forecast, lower80.clip(lower=0), upper80, lower95.clip(lower=0), upper95, None)


def baseline_as_result(series: pd.Series, horizon: int, mode: str) -> ForecastResult:
    baseline = seasonal_baseline(series, horizon, mode)
    return ForecastResult(pd.Series(dtype=float), baseline, baseline, baseline, baseline, baseline, None)
