from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "seasonal_baseline": "01 Sezónní baseline",
    "last_week": "02 Minulý týden",
    "four_week_mean": "03 4týdenní weekday mean",
    "rolling_mean": "04 Rolling mean 10D",
    "seasonal_trend": "05 Sezónní trend",
    "calendar_index": "06 Kalendářový index",
    "static_ytd": "07 Statický YTD",
    "dynamic_rolling": "08 Dynamický rolling",
    "ridge_internal": "09 Ridge interní",
    "ridge_external": "10 Ridge externí",
    "smart_blend": "11 Smart blend",
}

BASE_MODEL_KEYS = [key for key in MODEL_LABELS if key != "smart_blend"]
DAILY_LAGS = (1, 2, 5, 7, 14, 28)
DAILY_WINDOWS = (5, 10, 20)


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
    model_key: str
    label: str
    forecast: pd.Series
    fitted: pd.Series
    lower80: pd.Series
    upper80: pd.Series
    lower95: pd.Series
    upper95: pd.Series
    weights: Optional[Dict[str, float]] = None


@dataclass
class BacktestMetrics:
    model_key: str
    label: str
    wape: float
    mape: float
    bias: float
    mae: float
    n_windows: int


def model_options() -> Dict[str, str]:
    return MODEL_LABELS.copy()


def _metric_summary(actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    err = pred - actual
    abs_err = np.abs(err)
    actual_abs_sum = np.abs(actual).sum()
    wape = float(abs_err.sum() / actual_abs_sum) if actual_abs_sum > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(np.abs(actual) > 1e-9, abs_err / np.abs(actual), np.nan)
    mape = float(np.nanmean(ape)) if np.isfinite(np.nanmean(ape)) else np.nan
    bias = float(err.sum() / actual_abs_sum) if actual_abs_sum > 0 else np.nan
    mae = float(abs_err.mean()) if len(abs_err) else np.nan
    return wape, mape, bias, mae


def _clip_factor(value: float, lower: float = 0.7, upper: float = 1.35) -> float:
    if pd.isna(value):
        return 1.0
    return float(np.clip(value, lower, upper))


def _future_operating_dates(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    if series.empty or horizon <= 0:
        return pd.DatetimeIndex([])
    weekdays = sorted({int(ts.weekday()) for ts in pd.DatetimeIndex(series.index)})
    if not weekdays:
        weekdays = [0, 1, 2, 3, 4]
    dates: List[pd.Timestamp] = []
    current = pd.Timestamp(series.index.max()) + pd.Timedelta(days=1)
    while len(dates) < horizon:
        if current.weekday() in weekdays:
            dates.append(current.normalize())
        current += pd.Timedelta(days=1)
    return pd.DatetimeIndex(dates)


def _seasonal_template(train: pd.Series, future_date: pd.Timestamp) -> float:
    if train.empty:
        return 0.0
    idx = pd.DatetimeIndex(train.index)
    frame = pd.DataFrame({"value": train.values}, index=idx)
    frame["iso_week"] = idx.isocalendar().week.astype(int)
    frame["weekday"] = idx.weekday + 1
    same_slot = frame[(frame["iso_week"] == int(future_date.isocalendar().week)) & (frame["weekday"] == int(future_date.weekday() + 1))]
    if len(same_slot) >= 2:
        return float(same_slot["value"].median())
    same_weekday = frame[frame["weekday"] == int(future_date.weekday() + 1)]["value"]
    if len(same_weekday) >= 3:
        return float(same_weekday.tail(12).median())
    return float(frame["value"].tail(10).mean())


def _same_day_value(extended: pd.Series, day: pd.Timestamp, lag_days: int) -> Optional[float]:
    target = day - pd.Timedelta(days=lag_days)
    if target in extended.index:
        return float(extended.loc[target])
    return None


def _rolling_mean_before(extended: pd.Series, day: pd.Timestamp, window: int) -> float:
    hist = extended[extended.index < day].tail(window)
    if hist.empty:
        return 0.0
    return float(hist.mean())


def _recent_expected_ratio(train: pd.Series, lookback_days: int = 28) -> float:
    if train.empty:
        return 1.0
    recent = train.tail(min(len(train), lookback_days))
    if recent.empty:
        return 1.0
    expected = np.mean([_seasonal_template(train.iloc[:i], date) for i, date in enumerate(recent.index, start=len(train) - len(recent)) if i > 0])
    actual = float(recent.mean())
    if pd.isna(expected) or expected <= 1e-9:
        return 1.0
    return _clip_factor(actual / expected, 0.75, 1.25)


def _ytd_ratio(train: pd.Series, eval_year: int) -> float:
    idx = pd.DatetimeIndex(train.index)
    current = train[idx.year == eval_year]
    prior = train[idx.year < eval_year]
    if current.empty or prior.empty:
        return 1.0
    frame = pd.DataFrame({"value": train.values}, index=idx)
    frame["iso_week"] = idx.isocalendar().week.astype(int)
    frame["weekday"] = idx.weekday + 1
    current_pairs = frame.loc[current.index, ["iso_week", "weekday"]].drop_duplicates()
    hist_vals: List[float] = []
    for year in sorted(set(idx.year)):
        if year >= eval_year:
            continue
        sample = frame[idx.year == year].reset_index(drop=True)
        merged = sample.merge(current_pairs, on=["iso_week", "weekday"], how="inner")
        if not merged.empty:
            hist_vals.append(float(merged["value"].sum()))
    expected = float(np.mean(hist_vals)) if hist_vals else np.nan
    actual = float(current.sum())
    if pd.isna(expected) or expected <= 1e-9:
        return 1.0
    return _clip_factor(actual / expected, 0.75, 1.3)


def _calendar_index_value(train: pd.Series, future_date: pd.Timestamp) -> float:
    idx = pd.DatetimeIndex(train.index)
    frame = pd.DataFrame({"value": train.values}, index=idx)
    frame["year"] = idx.year
    frame["iso_week"] = idx.isocalendar().week.astype(int)
    frame["weekday"] = idx.weekday + 1
    medians = frame.groupby("year")["value"].median()
    ratios = []
    for year, med in medians.items():
        if pd.isna(med) or med <= 1e-9:
            continue
        sample = frame[(frame["year"] == year) & (frame["iso_week"] == int(future_date.isocalendar().week)) & (frame["weekday"] == int(future_date.weekday() + 1))]
        if not sample.empty:
            ratios.append(float(sample["value"].mean() / med))
    if not ratios:
        return _seasonal_template(train, future_date)
    standard_day = float(medians.tail(3).mean())
    idx_ratio = float(np.clip(np.mean(ratios), 0.4, 1.9))
    strength = _ytd_ratio(train, future_date.year)
    return max(0.0, standard_day * idx_ratio * strength)


def _calendar_features(date: pd.Timestamp, position: int) -> Dict[str, float]:
    row: Dict[str, float] = {"trend": float(position)}
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
    return row


def _feature_row(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    i: int,
    exog: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, float]]:
    min_hist = max(max(DAILY_LAGS, default=1), max(DAILY_WINDOWS, default=1))
    if i < min_hist:
        return None
    row = _calendar_features(dates[i], i)
    for lag in DAILY_LAGS:
        row[f"lag_{lag}"] = float(values[i - lag])
    for window in DAILY_WINDOWS:
        hist = values[i - window : i]
        row[f"roll_mean_{window}"] = float(np.nanmean(hist))
        row[f"roll_std_{window}"] = float(np.nanstd(hist))
    row["same_weekday_last_week"] = float(values[i - 7]) if i >= 7 else float(values[i - 1])
    row["same_weekday_last_4_weeks"] = float(np.nanmean([values[i - lag] for lag in (7, 14, 21, 28) if i >= lag]))
    if exog is not None and not exog.empty:
        ex_row = exog.reindex([dates[i]], method="ffill")
        if not ex_row.empty:
            for col, val in ex_row.iloc[0].items():
                row[col] = float(val) if pd.notna(val) else 0.0
    return row


def _design_matrix(series: pd.Series, exog: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    s = series.astype(float).copy()
    dates = pd.DatetimeIndex(s.index)
    values = s.values.astype(float)
    ex = exog.reindex(dates).ffill().bfill() if exog is not None and not exog.empty else None
    rows: List[Dict[str, float]] = []
    y: List[float] = []
    idx: List[pd.Timestamp] = []
    for i in range(len(values)):
        row = _feature_row(values, dates, i, ex)
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
    tmp = X.copy()
    for col in model.columns:
        if col not in tmp.columns:
            tmp[col] = 0.0
    tmp = tmp[model.columns].astype(float)
    xv = tmp.values
    xs = (xv - model.mu) / model.sigma
    xa = np.c_[np.ones(len(xs)), xs]
    return xa @ model.beta


def _ridge_forecast(
    series: pd.Series,
    future_dates: pd.DatetimeIndex,
    exog_hist: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    alpha: float = 12.0,
) -> ForecastResult:
    s = series.dropna().astype(float)
    if s.empty:
        empty = pd.Series(dtype=float)
        return ForecastResult("", "", empty, empty, empty, empty, empty, empty)
    X, y = _design_matrix(s, exog_hist)
    if len(X) < 90:
        baseline = _predict_single_model(s, future_dates, "seasonal_baseline")
        return ForecastResult("", "", baseline, pd.Series(dtype=float), baseline, baseline, baseline, baseline)
    model = _fit_ridge(X, y, alpha=alpha)
    fitted = pd.Series(model.fitted, index=X.index)
    full_values = list(s.values)
    full_dates = list(pd.DatetimeIndex(s.index))
    all_exog = pd.DataFrame()
    if exog_hist is not None and not exog_hist.empty:
        all_exog = exog_hist.copy()
    if exog_future is not None and not exog_future.empty:
        all_exog = pd.concat([all_exog, exog_future], axis=0) if not all_exog.empty else exog_future.copy()
    if not all_exog.empty:
        all_exog = all_exog.sort_index().ffill().bfill()
    preds: List[float] = []
    low80: List[float] = []
    high80: List[float] = []
    low95: List[float] = []
    high95: List[float] = []
    for step, future_date in enumerate(pd.DatetimeIndex(future_dates), start=1):
        full_dates.append(future_date)
        full_values.append(np.nan)
        row = _feature_row(np.asarray(full_values, dtype=float), pd.DatetimeIndex(full_dates), len(full_values) - 1, all_exog)
        xrow = pd.DataFrame([row], index=[future_date]).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        pred = max(0.0, float(_predict_ridge(model, xrow)[0]))
        full_values[-1] = pred
        preds.append(pred)
        widen = np.sqrt(1 + 0.25 * step)
        std = model.resid_std * widen
        low80.append(max(0.0, pred - 1.2816 * std))
        high80.append(pred + 1.2816 * std)
        low95.append(max(0.0, pred - 1.96 * std))
        high95.append(pred + 1.96 * std)
    idx = pd.DatetimeIndex(future_dates)
    return ForecastResult("", "", pd.Series(preds, index=idx), fitted, pd.Series(low80, index=idx), pd.Series(high80, index=idx), pd.Series(low95, index=idx), pd.Series(high95, index=idx))


def _predict_single_model(
    train: pd.Series,
    future_dates: pd.DatetimeIndex,
    model_key: str,
    exog_hist: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
) -> pd.Series:
    history = train.dropna().astype(float).sort_index()
    if history.empty or len(future_dates) == 0:
        return pd.Series(dtype=float)
    if model_key == "ridge_internal":
        return _ridge_forecast(history, future_dates, None, None).forecast
    if model_key == "ridge_external":
        return _ridge_forecast(history, future_dates, exog_hist, exog_future).forecast

    extended = history.copy()
    out: List[float] = []
    for day in pd.DatetimeIndex(future_dates):
        template = _seasonal_template(extended, day)
        recent_factor = _recent_expected_ratio(extended, 28)
        ytd_factor = _ytd_ratio(extended, day.year)
        if model_key == "seasonal_baseline":
            pred = template
        elif model_key == "last_week":
            pred = _same_day_value(extended, day, 7)
            pred = template if pred is None else pred
        elif model_key == "four_week_mean":
            vals = [_same_day_value(extended, day, lag) for lag in (7, 14, 21, 28)]
            vals = [val for val in vals if val is not None]
            pred = float(np.mean(vals)) if vals else template
        elif model_key == "rolling_mean":
            pred = _rolling_mean_before(extended, day, 10)
            if pred <= 0:
                pred = template
        elif model_key == "seasonal_trend":
            pred = template * recent_factor
        elif model_key == "calendar_index":
            pred = _calendar_index_value(extended, day)
        elif model_key == "static_ytd":
            pred = template * ytd_factor
        elif model_key == "dynamic_rolling":
            pred = template * _clip_factor(recent_factor * 0.55 + ytd_factor * 0.45, 0.72, 1.28)
        else:
            pred = template
        pred = max(0.0, float(pred))
        cap = max(float(history.quantile(0.98)) * 1.2, float(history.tail(20).max()) * 1.15, 1.0)
        pred = min(pred, cap)
        out.append(pred)
        extended.loc[day] = pred
    return pd.Series(out, index=pd.DatetimeIndex(future_dates))


def backtest_model(
    series: pd.Series,
    model_key: str,
    horizon: int,
    exog: Optional[pd.DataFrame] = None,
    n_origins: int = 8,
) -> BacktestMetrics:
    s = series.dropna().astype(float).sort_index()
    if len(s) < 60 or horizon <= 0:
        return BacktestMetrics(model_key, MODEL_LABELS[model_key], np.nan, np.nan, np.nan, np.nan, 0)
    origins = list(range(max(45, horizon), len(s) - horizon + 1))[-n_origins:]
    actual_all: List[float] = []
    pred_all: List[float] = []
    windows = 0
    for origin in origins:
        train = s.iloc[:origin]
        actual = s.iloc[origin : origin + horizon]
        if actual.empty:
            continue
        exog_hist = exog.reindex(train.index).ffill().bfill() if exog is not None and not exog.empty else None
        exog_future = exog.reindex(actual.index).ffill().bfill() if exog is not None and not exog.empty else None
        pred = _predict_single_model(train, pd.DatetimeIndex(actual.index), model_key, exog_hist, exog_future)
        if pred.empty:
            continue
        actual_all.extend(actual.values.tolist())
        pred_all.extend(pred.values.tolist())
        windows += 1
    if not windows:
        return BacktestMetrics(model_key, MODEL_LABELS[model_key], np.nan, np.nan, np.nan, np.nan, 0)
    wape, mape, bias, mae = _metric_summary(np.asarray(actual_all, dtype=float), np.asarray(pred_all, dtype=float))
    return BacktestMetrics(model_key, MODEL_LABELS[model_key], wape, mape, bias, mae, windows)


def _blend_weights(metrics_map: Dict[str, BacktestMetrics]) -> Dict[str, float]:
    valid = {key: value.wape for key, value in metrics_map.items() if pd.notna(value.wape) and value.wape > 0}
    if not valid:
        count = len(metrics_map)
        return {key: 1.0 / count for key in metrics_map} if count else {}
    inv = {key: 1.0 / score for key, score in valid.items()}
    total = sum(inv.values())
    return {key: val / total for key, val in inv.items()}


def backtest_smart_blend(
    series: pd.Series,
    horizon: int,
    exog: Optional[pd.DataFrame],
    base_metrics: Dict[str, BacktestMetrics],
    n_origins: int = 8,
) -> BacktestMetrics:
    s = series.dropna().astype(float).sort_index()
    if len(s) < 60 or horizon <= 0:
        return BacktestMetrics("smart_blend", MODEL_LABELS["smart_blend"], np.nan, np.nan, np.nan, np.nan, 0)
    weights = _blend_weights(base_metrics)
    origins = list(range(max(45, horizon), len(s) - horizon + 1))[-n_origins:]
    actual_all: List[float] = []
    pred_all: List[float] = []
    windows = 0
    for origin in origins:
        train = s.iloc[:origin]
        actual = s.iloc[origin : origin + horizon]
        if actual.empty:
            continue
        exog_hist = exog.reindex(train.index).ffill().bfill() if exog is not None and not exog.empty else None
        exog_future = exog.reindex(actual.index).ffill().bfill() if exog is not None and not exog.empty else None
        blend = pd.Series(0.0, index=actual.index)
        for key in BASE_MODEL_KEYS:
            pred = _predict_single_model(train, pd.DatetimeIndex(actual.index), key, exog_hist, exog_future)
            blend = blend.add(pred * weights.get(key, 0.0), fill_value=0.0)
        actual_all.extend(actual.values.tolist())
        pred_all.extend(blend.values.tolist())
        windows += 1
    if not windows:
        return BacktestMetrics("smart_blend", MODEL_LABELS["smart_blend"], np.nan, np.nan, np.nan, np.nan, 0)
    wape, mape, bias, mae = _metric_summary(np.asarray(actual_all, dtype=float), np.asarray(pred_all, dtype=float))
    return BacktestMetrics("smart_blend", MODEL_LABELS["smart_blend"], wape, mape, bias, mae, windows)


def forecast_all_models(
    series: pd.Series,
    future_dates: pd.DatetimeIndex,
    exog: Optional[pd.DataFrame] = None,
    horizon_for_backtest: Optional[int] = None,
) -> Tuple[Dict[str, ForecastResult], Dict[str, BacktestMetrics]]:
    horizon = horizon_for_backtest or max(5, len(future_dates))
    metrics_map = {key: backtest_model(series, key, horizon, exog) for key in BASE_MODEL_KEYS}
    metrics_map["smart_blend"] = backtest_smart_blend(series, horizon, exog, metrics_map)
    results: Dict[str, ForecastResult] = {}
    exog_hist = exog.reindex(series.index).ffill().bfill() if exog is not None and not exog.empty else None
    exog_future = exog.reindex(future_dates).ffill().bfill() if exog is not None and not exog.empty else None
    for key in BASE_MODEL_KEYS:
        if key == "ridge_internal":
            ridge = _ridge_forecast(series, future_dates, None, None)
            results[key] = ForecastResult(key, MODEL_LABELS[key], ridge.forecast, ridge.fitted, ridge.lower80, ridge.upper80, ridge.lower95, ridge.upper95)
        elif key == "ridge_external":
            ridge = _ridge_forecast(series, future_dates, exog_hist, exog_future)
            results[key] = ForecastResult(key, MODEL_LABELS[key], ridge.forecast, ridge.fitted, ridge.lower80, ridge.upper80, ridge.lower95, ridge.upper95)
        else:
            forecast = _predict_single_model(series, future_dates, key, exog_hist, exog_future)
            spread = pd.Series(
                np.maximum(
                    forecast.rolling(3, min_periods=1).std().fillna(0.0),
                    float(series.tail(20).std(ddof=0) if len(series) > 5 else 0.0) * 0.15,
                ),
                index=forecast.index,
            )
            results[key] = ForecastResult(
                key,
                MODEL_LABELS[key],
                forecast,
                pd.Series(dtype=float),
                (forecast - 1.2816 * spread).clip(lower=0),
                forecast + 1.2816 * spread,
                (forecast - 1.96 * spread).clip(lower=0),
                forecast + 1.96 * spread,
            )
    weights = _blend_weights({key: metrics_map[key] for key in BASE_MODEL_KEYS})
    blend = pd.Series(0.0, index=pd.DatetimeIndex(future_dates))
    base = pd.DataFrame({key: result.forecast for key, result in results.items() if not result.forecast.empty})
    for key, result in results.items():
        blend = blend.add(result.forecast * weights.get(key, 0.0), fill_value=0.0)
    spread = base.std(axis=1).fillna(0.0) if not base.empty else pd.Series(0.0, index=pd.DatetimeIndex(future_dates))
    results["smart_blend"] = ForecastResult(
        "smart_blend",
        MODEL_LABELS["smart_blend"],
        blend,
        pd.Series(dtype=float),
        (blend - 1.2816 * spread).clip(lower=0),
        blend + 1.2816 * spread,
        (blend - 1.96 * spread).clip(lower=0),
        blend + 1.96 * spread,
        weights=weights,
    )
    return results, metrics_map


def forecast_suite(
    series: pd.Series,
    future_dates: pd.DatetimeIndex,
    exog: Optional[pd.DataFrame] = None,
    model_key: str = "smart_blend",
    horizon_for_backtest: Optional[int] = None,
) -> Tuple[ForecastResult, Dict[str, BacktestMetrics]]:
    horizon = horizon_for_backtest or max(5, len(future_dates))
    if model_key == "smart_blend":
        results, metrics_map = forecast_all_models(series, future_dates, exog, horizon)
        return results[model_key], metrics_map

    exog_hist = exog.reindex(series.index).ffill().bfill() if exog is not None and not exog.empty else None
    exog_future = exog.reindex(future_dates).ffill().bfill() if exog is not None and not exog.empty else None
    metrics = backtest_model(series, model_key, horizon, exog)
    if model_key == "ridge_internal":
        ridge = _ridge_forecast(series, future_dates, None, None)
        result = ForecastResult(model_key, MODEL_LABELS[model_key], ridge.forecast, ridge.fitted, ridge.lower80, ridge.upper80, ridge.lower95, ridge.upper95)
    elif model_key == "ridge_external":
        ridge = _ridge_forecast(series, future_dates, exog_hist, exog_future)
        result = ForecastResult(model_key, MODEL_LABELS[model_key], ridge.forecast, ridge.fitted, ridge.lower80, ridge.upper80, ridge.lower95, ridge.upper95)
    else:
        forecast = _predict_single_model(series, future_dates, model_key, exog_hist, exog_future)
        spread = pd.Series(
            np.maximum(
                forecast.rolling(3, min_periods=1).std().fillna(0.0),
                float(series.tail(20).std(ddof=0) if len(series) > 5 else 0.0) * 0.15,
            ),
            index=forecast.index,
        )
        result = ForecastResult(
            model_key,
            MODEL_LABELS[model_key],
            forecast,
            pd.Series(dtype=float),
            (forecast - 1.2816 * spread).clip(lower=0),
            forecast + 1.2816 * spread,
            (forecast - 1.96 * spread).clip(lower=0),
            forecast + 1.96 * spread,
        )
    return result, {model_key: metrics}


def build_future_dates_for_week_window(
    series: pd.Series,
    eval_year: int,
    week_start: int,
    week_end: int,
    include_weekend: bool,
) -> pd.DatetimeIndex:
    if series.empty:
        return pd.DatetimeIndex([])
    last_date = pd.Timestamp(series.index.max()).normalize()
    observed_weekdays = {int(ts.weekday() + 1) for ts in pd.DatetimeIndex(series.index)}
    future: List[pd.Timestamp] = []
    for week in range(int(week_start), int(week_end) + 1):
        for weekday in range(1, 8):
            if not include_weekend and weekday > 5:
                continue
            if weekday not in observed_weekdays and weekday > 5:
                continue
            try:
                day = pd.Timestamp(datetime.date.fromisocalendar(int(eval_year), int(week), int(weekday))).normalize()
            except ValueError:
                continue
            if day > last_date:
                future.append(day)
    return pd.DatetimeIndex(sorted(set(future)))


def prepare_operational_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if df is None or df.empty or metric not in df.columns:
        return pd.Series(dtype=float)
    data = df[["date", metric]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["date"]).sort_values("date")
    if data.empty:
        return pd.Series(dtype=float)
    return data.groupby("date")[metric].sum().astype(float)


def backtest_table(metrics_map: Dict[str, BacktestMetrics]) -> pd.DataFrame:
    rows = []
    for key, metrics in metrics_map.items():
        rows.append(
            {
                "model_key": key,
                "model_name": metrics.label,
                "WAPE": metrics.wape,
                "MAPE": metrics.mape,
                "Bias": metrics.bias,
                "MAE": metrics.mae,
                "windows": metrics.n_windows,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["WAPE", "MAE"], ascending=[True, True]).reset_index(drop=True)


def future_operating_dates(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return _future_operating_dates(series, horizon)
