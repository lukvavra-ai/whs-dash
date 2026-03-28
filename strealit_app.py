#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from forecasting import align_known_exog, baseline_as_result, compute_blend_weights, ridge_forecast


st.set_page_config(page_title="Warehouse Dashboard - Balení & Nakládky", layout="wide")


EXPECTED_FILES = {
    "packed_daily": "packed_daily_kpis.csv",
    "packed_shift": "packed_shift_kpis.csv",
    "loaded_daily": "loaded_daily_kpis.csv",
    "loaded_shift": "loaded_shift_kpis.csv",
}

TREND_MODE_OPTIONS = ["hybrid", "calendar_index", "dynamic_rolling", "static_ytd"]
TREND_MODE_LABELS = {
    "hybrid": "Hybrid (kalendář × krátkodobý trend)",
    "calendar_index": "Kalendářový index",
    "dynamic_rolling": "Dynamický rolling",
    "static_ytd": "Statický YTD",
}

ADVANCED_MODELS = [
    "01 Seasonal baseline",
    "02 Same weekday median",
    "03 Rolling median 4 weeks",
    "04 Static YTD",
    "05 Dynamic rolling",
    "06 Calendar index",
    "07 Hybrid calendar x trend",
    "08 Ridge internal",
    "09 Ridge with drivers",
    "10 Smart blend",
]

FAST_BACKTEST_POINTS = 14

TREND_TO_ADVANCED_MODEL = {
    "static_ytd": "04 Static YTD",
    "dynamic_rolling": "05 Dynamic rolling",
    "calendar_index": "06 Calendar index",
    "hybrid": "07 Hybrid calendar x trend",
}

WEEKDAY_SHORT = {1: "Po", 2: "Út", 3: "St", 4: "Čt", 5: "Pá", 6: "So", 7: "Ne"}
WEEKDAY_NAME = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
WEEKDAY_CZ = {
    "Monday": "Po",
    "Tuesday": "Út",
    "Wednesday": "St",
    "Thursday": "Čt",
    "Friday": "Pá",
    "Saturday": "So",
    "Sunday": "Ne",
}


@dataclass
class Sources:
    packed_daily: Optional[pd.DataFrame]
    packed_shift: Optional[pd.DataFrame]
    loaded_daily: Optional[pd.DataFrame]
    loaded_shift: Optional[pd.DataFrame]


@dataclass
class StaffingBundle:
    actuals: Optional[pd.DataFrame]
    forecast: Optional[pd.DataFrame]
    horizon: Optional[pd.DataFrame]
    driver_backtests: Optional[pd.DataFrame]
    ratio_backtests: Optional[pd.DataFrame]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _xkey(iso_week: int, weekday: int) -> str:
    return f"KT{int(iso_week):02d} {WEEKDAY_SHORT.get(int(weekday), str(weekday))}"


def _sort_xkeys(x_values: Sequence[str]) -> List[str]:
    def sort_key(value: str) -> tuple[int, int]:
        parts = str(value).split()
        week = 0
        day = 0
        if parts:
            try:
                week = int(parts[0].replace("KT", ""))
            except ValueError:
                week = 0
        if len(parts) > 1:
            inv = {label: idx for idx, label in WEEKDAY_SHORT.items()}
            day = int(inv.get(parts[1], 0))
        return week, day

    return sorted({str(x) for x in x_values}, key=sort_key)


def _coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for col in ["date", "shift_start"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    if "iso_weekday" in frame.columns and "weekday" not in frame.columns:
        frame["weekday"] = pd.to_numeric(frame["iso_weekday"], errors="coerce")
    if "date" in frame.columns and frame["date"].notna().any():
        iso = frame["date"].dt.isocalendar()
        if "iso_year" not in frame.columns:
            frame["iso_year"] = iso.year.astype(int)
        if "iso_week" not in frame.columns:
            frame["iso_week"] = iso.week.astype(int)
        if "weekday" not in frame.columns:
            frame["weekday"] = iso.day.astype(int)
        frame["weekday"] = pd.to_numeric(frame["weekday"], errors="coerce").astype("Int64")
        if "weekday_name" not in frame.columns:
            frame["weekday_name"] = frame["weekday"].map(WEEKDAY_NAME)
    return frame


def available_metrics(df: pd.DataFrame) -> List[str]:
    skip = {
        "date",
        "shift_start",
        "shift",
        "shift_name",
        "iso_year",
        "iso_week",
        "iso_weekday",
        "weekday",
        "weekday_name",
    }
    out: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    frame = _coerce_date(df)
    metrics = available_metrics(frame)
    if not metrics:
        return pd.DataFrame()
    daily = frame.groupby("date", as_index=False)[metrics].sum()
    return _coerce_date(daily)


def _regularize_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = _aggregate_daily(df)
    if daily.empty or "date" not in daily.columns:
        return pd.DataFrame()
    metrics = available_metrics(daily)
    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    out = daily.set_index("date")[metrics].reindex(idx, fill_value=0.0).reset_index().rename(columns={"index": "date"})
    return _coerce_date(out)


def _daily_series(df: pd.DataFrame, metric: str) -> pd.Series:
    regular = _regularize_daily(df)
    if regular.empty or metric not in regular.columns:
        return pd.Series(dtype=float)
    return regular.set_index("date")[metric].astype(float).sort_index()


def _daily_metric_frame(df: pd.DataFrame, keep_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    regular = _regularize_daily(df)
    if regular.empty:
        return pd.DataFrame()
    metrics = list(keep_cols) if keep_cols is not None else available_metrics(regular)
    metrics = [col for col in metrics if col in regular.columns]
    if not metrics:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    return regular.set_index("date")[metrics].astype(float).sort_index()


@st.cache_data(show_spinner=False)
def load_sources(base_dir_str: str) -> Tuple[Sources, List[str]]:
    base_dir = Path(base_dir_str)
    missing: List[str] = []

    def try_load(key: str) -> Optional[pd.DataFrame]:
        path = base_dir / EXPECTED_FILES[key]
        if not path.exists():
            missing.append(path.name)
            return None
        return _coerce_date(_read_csv(path))

    src = Sources(
        packed_daily=try_load("packed_daily"),
        packed_shift=try_load("packed_shift"),
        loaded_daily=try_load("loaded_daily"),
        loaded_shift=try_load("loaded_shift"),
    )
    return src, missing


@st.cache_data(show_spinner=False)
def load_staffing_bundle(base_dir_str: str) -> StaffingBundle:
    base_dir = Path(base_dir_str) / "staffing_forecast_exports"

    def read_if_exists(name: str, parse_dates: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        path = base_dir / name
        if not path.exists():
            return None
        return pd.read_csv(path, parse_dates=parse_dates, encoding="utf-8-sig")

    return StaffingBundle(
        actuals=read_if_exists("staffing_capacity_actuals.csv", ["date"]),
        forecast=read_if_exists("staffing_forecast_daily.csv", ["date"]),
        horizon=read_if_exists("staffing_horizon_points.csv", ["date"]),
        driver_backtests=read_if_exists("staffing_driver_backtests.csv"),
        ratio_backtests=read_if_exists("staffing_ratio_backtest_summary.csv"),
    )


def _metric_summary(actual: pd.Series, pred: pd.Series) -> tuple[float, float, float, int]:
    actual_s = pd.to_numeric(actual, errors="coerce")
    pred_s = pd.to_numeric(pred, errors="coerce").reindex(actual_s.index)
    mask = actual_s.notna() & pred_s.notna()
    if not mask.any():
        return np.nan, np.nan, np.nan, 0
    actual_v = actual_s[mask].astype(float)
    pred_v = pred_s[mask].astype(float)
    err = pred_v - actual_v
    abs_sum = actual_v.abs().sum()
    wape = float(err.abs().sum() / abs_sum) if abs_sum > 0 else np.nan
    bias = float(err.sum() / abs_sum) if abs_sum > 0 else np.nan
    mae = float(err.abs().mean()) if len(err) else np.nan
    return wape, bias, mae, int(mask.sum())


def _backtest_row(model: str, actual: pd.Series, pred: pd.Series) -> Dict[str, object]:
    wape, bias, mae, n_points = _metric_summary(actual, pred)
    return {
        "model": model,
        "wape": wape,
        "bias": bias,
        "mae": mae,
        "n_points": n_points,
    }


def _predict_same_weekday(series: pd.Series, target_date: pd.Timestamp) -> float:
    hist = series[series.index < target_date]
    same = hist[hist.index.weekday == target_date.weekday()].tail(8)
    if len(same) >= 3:
        return float(same.median())
    recent = hist.tail(28)
    if len(recent) >= 5:
        return float(recent.median())
    return float(hist.tail(1).mean()) if not hist.empty else 0.0


def _predict_rolling_median(series: pd.Series, target_date: pd.Timestamp) -> float:
    hist = series[series.index < target_date]
    recent = hist.tail(28)
    if len(recent) >= 7:
        return float(recent.median())
    return float(hist.tail(1).mean()) if not hist.empty else 0.0


def _recursive_forecast(series: pd.Series, future_idx: pd.DatetimeIndex, predictor) -> pd.Series:
    full = series.copy()
    preds: List[float] = []
    for target_date in future_idx:
        pred = max(0.0, float(predictor(full, pd.Timestamp(target_date))))
        full.loc[pd.Timestamp(target_date)] = pred
        preds.append(pred)
    return pd.Series(preds, index=future_idx)


def _apply_week_filter(df: pd.DataFrame, include_weekend: bool) -> pd.DataFrame:
    if include_weekend:
        return df.copy()
    return df[df["weekday"].astype(int).between(1, 5)].copy()


def _legacy_components(
    df: pd.DataFrame,
    metric: str,
    target_date: pd.Timestamp,
    lookback_weeks: int,
    include_weekend: bool,
) -> Dict[str, float]:
    hist = df[df["date"] < target_date].copy()
    hist = hist[hist["date"].notna()].copy()
    if hist.empty:
        return {"baseline": 0.0, "static_factor": 1.0, "dynamic_factor": 1.0, "idx_value": 1.0, "calendar_base": 0.0}

    target_iso = target_date.isocalendar()
    eval_year = int(target_iso.year)
    train_years = sorted(int(y) for y in hist["iso_year"].dropna().unique().tolist() if int(y) < eval_year)
    if not train_years:
        last_val = float(hist[metric].tail(1).mean()) if metric in hist.columns else 0.0
        return {"baseline": last_val, "static_factor": 1.0, "dynamic_factor": 1.0, "idx_value": 1.0, "calendar_base": last_val}

    hist = _apply_week_filter(hist, include_weekend)
    current_year = hist[hist["iso_year"] == eval_year].copy()
    baseline = (
        hist[hist["iso_year"].isin(train_years)]
        .groupby(["iso_week", "weekday"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    def baseline_value(iso_week: int, weekday: int) -> float:
        match = baseline[(baseline["iso_week"] == iso_week) & (baseline["weekday"] == weekday)][metric]
        if not match.empty:
            return float(match.mean())
        fallback = baseline[baseline["weekday"] == weekday][metric]
        if not fallback.empty:
            return float(fallback.mean())
        return float(hist[metric].tail(28).mean()) if not hist.empty else 0.0

    wk = int(target_iso.week)
    wd = int(target_iso.weekday)
    base = baseline_value(wk, wd)

    static_factor = 1.0
    current_ytd = current_year.copy()
    if not current_ytd.empty:
        merged = current_ytd[["iso_week", "weekday", metric]].merge(
            baseline,
            on=["iso_week", "weekday"],
            how="left",
            suffixes=("", "_base"),
        )
        expected_ytd = float(merged[f"{metric}_base"].fillna(0.0).sum())
        actual_ytd = float(current_ytd[metric].sum())
        static_factor = (actual_ytd / expected_ytd) if expected_ytd > 0 else 1.0

    dynamic_factor = static_factor
    if lookback_weeks > 0 and not current_year.empty:
        end_ref = min(pd.Timestamp(target_date) - pd.Timedelta(days=1), current_year["date"].max())
        start_ref = end_ref - pd.Timedelta(days=7 * lookback_weeks)
        recent = current_year[(current_year["date"] >= start_ref) & (current_year["date"] <= end_ref)].copy()
        if not recent.empty:
            merged = recent[["iso_week", "weekday", metric]].merge(
                baseline,
                on=["iso_week", "weekday"],
                how="left",
                suffixes=("", "_base"),
            )
            expected_recent = float(merged[f"{metric}_base"].fillna(0.0).sum())
            actual_recent = float(recent[metric].sum())
            dynamic_factor = (actual_recent / expected_recent) if expected_recent > 0 else static_factor

    train_filtered = hist[hist["iso_year"].isin(train_years)].copy()
    std_by_year: Dict[int, float] = {}
    for year in train_years:
        vals = train_filtered[train_filtered["iso_year"] == year][metric].astype(float)
        vals = vals[vals > 0]
        if not vals.empty:
            std_by_year[year] = float(vals.median())
    if std_by_year:
        ordered = sorted(std_by_year)
        weights = {year: idx + 1 for idx, year in enumerate(ordered)}
        standard_day_hist = sum(std_by_year[year] * weights[year] for year in ordered) / sum(weights.values())
    else:
        standard_day_hist = base

    idx_rows: List[pd.DataFrame] = []
    for year, std_value in std_by_year.items():
        if std_value <= 0:
            continue
        year_df = train_filtered[train_filtered["iso_year"] == year].groupby(["iso_week", "weekday"], as_index=False)[metric].sum()
        year_df["idx"] = year_df[metric].astype(float) / float(std_value)
        idx_rows.append(year_df[["iso_week", "weekday", "idx"]])
    if idx_rows:
        idx_profile = pd.concat(idx_rows, ignore_index=True).groupby(["iso_week", "weekday"], as_index=False)["idx"].mean()
        idx_match = idx_profile[(idx_profile["iso_week"] == wk) & (idx_profile["weekday"] == wd)]["idx"]
        idx_value = float(idx_match.mean()) if not idx_match.empty else 1.0
    else:
        idx_value = 1.0

    strength_factor = 1.0
    if not current_year.empty:
        pairs = current_year[["iso_week", "weekday"]].drop_duplicates()
        hist_sums = []
        for year in train_years:
            year_df = train_filtered[train_filtered["iso_year"] == year].merge(pairs, on=["iso_week", "weekday"], how="inner")
            if not year_df.empty:
                hist_sums.append(float(year_df[metric].sum()))
        hist_mean = float(np.mean(hist_sums)) if hist_sums else 0.0
        current_sum = float(current_year[metric].sum())
        if hist_mean > 0:
            strength_factor = current_sum / hist_mean

    calendar_base = standard_day_hist * strength_factor
    return {
        "baseline": float(max(base, 0.0)),
        "static_factor": float(max(static_factor, 0.0)),
        "dynamic_factor": float(max(dynamic_factor, 0.0)),
        "idx_value": float(max(idx_value, 0.0)),
        "calendar_base": float(max(calendar_base, 0.0)),
    }


def _legacy_point_forecast(
    df: pd.DataFrame,
    metric: str,
    target_date: pd.Timestamp,
    mode: str,
    lookback_weeks: int,
    include_weekend: bool,
    manual_adj_pct: float = 0.0,
) -> float:
    comp = _legacy_components(df, metric, target_date, lookback_weeks, include_weekend)
    baseline = comp["baseline"]
    static_factor = comp["static_factor"]
    dynamic_factor = comp["dynamic_factor"]
    idx_value = comp["idx_value"]
    calendar_base = comp["calendar_base"]
    manual_factor = 1.0 + (manual_adj_pct / 100.0)

    if mode == "static_ytd":
        yhat = baseline * static_factor
    elif mode == "dynamic_rolling":
        yhat = baseline * dynamic_factor
    elif mode == "calendar_index":
        yhat = calendar_base * idx_value
    else:
        adj = (dynamic_factor / static_factor) if static_factor > 0 else dynamic_factor
        yhat = calendar_base * idx_value * adj
    return max(0.0, float(yhat * manual_factor))


def _legacy_forecast_series(
    df: pd.DataFrame,
    metric: str,
    future_idx: pd.DatetimeIndex,
    mode: str,
    lookback_weeks: int,
    include_weekend: bool,
    manual_adj_pct: float = 0.0,
) -> pd.Series:
    regular = _regularize_daily(df)
    preds = [
        _legacy_point_forecast(
            regular,
            metric=metric,
            target_date=pd.Timestamp(day),
            mode=mode,
            lookback_weeks=lookback_weeks,
            include_weekend=include_weekend,
            manual_adj_pct=manual_adj_pct,
        )
        for day in future_idx
    ]
    return pd.Series(preds, index=future_idx)


def _build_exog_features(
    source_df: pd.DataFrame,
    linked_df: Optional[pd.DataFrame],
    target_metric: str,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    self_metrics = [col for col in available_metrics(source_df) if col != target_metric]
    self_frame = _daily_metric_frame(source_df, self_metrics)
    if not self_frame.empty:
        parts.append(self_frame.add_prefix("self_"))

    if linked_df is not None and not linked_df.empty:
        linked_metrics = available_metrics(linked_df)
        linked_frame = _daily_metric_frame(linked_df, linked_metrics)
        if not linked_frame.empty:
            parts.append(linked_frame.add_prefix("linked_"))

    if not parts:
        return pd.DataFrame(index=target_index)

    exog = pd.concat(parts, axis=1).sort_index()
    full_idx = pd.date_range(min(exog.index.min(), target_index.min()), max(exog.index.max(), target_index.max()), freq="D")
    exog = exog.reindex(full_idx, fill_value=0.0).ffill().bfill().fillna(0.0)
    return exog


def _predict_model_one_step(
    model: str,
    raw_df: pd.DataFrame,
    series: pd.Series,
    exog_shifted: pd.DataFrame,
    metric: str,
    target_date: pd.Timestamp,
    lookback_weeks: int,
    include_weekend: bool,
) -> float:
    train_series = series[series.index < target_date]
    if train_series.empty:
        return 0.0

    if model == "01 Seasonal baseline":
        forecast = baseline_as_result(train_series, 1, "daily").forecast
        return max(0.0, float(forecast.iloc[0]))
    if model == "02 Same weekday median":
        return max(0.0, _predict_same_weekday(train_series, target_date))
    if model == "03 Rolling median 4 weeks":
        return max(0.0, _predict_rolling_median(train_series, target_date))
    if model == "04 Static YTD":
        return _legacy_point_forecast(raw_df, metric, target_date, "static_ytd", lookback_weeks, include_weekend)
    if model == "05 Dynamic rolling":
        return _legacy_point_forecast(raw_df, metric, target_date, "dynamic_rolling", lookback_weeks, include_weekend)
    if model == "06 Calendar index":
        return _legacy_point_forecast(raw_df, metric, target_date, "calendar_index", lookback_weeks, include_weekend)
    if model == "07 Hybrid calendar x trend":
        return _legacy_point_forecast(raw_df, metric, target_date, "hybrid", lookback_weeks, include_weekend)
    if model == "08 Ridge internal":
        forecast = ridge_forecast(train_series, 1, "daily").forecast
        return max(0.0, float(forecast.iloc[0]))
    if model == "09 Ridge with drivers":
        if exog_shifted.empty:
            forecast = ridge_forecast(train_series, 1, "daily").forecast
            return max(0.0, float(forecast.iloc[0]))
        hist = exog_shifted.reindex(train_series.index).ffill().bfill()
        fut = exog_shifted.reindex(pd.DatetimeIndex([target_date])).ffill().bfill()
        forecast = ridge_forecast(train_series, 1, "daily", exog_hist=hist, exog_future=fut).forecast
        return max(0.0, float(forecast.iloc[0]))
    raise ValueError(f"Unknown model: {model}")


def _model_scores_for_weights(score_df: pd.DataFrame) -> Dict[str, float]:
    metric_map = {}
    for row in score_df.itertuples():
        if row.model == "10 Smart blend":
            continue
        metric_map[row.model] = type("Metrics", (), {"wape": row.wape})()
    return compute_blend_weights(metric_map)


@st.cache_data(show_spinner=False, max_entries=64)
def compute_model_suite(
    raw_df: pd.DataFrame,
    linked_df: Optional[pd.DataFrame],
    metric: str,
    horizon_days: int,
    lookback_weeks: int,
    include_weekend: bool,
) -> Dict[str, object]:
    series = _daily_series(raw_df, metric)
    if series.empty:
        return {"future": pd.DataFrame(), "backtest": pd.DataFrame(), "scores": pd.DataFrame(), "weights": pd.DataFrame()}

    horizon_days = max(int(horizon_days), 1)
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    exog_full = _build_exog_features(raw_df, linked_df, metric, series.index)
    exog_shifted = exog_full.shift(1).ffill().bfill() if not exog_full.empty else pd.DataFrame(index=series.index)

    future = pd.DataFrame(index=future_idx)
    future["01 Seasonal baseline"] = baseline_as_result(series, horizon_days, "daily").forecast.reindex(future_idx).values
    future["02 Same weekday median"] = _recursive_forecast(series, future_idx, _predict_same_weekday).values
    future["03 Rolling median 4 weeks"] = _recursive_forecast(series, future_idx, _predict_rolling_median).values
    future["04 Static YTD"] = _legacy_forecast_series(raw_df, metric, future_idx, "static_ytd", lookback_weeks, include_weekend).values
    future["05 Dynamic rolling"] = _legacy_forecast_series(raw_df, metric, future_idx, "dynamic_rolling", lookback_weeks, include_weekend).values
    future["06 Calendar index"] = _legacy_forecast_series(raw_df, metric, future_idx, "calendar_index", lookback_weeks, include_weekend).values
    future["07 Hybrid calendar x trend"] = _legacy_forecast_series(raw_df, metric, future_idx, "hybrid", lookback_weeks, include_weekend).values
    future["08 Ridge internal"] = ridge_forecast(series, horizon_days, "daily").forecast.reindex(future_idx).values

    hist_exog, future_exog = align_known_exog(exog_full, future_idx, lag_periods=1) if not exog_full.empty else (pd.DataFrame(), pd.DataFrame())
    hist_exog = hist_exog.reindex(series.index).ffill().bfill() if not hist_exog.empty else hist_exog
    if hist_exog is not None and not hist_exog.empty:
        future["09 Ridge with drivers"] = ridge_forecast(series, horizon_days, "daily", exog_hist=hist_exog, exog_future=future_exog).forecast.reindex(future_idx).values
    else:
        future["09 Ridge with drivers"] = future["08 Ridge internal"].values

    min_train = min(max(90, len(series) // 2), max(len(series) - 21, 30))
    eval_dates = list(series.index[min_train:]) if len(series) > min_train else []
    if len(eval_dates) > FAST_BACKTEST_POINTS:
        eval_dates = eval_dates[-FAST_BACKTEST_POINTS:]

    regular_raw = _regularize_daily(raw_df)
    backtest_rows: List[Dict[str, object]] = []
    for target_date in eval_dates:
        row: Dict[str, object] = {"date": target_date, "actual": float(series.loc[target_date])}
        for model in ADVANCED_MODELS[:-1]:
            row[model] = _predict_model_one_step(
                model=model,
                raw_df=regular_raw,
                series=series,
                exog_shifted=exog_shifted,
                metric=metric,
                target_date=pd.Timestamp(target_date),
                lookback_weeks=lookback_weeks,
                include_weekend=include_weekend,
            )
        backtest_rows.append(row)
    backtest = pd.DataFrame(backtest_rows)

    score_rows: List[Dict[str, object]] = []
    if not backtest.empty:
        actual = backtest.set_index("date")["actual"]
        for model in ADVANCED_MODELS[:-1]:
            score_rows.append(_backtest_row(model, actual, backtest.set_index("date")[model]))

    scores = pd.DataFrame(score_rows).sort_values(["wape", "mae"], na_position="last").reset_index(drop=True) if score_rows else pd.DataFrame()
    weights = _model_scores_for_weights(scores) if not scores.empty else {}

    if weights:
        future["10 Smart blend"] = 0.0
        for model, weight in weights.items():
            if model in future.columns:
                future["10 Smart blend"] = future["10 Smart blend"] + future[model] * float(weight)
    else:
        future["10 Smart blend"] = future["07 Hybrid calendar x trend"].values

    if not backtest.empty:
        backtest["10 Smart blend"] = 0.0
        if weights:
            for model, weight in weights.items():
                if model in backtest.columns:
                    backtest["10 Smart blend"] = backtest["10 Smart blend"] + backtest[model] * float(weight)
        else:
            backtest["10 Smart blend"] = backtest["07 Hybrid calendar x trend"]
        actual = backtest.set_index("date")["actual"]
        score_rows.append(_backtest_row("10 Smart blend", actual, backtest.set_index("date")["10 Smart blend"]))
        scores = pd.DataFrame(score_rows).sort_values(["wape", "mae"], na_position="last").reset_index(drop=True)

    weight_df = pd.DataFrame(
        [{"model": model, "weight": float(weight)} for model, weight in weights.items()]
    ).sort_values("weight", ascending=False) if weights else pd.DataFrame(columns=["model", "weight"])

    future = future.reset_index().rename(columns={"index": "date"})
    return {"future": future, "backtest": backtest, "scores": scores, "weights": weight_df}


def _future_dates_for_window(last_actual_date: pd.Timestamp, eval_year: int, week_start: int, week_end: int, include_weekend: bool) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    for week in range(int(week_start), int(week_end) + 1):
        for weekday in range(1, 8):
            if not include_weekend and weekday > 5:
                continue
            try:
                day = pd.Timestamp(dt.date.fromisocalendar(int(eval_year), int(week), int(weekday)))
            except ValueError:
                continue
            if day > last_actual_date:
                dates.append(day)
    return sorted(dates)


def _week_day_grid(
    df: pd.DataFrame,
    metric: str,
    week_start: int,
    week_end: int,
    include_weekend: bool,
) -> pd.DataFrame:
    frame = _coerce_date(df)
    frame = frame[frame["iso_week"].between(week_start, week_end, inclusive="both")].copy()
    if not include_weekend:
        frame = frame[frame["weekday"].astype(int).between(1, 5)].copy()
    grouped = frame.groupby(["iso_year", "iso_week", "weekday"], dropna=False, as_index=False)[metric].sum()
    grouped["x"] = grouped.apply(lambda row: _xkey(row["iso_week"], row["weekday"]), axis=1)
    x_categories = []
    for week in range(int(week_start), int(week_end) + 1):
        for weekday in range(1, 8):
            if not include_weekend and weekday > 5:
                continue
            x_categories.append(_xkey(week, weekday))
    grouped["x"] = pd.Categorical(grouped["x"], categories=x_categories, ordered=True)
    return grouped.sort_values(["x", "iso_year"]).reset_index(drop=True)


def _format_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1%}"


def _format_num(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.{digits}f}".replace(",", " ")


def _score_lookup(scores: pd.DataFrame, model: str) -> Optional[pd.Series]:
    if scores is None or scores.empty:
        return None
    match = scores[scores["model"] == model]
    if match.empty:
        return None
    return match.iloc[0]


def _apply_manual_adjustment(series: pd.Series, manual_adj_pct: float) -> pd.Series:
    factor = 1.0 + (float(manual_adj_pct) / 100.0)
    return series.astype(float) * factor


def _render_backtest_summary(scores: pd.DataFrame, model_name: str, backtest: pd.DataFrame) -> None:
    score = _score_lookup(scores, model_name)
    if score is None:
        st.info("K tomuhle modelu zatím nemám backtest.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model_name)
    c2.metric("WAPE", _format_pct(float(score["wape"])))
    c3.metric("Bias", _format_pct(float(score["bias"])))
    c4.metric("MAE", _format_num(float(score["mae"]), 2))

    if backtest is None or backtest.empty or model_name not in backtest.columns:
        return

    tail = backtest[["date", "actual", model_name]].dropna().tail(10).copy()
    if tail.empty:
        return
    tail["abs_error"] = (tail["actual"] - tail[model_name]).abs()
    tail = tail.rename(columns={model_name: "forecast"})
    st.caption("Poslední backtest body pro aktivní model")
    st.dataframe(tail, use_container_width=True, hide_index=True)


def _render_operational_tab(
    tab_name: str,
    daily_df: pd.DataFrame,
    shift_df: Optional[pd.DataFrame],
    linked_daily_df: Optional[pd.DataFrame],
    metric_labels: Dict[str, str],
    default_metrics: List[str],
    shift_format,
    prefix: str,
    include_weekend_toggle: bool,
) -> None:
    st.subheader(tab_name)

    shift_opts = ["all"]
    shift_col = None
    if shift_df is not None and not shift_df.empty:
        for candidate in ["shift_name", "shift"]:
            if candidate in shift_df.columns:
                shift_col = candidate
                break
        if shift_col:
            shift_opts += sorted(shift_df[shift_col].dropna().astype(str).unique().tolist())

    shift_sel = st.selectbox("Směna", shift_opts, index=0, key=f"{prefix}_shift_sel", format_func=shift_format)

    if shift_sel == "all" or shift_df is None or shift_df.empty or shift_col is None:
        df = daily_df.copy()
    else:
        frame = shift_df[shift_df[shift_col].astype(str) == shift_sel].copy()
        df = _aggregate_daily(frame)

    if df.empty:
        st.warning("Pro zvolený výřez nemám data.")
        return

    metrics_all = [metric for metric in metric_labels if metric in df.columns]
    if not metrics_all:
        st.error("Nenašel jsem očekávané metriky.")
        return

    years_all = sorted(df["iso_year"].dropna().astype(int).unique().tolist())
    eval_year = max(years_all)

    st.markdown("### Pohled: vybrané KT → dny vedle sebe + predikce")
    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
    with c1:
        metrics_sel = st.multiselect(
            "Metriky v grafech",
            options=metrics_all,
            default=[metric for metric in default_metrics if metric in metrics_all] or metrics_all[:1],
            key=f"{prefix}_metrics_sel",
            format_func=lambda item: metric_labels.get(item, item),
        )
    with c2:
        week_start = int(st.number_input("KT od", min_value=1, max_value=53, value=6, step=1, key=f"{prefix}_week_start"))
    with c3:
        week_end = int(st.number_input("KT do", min_value=1, max_value=53, value=14, step=1, key=f"{prefix}_week_end"))
    with c4:
        if include_weekend_toggle:
            include_weekend = st.checkbox("Zahrnout víkendy", value=False, key=f"{prefix}_weekend")
        else:
            include_weekend = False
            st.checkbox("Zahrnout víkendy", value=False, key=f"{prefix}_weekend_disabled", disabled=True)

    c5, c6 = st.columns([4, 3])
    with c5:
        years_sel = st.multiselect("Roky v grafu", options=years_all, default=years_all, key=f"{prefix}_years")
    with c6:
        show_forecast = st.checkbox(f"Zobrazit predikci ({eval_year})", value=True, key=f"{prefix}_show_fc")

    if week_end < week_start:
        week_start, week_end = week_end, week_start
    if eval_year not in years_sel:
        years_sel = years_sel + [eval_year]
    if not metrics_sel:
        st.info("Vyber aspoň jednu metriku.")
        return

    state_key = f"{prefix}_metrics_order"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(metrics_sel)
    order = [metric for metric in st.session_state[state_key] if metric in metrics_sel]
    for metric in metrics_sel:
        if metric not in order:
            order.append(metric)
    st.session_state[state_key] = order

    st.markdown("#### Pořadí metrik")
    op1, op2, op3, op4 = st.columns([3, 1, 1, 3])
    with op1:
        picked = st.selectbox("Vybraná metrika", order, key=f"{prefix}_order_pick", format_func=lambda item: metric_labels.get(item, item))
    with op2:
        if st.button("Nahoru", key=f"{prefix}_order_up"):
            idx = order.index(picked)
            if idx > 0:
                order[idx - 1], order[idx] = order[idx], order[idx - 1]
                st.session_state[state_key] = order
    with op3:
        if st.button("Dolů", key=f"{prefix}_order_down"):
            idx = order.index(picked)
            if idx < len(order) - 1:
                order[idx + 1], order[idx] = order[idx], order[idx + 1]
                st.session_state[state_key] = order
    with op4:
        st.caption("Pořadí se použije pro vykreslení grafů pod sebou.")

    fc1, fc2, fc3 = st.columns([2, 3, 2])
    with fc1:
        trend_mode = st.selectbox(
            "Trend mód",
            TREND_MODE_OPTIONS,
            index=0,
            key=f"{prefix}_trend_mode",
            disabled=not show_forecast,
            format_func=lambda item: TREND_MODE_LABELS[item],
        )
    with fc2:
        lookback_weeks = int(st.slider("Trend okno (týdny)", 2, 26, 8, key=f"{prefix}_lookback", disabled=not show_forecast))
    with fc3:
        manual_adj = int(st.slider("Korekce predikce (%)", -30, 30, 0, key=f"{prefix}_adj", disabled=not show_forecast))

    selected_model_name = TREND_TO_ADVANCED_MODEL[trend_mode]
    grids: Dict[str, pd.DataFrame] = {}
    suites: Dict[str, Dict[str, object]] = {}
    forecast_frames: Dict[str, pd.DataFrame] = {}

    last_actual_date = pd.to_datetime(df["date"]).max()
    future_window_dates = _future_dates_for_window(last_actual_date, eval_year, week_start, week_end, include_weekend)
    max_horizon = max((future_window_dates[-1] - last_actual_date).days, 1) if future_window_dates else 1

    for metric in order:
        st.markdown(f"#### {metric_labels.get(metric, metric)}")
        grid = _week_day_grid(df[df["iso_year"].isin(years_sel)], metric, week_start, week_end, include_weekend)
        grids[metric] = grid
        if grid.empty:
            st.info("V tomto výřezu nejsou data.")
            continue

        suite = compute_model_suite(
            raw_df=df,
            linked_df=linked_daily_df,
            metric=metric,
            horizon_days=max_horizon,
            lookback_weeks=lookback_weeks,
            include_weekend=include_weekend,
        )
        suites[metric] = suite

        forecast_df = pd.DataFrame(columns=["date", "forecast", "iso_week", "weekday", "x"])
        if show_forecast and future_window_dates and not suite["future"].empty and selected_model_name in suite["future"].columns:
            future_table = suite["future"].copy()
            future_table["date"] = pd.to_datetime(future_table["date"])
            series = future_table.set_index("date")[selected_model_name]
            series = _apply_manual_adjustment(series, manual_adj)
            series = series.reindex(pd.DatetimeIndex(future_window_dates))
            forecast_df = series.reset_index().rename(columns={"index": "date", selected_model_name: "forecast"})
            forecast_df["iso_week"] = forecast_df["date"].dt.isocalendar().week.astype(int)
            forecast_df["weekday"] = forecast_df["date"].dt.isocalendar().day.astype(int)
            forecast_df["x"] = forecast_df.apply(lambda row: _xkey(row["iso_week"], row["weekday"]), axis=1)
            forecast_df = forecast_df.dropna(subset=["forecast"])
        forecast_frames[metric] = forecast_df

        score = _score_lookup(suite["scores"], selected_model_name)
        if show_forecast and score is not None:
            st.caption(
                f"Aktivní forecast: {selected_model_name} | "
                f"WAPE {_format_pct(float(score['wape']))} | "
                f"Bias {_format_pct(float(score['bias']))} | "
                f"MAE {_format_num(float(score['mae']), 2)}"
            )

        long_actual = grid.rename(columns={metric: "value"}).copy()
        long_actual["series"] = long_actual["iso_year"].astype(int).astype(str)
        long_plot = long_actual[["x", "value", "series"]].copy()

        if show_forecast and not forecast_df.empty:
            fc_long = forecast_df[["x", "forecast"]].rename(columns={"forecast": "value"}).copy()
            fc_long["series"] = f"Predikce {eval_year}"
            long_plot = pd.concat([long_plot, fc_long], ignore_index=True)

        fig = px.line(long_plot, x="x", y="value", color="series", markers=True)
        fig.update_traces(connectgaps=False)
        fig.update_layout(
            xaxis_title="KT + den",
            yaxis_title=metric_labels.get(metric, metric),
            legend_title_text="Rok / predikce",
        )
        for trace in fig.data:
            name = str(trace.name)
            if name == f"Predikce {eval_year}":
                trace.update(line=dict(width=4, dash="dash"))
            elif name == str(eval_year):
                trace.update(line=dict(width=4))
            else:
                trace.update(opacity=0.45, line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Detail a přesnost modelu")

    detail_metric = st.selectbox(
        "Detail metrika",
        options=order,
        index=0,
        key=f"{prefix}_detail_metric",
        format_func=lambda item: metric_labels.get(item, item),
    )
    detail_grid = grids.get(detail_metric, pd.DataFrame())
    detail_suite = suites.get(detail_metric, {})
    detail_fc = forecast_frames.get(detail_metric, pd.DataFrame())

    if detail_grid.empty:
        st.info("Pro detail metriku nejsou data.")
        return

    pivot = detail_grid.pivot_table(index="x", columns="iso_year", values=detail_metric, aggfunc="sum", fill_value=0).reset_index()
    keep_years = [year for year in years_sel if year in pivot.columns]
    pivot = pivot[["x"] + keep_years]

    if show_forecast and not detail_fc.empty:
        fc_col = f"Predikce {eval_year}"
        fc_table = detail_fc[["x", "forecast"]].rename(columns={"forecast": fc_col})
        pivot = pivot.merge(fc_table, on="x", how="left")

    pivot["sort_x"] = pivot["x"].astype(str)
    pivot = pivot.sort_values("sort_x", key=lambda s: s.map({val: idx for idx, val in enumerate(_sort_xkeys(s))})).drop(columns=["sort_x"])
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    if show_forecast and detail_suite:
        _render_backtest_summary(detail_suite.get("scores", pd.DataFrame()), selected_model_name, detail_suite.get("backtest", pd.DataFrame()))


def tab_baleni(src: Sources) -> None:
    if src.packed_daily is None or src.packed_daily.empty:
        st.error("Chybí packed_daily_kpis.csv")
        return

    metric_labels = {
        "binhits": "Binhits",
        "gross_tons": "GW (t)",
        "cartons_count": "Kartony",
        "pallets_count": "Palety",
        "vydejky_unique": "Výdejky",
    }

    _render_operational_tab(
        tab_name="Balení (Kompletace) 📦",
        daily_df=src.packed_daily,
        shift_df=src.packed_shift,
        linked_daily_df=src.loaded_daily,
        metric_labels=metric_labels,
        default_metrics=["binhits", "gross_tons", "pallets_count", "cartons_count"],
        shift_format=lambda item: "Celkem" if item == "all" else ("Denní" if item == "day" else ("Noční" if item == "night" else item)),
        prefix="pk",
        include_weekend_toggle=True,
    )


def tab_nakladky(src: Sources) -> None:
    if src.loaded_daily is None or src.loaded_daily.empty:
        st.error("Chybí loaded_daily_kpis.csv")
        return

    metric_labels = {
        "trips_total": "Trips celkem",
        "trips_export": "Trips export",
        "trips_europe": "Trips Evropa",
        "containers_count": "Kontejnery",
        "orders_nunique": "Objednávky",
        "gross_tons": "GW (t)",
    }

    _render_operational_tab(
        tab_name="Nakládky (Výdeje) 🚚",
        daily_df=src.loaded_daily,
        shift_df=src.loaded_shift,
        linked_daily_df=src.packed_daily,
        metric_labels=metric_labels,
        default_metrics=["trips_total", "gross_tons", "containers_count"],
        shift_format=lambda item: "Celkem" if item == "all" else ("Ranní" if item == "morning" else ("Odpolední" if item == "afternoon" else item)),
        prefix="ld",
        include_weekend_toggle=False,
    )


def _render_prediction_models(src: Sources) -> None:
    source_options = {
        "Balení (Kompletace)": ("packed", src.packed_daily, src.loaded_daily),
        "Nakládky (Výdeje)": ("loaded", src.loaded_daily, src.packed_daily),
    }
    choice = st.selectbox("Zdroj", list(source_options.keys()), key="pred_source")
    source_key, df, linked_df = source_options[choice]

    if df is None or df.empty:
        st.warning("Pro zvolený zdroj nemám data.")
        return

    metric_defaults = {
        "packed": "binhits",
        "loaded": "trips_total",
    }
    metric_labels = {
        "binhits": "Binhits",
        "gross_tons": "GW (t)",
        "cartons_count": "Kartony",
        "pallets_count": "Palety",
        "vydejky_unique": "Výdejky",
        "trips_total": "Trips celkem",
        "trips_export": "Trips export",
        "trips_europe": "Trips Evropa",
        "containers_count": "Kontejnery",
        "orders_nunique": "Objednávky",
    }

    metrics = available_metrics(df)
    default_metric = metric_defaults[source_key] if metric_defaults[source_key] in metrics else metrics[0]
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        metric = st.selectbox("Metrika", metrics, index=metrics.index(default_metric), key="pred_metric", format_func=lambda item: metric_labels.get(item, item))
    with c2:
        model_name = st.selectbox("Model", ADVANCED_MODELS, index=ADVANCED_MODELS.index("10 Smart blend"), key="pred_model")
    with c3:
        horizon = int(st.slider("Horizont (dní)", 5, 120, 30, key="pred_horizon"))
    with c4:
        manual_adj = int(st.slider("Korekce (%)", -30, 30, 0, key="pred_adj"))

    lookback_weeks = int(st.slider("Trend okno pro trendové modely", 2, 26, 8, key="pred_lookback"))
    if source_key == "packed":
        include_weekend = bool(st.checkbox("Zahrnout víkendy do zobrazení a forecastu", value=False, key="pred_weekend"))
    else:
        include_weekend = False
        st.caption("Nakládky se na kartě Predikce zobrazují bez víkendů.")

    effective_horizon = int(horizon if include_weekend else max(horizon + 12, round(horizon * 1.6)))

    with st.spinner("Počítám forecast a backtest modelů..."):
        suite = compute_model_suite(df, linked_df, metric, effective_horizon, lookback_weeks, include_weekend)
    if suite["future"].empty:
        st.info("Na predikci zatím není dost dat.")
        return

    future = suite["future"].copy()
    future["date"] = pd.to_datetime(future["date"])
    if not include_weekend:
        future = future[future["date"].dt.weekday < 5].copy()
    future = future.head(horizon).copy()
    forecast_series = future.set_index("date")[model_name]
    forecast_series = _apply_manual_adjustment(forecast_series, manual_adj)
    scores = suite["scores"].copy()
    backtest = suite["backtest"].copy()
    weights = suite["weights"].copy()

    actual = _daily_series(df, metric)
    if not include_weekend:
        actual = actual[actual.index.weekday < 5]
    tail = actual.tail(90).reset_index()
    tail.columns = ["date", "value"]
    tail["series"] = "Historie"

    future_plot = forecast_series.reset_index()
    future_plot.columns = ["date", "value"]
    future_plot["series"] = model_name
    plot_df = pd.concat([tail, future_plot], ignore_index=True)

    score = _score_lookup(scores, model_name)
    sum_10 = float(forecast_series.head(min(10, len(forecast_series))).sum())
    mean_10 = float(forecast_series.head(min(10, len(forecast_series))).mean())
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Model", model_name)
    c6.metric("WAPE", _format_pct(float(score["wape"])) if score is not None else "n/a")
    c7.metric("Součet 10 dní", _format_num(sum_10, 1))
    c8.metric("Průměr den", _format_num(mean_10, 1))

    fig = px.line(plot_df, x="date", y="value", color="series", markers=True, title=f"{metric_labels.get(metric, metric)} - historie + predikce")
    fig.update_traces(connectgaps=False)
    st.plotly_chart(fig, use_container_width=True)

    if model_name == "10 Smart blend" and not weights.empty:
        st.markdown("#### Váhy Smart blend")
        show_weights = weights.copy()
        show_weights["weight"] = show_weights["weight"].map(lambda value: f"{value:.1%}")
        st.dataframe(show_weights, use_container_width=True, hide_index=True)

    st.markdown("#### Backtest modelů")
    if not scores.empty:
        show_scores = scores.copy()
        for col in ["wape", "bias"]:
            show_scores[col] = show_scores[col].map(_format_pct)
        show_scores["mae"] = show_scores["mae"].map(lambda value: _format_num(value, 2))
        st.dataframe(show_scores, use_container_width=True, hide_index=True)
    else:
        st.info("Backtest zatím není k dispozici.")

    if not backtest.empty and model_name in backtest.columns:
        bt = backtest[["date", "actual", model_name]].dropna().tail(20).copy()
        bt = bt.rename(columns={model_name: "forecast"})
        bt["abs_error"] = (bt["actual"] - bt["forecast"]).abs()
        st.markdown("#### Poslední backtest body")
        st.dataframe(bt, use_container_width=True, hide_index=True)

    st.markdown("#### Forecast tabulka")
    export_fc = future[["date"]].copy()
    export_fc["forecast"] = forecast_series.values
    export_fc["weekday"] = export_fc["date"].dt.day_name().map(WEEKDAY_CZ)
    export_fc["iso_week"] = export_fc["date"].dt.isocalendar().week.astype(int)
    st.dataframe(export_fc, use_container_width=True, hide_index=True)


def _render_staffing_tab(bundle: StaffingBundle) -> None:
    if bundle.forecast is None or bundle.forecast.empty:
        st.warning("Chybí staffing forecast exporty.")
        return

    forecast = bundle.forecast.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    actuals = bundle.actuals.copy() if bundle.actuals is not None else pd.DataFrame()
    if not actuals.empty:
        actuals["date"] = pd.to_datetime(actuals["date"])
    horizon = bundle.horizon.copy() if bundle.horizon is not None else pd.DataFrame()
    if not horizon.empty:
        horizon["date"] = pd.to_datetime(horizon["date"])

    metric_map = {
        "Binhits": ("binhits", "forecast_binhits"),
        "Potřebný headcount": ("att_headcount", "required_headcount_ceiling"),
        "Placené hodiny": ("att_hours", "required_paid_hours"),
        "Produktivní workers": ("prod_workers", "required_productive_workers_ceiling"),
    }

    c1, c2 = st.columns([2, 3])
    with c1:
        metric_label = st.selectbox("Staffing metrika", list(metric_map.keys()), key="staff_metric")
    with c2:
        horizon_days = int(st.slider("Horizont staffing grafu", 5, 90, 30, key="staff_horizon"))

    actual_col, forecast_col = metric_map[metric_label]

    plot_rows = []
    if not actuals.empty and actual_col in actuals.columns:
        actual_plot = actuals[["date", actual_col]].dropna().tail(60).rename(columns={actual_col: "value"})
        actual_plot["series"] = "Historie"
        plot_rows.append(actual_plot)
    future_plot = forecast[["date", forecast_col]].head(horizon_days).rename(columns={forecast_col: "value"})
    future_plot["series"] = "Predikce"
    plot_rows.append(future_plot)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    fig = px.line(plot_df, x="date", y="value", color="series", markers=True, title=metric_label)
    fig.update_traces(connectgaps=False)
    st.plotly_chart(fig, use_container_width=True)

    card_horizons = [5, 10, 20, 30]
    cols = st.columns(len(card_horizons))
    for col, horizon_day in zip(cols, card_horizons):
        row = horizon[horizon["horizon_days"] == horizon_day] if not horizon.empty else pd.DataFrame()
        if row.empty:
            col.metric(f"{horizon_day} dní", "n/a")
            continue
        val = float(row.iloc[0][forecast_col]) if forecast_col in row.columns else np.nan
        col.metric(f"{horizon_day} dní", _format_num(val, 1))

    if bundle.driver_backtests is not None and not bundle.driver_backtests.empty:
        st.markdown("#### Backtest driver modelů")
        bt = bundle.driver_backtests.copy()
        for col in ["wape", "bias"]:
            bt[col] = bt[col].map(_format_pct)
        bt["mae"] = bt["mae"].map(lambda value: _format_num(value, 2))
        st.dataframe(bt, use_container_width=True, hide_index=True)

    if bundle.ratio_backtests is not None and not bundle.ratio_backtests.empty:
        st.markdown("#### Backtest staffing převodu")
        ratio = bundle.ratio_backtests.copy()
        ratio["wape"] = ratio["wape"].map(_format_pct)
        ratio["mae"] = ratio["mae"].map(lambda value: _format_num(value, 2))
        st.dataframe(ratio, use_container_width=True, hide_index=True)

    if not horizon.empty:
        st.markdown("#### Horizon points")
        st.dataframe(horizon, use_container_width=True, hide_index=True)

    st.markdown("#### Staffing forecast tabulka")
    st.dataframe(
        forecast[
            [
                "date",
                "weekday",
                "forecast_binhits",
                "required_headcount_ceiling",
                "required_kmen",
                "required_agency",
                "required_day_shift_workers",
                "required_night_shift_workers",
            ]
        ].head(horizon_days),
        use_container_width=True,
        hide_index=True,
    )


def tab_predikce(src: Sources, staffing_bundle: StaffingBundle) -> None:
    st.subheader("Predikce dopředu 📈")
    sub1, sub2 = st.tabs(["Modely výkonu", "Staffing"])

    with sub1:
        _render_prediction_models(src)

    with sub2:
        _render_staffing_tab(staffing_bundle)


def main() -> None:
    st.title("Warehouse Dashboard - Balení & Nakládky")

    with st.sidebar:
        st.header("Data")
        base = st.text_input("Složka s KPI CSV", value=".")
        base_dir = Path(base).resolve()
        st.caption("Očekává: packed/loaded KPI CSV a volitelně staffing_forecast_exports")

    src, missing = load_sources(str(base_dir))
    staffing_bundle = load_staffing_bundle(str(base_dir))

    if missing:
        st.warning("Chybí některé KPI soubory: " + ", ".join(missing))

    tab1, tab2, tab3 = st.tabs(["Balení", "Nakládky", "Predikce dopředu"])

    with tab1:
        tab_baleni(src)
    with tab2:
        tab_nakladky(src)
    with tab3:
        tab_predikce(src, staffing_bundle)


if __name__ == "__main__":
    main()
