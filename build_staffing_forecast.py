from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import unicodedata

import numpy as np
import pandas as pd

from finance_dashboard_builder import parse_datetime_mixed
from forecasting import align_known_exog, baseline_as_result, ridge_forecast, rolling_backtest


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "staffing_forecast_exports"
DATE_NOW = pd.Timestamp("2026-03-27")
HORIZON_DAYS = 91
PRODUCTIVE_ACTIVITIES = {"Kompletace", "Kontrola", "ZaskladnÄ›nĂ­", "VyklĂˇdka", "PĹ™esuny"}
ATTENDANCE_BUCKETS = ("kmen_early", "kmen_late", "agency_day", "agency_night")


def _norm(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


def _metrics(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    actual = actual.astype(float)
    pred = pred.astype(float).reindex(actual.index)
    err = (pred - actual).abs()
    denom = actual.abs().sum()
    wape = float(err.sum() / denom) if denom > 0 else np.nan
    mae = float(err.mean()) if len(err) else np.nan
    return wape, mae


def _safe_median(values: pd.Series, default: float = 0.0) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return float(default)
    return float(series.median())


def load_packed_daily() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "packed_daily_kpis.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_packed_shift() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "packed_shift_kpis.csv", parse_dates=["date"])
    pivot = (
        df.pivot_table(index="date", columns="shift", values="binhits", aggfunc="sum")
        .sort_index()
        .reset_index()
    )
    renamed = {}
    for col in pivot.columns:
        norm_col = _norm(col)
        if norm_col == "day":
            renamed[col] = "packed_day_binhits"
        elif norm_col == "night":
            renamed[col] = "packed_night_binhits"
    pivot = pivot.rename(columns=renamed)
    for col in ["packed_day_binhits", "packed_night_binhits"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot[["packed_day_binhits", "packed_night_binhits"]] = pivot[["packed_day_binhits", "packed_night_binhits"]].fillna(0.0)
    return pivot[["date", "packed_day_binhits", "packed_night_binhits"]].sort_values("date").reset_index(drop=True)


def load_loaded_daily() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "loaded_daily_kpis.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_tm_productive_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ROOT / "Reporty ESAB" / "Reporting" / "TimeManagement" / "Data"
    frames = []
    for path in sorted(root.glob("*.csv")):
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        cols = {_norm(c): c for c in df.columns}
        required = ["datum smeny", "smena", "zdroj", "skladnik", "cinnost", "trvani min.", "vyskytu"]
        if not all(key in cols for key in required):
            continue
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(df[cols["datum smeny"]], dayfirst=True, errors="coerce"),
                "shift": df[cols["smena"]].astype(str).str.strip(),
                "source": df[cols["zdroj"]].astype(str).str.strip(),
                "worker": df[cols["skladnik"]].astype(str).str.strip(),
                "activity": df[cols["cinnost"]].astype(str).str.strip(),
                "duration_min": pd.to_numeric(df[cols["trvani min."]], errors="coerce").fillna(0.0),
                "binhits_tm": pd.to_numeric(df[cols["vyskytu"]], errors="coerce").fillna(0.0),
            }
        )
        frames.append(part)

    data = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    data["hours"] = data["duration_min"] / 60.0
    productive = data[data["activity"].isin(PRODUCTIVE_ACTIVITIES)].copy()

    daily = (
        productive.groupby("date", as_index=False)
        .agg(
            prod_workers=("worker", "nunique"),
            prod_hours=("hours", "sum"),
            prod_binhits_tm=("binhits_tm", "sum"),
        )
        .sort_values("date")
    )

    shift = (
        productive.groupby(["date", "shift"], as_index=False)
        .agg(
            shift_prod_workers=("worker", "nunique"),
            shift_prod_hours=("hours", "sum"),
            shift_prod_binhits=("binhits_tm", "sum"),
        )
        .sort_values(["date", "shift"])
    )
    return daily, shift


def load_attendance_daily() -> pd.DataFrame:
    path = ROOT / "Reporty ESAB" / "Reporting" / "TimeManagement" / "Timemanagement.xlsx"
    xl = pd.ExcelFile(path)
    sheet = next(name for name in xl.sheet_names if "och" in _norm(name))
    df = xl.parse(sheet)
    cols = {_norm(c): c for c in df.columns}

    source_col = cols["zdroj"]
    name_col = cols["jmeno"]
    in_col = cols["prichod"]
    out_col = cols["odchod"]
    break_col = cols["prestavka (min)"]

    att = pd.DataFrame(
        {
            "source": df[source_col].astype(str).str.strip(),
            "person": df[name_col].astype(str).str.strip(),
            "in_dt": df[in_col].apply(parse_datetime_mixed),
            "out_dt": df[out_col].apply(parse_datetime_mixed),
            "break_min": pd.to_numeric(df[break_col], errors="coerce").fillna(0.0),
        }
    )
    att["date"] = att["in_dt"].dt.floor("D")
    att["hours"] = ((att["out_dt"] - att["in_dt"]).dt.total_seconds() / 3600.0) - (att["break_min"] / 60.0)
    att["hours"] = att["hours"].clip(lower=0.0).fillna(0.0)
    att = att.dropna(subset=["date"])
    att["source_norm"] = att["source"].map(_norm)
    att["start_hour"] = att["in_dt"].dt.hour + (att["in_dt"].dt.minute / 60.0)

    att["bucket"] = ""
    kmen_mask = att["source_norm"].eq("kmen")
    agency_mask = att["source_norm"].eq("agentura")
    att.loc[kmen_mask, "bucket"] = np.where(att.loc[kmen_mask, "start_hour"] < 10, "kmen_early", "kmen_late")
    att.loc[agency_mask, "bucket"] = np.where(att.loc[agency_mask, "start_hour"] < 12, "agency_day", "agency_night")

    daily = (
        att.groupby("date", as_index=False)
        .agg(
            att_headcount=("person", "nunique"),
            att_hours=("hours", "sum"),
            att_kmen_headcount=("person", lambda s: att.loc[s.index, "source_norm"].eq("kmen").sum()),
            att_agency_headcount=("person", lambda s: att.loc[s.index, "source_norm"].eq("agentura").sum()),
            att_kmen_hours=("hours", lambda s: att.loc[s.index, "source_norm"].eq("kmen").mul(s).sum()),
            att_agency_hours=("hours", lambda s: att.loc[s.index, "source_norm"].eq("agentura").mul(s).sum()),
        )
        .sort_values("date")
    )

    bucket_rows = att[att["bucket"].isin(ATTENDANCE_BUCKETS)].copy()
    bucket_headcount = bucket_rows.groupby(["date", "bucket"])["person"].nunique().unstack(fill_value=0)
    bucket_hours = bucket_rows.groupby(["date", "bucket"])["hours"].sum().unstack(fill_value=0.0)

    for bucket in ATTENDANCE_BUCKETS:
        daily[f"att_{bucket}_headcount"] = bucket_headcount.get(bucket, pd.Series(dtype=float)).reindex(daily["date"]).fillna(0.0).values
        daily[f"att_{bucket}_hours"] = bucket_hours.get(bucket, pd.Series(dtype=float)).reindex(daily["date"]).fillna(0.0).values
        daily[f"{bucket}_hours_per_person"] = daily[f"att_{bucket}_hours"] / daily[f"att_{bucket}_headcount"].replace(0, np.nan)

    daily["day_att_headcount"] = daily["att_kmen_early_headcount"] + daily["att_agency_day_headcount"]
    daily["night_att_headcount"] = daily["att_kmen_late_headcount"] + daily["att_agency_night_headcount"]
    daily["day_paid_hours"] = daily["att_kmen_early_hours"] + daily["att_agency_day_hours"]
    daily["night_paid_hours"] = daily["att_kmen_late_hours"] + daily["att_agency_night_hours"]
    daily["day_paid_hours_per_person"] = daily["day_paid_hours"] / daily["day_att_headcount"].replace(0, np.nan)
    daily["night_paid_hours_per_person"] = daily["night_paid_hours"] / daily["night_att_headcount"].replace(0, np.nan)
    return daily


def build_staffing_actuals() -> tuple[pd.DataFrame, pd.DataFrame]:
    packed = load_packed_daily()
    packed_shift = load_packed_shift()
    loaded = load_loaded_daily().rename(
        columns={
            "orders_nunique": "loaded_orders_nunique",
            "gross_tons": "loaded_gross_tons",
        }
    )
    prod_daily, prod_shift = load_tm_productive_daily()
    attendance = load_attendance_daily()

    daily = packed.merge(packed_shift, on="date", how="left")
    daily = daily.merge(
        loaded[["date", "trips_total", "loaded_orders_nunique", "loaded_gross_tons"]],
        on="date",
        how="left",
    )
    daily = daily.merge(prod_daily, on="date", how="left").merge(attendance, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["weekday"] = daily["date"].dt.day_name()
    daily["bh_per_paid_hour"] = daily["binhits"] / daily["att_hours"].replace(0, np.nan)
    daily["bh_per_prod_hour"] = daily["binhits"] / daily["prod_hours"].replace(0, np.nan)
    daily["paid_hours_per_person"] = daily["att_hours"] / daily["att_headcount"].replace(0, np.nan)
    daily["prod_hours_per_worker"] = daily["prod_hours"] / daily["prod_workers"].replace(0, np.nan)
    daily["kmen_share"] = daily["att_kmen_headcount"] / daily["att_headcount"].replace(0, np.nan)
    daily["agency_share"] = daily["att_agency_headcount"] / daily["att_headcount"].replace(0, np.nan)
    daily["day_bh_per_paid_hour"] = daily["packed_day_binhits"] / daily["day_paid_hours"].replace(0, np.nan)
    daily["night_bh_per_paid_hour"] = daily["packed_night_binhits"] / daily["night_paid_hours"].replace(0, np.nan)
    daily["day_bh_per_person"] = daily["packed_day_binhits"] / daily["day_att_headcount"].replace(0, np.nan)
    daily["night_bh_per_person"] = daily["packed_night_binhits"] / daily["night_att_headcount"].replace(0, np.nan)
    daily["day_shift_share"] = daily["packed_day_binhits"] / daily["binhits"].replace(0, np.nan)
    daily["night_shift_share"] = daily["packed_night_binhits"] / daily["binhits"].replace(0, np.nan)

    shift_pivot = prod_shift.pivot(index="date", columns="shift", values=["shift_prod_workers", "shift_prod_hours"]).sort_index()
    shift_pivot.columns = [f"{metric}_{shift.lower()}" for metric, shift in shift_pivot.columns]
    shift_pivot = shift_pivot.reset_index()
    daily = daily.merge(shift_pivot, on="date", how="left")
    return daily, prod_shift


@dataclass
class SeriesForecast:
    target: str
    selected_model: str
    forecast: pd.Series
    scores: pd.DataFrame


def forecast_daily_series(series: pd.Series, exog_full: pd.DataFrame | None = None, backtest_horizon: int = 10) -> SeriesForecast:
    s = series.dropna().astype(float)
    gap_days = max(0, (DATE_NOW.normalize() - pd.Timestamp(s.index.max()).normalize()).days)
    horizon = HORIZON_DAYS + gap_days
    future_idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    hist_exog, future_exog = pd.DataFrame(), pd.DataFrame()
    if exog_full is not None and not exog_full.empty:
        exog_full = exog_full.sort_index()
        exog_full = exog_full[~exog_full.index.duplicated(keep="last")]
        hist_exog, future_exog = align_known_exog(exog_full, future_idx, lag_periods=1)
        hist_exog = hist_exog.reindex(s.index).ffill().bfill()

    baseline = baseline_as_result(s, horizon, "daily")
    internal = ridge_forecast(s, horizon, "daily")
    global_result = ridge_forecast(s, horizon, "daily", exog_hist=hist_exog, exog_future=future_exog) if not hist_exog.empty else None

    score_rows = []
    for model in ["baseline", "internal", "global"]:
        if model == "global" and hist_exog.empty:
            continue
        metrics = rolling_backtest(s, "daily", backtest_horizon, model, exog_aligned=hist_exog if model == "global" else None, n_origins=8)
        score_rows.append(
            {
                "target": series.name or "target",
                "model": model,
                "wape": metrics.wape,
                "bias": metrics.bias,
                "mae": metrics.mae,
                "n_windows": metrics.n_windows,
            }
        )
    scores = pd.DataFrame(score_rows).sort_values(["wape", "mae"], na_position="last").reset_index(drop=True)
    best = str(scores.iloc[0]["model"]) if not scores.empty else "internal"
    selected = {"baseline": baseline, "internal": internal, "global": global_result}.get(best, internal)
    forecast = selected.forecast[selected.forecast.index > DATE_NOW.normalize()].copy()
    return SeriesForecast(target=series.name or "target", selected_model=best, forecast=forecast, scores=scores)


def _recent_profile(history: pd.DataFrame, target_date: pd.Timestamp, col: str, default: float) -> float:
    same_weekday = history[(history["date"] < target_date) & (history["date"].dt.weekday == target_date.weekday())][col].dropna().tail(8)
    if len(same_weekday) >= 3:
        return float(same_weekday.median())
    recent = history[history["date"] < target_date][col].dropna().tail(42)
    if len(recent) >= 5:
        return float(recent.median())
    return float(default)


def _recent_positive_profile(history: pd.DataFrame, target_date: pd.Timestamp, col: str, default: float) -> float:
    same_weekday = history[(history["date"] < target_date) & (history["date"].dt.weekday == target_date.weekday())][col].dropna()
    same_weekday = same_weekday[same_weekday > 0].tail(8)
    if len(same_weekday) >= 2:
        return float(same_weekday.median())

    recent = history[history["date"] < target_date][col].dropna()
    recent = recent[recent > 0].tail(42)
    if len(recent) >= 4:
        return float(recent.median())

    return float(default)


def _recent_fixed_capacity(history: pd.DataFrame, target_date: pd.Timestamp, col: str, default: float) -> float:
    prior = history[(history["date"] < target_date) & history[col].notna()].copy()
    if prior.empty:
        return float(default)

    same_weekday_all = prior[prior["date"].dt.weekday == target_date.weekday()][col].dropna().tail(12)
    if len(same_weekday_all) >= 3:
        if float(same_weekday_all.median()) <= 0:
            return 0.0
        same_weekday_positive = same_weekday_all[same_weekday_all > 0]
        if len(same_weekday_positive) >= 2:
            return float(same_weekday_positive.median())

    positive = prior[prior[col] > 0].copy()
    if positive.empty:
        return float(default)

    recent_window = positive[positive["date"] >= (target_date - pd.Timedelta(days=84))][col]
    if len(recent_window) >= 4:
        return float(recent_window.median())

    return float(positive[col].tail(20).median())


def _planned_shift_staff(required_hours: float, fixed_cap: float, fixed_hours_per_person: float, flex_hours_per_person: float) -> tuple[int, int, float]:
    fixed_people = max(int(round(fixed_cap)), 0)
    if required_hours <= 0:
        return 0, 0, 0.0

    fixed_hours = fixed_people * max(fixed_hours_per_person, 0.0)
    flex_hours = max(required_hours - fixed_hours, 0.0)
    flex_people = int(math.ceil(flex_hours / max(flex_hours_per_person, 1e-6))) if flex_hours > 0 else 0
    return fixed_people, flex_people, flex_hours


def _guard_shift_forecast(history: pd.DataFrame, target_date: pd.Timestamp, col: str, raw_value: float) -> float:
    same_weekday = history[(history["date"] < target_date) & (history["date"].dt.weekday == target_date.weekday())][col].dropna().tail(12)
    if len(same_weekday) < 3:
        return float(max(raw_value, 0.0))

    positive = same_weekday[same_weekday > 0]
    if positive.empty:
        return 0.0

    positive_share = len(positive) / len(same_weekday)
    if positive_share <= 0.25:
        return 0.0

    median_val = float(positive.median())
    q75 = float(positive.quantile(0.75))
    q90 = float(positive.quantile(0.90))
    cap = max(median_val * 1.35, q75 * 1.15, q90)
    return float(min(max(raw_value, 0.0), cap))


def staffing_ratio_backtest(actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = [
        "date",
        "packed_day_binhits",
        "packed_night_binhits",
        "att_headcount",
        "att_hours",
        "att_kmen_headcount",
        "att_agency_headcount",
        "day_att_headcount",
        "night_att_headcount",
    ]
    data = actuals.dropna(subset=required_cols).copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    defaults = {
        "day_bh_per_paid_hour": _safe_median(data["day_bh_per_paid_hour"], 2.7),
        "night_bh_per_paid_hour": _safe_median(data["night_bh_per_paid_hour"], 2.5),
        "bh_per_prod_hour": _safe_median(data["bh_per_prod_hour"], 7.0),
        "prod_hours_per_worker": _safe_median(data["prod_hours_per_worker"], 6.5),
        "kmen_early_hours_per_person": _safe_median(data["kmen_early_hours_per_person"], 7.75),
        "kmen_late_hours_per_person": _safe_median(data["kmen_late_hours_per_person"], 7.75),
        "agency_day_hours_per_person": _safe_median(data["agency_day_hours_per_person"], 11.0),
        "agency_night_hours_per_person": _safe_median(data["agency_night_hours_per_person"], 11.0),
        "att_kmen_early_headcount": _safe_median(data.loc[data["att_kmen_early_headcount"] > 0, "att_kmen_early_headcount"], 0.0),
        "att_kmen_late_headcount": _safe_median(data.loc[data["att_kmen_late_headcount"] > 0, "att_kmen_late_headcount"], 0.0),
        "att_agency_day_headcount": _safe_median(data.loc[data["att_agency_day_headcount"] > 0, "att_agency_day_headcount"], 0.0),
        "att_agency_night_headcount": _safe_median(data.loc[data["att_agency_night_headcount"] > 0, "att_agency_night_headcount"], 0.0),
    }

    preds = []
    for row in data.itertuples():
        target_date = pd.Timestamp(row.date)
        hist = data[data["date"] < target_date]
        if len(hist) < 21:
            continue

        day_bh_per_paid_hour = _recent_positive_profile(hist, target_date, "day_bh_per_paid_hour", defaults["day_bh_per_paid_hour"])
        night_bh_per_paid_hour = _recent_positive_profile(hist, target_date, "night_bh_per_paid_hour", defaults["night_bh_per_paid_hour"])
        bh_per_prod_hour = _recent_positive_profile(hist, target_date, "bh_per_prod_hour", defaults["bh_per_prod_hour"])
        prod_hours_per_worker = _recent_positive_profile(hist, target_date, "prod_hours_per_worker", defaults["prod_hours_per_worker"])
        kmen_early_hpp = _recent_positive_profile(hist, target_date, "kmen_early_hours_per_person", defaults["kmen_early_hours_per_person"])
        kmen_late_hpp = _recent_positive_profile(hist, target_date, "kmen_late_hours_per_person", defaults["kmen_late_hours_per_person"])
        agency_day_hpp = _recent_positive_profile(hist, target_date, "agency_day_hours_per_person", defaults["agency_day_hours_per_person"])
        agency_night_hpp = _recent_positive_profile(hist, target_date, "agency_night_hours_per_person", defaults["agency_night_hours_per_person"])
        kmen_early_cap = _recent_fixed_capacity(hist, target_date, "att_kmen_early_headcount", defaults["att_kmen_early_headcount"])
        kmen_late_cap = _recent_fixed_capacity(hist, target_date, "att_kmen_late_headcount", defaults["att_kmen_late_headcount"])

        required_day_paid_hours = row.packed_day_binhits / max(day_bh_per_paid_hour, 1e-6)
        required_night_paid_hours = row.packed_night_binhits / max(night_bh_per_paid_hour, 1e-6)
        required_prod_hours = (row.packed_day_binhits + row.packed_night_binhits) / max(bh_per_prod_hour, 1e-6)
        required_prod_workers = required_prod_hours / max(prod_hours_per_worker, 1e-6)

        required_kmen_early, required_agency_day, _ = _planned_shift_staff(required_day_paid_hours, kmen_early_cap, kmen_early_hpp, agency_day_hpp)
        required_kmen_late, required_agency_night, _ = _planned_shift_staff(required_night_paid_hours, kmen_late_cap, kmen_late_hpp, agency_night_hpp)

        pred_paid_hours = (
            required_kmen_early * kmen_early_hpp
            + required_agency_day * agency_day_hpp
            + required_kmen_late * kmen_late_hpp
            + required_agency_night * agency_night_hpp
        )
        pred_kmen = required_kmen_early + required_kmen_late
        pred_agency = required_agency_day + required_agency_night
        pred_day_shift = required_kmen_early + required_agency_day
        pred_night_shift = required_kmen_late + required_agency_night

        preds.append(
            {
                "date": target_date,
                "pred_paid_hours": pred_paid_hours,
                "pred_headcount": pred_kmen + pred_agency,
                "pred_kmen": pred_kmen,
                "pred_agency": pred_agency,
                "pred_day_shift": pred_day_shift,
                "pred_night_shift": pred_night_shift,
                "pred_prod_workers": required_prod_workers,
                "actual_paid_hours": row.att_hours,
                "actual_headcount": row.att_headcount,
                "actual_kmen": row.att_kmen_headcount,
                "actual_agency": row.att_agency_headcount,
                "actual_day_shift": row.day_att_headcount,
                "actual_night_shift": row.night_att_headcount,
                "actual_prod_workers": row.prod_workers,
            }
        )

    pred_df = pd.DataFrame(preds)
    if pred_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for actual_col, pred_col, metric_name in [
        ("actual_paid_hours", "pred_paid_hours", "paid_hours"),
        ("actual_headcount", "pred_headcount", "headcount_total"),
        ("actual_kmen", "pred_kmen", "kmen_total"),
        ("actual_agency", "pred_agency", "agency_total"),
        ("actual_day_shift", "pred_day_shift", "day_shift_total"),
        ("actual_night_shift", "pred_night_shift", "night_shift_total"),
        ("actual_prod_workers", "pred_prod_workers", "productive_workers"),
    ]:
        wape, mae = _metrics(pred_df[actual_col], pred_df[pred_col])
        rows.append({"metric": metric_name, "wape": wape, "mae": mae, "n_days": len(pred_df)})
    return pred_df, pd.DataFrame(rows)


def build_forecasts() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    packed = load_packed_daily().set_index("date")
    packed_shift = load_packed_shift().set_index("date")
    loaded = load_loaded_daily().set_index("date")
    staffing_actuals, _prod_shift = build_staffing_actuals()
    staffing_actuals.to_csv(EXPORT_DIR / "staffing_capacity_actuals.csv", index=False, encoding="utf-8-sig")

    loaded_orders_fc = forecast_daily_series(loaded["orders_nunique"].rename("loaded_orders_nunique"))
    trips_fc = forecast_daily_series(loaded["trips_total"].rename("trips_total"))
    loaded_gross_fc = forecast_daily_series(loaded["gross_tons"].rename("loaded_gross_tons"))
    day_binhits_fc = forecast_daily_series(packed_shift["packed_day_binhits"].rename("forecast_day_binhits"))
    night_binhits_fc = forecast_daily_series(packed_shift["packed_night_binhits"].rename("forecast_night_binhits"))

    driver_scores = pd.concat(
        [
            loaded_orders_fc.scores,
            trips_fc.scores,
            loaded_gross_fc.scores,
            day_binhits_fc.scores,
            night_binhits_fc.scores,
        ],
        ignore_index=True,
    )

    future_dates = pd.date_range(DATE_NOW.normalize() + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    future = pd.DataFrame({"date": future_dates})
    future["forecast_day_binhits"] = day_binhits_fc.forecast.reindex(future_dates).ffill().bfill().values
    future["forecast_night_binhits"] = night_binhits_fc.forecast.reindex(future_dates).ffill().bfill().values
    shift_history = packed_shift.reset_index()[["date", "packed_day_binhits", "packed_night_binhits"]].copy()
    future["forecast_day_binhits"] = [
        _guard_shift_forecast(shift_history, pd.Timestamp(day), "packed_day_binhits", float(value))
        for day, value in zip(future["date"], future["forecast_day_binhits"])
    ]
    future["forecast_night_binhits"] = [
        _guard_shift_forecast(shift_history, pd.Timestamp(day), "packed_night_binhits", float(value))
        for day, value in zip(future["date"], future["forecast_night_binhits"])
    ]
    future["forecast_binhits"] = future["forecast_day_binhits"] + future["forecast_night_binhits"]
    future["forecast_loaded_orders"] = loaded_orders_fc.forecast.reindex(future_dates).ffill().bfill().values
    future["forecast_trips_total"] = trips_fc.forecast.reindex(future_dates).ffill().bfill().values
    future["forecast_loaded_gross_tons"] = loaded_gross_fc.forecast.reindex(future_dates).ffill().bfill().values
    future["weekday"] = future["date"].dt.day_name()

    actuals = staffing_actuals.dropna(subset=["att_hours", "att_headcount", "packed_day_binhits", "packed_night_binhits"]).copy()
    defaults = {
        "bh_per_prod_hour": _safe_median(actuals["bh_per_prod_hour"], 7.0),
        "prod_hours_per_worker": _safe_median(actuals["prod_hours_per_worker"], 6.5),
        "day_bh_per_paid_hour": _safe_median(actuals["day_bh_per_paid_hour"], 2.7),
        "night_bh_per_paid_hour": _safe_median(actuals["night_bh_per_paid_hour"], 2.5),
        "kmen_early_hours_per_person": _safe_median(actuals["kmen_early_hours_per_person"], 7.75),
        "kmen_late_hours_per_person": _safe_median(actuals["kmen_late_hours_per_person"], 7.75),
        "agency_day_hours_per_person": _safe_median(actuals["agency_day_hours_per_person"], 11.0),
        "agency_night_hours_per_person": _safe_median(actuals["agency_night_hours_per_person"], 11.0),
        "att_kmen_early_headcount": _safe_median(actuals.loc[actuals["att_kmen_early_headcount"] > 0, "att_kmen_early_headcount"], 0.0),
        "att_kmen_late_headcount": _safe_median(actuals.loc[actuals["att_kmen_late_headcount"] > 0, "att_kmen_late_headcount"], 0.0),
        "att_agency_day_headcount": _safe_median(actuals.loc[actuals["att_agency_day_headcount"] > 0, "att_agency_day_headcount"], 0.0),
        "att_agency_night_headcount": _safe_median(actuals.loc[actuals["att_agency_night_headcount"] > 0, "att_agency_night_headcount"], 0.0),
    }

    plan_rows = []
    for row in future.itertuples():
        day = pd.Timestamp(row.date)
        day_bh_per_paid_hour = _recent_positive_profile(actuals, day, "day_bh_per_paid_hour", defaults["day_bh_per_paid_hour"])
        night_bh_per_paid_hour = _recent_positive_profile(actuals, day, "night_bh_per_paid_hour", defaults["night_bh_per_paid_hour"])
        bh_per_prod_hour = _recent_positive_profile(actuals, day, "bh_per_prod_hour", defaults["bh_per_prod_hour"])
        prod_hours_per_worker = _recent_positive_profile(actuals, day, "prod_hours_per_worker", defaults["prod_hours_per_worker"])
        kmen_early_hpp = _recent_positive_profile(actuals, day, "kmen_early_hours_per_person", defaults["kmen_early_hours_per_person"])
        kmen_late_hpp = _recent_positive_profile(actuals, day, "kmen_late_hours_per_person", defaults["kmen_late_hours_per_person"])
        agency_day_hpp = _recent_positive_profile(actuals, day, "agency_day_hours_per_person", defaults["agency_day_hours_per_person"])
        agency_night_hpp = _recent_positive_profile(actuals, day, "agency_night_hours_per_person", defaults["agency_night_hours_per_person"])

        kmen_early_cap = _recent_fixed_capacity(actuals, day, "att_kmen_early_headcount", defaults["att_kmen_early_headcount"])
        kmen_late_cap = _recent_fixed_capacity(actuals, day, "att_kmen_late_headcount", defaults["att_kmen_late_headcount"])
        agency_day_base = _recent_fixed_capacity(actuals, day, "att_agency_day_headcount", defaults["att_agency_day_headcount"])
        agency_night_base = _recent_fixed_capacity(actuals, day, "att_agency_night_headcount", defaults["att_agency_night_headcount"])

        required_day_paid_hours = row.forecast_day_binhits / max(day_bh_per_paid_hour, 1e-6)
        required_night_paid_hours = row.forecast_night_binhits / max(night_bh_per_paid_hour, 1e-6)
        required_prod_hours = row.forecast_binhits / max(bh_per_prod_hour, 1e-6)
        required_prod_workers = required_prod_hours / max(prod_hours_per_worker, 1e-6)

        required_kmen_early, required_agency_day, day_flex_hours = _planned_shift_staff(required_day_paid_hours, kmen_early_cap, kmen_early_hpp, agency_day_hpp)
        required_kmen_late, required_agency_night, night_flex_hours = _planned_shift_staff(required_night_paid_hours, kmen_late_cap, kmen_late_hpp, agency_night_hpp)

        required_kmen = required_kmen_early + required_kmen_late
        required_agency = required_agency_day + required_agency_night
        required_day_shift_workers = required_kmen_early + required_agency_day
        required_night_shift_workers = required_kmen_late + required_agency_night
        required_headcount_ceiling = required_kmen + required_agency
        recent_total_capacity = kmen_early_cap + kmen_late_cap + agency_day_base + agency_night_base
        capacity_gap = max(required_headcount_ceiling - int(math.ceil(recent_total_capacity)), 0)
        required_paid_hours = required_day_paid_hours + required_night_paid_hours

        plan_rows.append(
            {
                "date": day,
                "weekday": row.weekday,
                "forecast_day_binhits": float(row.forecast_day_binhits),
                "forecast_night_binhits": float(row.forecast_night_binhits),
                "forecast_binhits": float(row.forecast_binhits),
                "forecast_loaded_orders": float(row.forecast_loaded_orders),
                "forecast_trips_total": float(row.forecast_trips_total),
                "forecast_loaded_gross_tons": float(row.forecast_loaded_gross_tons),
                "expected_day_bh_per_paid_hour": day_bh_per_paid_hour,
                "expected_night_bh_per_paid_hour": night_bh_per_paid_hour,
                "expected_bh_per_paid_hour": row.forecast_binhits / max(required_paid_hours, 1e-6),
                "expected_bh_per_prod_hour": bh_per_prod_hour,
                "required_day_paid_hours": required_day_paid_hours,
                "required_night_paid_hours": required_night_paid_hours,
                "required_paid_hours": required_paid_hours,
                "required_productive_hours": required_prod_hours,
                "required_headcount": float(required_headcount_ceiling),
                "required_headcount_ceiling": required_headcount_ceiling,
                "required_productive_workers": required_prod_workers,
                "required_productive_workers_ceiling": int(math.ceil(required_prod_workers)),
                "kmen_share": (required_kmen / required_headcount_ceiling) if required_headcount_ceiling else 0.0,
                "agency_share": (required_agency / required_headcount_ceiling) if required_headcount_ceiling else 0.0,
                "kmen_capacity_cap": kmen_early_cap + kmen_late_cap,
                "agency_recent_base": agency_day_base + agency_night_base,
                "recent_total_capacity": recent_total_capacity,
                "capacity_gap_headcount": capacity_gap,
                "kmen_early_capacity_cap": kmen_early_cap,
                "kmen_late_capacity_cap": kmen_late_cap,
                "agency_day_recent_base": agency_day_base,
                "agency_night_recent_base": agency_night_base,
                "required_kmen_early": required_kmen_early,
                "required_kmen_late": required_kmen_late,
                "required_agency_day": required_agency_day,
                "required_agency_night": required_agency_night,
                "required_kmen": required_kmen,
                "required_agency": required_agency,
                "required_day_shift_workers": required_day_shift_workers,
                "required_night_shift_workers": required_night_shift_workers,
                "day_flex_hours": day_flex_hours,
                "night_flex_hours": night_flex_hours,
                "binhits_model": f"day:{day_binhits_fc.selected_model}; night:{night_binhits_fc.selected_model}",
            }
        )

    plan = pd.DataFrame(plan_rows)
    plan.to_csv(EXPORT_DIR / "staffing_forecast_daily.csv", index=False, encoding="utf-8-sig")

    ratio_pred_df, ratio_metrics = staffing_ratio_backtest(actuals)
    if not ratio_pred_df.empty:
        ratio_pred_df.to_csv(EXPORT_DIR / "staffing_ratio_backtest_daily.csv", index=False, encoding="utf-8-sig")
    ratio_metrics.to_csv(EXPORT_DIR / "staffing_ratio_backtest_summary.csv", index=False, encoding="utf-8-sig")

    driver_scores.to_csv(EXPORT_DIR / "staffing_driver_backtests.csv", index=False, encoding="utf-8-sig")

    horizon_rows = []
    for horizon in [5, 10, 15, 20, 30, 60, 90]:
        row = plan.iloc[horizon - 1]
        horizon_rows.append(
            {
                "horizon_days": horizon,
                "date": row["date"],
                "forecast_day_binhits": row["forecast_day_binhits"],
                "forecast_night_binhits": row["forecast_night_binhits"],
                "forecast_binhits": row["forecast_binhits"],
                "required_paid_hours": row["required_paid_hours"],
                "required_headcount_ceiling": row["required_headcount_ceiling"],
                "capacity_gap_headcount": row["capacity_gap_headcount"],
                "required_kmen_early": row["required_kmen_early"],
                "required_kmen_late": row["required_kmen_late"],
                "required_agency_day": row["required_agency_day"],
                "required_agency_night": row["required_agency_night"],
                "required_kmen": row["required_kmen"],
                "required_agency": row["required_agency"],
                "required_day_shift_workers": row["required_day_shift_workers"],
                "required_night_shift_workers": row["required_night_shift_workers"],
            }
        )
    pd.DataFrame(horizon_rows).to_csv(EXPORT_DIR / "staffing_horizon_points.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "# Staffing Forecast",
        "",
        f"- Run date: {DATE_NOW.date()}",
        f"- Day binhits forecast model: {day_binhits_fc.selected_model}",
        f"- Night binhits forecast model: {night_binhits_fc.selected_model}",
        f"- Binhits last actual date: {packed.index.max().date()}",
        f"- Attendance last actual date: {actuals['date'].max().date()}",
        f"- Median day binhits per paid hour: {defaults['day_bh_per_paid_hour']:.2f}",
        f"- Median night binhits per paid hour: {defaults['night_bh_per_paid_hour']:.2f}",
        "",
        "## Staffing Logic",
        "",
        "- Forecast day and night binhits separately.",
        "- Convert shift binhits to paid hours with recent weekday productivity for the same shift.",
        "- Hold `kmen` in fixed buckets: early (06-14 proxy) and late (14-22 proxy).",
        "- Let agency absorb flex separately for day and night based on remaining required hours.",
        "- Keep a total productive-workers estimate as a secondary view for operations.",
        "",
        "## Notes",
        "",
        "- `required_kmen_*` is anchored to recent positive observed weekday capacity, not free reallocation.",
        "- End-to-end planning error is staffing-ratio error plus shift-level binhits forecast error.",
    ]
    (EXPORT_DIR / "STAFFING_FORECAST.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    build_forecasts()
