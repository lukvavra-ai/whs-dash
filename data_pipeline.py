from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:  # pragma: no cover - fallback for offline/runtime environments
    holidays = None


REQUIRED_FILES = {
    "packed_daily_kpis.csv",
    "packed_shift_kpis.csv",
    "loaded_daily_kpis.csv",
    "loaded_shift_kpis.csv",
}


def find_data_dir(base: Optional[Path] = None) -> Path:
    anchor = Path(base) if base is not None else Path(__file__).resolve().parent
    candidates = [
        anchor / "data",
        anchor,
        Path.cwd() / "data",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate.exists():
            files = {p.name for p in candidate.glob("*.csv")}
            if REQUIRED_FILES.issubset(files):
                return candidate
    return anchor


def ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _standardize_dataset(df: pd.DataFrame, key: str) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    if key.startswith("packed"):
        if "orders_nunique" in out.columns and "vydejky_unique" not in out.columns:
            rename_map["orders_nunique"] = "vydejky_unique"
        if "iso_weekday" in out.columns and "weekday" not in out.columns:
            rename_map["iso_weekday"] = "weekday"
    out = out.rename(columns=rename_map)
    return out


def add_calendar_columns(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    dt = out[date_col]
    valid_years = dt.dt.year.dropna()
    min_year = int(valid_years.min()) if not valid_years.empty else 2022
    max_year = int(valid_years.max()) if not valid_years.empty else 2026
    cz_holidays = holidays.Czechia(years=range(min_year - 1, max_year + 2)) if holidays is not None else {}
    month_period = dt.dt.to_period("M")
    week_period = dt.dt.to_period("W-MON")
    holiday_dates = pd.to_datetime(list(cz_holidays.keys())) if holidays is not None else pd.DatetimeIndex([])
    is_holiday = dt.dt.normalize().isin(holiday_dates)
    is_weekend = dt.dt.weekday >= 5
    is_business_day = (~is_weekend) & (~is_holiday)

    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["weekday_num"] = dt.dt.weekday
    out["weekday_name"] = dt.dt.day_name()
    out["week_start"] = dt - pd.to_timedelta(dt.dt.weekday, unit="D")
    out["is_holiday"] = is_holiday.astype(int)
    out["is_business_day"] = is_business_day.astype(int)
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)
    out["days_to_month_end"] = (dt.dt.days_in_month - dt.dt.day).astype("Int64")
    out["workday_in_month"] = (
        out.assign(_month=month_period, _work=is_business_day.astype(int))
        .groupby("_month")["_work"]
        .cumsum()
        .astype("Int64")
    )
    out["number_of_workdays_in_month"] = (
        out.assign(_month=month_period, _work=is_business_day.astype(int))
        .groupby("_month")["_work"]
        .transform("sum")
        .astype("Int64")
    )
    out["number_of_workdays_in_week"] = (
        out.assign(_week=week_period, _work=is_business_day.astype(int))
        .groupby("_week")["_work"]
        .transform("sum")
        .astype("Int64")
    )
    out["first_business_day"] = ((out["workday_in_month"] == 1) & is_business_day).astype(int)
    out["last_business_day"] = (
        (out["days_to_month_end"] <= 3) & is_business_day & (out["number_of_workdays_in_month"] == out["workday_in_month"])
    ).astype(int)

    prev_day = (dt - pd.Timedelta(days=1)).dt.normalize()
    next_day = (dt + pd.Timedelta(days=1)).dt.normalize()
    out["pre_holiday"] = (is_business_day & next_day.isin(holiday_dates)).astype(int)
    out["post_holiday"] = (is_business_day & prev_day.isin(holiday_dates)).astype(int)
    bridge_prev = prev_day.isin(holiday_dates) | ((dt.dt.weekday == 0) & (prev_day.dt.weekday == 6))
    bridge_next = next_day.isin(holiday_dates) | ((dt.dt.weekday == 4) & (next_day.dt.weekday == 5))
    out["bridge_day"] = (is_business_day & (bridge_prev | bridge_next) & ~out["pre_holiday"].astype(bool) & ~out["post_holiday"].astype(bool)).astype(int)
    return out


def load_local_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "packed_daily": data_dir / "packed_daily_kpis.csv",
        "packed_shift": data_dir / "packed_shift_kpis.csv",
        "loaded_daily": data_dir / "loaded_daily_kpis.csv",
        "loaded_shift": data_dir / "loaded_shift_kpis.csv",
        "packed_typobalu_audit": data_dir / "packed_typobalu_audit.csv",
    }
    data: Dict[str, pd.DataFrame] = {}
    for key, path in files.items():
        if not path.exists():
            data[key] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        df = _standardize_dataset(df, key)
        if "date" in df.columns:
            df = ensure_datetime(df, ["date"])
            df = add_calendar_columns(df, "date")
        if "shift_start" in df.columns:
            df = ensure_datetime(df, ["shift_start"])
        if key == "loaded_shift" and "shift_name" in df.columns:
            df["shift_label"] = df["shift_name"].replace({"morning": "Ranní", "afternoon": "Odpolední"})
        if key == "packed_shift" and "shift" in df.columns:
            df["shift_label"] = df["shift"].replace({"day": "Denní", "night": "Noční"})
        data[key] = df
    return data


def coverage_table(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=["year", "rows", "first_date", "last_date"])
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return pd.DataFrame(columns=["year", "rows", "first_date", "last_date"])
    out = (
        pd.DataFrame({"date": d})
        .assign(year=lambda x: x["date"].dt.year)
        .groupby("year")
        .agg(rows=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index()
    )
    return out


def infer_expected_weekdays(df: pd.DataFrame, date_col: str = "date") -> set[int]:
    if df.empty or date_col not in df.columns:
        return set()
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return set()
    shares = d.dt.weekday.value_counts(normalize=True)
    selected = set(int(idx) for idx, share in shares.items() if share >= 0.05)
    if not selected:
        selected = set(int(x) for x in d.dt.weekday.unique().tolist())
    return selected


def expected_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DatetimeIndex:
    if df.empty or date_col not in df.columns:
        return pd.DatetimeIndex([])
    d = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if d.empty:
        return pd.DatetimeIndex([])
    weekdays = infer_expected_weekdays(df, date_col)
    rng = pd.date_range(d.min(), d.max(), freq="D")
    return pd.DatetimeIndex([x for x in rng if x.weekday() in weekdays])


def missing_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=["date"])
    observed = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.normalize().drop_duplicates()
    expected = expected_dates(df, date_col)
    missing = expected.difference(pd.DatetimeIndex(observed))
    return pd.DataFrame({"date": missing})


def missing_weeks(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=["week_start"])
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return pd.DataFrame(columns=["week_start"])
    observed = (d - pd.to_timedelta(d.dt.weekday, unit="D")).drop_duplicates().sort_values()
    expected = pd.date_range(observed.min(), observed.max(), freq="W-MON")
    missing = expected.difference(pd.DatetimeIndex(observed))
    return pd.DataFrame({"week_start": missing})


def regular_daily_series(df: pd.DataFrame, value_col: str, date_col: str = "date") -> pd.Series:
    if df.empty or value_col not in df.columns or date_col not in df.columns:
        return pd.Series(dtype=float)
    x = df[[date_col, value_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0)
    x = x.dropna(subset=[date_col])
    if x.empty:
        return pd.Series(dtype=float)
    s = x.groupby(date_col)[value_col].sum().sort_index().astype(float)
    idx = expected_dates(df, date_col)
    if idx.empty:
        idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx, fill_value=0.0)


def infer_expected_days_per_week(df: pd.DataFrame, date_col: str = "date") -> int:
    if df.empty or date_col not in df.columns:
        return 5
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return 5
    week_counts = (d - pd.to_timedelta(d.dt.weekday, unit="D")).value_counts()
    if len(week_counts) > 2:
        week_counts = week_counts.iloc[1:-1]
    if week_counts.empty:
        week_counts = (d - pd.to_timedelta(d.dt.weekday, unit="D")).value_counts()
    return int(round(float(week_counts.median()))) if not week_counts.empty else 5


def aggregate_daily_to_weekly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["week_start", value_col, "observed_days", "expected_days", "is_complete"])
    x = df[["date", value_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna(subset=["date"])
    x["week_start"] = x["date"] - pd.to_timedelta(x["date"].dt.weekday, unit="D")
    expected_days = infer_expected_days_per_week(df)
    weekly = (
        x.groupby("week_start")
        .agg(**{value_col: (value_col, "sum"), "observed_days": ("date", "nunique")})
        .sort_index()
        .reset_index()
    )
    weekly["expected_days"] = expected_days
    weekly["is_complete"] = weekly["observed_days"] >= expected_days
    return weekly


def weekly_from_daily(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"])
    x["week_start"] = x["date"] - pd.to_timedelta(x["date"].dt.weekday, unit="D")
    excluded = {"iso_year", "iso_week", "iso_weekday", "weekday", "weekday_num", "month", "year", "quarter"}
    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and c not in excluded]
    weekly = x.groupby("week_start")[numeric_cols].sum(min_count=1).sort_index()
    weekly.columns = [f"{prefix}_{c}" for c in weekly.columns]
    weekly[f"{prefix}_observed_days"] = x.groupby("week_start")["date"].nunique()
    weekly[f"{prefix}_expected_days"] = infer_expected_days_per_week(df)
    weekly[f"{prefix}_is_complete"] = weekly[f"{prefix}_observed_days"] >= weekly[f"{prefix}_expected_days"]
    return weekly


def safe_div(numerator, denominator):
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        if not isinstance(numerator, pd.Series):
            numerator = pd.Series(numerator, index=denominator.index)
        if not isinstance(denominator, pd.Series):
            denominator = pd.Series(denominator, index=numerator.index)
        out = numerator / denominator.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)
    if denominator in (0, None):
        return np.nan
    return numerator / denominator


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_internal_weekly(local_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    loaded = weekly_from_daily(local_data.get("loaded_daily", pd.DataFrame()), "loaded")
    packed = weekly_from_daily(local_data.get("packed_daily", pd.DataFrame()), "packed")
    weekly = loaded.join(packed, how="outer").sort_index()
    if weekly.empty:
        return weekly
    weekly = weekly.fillna(0.0)
    weekly["loaded_export_share"] = safe_div(weekly.get("loaded_trips_export", 0.0), weekly.get("loaded_trips_total", np.nan)).fillna(0.0)
    weekly["loaded_europe_share"] = safe_div(weekly.get("loaded_trips_europe", 0.0), weekly.get("loaded_trips_total", np.nan)).fillna(0.0)
    weekly["loaded_container_share"] = safe_div(weekly.get("loaded_containers_count", 0.0), weekly.get("loaded_trips_total", np.nan)).fillna(0.0)
    weekly["loaded_orders_per_trip"] = safe_div(weekly.get("loaded_orders_nunique", 0.0), weekly.get("loaded_trips_total", np.nan)).fillna(0.0)
    weekly["loaded_tons_per_trip"] = safe_div(weekly.get("loaded_gross_tons", 0.0), weekly.get("loaded_trips_total", np.nan)).fillna(0.0)
    weekly["packed_cartons_per_pallet"] = safe_div(weekly.get("packed_cartons_count", 0.0), weekly.get("packed_pallets_count", np.nan)).fillna(0.0)
    weekly["packed_tons_per_pallet"] = safe_div(weekly.get("packed_gross_tons", 0.0), weekly.get("packed_pallets_count", np.nan)).fillna(0.0)
    weekly["packed_binhits_per_vydejka"] = safe_div(weekly.get("packed_binhits", 0.0), weekly.get("packed_vydejky_unique", np.nan)).fillna(0.0)
    pressure_parts = []
    for col in ["loaded_orders_nunique", "loaded_trips_total", "packed_pallets_count", "packed_binhits"]:
        if col in weekly.columns:
            pressure_parts.append(zscore(weekly[col]))
    weekly["warehouse_pressure_index"] = 100 + 10 * (sum(pressure_parts) / len(pressure_parts)) if pressure_parts else np.nan
    return weekly


def summarize_years(df: pd.DataFrame, date_col: str = "date") -> str:
    cov = coverage_table(df, date_col)
    if cov.empty:
        return ""
    return ", ".join(f"{int(row.year)}:{int(row.rows)}" for row in cov.itertuples())


def pipeline_audit(daily_df: pd.DataFrame, external_weekly: Optional[pd.DataFrame] = None, metric: Optional[str] = None) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["step", "rows", "min_date", "max_date", "years", "note"])
    parsed = daily_df.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    parsed = parsed.dropna(subset=["date"]).sort_values("date")
    metric_col = metric or next((c for c in parsed.columns if pd.api.types.is_numeric_dtype(parsed[c]) and c not in {"iso_year", "iso_week", "weekday"}), None)

    def audit_row(name: str, frame: pd.DataFrame, date_col: str, note: str = "") -> dict:
        if frame.empty or date_col not in frame.columns:
            return {"step": name, "rows": 0, "min_date": pd.NaT, "max_date": pd.NaT, "years": "", "note": note}
        d = pd.to_datetime(frame[date_col], errors="coerce").dropna()
        return {
            "step": name,
            "rows": int(len(frame)),
            "min_date": d.min() if not d.empty else pd.NaT,
            "max_date": d.max() if not d.empty else pd.NaT,
            "years": summarize_years(frame.rename(columns={date_col: "date"}), "date"),
            "note": note,
        }

    out = [audit_row("raw_csv", daily_df, "date", "Vstupní CSV bez zásahu.")]
    out.append(audit_row("parsed_dates", parsed, "date", "Po převodu date na datetime."))

    expected = expected_dates(parsed)
    if metric_col is not None and not expected.empty:
        regular = regular_daily_series(parsed, metric_col).to_frame(metric_col).reset_index().rename(columns={"index": "date"})
        out.append(audit_row("regularized_daily", regular, "date", "Reindex na očekávané provozní dny, chybějící dny vyplněny nulou."))

    weekly = weekly_from_daily(parsed, "metric").reset_index()
    if not weekly.empty:
        out.append(audit_row("weekly_agg", weekly.rename(columns={"week_start": "date"}), "date", "Agregace po týdnech."))

    if external_weekly is not None and not external_weekly.empty and not weekly.empty:
        ext = external_weekly.copy()
        ext.index = pd.to_datetime(ext.index, errors="coerce")
        left_join = weekly.set_index("week_start").join(ext, how="left").reset_index()
        out.append(audit_row("weekly_left_join_world", left_join.rename(columns={"week_start": "date"}), "date", "Správný left join na externí faktory."))

        risky_inner = weekly.set_index("week_start").join(ext.dropna(how="all"), how="inner").reset_index()
        out.append(audit_row("weekly_inner_join_world", risky_inner.rename(columns={"week_start": "date"}), "date", "Rizikový inner join. Tady může zmizet 2023."))

        risky_dropna = left_join.dropna()
        out.append(audit_row("weekly_dropna_all", risky_dropna.rename(columns={"week_start": "date"}), "date", "Rizikový dropna přes celý dataframe po merge."))

    return pd.DataFrame(out)
