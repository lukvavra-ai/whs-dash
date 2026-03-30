from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import unicodedata

import numpy as np
import pandas as pd

from forecasting import align_known_exog, baseline_as_result, ridge_forecast, rolling_backtest


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "staffing_forecast_exports"
MANUAL_HISTORY_PATH = ROOT / "staffing_manual_history.csv"
MANUAL_README_PATH = ROOT / "STAFFING_MANUAL_INPUT_README.md"
HORIZON_DAYS = 91

EXCLUDED_ATTENDANCE_NAMES = {
    "Adam Zeman",
    "Denisa Pavcová",
    "Diana Jeřábková",
    "Eva Bolfová",
    "Filip Branda",
    "Lukáš Fišer",
    "Marek Špáta",
    "Marián Ondruška",
    "Tsanko Nachev",
}

PICK_ACTIVITIES = {"Kompletace", "Kontrola"}
INBOUND_ACTIVITIES = {"Vykládka", "Příjem", "Zaskladnění"}
IGNORE_ACTIVITIES = {"Příchod", "Odchod", "Standardní pauza", "Dlouhá pauza"}

MANUAL_CATEGORY_COLUMNS = [
    "kmen_inbound_morning",
    "kmen_inbound_afternoon",
    "kmen_loading_morning",
    "kmen_loading_afternoon",
    "kmen_pick_morning",
    "kmen_pick_afternoon",
    "kmen_other_morning",
    "kmen_other_afternoon",
    "agency_inbound_day",
    "agency_inbound_night",
    "agency_loading_day",
    "agency_loading_night",
    "agency_pick_day",
    "agency_pick_night",
    "agency_other_day",
    "agency_other_night",
]
KMEAN_CATEGORY_COLUMNS = [col for col in MANUAL_CATEGORY_COLUMNS if col.startswith("kmen_")]
AGENCY_CATEGORY_COLUMNS = [col for col in MANUAL_CATEGORY_COLUMNS if col.startswith("agency_")]
CATEGORY_WEIGHTS = {col: (1.0 if col.startswith("kmen_") else 1.5) for col in MANUAL_CATEGORY_COLUMNS}


def _norm(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


def _safe_median(values: pd.Series, default: float = 0.0) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return float(default)
    return float(series.median())


def _metrics(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    actual = pd.to_numeric(actual, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce").reindex(actual.index)
    mask = actual.notna() & pred.notna()
    if not mask.any():
        return np.nan, np.nan
    actual = actual.loc[mask].astype(float)
    pred = pred.loc[mask].astype(float)
    err = (pred - actual).abs()
    denom = actual.abs().sum()
    wape = float(err.sum() / denom) if denom > 0 else np.nan
    mae = float(err.mean()) if len(err) else np.nan
    return wape, mae


def _latest_known_date(*dates: object) -> pd.Timestamp:
    valid = [pd.Timestamp(value).normalize() for value in dates if pd.notna(value)]
    if not valid:
        return pd.Timestamp.today().normalize()
    return max(valid)


def _find_sheet_name(path: Path, target_norm: str) -> str:
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        if _norm(sheet) == target_norm:
            return sheet
    return xls.sheet_names[0]


def load_packed_daily() -> pd.DataFrame:
    return pd.read_csv(ROOT / "packed_daily_kpis.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def load_packed_shift() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "packed_shift_kpis.csv", parse_dates=["date"])
    pivot = df.pivot_table(index="date", columns="shift", values="binhits", aggfunc="sum").sort_index().reset_index()
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
    return pd.read_csv(ROOT / "loaded_daily_kpis.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def load_warehouse_state_daily() -> pd.DataFrame:
    path = ROOT / "warehouse_state_exports" / "warehouse_state_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def _load_tm_activity_rows() -> pd.DataFrame:
    root = ROOT / "Reporty ESAB" / "Reporting" / "TimeManagement" / "Data"
    frames = []
    for path in sorted(root.glob("*.csv")):
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        cols = {_norm(c): c for c in df.columns}
        required = ["datum smeny", "smena", "zdroj", "skladnik", "cinnost", "trvani min.", "vyskytu", "zakaznik"]
        if not all(key in cols for key in required):
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(df[cols["datum smeny"]], dayfirst=True, errors="coerce"),
                    "shift": df[cols["smena"]].astype(str).str.strip(),
                    "source": df[cols["zdroj"]].astype(str).str.strip(),
                    "worker": df[cols["skladnik"]].astype(str).str.strip(),
                    "activity": df[cols["cinnost"]].astype(str).str.strip(),
                    "customer": df[cols["zakaznik"]].astype(str).str.strip(),
                    "duration_min": pd.to_numeric(df[cols["trvani min."]], errors="coerce").fillna(0.0),
                    "binhits_tm": pd.to_numeric(df[cols["vyskytu"]], errors="coerce").fillna(0.0),
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["date", "shift", "source", "worker", "activity", "customer", "duration_min", "binhits_tm"])

    data = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    data["source_norm"] = data["source"].map(_norm)
    data["shift_norm"] = data["shift"].map(_norm)
    data["activity_norm"] = data["activity"].map(_norm)
    data["customer_norm"] = data["customer"].map(_norm)
    data["hours"] = data["duration_min"] / 60.0
    return data


def _bucket_for_row(source_norm: str, shift_norm: str) -> str:
    if source_norm == "kmen":
        return "morning" if "den" in shift_norm else "afternoon"
    if source_norm == "agentura":
        return "day" if "den" in shift_norm else "night"
    return ""


def _category_for_activity(activity: str) -> str:
    if activity in PICK_ACTIVITIES:
        return "pick"
    if activity in INBOUND_ACTIVITIES:
        return "inbound"
    return "other"


def load_activity_scope_daily() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _load_tm_activity_rows()
    if data.empty:
        empty = pd.DataFrame(columns=["date"])
        return empty, empty, empty

    data["bucket"] = [_bucket_for_row(src, sh) for src, sh in zip(data["source_norm"], data["shift_norm"])]
    scope_people = data[["date", "worker", "source_norm", "bucket"]].drop_duplicates()

    actuals = (
        scope_people.groupby("date", as_index=False)
        .agg(
            att_headcount=("worker", "nunique"),
            att_kmen_headcount=("worker", lambda s: int(scope_people.loc[s.index, "source_norm"].eq("kmen").sum())),
            att_agency_headcount=("worker", lambda s: int(scope_people.loc[s.index, "source_norm"].eq("agentura").sum())),
            day_att_headcount=("worker", lambda s: int(scope_people.loc[s.index, "bucket"].isin({"morning", "day"}).sum())),
            night_att_headcount=("worker", lambda s: int(scope_people.loc[s.index, "bucket"].isin({"afternoon", "night"}).sum())),
            att_kmen_early_headcount=("worker", lambda s: int(scope_people.loc[s.index, "bucket"].eq("morning").sum())),
            att_kmen_late_headcount=("worker", lambda s: int(scope_people.loc[s.index, "bucket"].eq("afternoon").sum())),
            att_agency_day_headcount=("worker", lambda s: int(scope_people.loc[s.index, "bucket"].eq("day").sum())),
            att_agency_night_headcount=("worker", lambda s: int(scope_people.loc[s.index, "bucket"].eq("night").sum())),
        )
        .sort_values("date")
    )

    work_rows = data[~data["activity"].isin(IGNORE_ACTIVITIES)].copy()
    work_rows["category"] = work_rows["activity"].map(_category_for_activity)
    dominant = (
        work_rows.groupby(["date", "source_norm", "bucket", "worker", "category"], as_index=False)["hours"]
        .sum()
        .sort_values(["date", "source_norm", "bucket", "worker", "hours"], ascending=[True, True, True, True, False])
        .drop_duplicates(["date", "source_norm", "bucket", "worker"], keep="first")
    )

    label_map = {
        ("kmen", "morning", "inbound"): "suggested_kmen_inbound_morning",
        ("kmen", "afternoon", "inbound"): "suggested_kmen_inbound_afternoon",
        ("kmen", "morning", "pick"): "suggested_kmen_pick_morning",
        ("kmen", "afternoon", "pick"): "suggested_kmen_pick_afternoon",
        ("kmen", "morning", "other"): "suggested_kmen_other_morning",
        ("kmen", "afternoon", "other"): "suggested_kmen_other_afternoon",
        ("agentura", "day", "inbound"): "suggested_agency_inbound_day",
        ("agentura", "night", "inbound"): "suggested_agency_inbound_night",
        ("agentura", "day", "pick"): "suggested_agency_pick_day",
        ("agentura", "night", "pick"): "suggested_agency_pick_night",
        ("agentura", "day", "other"): "suggested_agency_other_day",
        ("agentura", "night", "other"): "suggested_agency_other_night",
    }
    dominant["suggested_col"] = [
        label_map.get((src, bucket, category), "")
        for src, bucket, category in zip(dominant["source_norm"], dominant["bucket"], dominant["category"])
    ]
    suggested = dominant[dominant["suggested_col"] != ""]
    if suggested.empty:
        suggested_daily = pd.DataFrame(columns=["date"])
    else:
        suggested_daily = (
            suggested.groupby(["date", "suggested_col"])["worker"]
            .nunique()
            .unstack(fill_value=0)
            .reset_index()
            .sort_values("date")
        )

    for col in [
        "suggested_kmen_inbound_morning",
        "suggested_kmen_inbound_afternoon",
        "suggested_kmen_pick_morning",
        "suggested_kmen_pick_afternoon",
        "suggested_kmen_other_morning",
        "suggested_kmen_other_afternoon",
        "suggested_agency_inbound_day",
        "suggested_agency_inbound_night",
        "suggested_agency_pick_day",
        "suggested_agency_pick_night",
        "suggested_agency_other_day",
        "suggested_agency_other_night",
    ]:
        if col not in suggested_daily.columns:
            suggested_daily[col] = 0
    suggested_daily["suggested_kmen_loading_morning"] = 0
    suggested_daily["suggested_kmen_loading_afternoon"] = 0
    suggested_daily["suggested_agency_loading_day"] = 0
    suggested_daily["suggested_agency_loading_night"] = 0

    share_base = data[~data["customer_norm"].isin({"", "nan"})].copy()
    share_base = share_base[share_base["customer_norm"] != "esab/gce"]
    share_base["week_end"] = share_base["date"] + pd.to_timedelta(6 - share_base["date"].dt.weekday, unit="D")
    weekly = (
        share_base.groupby(["week_end", "customer_norm"], as_index=False)["hours"]
        .sum()
        .pivot(index="week_end", columns="customer_norm", values="hours")
        .fillna(0.0)
        .sort_index()
    )
    if "esab" not in weekly.columns:
        weekly["esab"] = 0.0
    if "gce" not in weekly.columns:
        weekly["gce"] = 0.0
    weekly["esab_activity_share"] = weekly["esab"] / (weekly["esab"] + weekly["gce"]).replace(0, np.nan)
    valid_share = weekly["esab_activity_share"].where((weekly["esab_activity_share"] >= 0.10) & (weekly["esab_activity_share"] <= 0.95))
    fallback_share = valid_share.dropna().tail(13).median()
    if pd.isna(fallback_share):
        fallback_share = valid_share.dropna().median()
    if pd.isna(fallback_share):
        fallback_share = 0.70
    weekly["esab_activity_share"] = weekly["esab_activity_share"].where((weekly["esab_activity_share"] >= 0.10) & (weekly["esab_activity_share"] <= 0.95), fallback_share)

    share = data[["date"]].drop_duplicates().sort_values("date")
    share["week_end"] = share["date"] + pd.to_timedelta(6 - share["date"].dt.weekday, unit="D")
    share = share.merge(weekly[["esab_activity_share"]].reset_index(), on="week_end", how="left").drop(columns=["week_end"])
    share["esab_activity_share"] = share["esab_activity_share"].ffill().bfill().fillna(float(fallback_share))
    return actuals, suggested_daily, share[["date", "esab_activity_share"]]


def load_paid_hours_daily() -> pd.DataFrame:
    path = ROOT / "Reporty ESAB" / "Reporting" / "TimeManagement" / "Timemanagement.xlsx"
    sheet_name = _find_sheet_name(path, "dochazka")
    raw = pd.read_excel(path, sheet_name=sheet_name)
    cols = {_norm(c): c for c in raw.columns}
    required = ["jmeno", "prichod", "odchod", "prestavka (min)"]
    if not all(key in cols for key in required):
        return pd.DataFrame(columns=["date", "paid_hours_total"])

    df = pd.DataFrame(
        {
            "name": raw[cols["jmeno"]].astype(str).str.strip(),
            "arrival": pd.to_datetime(raw[cols["prichod"]], errors="coerce", dayfirst=True),
            "departure": pd.to_datetime(raw[cols["odchod"]], errors="coerce", dayfirst=True),
            "break_min": pd.to_numeric(raw[cols["prestavka (min)"]], errors="coerce").fillna(0.0),
        }
    )
    df = df[~df["name"].isin(EXCLUDED_ATTENDANCE_NAMES)].dropna(subset=["arrival", "departure"])
    df["date"] = df["arrival"].dt.normalize()
    df["paid_hours_total"] = ((df["departure"] - df["arrival"]).dt.total_seconds() / 3600.0) - (df["break_min"] / 60.0)
    return df.groupby("date", as_index=False)["paid_hours_total"].sum().sort_values("date")


def _manual_helper_base() -> pd.DataFrame:
    packed = load_packed_daily()[["date", "binhits", "gross_tons"]]
    loaded = load_loaded_daily()[["date", "trips_total", "containers_count", "orders_nunique", "gross_tons"]].rename(
        columns={"gross_tons": "loaded_gross_tons"}
    )
    warehouse = load_warehouse_state_daily()
    activity_actuals, suggested, esab_share = load_activity_scope_daily()
    paid = load_paid_hours_daily()

    helper = packed.merge(loaded, on="date", how="outer")
    helper = helper.merge(
        warehouse[["date", "inbound_gross_kg", "inbound_docs", "packing_orders", "packing_pallets"]] if not warehouse.empty else pd.DataFrame(columns=["date"]),
        on="date",
        how="left",
    )
    helper = helper.merge(activity_actuals, on="date", how="left")
    helper = helper.merge(esab_share, on="date", how="left")
    helper = helper.merge(paid, on="date", how="left")
    helper = helper.merge(suggested, on="date", how="left")
    helper = helper.sort_values("date").reset_index(drop=True)
    helper["esab_activity_share"] = helper["esab_activity_share"].ffill().bfill().fillna(1.0)
    helper["esab_paid_hours"] = helper["paid_hours_total"] * helper["esab_activity_share"]
    helper["esab_fte_8h"] = helper["esab_paid_hours"] / 8.0
    return helper


def ensure_manual_history() -> pd.DataFrame:
    helper = _manual_helper_base().copy()
    if helper.empty:
        helper = pd.DataFrame(columns=["date"])

    helper["customer"] = "ESAB"
    helper["kmen_total_people"] = np.nan
    helper["agency_total_people"] = np.nan
    helper["notes"] = ""
    for col in MANUAL_CATEGORY_COLUMNS:
        helper[col] = np.nan

    helper = helper.rename(
        columns={
            "binhits": "helper_binhits",
            "gross_tons": "helper_packing_gw_t",
            "trips_total": "helper_trips_total",
            "containers_count": "helper_containers_count",
            "orders_nunique": "helper_loaded_orders",
            "loaded_gross_tons": "helper_loaded_gw_t",
            "inbound_gross_kg": "helper_inbound_kg",
            "inbound_docs": "helper_inbound_docs",
            "packing_orders": "helper_packing_orders",
            "packing_pallets": "helper_packing_pallets",
            "paid_hours_total": "helper_paid_hours_total",
            "esab_activity_share": "helper_esab_share",
            "esab_paid_hours": "helper_esab_paid_hours",
            "esab_fte_8h": "helper_esab_fte_8h",
            "att_headcount": "helper_activity_people",
            "att_kmen_headcount": "helper_activity_kmen_people",
            "att_agency_headcount": "helper_activity_agency_people",
            "day_att_headcount": "helper_activity_day_people",
            "night_att_headcount": "helper_activity_night_people",
        }
    )

    ordered_cols = (
        ["date", "customer", "kmen_total_people", "agency_total_people"]
        + MANUAL_CATEGORY_COLUMNS
        + ["notes"]
        + [col for col in helper.columns if col.startswith("helper_") or col.startswith("suggested_")]
    )
    helper = helper.reindex(columns=ordered_cols)

    if MANUAL_HISTORY_PATH.exists():
        existing = pd.read_csv(MANUAL_HISTORY_PATH, parse_dates=["date"])
        for col in ordered_cols:
            if col not in existing.columns:
                existing[col] = np.nan if col not in {"customer", "notes"} else ""
        existing = existing[ordered_cols]
        merged = helper.merge(existing, on="date", how="left", suffixes=("_new", ""))
        result = pd.DataFrame({"date": merged["date"]})
        for col in ordered_cols:
            if col == "date":
                continue
            if f"{col}_new" in merged.columns:
                if col.startswith("helper_") or col.startswith("suggested_") or col == "customer":
                    result[col] = merged[f"{col}_new"]
                else:
                    result[col] = merged[col]
            else:
                result[col] = merged[col]
        result = result[ordered_cols]
    else:
        result = helper[ordered_cols]

    result.to_csv(MANUAL_HISTORY_PATH, index=False, encoding="utf-8-sig")
    MANUAL_README_PATH.write_text(
        "\n".join(
            [
                "# Staffing Manual Input",
                "",
                "Soubor `staffing_manual_history.csv` slouzi jako rucne vyplnitelna historie staffing rozpadů pro ESAB.",
                "",
                "Vyplnuj pouze minulost. Budoucnost pocita forecast.",
                "",
                "Logika poli:",
                "- `kmen_*_morning` = kmen 06:00-14:00",
                "- `kmen_*_afternoon` = kmen 14:00-22:00",
                "- `agency_*_day` = agentura 07:00-19:00",
                "- `agency_*_night` = agentura 19:00-07:00",
                "",
                "Kategorie:",
                "- `inbound` = vykladka / prijem / zaskladneni",
                "- `loading` = nakladky",
                "- `pick` = pick / kompletace",
                "- `other` = ostatni cinnosti",
                "",
                "Pomocne sloupce `helper_*` a `suggested_*` jsou generovane automaticky. Nejsou povinne pro editaci.",
                "",
                "Forecast je ukotveny pres `helper_esab_paid_hours` a `helper_esab_fte_8h`.",
            ]
        ),
        encoding="utf-8",
    )
    return result


def _filled_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")


def _manual_actuals(manual: pd.DataFrame) -> pd.DataFrame:
    df = manual.copy()
    for col in MANUAL_CATEGORY_COLUMNS:
        df[col] = _filled_numeric(df, col)

    df["manual_kmen"] = df[KMEAN_CATEGORY_COLUMNS].sum(axis=1, min_count=1)
    df["manual_agency"] = df[AGENCY_CATEGORY_COLUMNS].sum(axis=1, min_count=1)
    df["manual_total_people"] = df["manual_kmen"] + df["manual_agency"]
    df["manual_any_filled"] = df[MANUAL_CATEGORY_COLUMNS].fillna(0.0).sum(axis=1) > 0

    suggested_map = {
        "kmen_inbound_morning": "suggested_kmen_inbound_morning",
        "kmen_inbound_afternoon": "suggested_kmen_inbound_afternoon",
        "kmen_loading_morning": "suggested_kmen_loading_morning",
        "kmen_loading_afternoon": "suggested_kmen_loading_afternoon",
        "kmen_pick_morning": "suggested_kmen_pick_morning",
        "kmen_pick_afternoon": "suggested_kmen_pick_afternoon",
        "kmen_other_morning": "suggested_kmen_other_morning",
        "kmen_other_afternoon": "suggested_kmen_other_afternoon",
        "agency_inbound_day": "suggested_agency_inbound_day",
        "agency_inbound_night": "suggested_agency_inbound_night",
        "agency_loading_day": "suggested_agency_loading_day",
        "agency_loading_night": "suggested_agency_loading_night",
        "agency_pick_day": "suggested_agency_pick_day",
        "agency_pick_night": "suggested_agency_pick_night",
        "agency_other_day": "suggested_agency_other_day",
        "agency_other_night": "suggested_agency_other_night",
    }
    for manual_col, suggested_col in suggested_map.items():
        df[f"selected_{manual_col}"] = _filled_numeric(df, manual_col).where(_filled_numeric(df, manual_col).notna(), _filled_numeric(df, suggested_col))

    df["selected_kmen_early"] = df[[f"selected_{c}" for c in ["kmen_inbound_morning", "kmen_loading_morning", "kmen_pick_morning", "kmen_other_morning"]]].sum(axis=1, min_count=1)
    df["selected_kmen_late"] = df[[f"selected_{c}" for c in ["kmen_inbound_afternoon", "kmen_loading_afternoon", "kmen_pick_afternoon", "kmen_other_afternoon"]]].sum(axis=1, min_count=1)
    df["selected_agency_day"] = df[[f"selected_{c}" for c in ["agency_inbound_day", "agency_loading_day", "agency_pick_day", "agency_other_day"]]].sum(axis=1, min_count=1)
    df["selected_agency_night"] = df[[f"selected_{c}" for c in ["agency_inbound_night", "agency_loading_night", "agency_pick_night", "agency_other_night"]]].sum(axis=1, min_count=1)
    df["selected_kmen"] = df["selected_kmen_early"] + df["selected_kmen_late"]
    df["selected_agency"] = df["selected_agency_day"] + df["selected_agency_night"]
    df["selected_total_people"] = df["selected_kmen"] + df["selected_agency"]
    df["selected_total_fte_8h"] = df["selected_kmen_early"] + df["selected_kmen_late"] + 1.5 * df["selected_agency_day"] + 1.5 * df["selected_agency_night"]
    return df


def build_staffing_actuals() -> tuple[pd.DataFrame, pd.DataFrame]:
    packed = load_packed_daily()
    packed_shift = load_packed_shift()
    loaded = load_loaded_daily().rename(columns={"orders_nunique": "loaded_orders_nunique", "gross_tons": "loaded_gross_tons"})
    warehouse = load_warehouse_state_daily()
    activity_actuals, _suggested_daily, _share = load_activity_scope_daily()
    manual = _manual_actuals(ensure_manual_history())

    daily = packed.merge(packed_shift, on="date", how="left")
    daily = daily.merge(
        loaded[["date", "trips_total", "containers_count", "loaded_orders_nunique", "loaded_gross_tons"]],
        on="date",
        how="left",
    )
    if not warehouse.empty:
        daily = daily.merge(
            warehouse[["date", "inbound_gross_kg", "inbound_docs", "packing_orders", "packing_pallets"]],
            on="date",
            how="left",
        )
    daily = daily.merge(activity_actuals, on="date", how="left")
    daily = daily.merge(
        manual[
            [
                "date",
                "helper_paid_hours_total",
                "helper_esab_share",
                "helper_esab_paid_hours",
                "helper_esab_fte_8h",
                "manual_any_filled",
                "selected_total_people",
                "selected_kmen",
                "selected_agency",
                "selected_kmen_early",
                "selected_kmen_late",
                "selected_agency_day",
                "selected_agency_night",
            ]
        ],
        on="date",
        how="left",
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["weekday"] = daily["date"].dt.day_name()
    daily["paid_hours_total"] = pd.to_numeric(daily["helper_paid_hours_total"], errors="coerce")
    daily["esab_share"] = pd.to_numeric(daily["helper_esab_share"], errors="coerce")
    daily["att_hours"] = pd.to_numeric(daily["helper_esab_paid_hours"], errors="coerce")
    daily["esab_fte_8h"] = pd.to_numeric(daily["helper_esab_fte_8h"], errors="coerce")

    daily["att_headcount"] = pd.to_numeric(daily["selected_total_people"], errors="coerce").where(
        pd.to_numeric(daily["selected_total_people"], errors="coerce").notna(),
        pd.to_numeric(daily["att_headcount"], errors="coerce"),
    )
    daily["att_kmen_headcount"] = pd.to_numeric(daily["selected_kmen"], errors="coerce").where(
        pd.to_numeric(daily["selected_kmen"], errors="coerce").notna(),
        pd.to_numeric(daily["att_kmen_headcount"], errors="coerce"),
    )
    daily["att_agency_headcount"] = pd.to_numeric(daily["selected_agency"], errors="coerce").where(
        pd.to_numeric(daily["selected_agency"], errors="coerce").notna(),
        pd.to_numeric(daily["att_agency_headcount"], errors="coerce"),
    )
    daily["att_kmen_early_headcount"] = pd.to_numeric(daily["selected_kmen_early"], errors="coerce").where(
        pd.to_numeric(daily["selected_kmen_early"], errors="coerce").notna(),
        pd.to_numeric(daily["att_kmen_early_headcount"], errors="coerce"),
    )
    daily["att_kmen_late_headcount"] = pd.to_numeric(daily["selected_kmen_late"], errors="coerce").where(
        pd.to_numeric(daily["selected_kmen_late"], errors="coerce").notna(),
        pd.to_numeric(daily["att_kmen_late_headcount"], errors="coerce"),
    )
    daily["att_agency_day_headcount"] = pd.to_numeric(daily["selected_agency_day"], errors="coerce").where(
        pd.to_numeric(daily["selected_agency_day"], errors="coerce").notna(),
        pd.to_numeric(daily["att_agency_day_headcount"], errors="coerce"),
    )
    daily["att_agency_night_headcount"] = pd.to_numeric(daily["selected_agency_night"], errors="coerce").where(
        pd.to_numeric(daily["selected_agency_night"], errors="coerce").notna(),
        pd.to_numeric(daily["att_agency_night_headcount"], errors="coerce"),
    )
    daily["day_att_headcount"] = daily["att_kmen_early_headcount"] + daily["att_agency_day_headcount"]
    daily["night_att_headcount"] = daily["att_kmen_late_headcount"] + daily["att_agency_night_headcount"]
    daily["required_reference_fte_8h"] = daily["esab_fte_8h"]
    return daily, manual


@dataclass
class SeriesForecast:
    target: str
    selected_model: str
    forecast: pd.Series
    scores: pd.DataFrame


def forecast_daily_series(
    series: pd.Series,
    exog_full: pd.DataFrame | None = None,
    backtest_horizon: int = 10,
    anchor_date: pd.Timestamp | None = None,
) -> SeriesForecast:
    s = series.dropna().astype(float)
    anchor = pd.Timestamp(anchor_date).normalize() if anchor_date is not None else pd.Timestamp(s.index.max()).normalize()
    gap_days = max(0, (anchor - pd.Timestamp(s.index.max()).normalize()).days)
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
    forecast = selected.forecast[selected.forecast.index > anchor].copy().clip(lower=0.0)
    return SeriesForecast(target=series.name or "target", selected_model=best, forecast=forecast, scores=scores)


def _recent_category_shares(history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, float]:
    prior = history[history["date"] < target_date].copy()
    if prior.empty:
        return {}
    same_weekday = prior[prior["date"].dt.weekday == target_date.weekday()].tail(8)
    recent = prior.tail(28)
    base = same_weekday if len(same_weekday) >= 3 else recent
    if base.empty:
        return {}

    medians: dict[str, float] = {}
    total = 0.0
    for col in MANUAL_CATEGORY_COLUMNS:
        share_col = f"{col}_fte_share"
        value = _safe_median(base[share_col], 0.0)
        medians[col] = max(value, 0.0)
        total += medians[col]
    if total <= 0:
        return {}
    return {col: medians[col] / total for col in MANUAL_CATEGORY_COLUMNS}


def _allocate_counts_from_fte(target_fte: float, shares: dict[str, float]) -> dict[str, int]:
    if target_fte <= 0 or not shares:
        return {col: 0 for col in MANUAL_CATEGORY_COLUMNS}

    normalized = {col: max(shares.get(col, 0.0), 0.0) for col in MANUAL_CATEGORY_COLUMNS}
    total_share = sum(normalized.values())
    if total_share <= 0:
        return {col: 0 for col in MANUAL_CATEGORY_COLUMNS}
    normalized = {col: normalized[col] / total_share for col in MANUAL_CATEGORY_COLUMNS}

    raw = {col: (target_fte * normalized[col] / CATEGORY_WEIGHTS[col]) for col in MANUAL_CATEGORY_COLUMNS}
    counts = {col: int(math.floor(max(raw[col], 0.0))) for col in MANUAL_CATEGORY_COLUMNS}
    current_fte = sum(counts[col] * CATEGORY_WEIGHTS[col] for col in MANUAL_CATEGORY_COLUMNS)
    order = sorted(MANUAL_CATEGORY_COLUMNS, key=lambda col: raw[col] - counts[col], reverse=True)
    while current_fte + 0.5 < target_fte and order:
        best = min(order, key=lambda col: abs((current_fte + CATEGORY_WEIGHTS[col]) - target_fte))
        counts[best] += 1
        current_fte += CATEGORY_WEIGHTS[best]
        if current_fte >= target_fte:
            break
    return counts


def build_category_profile_history(manual: pd.DataFrame) -> pd.DataFrame:
    hist = manual.copy()
    for col in MANUAL_CATEGORY_COLUMNS:
        hist[col] = pd.to_numeric(hist[f"selected_{col}"], errors="coerce").fillna(0.0)
        hist[f"{col}_fte"] = hist[col] * CATEGORY_WEIGHTS[col]
    hist["total_fte"] = hist[[f"{col}_fte" for col in MANUAL_CATEGORY_COLUMNS]].sum(axis=1)
    hist = hist[hist["total_fte"] > 0].copy()
    for col in MANUAL_CATEGORY_COLUMNS:
        hist[f"{col}_fte_share"] = hist[f"{col}_fte"] / hist["total_fte"]
    hist["date"] = pd.to_datetime(hist["date"])
    hist["weekday"] = hist["date"].dt.day_name()
    return hist


def staffing_ratio_backtest(actuals: pd.DataFrame, profile_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = actuals.dropna(subset=["date", "esab_fte_8h", "att_headcount"]).copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for row in data.itertuples():
        day = pd.Timestamp(row.date)
        shares = _recent_category_shares(profile_history, day)
        if not shares:
            continue
        counts = _allocate_counts_from_fte(float(row.esab_fte_8h), shares)
        pred_kmen_early = counts["kmen_inbound_morning"] + counts["kmen_loading_morning"] + counts["kmen_pick_morning"] + counts["kmen_other_morning"]
        pred_kmen_late = counts["kmen_inbound_afternoon"] + counts["kmen_loading_afternoon"] + counts["kmen_pick_afternoon"] + counts["kmen_other_afternoon"]
        pred_agency_day = counts["agency_inbound_day"] + counts["agency_loading_day"] + counts["agency_pick_day"] + counts["agency_other_day"]
        pred_agency_night = counts["agency_inbound_night"] + counts["agency_loading_night"] + counts["agency_pick_night"] + counts["agency_other_night"]
        rows.append(
            {
                "date": day,
                "pred_fte_8h": float(sum(counts[c] * CATEGORY_WEIGHTS[c] for c in MANUAL_CATEGORY_COLUMNS)),
                "actual_fte_8h": row.esab_fte_8h,
                "pred_headcount": pred_kmen_early + pred_kmen_late + pred_agency_day + pred_agency_night,
                "actual_headcount": row.att_headcount,
                "pred_kmen": pred_kmen_early + pred_kmen_late,
                "actual_kmen": row.att_kmen_headcount,
                "pred_agency": pred_agency_day + pred_agency_night,
                "actual_agency": row.att_agency_headcount,
                "pred_day_shift": pred_kmen_early + pred_agency_day,
                "actual_day_shift": row.day_att_headcount,
                "pred_night_shift": pred_kmen_late + pred_agency_night,
                "actual_night_shift": row.night_att_headcount,
            }
        )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return pred_df, pd.DataFrame()

    summary_rows = []
    for actual_col, pred_col, metric_name in [
        ("actual_fte_8h", "pred_fte_8h", "fte_8h"),
        ("actual_headcount", "pred_headcount", "headcount_total"),
        ("actual_kmen", "pred_kmen", "kmen_total"),
        ("actual_agency", "pred_agency", "agency_total"),
        ("actual_day_shift", "pred_day_shift", "day_shift_total"),
        ("actual_night_shift", "pred_night_shift", "night_shift_total"),
    ]:
        wape, mae = _metrics(pred_df[actual_col], pred_df[pred_col])
        summary_rows.append({"metric": metric_name, "wape": wape, "mae": mae, "n_days": len(pred_df)})
    return pred_df, pd.DataFrame(summary_rows)


def build_forecasts() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    packed = load_packed_daily().set_index("date")
    packed_shift = load_packed_shift().set_index("date")
    loaded = load_loaded_daily().set_index("date")
    warehouse_daily = load_warehouse_state_daily()
    warehouse = warehouse_daily.set_index("date") if not warehouse_daily.empty else pd.DataFrame()

    staffing_actuals, manual_history = build_staffing_actuals()
    staffing_actuals.to_csv(EXPORT_DIR / "staffing_capacity_actuals.csv", index=False, encoding="utf-8-sig")

    anchor_date = _latest_known_date(
        packed.index.max() if not packed.empty else pd.NaT,
        packed_shift.index.max() if not packed_shift.empty else pd.NaT,
        loaded.index.max() if not loaded.empty else pd.NaT,
        staffing_actuals["date"].max() if not staffing_actuals.empty else pd.NaT,
    )
    future_dates = pd.date_range(anchor_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")

    orders_fc = forecast_daily_series(loaded["orders_nunique"].rename("loaded_orders_nunique"), anchor_date=anchor_date)
    trips_fc = forecast_daily_series(loaded["trips_total"].rename("trips_total"), anchor_date=anchor_date)
    containers_fc = forecast_daily_series(loaded["containers_count"].rename("containers_count"), anchor_date=anchor_date)
    loaded_gross_fc = forecast_daily_series(loaded["gross_tons"].rename("loaded_gross_tons"), anchor_date=anchor_date)
    day_binhits_fc = forecast_daily_series(packed_shift["packed_day_binhits"].rename("forecast_day_binhits"), anchor_date=anchor_date)
    night_binhits_fc = forecast_daily_series(packed_shift["packed_night_binhits"].rename("forecast_night_binhits"), anchor_date=anchor_date)

    inbound_fc = None
    if isinstance(warehouse, pd.DataFrame) and not warehouse.empty and "inbound_gross_kg" in warehouse.columns:
        inbound_fc = forecast_daily_series(warehouse["inbound_gross_kg"].rename("inbound_gross_kg"), anchor_date=anchor_date)

    actuals_idx = staffing_actuals.set_index("date").sort_index()
    target = actuals_idx["esab_fte_8h"].dropna().rename("esab_fte_8h")

    exog_hist = pd.DataFrame(index=actuals_idx.index)
    exog_hist["binhits"] = actuals_idx["binhits"]
    exog_hist["packed_day_binhits"] = actuals_idx["packed_day_binhits"]
    exog_hist["packed_night_binhits"] = actuals_idx["packed_night_binhits"]
    exog_hist["trips_total"] = actuals_idx["trips_total"]
    exog_hist["containers_count"] = actuals_idx["containers_count"]
    exog_hist["loaded_orders_nunique"] = actuals_idx["loaded_orders_nunique"]
    exog_hist["loaded_gross_tons"] = actuals_idx["loaded_gross_tons"]
    if "inbound_gross_kg" in actuals_idx.columns:
        exog_hist["inbound_gross_kg"] = actuals_idx["inbound_gross_kg"]

    exog_future = pd.DataFrame(index=future_dates)
    exog_future["packed_day_binhits"] = day_binhits_fc.forecast.reindex(future_dates).ffill().bfill().values
    exog_future["packed_night_binhits"] = night_binhits_fc.forecast.reindex(future_dates).ffill().bfill().values
    exog_future["binhits"] = exog_future["packed_day_binhits"] + exog_future["packed_night_binhits"]
    exog_future["trips_total"] = trips_fc.forecast.reindex(future_dates).ffill().bfill().values
    exog_future["containers_count"] = containers_fc.forecast.reindex(future_dates).ffill().bfill().values
    exog_future["loaded_orders_nunique"] = orders_fc.forecast.reindex(future_dates).ffill().bfill().values
    exog_future["loaded_gross_tons"] = loaded_gross_fc.forecast.reindex(future_dates).ffill().bfill().values
    if inbound_fc is not None:
        exog_future["inbound_gross_kg"] = inbound_fc.forecast.reindex(future_dates).ffill().bfill().values
        exog_hist["inbound_gross_kg"] = exog_hist["inbound_gross_kg"].ffill().bfill()

    fte_exog_full = pd.concat([exog_hist, exog_future], axis=0).sort_index().ffill().bfill()
    fte_fc = forecast_daily_series(target, exog_full=fte_exog_full, anchor_date=anchor_date)

    driver_scores = pd.concat(
        [
            orders_fc.scores,
            trips_fc.scores,
            containers_fc.scores,
            loaded_gross_fc.scores,
            day_binhits_fc.scores,
            night_binhits_fc.scores,
            fte_fc.scores,
        ]
        + ([inbound_fc.scores] if inbound_fc is not None else []),
        ignore_index=True,
    )

    profile_history = build_category_profile_history(manual_history)
    fte_future = fte_fc.forecast.reindex(future_dates).ffill().bfill()
    plan_rows = []
    for day in future_dates:
        shares = _recent_category_shares(profile_history, pd.Timestamp(day))
        required_fte = max(float(fte_future.loc[day]), 0.0)
        counts = _allocate_counts_from_fte(required_fte, shares) if shares else {col: 0 for col in MANUAL_CATEGORY_COLUMNS}

        required_kmen_early = counts["kmen_inbound_morning"] + counts["kmen_loading_morning"] + counts["kmen_pick_morning"] + counts["kmen_other_morning"]
        required_kmen_late = counts["kmen_inbound_afternoon"] + counts["kmen_loading_afternoon"] + counts["kmen_pick_afternoon"] + counts["kmen_other_afternoon"]
        required_agency_day = counts["agency_inbound_day"] + counts["agency_loading_day"] + counts["agency_pick_day"] + counts["agency_other_day"]
        required_agency_night = counts["agency_inbound_night"] + counts["agency_loading_night"] + counts["agency_pick_night"] + counts["agency_other_night"]
        required_kmen = required_kmen_early + required_kmen_late
        required_agency = required_agency_day + required_agency_night
        required_headcount = required_kmen + required_agency

        recent_same_weekday = staffing_actuals[
            (staffing_actuals["date"] < day) & (staffing_actuals["date"].dt.weekday == pd.Timestamp(day).weekday())
        ]
        recent_total = _safe_median(recent_same_weekday["att_headcount"], _safe_median(staffing_actuals["att_headcount"], 0.0))
        capacity_gap = max(required_headcount - int(math.ceil(recent_total)), 0)

        row = {
            "date": day,
            "weekday": pd.Timestamp(day).day_name(),
            "forecast_day_binhits": float(exog_future.loc[day, "packed_day_binhits"]),
            "forecast_night_binhits": float(exog_future.loc[day, "packed_night_binhits"]),
            "forecast_binhits": float(exog_future.loc[day, "binhits"]),
            "forecast_loaded_orders": float(exog_future.loc[day, "loaded_orders_nunique"]),
            "forecast_trips_total": float(exog_future.loc[day, "trips_total"]),
            "forecast_containers_count": float(exog_future.loc[day, "containers_count"]),
            "forecast_loaded_gross_tons": float(exog_future.loc[day, "loaded_gross_tons"]),
            "forecast_inbound_gross_kg": float(exog_future.loc[day, "inbound_gross_kg"]) if "inbound_gross_kg" in exog_future.columns else np.nan,
            "required_fte_8h": required_fte,
            "required_paid_hours": required_fte * 8.0,
            "required_headcount": float(required_headcount),
            "required_headcount_ceiling": int(required_headcount),
            "capacity_gap_headcount": int(capacity_gap),
            "required_kmen_early": int(required_kmen_early),
            "required_kmen_late": int(required_kmen_late),
            "required_agency_day": int(required_agency_day),
            "required_agency_night": int(required_agency_night),
            "required_kmen": int(required_kmen),
            "required_agency": int(required_agency),
            "required_day_shift_workers": int(required_kmen_early + required_agency_day),
            "required_night_shift_workers": int(required_kmen_late + required_agency_night),
            "required_productive_workers": int(round(required_fte)),
            "required_productive_workers_ceiling": int(math.ceil(required_fte)),
            "fte_model": fte_fc.selected_model,
            "fte_profile_source": "manual_or_suggested_history" if shares else "missing_profile",
        }
        for col in MANUAL_CATEGORY_COLUMNS:
            row[f"required_{col}"] = int(counts[col])
        plan_rows.append(row)

    plan = pd.DataFrame(plan_rows)
    plan.to_csv(EXPORT_DIR / "staffing_forecast_daily.csv", index=False, encoding="utf-8-sig")

    ratio_pred_df, ratio_metrics = staffing_ratio_backtest(staffing_actuals, profile_history)
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
                "required_fte_8h": row["required_fte_8h"],
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
        f"- Run date: {anchor_date.date()}",
        f"- FTE model: {fte_fc.selected_model}",
        f"- Last packed actual: {packed.index.max().date()}",
        f"- Last staffing actual: {staffing_actuals['date'].max().date()}",
        "",
        "## Staffing Logic",
        "",
        "- Historie lidi se bere z workeru, kteri meli nejakou cinnost v `TimeManagement/Data`.",
        "- Celkovy forecast je ukotveny pres `ESAB placene hodiny / 8 = FTE`.",
        "- Rozpad na kmen/agenturu a cinnosti bere prioritu z rucne vyplneneho `staffing_manual_history.csv`.",
        "- Pokud manualni rozpad chybi, fallback je na auto-suggest z aktivit. Nakladka je v auto-suggestu nulova a ma se dopsat rucne.",
        "",
        "## Manual Input",
        "",
        "- Vyplnuj `staffing_manual_history.csv` jen pro minulost.",
        "- Kmen: morning 06-14, afternoon 14-22.",
        "- Agentura: day 07-19, night 19-07.",
        "- `helper_esab_paid_hours` a `helper_esab_fte_8h` slouzi jako kotva reality.",
    ]
    (EXPORT_DIR / "STAFFING_FORECAST.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    build_forecasts()
