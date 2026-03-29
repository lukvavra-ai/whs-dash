from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting import align_known_exog, baseline_as_result, blend_forecasts, compute_blend_weights, ridge_forecast, rolling_backtest


ROOT = Path(__file__).resolve().parent
REPORTING_DIR = ROOT / "Reporty ESAB" / "Reporting"
EXPORT_DIR = ROOT / "warehouse_state_exports"
HORIZON_DAYS = 91
HORIZON_WEEKS = 13
DATE_NOW = pd.Timestamp("2026-03-27")
MIN_VALID_DATE = pd.Timestamp("2022-01-01")
MAX_VALID_DATE = DATE_NOW.normalize() + pd.Timedelta(days=180)


def _norm(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.lower().strip()


def _to_number(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    text = text.str.replace("\xa0", "", regex=False).str.replace(" ", "", regex=False)

    has_comma = text.str.contains(",", regex=False, na=False)
    has_dot = text.str.contains(".", regex=False, na=False)
    both = has_comma & has_dot
    text.loc[both] = text.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    text.loc[has_comma & ~has_dot] = text.loc[has_comma & ~has_dot].str.replace(",", ".", regex=False)
    text = text.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _load_csv(path: Path) -> pd.DataFrame:
    candidates = [
        ("utf-8-sig", ";"),
        ("cp1250", ";"),
        ("latin1", ";"),
        ("utf-8-sig", ","),
        ("cp1250", ","),
        ("latin1", ","),
    ]
    for encoding, sep in candidates:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False)
        except Exception:
            continue
        if len(df.columns) >= 3:
            return df
    raise RuntimeError(f"Unable to parse {path}")


def _short_dir(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return path
    target_name = _norm(path.name)
    for child in parent.iterdir():
        if _norm(child.name) == target_name:
            return child
    raise FileNotFoundError(path)


def _columns(df: pd.DataFrame) -> dict[str, str]:
    return {_norm(col): col for col in df.columns}


def _find_col(df: pd.DataFrame, *candidates: str, contains: bool = False) -> str | None:
    cols = _columns(df)
    for candidate in candidates:
        needle = _norm(candidate)
        if not contains and needle in cols:
            return cols[needle]
        for key, original in cols.items():
            if contains and needle in key:
                return original
    return None


def _parse_day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce").dt.normalize()


def _read_yearly_folder(folder: Path) -> list[tuple[str, pd.DataFrame]]:
    files = sorted(p for p in folder.glob("*.csv") if p.is_file())
    out: list[tuple[str, pd.DataFrame]] = []
    for path in files:
        out.append((path.name, _load_csv(path)))
    return out


def _daily_agg(frame: pd.DataFrame, date_col: str, agg_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    x = frame.copy()
    x["date"] = _parse_day(x[date_col])
    x = x.dropna(subset=["date"])
    x = x[(x["date"] >= MIN_VALID_DATE) & (x["date"] <= MAX_VALID_DATE)]
    grouped = x.groupby("date").agg(**agg_map).sort_index()
    return grouped


def load_outbound_daily(root: Path) -> pd.DataFrame:
    frames = []
    for source_name, df in _read_yearly_folder(root / "Vydeje"):
        cols = _columns(df)
        date_col = cols.get("vydano")
        if not date_col:
            continue
        qty_col = _find_col(df, "mnozstvi", contains=True)
        gross_col = _find_col(df, "brutto kg")
        cbm_col = _find_col(df, "cbm")
        load_units_col = _find_col(df, "nakladove kusy")
        picks_col = _find_col(df, "pocet picku")
        full_picks_col = _find_col(df, "pocet celopicku")
        part_picks_col = _find_col(df, "pocet necelopicku")

        work = pd.DataFrame(
            {
                "date_raw": df[date_col],
                "doklad": df[cols["doklad"]].astype(str),
                "objednavka": df[cols["objednavka"]].astype(str),
                "outbound_qty": _to_number(df[qty_col]) if qty_col else 0.0,
                "outbound_gross_kg": _to_number(df[gross_col]) if gross_col else 0.0,
                "outbound_cbm": _to_number(df[cbm_col]) if cbm_col else 0.0,
                "outbound_load_units": _to_number(df[load_units_col]) if load_units_col else 0.0,
                "outbound_picks": _to_number(df[picks_col]) if picks_col else 0.0,
                "outbound_full_picks": _to_number(df[full_picks_col]) if full_picks_col else 0.0,
                "outbound_partial_picks": _to_number(df[part_picks_col]) if part_picks_col else 0.0,
            }
        )
        work["source_file"] = source_name
        frames.append(work)

    combined = pd.concat(frames, ignore_index=True)
    daily = _daily_agg(
        combined,
        "date_raw",
        {
            "outbound_docs": ("doklad", "nunique"),
            "outbound_orders": ("objednavka", "nunique"),
            "outbound_qty": ("outbound_qty", "sum"),
            "outbound_gross_kg": ("outbound_gross_kg", "sum"),
            "outbound_cbm": ("outbound_cbm", "sum"),
            "outbound_load_units": ("outbound_load_units", "sum"),
            "outbound_picks": ("outbound_picks", "sum"),
            "outbound_full_picks": ("outbound_full_picks", "sum"),
            "outbound_partial_picks": ("outbound_partial_picks", "sum"),
        },
    )
    return daily


def load_packing_daily(root: Path) -> pd.DataFrame:
    frames = []
    for source_name, df in _read_yearly_folder(root / "Kompletace"):
        date_col = _find_col(df, "datum expedice")
        if not date_col:
            continue
        order_col = _find_col(df, "objednavka")
        pallet_col = _find_col(df, "paleta cislo")
        gross_col = _find_col(df, "brutto palety kg")
        net_col = _find_col(df, "dilci netto kg")
        volume_col = _find_col(df, "objem palety dm3")

        work = pd.DataFrame(
            {
                "date_raw": df[date_col],
                "packing_order": df[order_col].astype(str) if order_col else "",
                "packing_pallet": df[pallet_col].astype(str) if pallet_col else "",
                "packing_gross_kg": _to_number(df[gross_col]) if gross_col else 0.0,
                "packing_net_kg": _to_number(df[net_col]) if net_col else 0.0,
                "packing_volume_dm3": _to_number(df[volume_col]) if volume_col else 0.0,
            }
        )
        work["packing_lines"] = 1
        work["source_file"] = source_name
        frames.append(work)

    combined = pd.concat(frames, ignore_index=True)
    daily = _daily_agg(
        combined,
        "date_raw",
        {
            "packing_lines": ("packing_lines", "sum"),
            "packing_orders": ("packing_order", "nunique"),
            "packing_pallets": ("packing_pallet", "nunique"),
            "packing_gross_kg": ("packing_gross_kg", "sum"),
            "packing_net_kg": ("packing_net_kg", "sum"),
            "packing_volume_dm3": ("packing_volume_dm3", "sum"),
        },
    )
    return daily


def _receipt_files(reporting_dir: Path) -> list[Path]:
    hist_dir = _short_dir(reporting_dir / "Příjmy s DK")
    current_week_dir = _short_dir(reporting_dir / "Příjmy s DK aktuální týden")
    files = sorted(hist_dir.glob("*.csv")) + sorted(hist_dir.glob("*.CSV")) + sorted(current_week_dir.glob("*.csv")) + sorted(current_week_dir.glob("*.CSV"))
    return files


def load_receipts_daily(reporting_dir: Path) -> pd.DataFrame:
    rows = []
    for path in _receipt_files(reporting_dir):
        df = _load_csv(path)
        qty_col = _find_col(df, "mnozstvi")
        gross_col = _find_col(df, "brutto kg")
        cbm_col = _find_col(df, "cbm")
        date_col = _find_col(df, "prijato")
        doc_col = _find_col(df, "doklad")
        avizo_col = _find_col(df, "avizo")
        artikl_col = _find_col(df, "artikl")
        paleta_col = _find_col(df, "paleta")
        lot_col = _find_col(df, "lot")
        if not date_col or not doc_col:
            continue
        work = pd.DataFrame(
            {
                "date_raw": df[date_col],
                "receipt_doc": df[doc_col].astype(str),
                "receipt_avizo": df[avizo_col].astype(str) if avizo_col else "",
                "receipt_artikl": df[artikl_col].astype(str) if artikl_col else "",
                "receipt_paleta": df[paleta_col].astype(str) if paleta_col else "",
                "receipt_lot": df[lot_col].astype(str) if lot_col else "",
                "inbound_qty": _to_number(df[qty_col]) if qty_col else 0.0,
                "inbound_gross_kg": _to_number(df[gross_col]) if gross_col else 0.0,
                "inbound_cbm": _to_number(df[cbm_col]) if cbm_col else 0.0,
            }
        )
        work["source_file"] = path.name
        rows.append(work)

    combined = pd.concat(rows, ignore_index=True)
    combined["date"] = _parse_day(combined["date_raw"])
    combined = combined[(combined["date"] >= MIN_VALID_DATE) & (combined["date"] <= MAX_VALID_DATE)]
    dedupe_cols = ["date", "receipt_doc", "receipt_artikl", "receipt_paleta", "receipt_lot", "inbound_qty", "inbound_gross_kg"]
    combined = combined.dropna(subset=["date"]).drop_duplicates(subset=dedupe_cols)
    daily = (
        combined.groupby("date")
        .agg(
            inbound_docs=("receipt_doc", "nunique"),
            inbound_avizo=("receipt_avizo", "nunique"),
            inbound_articles=("receipt_artikl", "nunique"),
            inbound_lines=("receipt_doc", "size"),
            inbound_qty=("inbound_qty", "sum"),
            inbound_gross_kg=("inbound_gross_kg", "sum"),
            inbound_cbm=("inbound_cbm", "sum"),
        )
        .sort_index()
    )
    return daily


def _inventory_group_map(reporting_dir: Path) -> dict[str, str]:
    path = _short_dir(reporting_dir / "Inventura skladu") / "ESAB - Inventura skladu.csv"
    df = _load_csv(path)
    artikl_col = _find_col(df, "artikl")
    group_col = _find_col(df, "nazev skupiny")
    if not artikl_col or not group_col:
        return {}
    mapping = (
        pd.DataFrame({"artikl": df[artikl_col].astype(str), "group_name": df[group_col].astype(str)})
        .drop_duplicates(subset=["artikl"])
    )
    return dict(zip(mapping["artikl"], mapping["group_name"]))


def load_receipts_group_daily(reporting_dir: Path) -> pd.DataFrame:
    group_map = _inventory_group_map(reporting_dir)
    rows = []
    for path in _receipt_files(reporting_dir):
        df = _load_csv(path)
        gross_col = _find_col(df, "brutto kg")
        date_col = _find_col(df, "prijato")
        artikl_col = _find_col(df, "artikl")
        if not date_col or not artikl_col or not gross_col:
            continue
        work = pd.DataFrame(
            {
                "date": _parse_day(df[date_col]),
                "artikl": df[artikl_col].astype(str),
                "gross_kg": _to_number(df[gross_col]),
            }
        ).dropna(subset=["date"])
        work = work[(work["date"] >= MIN_VALID_DATE) & (work["date"] <= MAX_VALID_DATE)]
        work["group_name"] = work["artikl"].map(group_map).fillna("Unknown")
        rows.append(work)

    combined = pd.concat(rows, ignore_index=True)
    daily = (
        combined.groupby(["date", "group_name"])["gross_kg"]
        .sum(min_count=1)
        .unstack(fill_value=0.0)
        .sort_index()
    )
    rename = {
        "Consumables": "inbound_consumables_gross_kg",
        "Equipment": "inbound_equipment_gross_kg",
        "Ostatní": "inbound_other_gross_kg",
        "Proclený": "inbound_customs_gross_kg",
        "Unknown": "inbound_unknown_gross_kg",
    }
    daily = daily.rename(columns={col: rename.get(col, f"inbound_group_{_norm(col).replace(' ', '_')}_gross_kg") for col in daily.columns})
    return daily


def load_inventory_snapshot(reporting_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = _short_dir(reporting_dir / "Inventura skladu") / "ESAB - Inventura skladu.csv"
    df = _load_csv(path)
    qty_col = _find_col(df, "mnozstvi")
    gross_col = _find_col(df, "brutto kg")
    days_col = _find_col(df, "dny lozeni")
    block_loc_col = _find_col(df, "blokovana lokace")
    block_pallet_col = _find_col(df, "blokovana paleta")
    group_col = _find_col(df, "nazev skupiny")
    artikl_col = _find_col(df, "artikl")
    pallet_col = _find_col(df, "paleta")
    location_col = _find_col(df, "popis lokace")

    work = pd.DataFrame(
        {
            "inventory_qty": _to_number(df[qty_col]) if qty_col else 0.0,
            "inventory_gross_kg": _to_number(df[gross_col]) if gross_col else 0.0,
            "inventory_days": _to_number(df[days_col]) if days_col else np.nan,
            "blocked_location": df[block_loc_col].astype(str).str.lower().eq("ano") if block_loc_col else False,
            "blocked_pallet": df[block_pallet_col].astype(str).str.lower().eq("ano") if block_pallet_col else False,
            "group_name": df[group_col].astype(str) if group_col else "Unknown",
            "artikl": df[artikl_col].astype(str) if artikl_col else "",
            "pallet_id": df[pallet_col].astype(str) if pallet_col else "",
            "location": df[location_col].astype(str) if location_col else "",
        }
    )
    summary = pd.DataFrame(
        [
            {
                "snapshot_date": DATE_NOW.normalize(),
                "inventory_lines": int(len(work)),
                "inventory_articles": int(work["artikl"].nunique()),
                "inventory_pallets": int(work["pallet_id"].nunique()),
                "inventory_locations": int(work["location"].nunique()),
                "inventory_qty": float(work["inventory_qty"].sum()),
                "inventory_gross_kg": float(work["inventory_gross_kg"].sum()),
                "inventory_blocked_lines": int((work["blocked_location"] | work["blocked_pallet"]).sum()),
                "inventory_blocked_qty": float(work.loc[work["blocked_location"] | work["blocked_pallet"], "inventory_qty"].sum()),
                "inventory_avg_days": float(work["inventory_days"].dropna().mean()) if work["inventory_days"].notna().any() else np.nan,
            }
        ]
    )
    by_group = (
        work.groupby("group_name")
        .agg(
            inventory_lines=("artikl", "size"),
            inventory_articles=("artikl", "nunique"),
            inventory_pallets=("pallet_id", "nunique"),
            inventory_qty=("inventory_qty", "sum"),
            inventory_gross_kg=("inventory_gross_kg", "sum"),
            inventory_avg_days=("inventory_days", "mean"),
        )
        .reset_index()
        .sort_values(["inventory_gross_kg", "inventory_qty"], ascending=False)
    )
    return summary, by_group


def load_invoice_weekly(reporting_dir: Path) -> pd.DataFrame:
    invoice_dir = _short_dir(reporting_dir / "Fakturace")
    frames = []
    for path in sorted(invoice_dir.glob("*.csv")) + sorted(invoice_dir.glob("*.CSV")):
        df = _load_csv(path)
        service_col = _find_col(df, "popis sluzby")
        date_col = _find_col(df, "datum sluzby")
        qty_col = _find_col(df, "mnozstvi")
        if not service_col or not date_col or not qty_col:
            continue
        work = pd.DataFrame(
            {
                "week_end": _parse_day(df[date_col]),
                "service": df[service_col].astype(str),
                "qty": _to_number(df[qty_col]),
            }
        ).dropna(subset=["week_end"])
        work = work[(work["week_end"] >= MIN_VALID_DATE) & (work["week_end"] <= MAX_VALID_DATE)]
        frames.append(work)
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(index="week_end", columns="service", values="qty", aggfunc="sum", fill_value=0.0).sort_index()
    rename = {col: f"invoice_{_norm(col).replace(' - ', '_').replace(' ', '_')}" for col in pivot.columns}
    return pivot.rename(columns=rename)


def load_manipulation_daily(reporting_dir: Path) -> pd.DataFrame:
    manip_dir = _short_dir(reporting_dir / "TimeManagement") / "FA_manipulace"
    frames = []
    for path in sorted(manip_dir.glob("ESAB*.csv")) + sorted(manip_dir.glob("ESAB*.CSV")):
        df = _load_csv(path)
        service_col = _find_col(df, "popis sluzby")
        date_col = _find_col(df, "datum sluzby")
        qty_col = _find_col(df, "mnozstvi")
        int_ref_col = _find_col(df, "int.ref.")
        ext_ref_col = _find_col(df, "ext.ref.")
        if not service_col or not date_col or not qty_col:
            continue
        work = pd.DataFrame(
            {
                "date": _parse_day(df[date_col]),
                "service": df[service_col].astype(str),
                "qty": _to_number(df[qty_col]),
                "int_ref": df[int_ref_col].astype(str) if int_ref_col else "",
                "ext_ref": df[ext_ref_col].astype(str) if ext_ref_col else "",
            }
        ).dropna(subset=["date"])
        work = work[(work["date"] >= MIN_VALID_DATE) & (work["date"] <= MAX_VALID_DATE)]
        work = work.drop_duplicates(subset=["date", "service", "qty", "int_ref", "ext_ref"])
        frames.append(work)
    combined = pd.concat(frames, ignore_index=True)
    combined["service_norm"] = combined["service"].map(_norm)
    combined["handling_in_qty"] = np.where(combined["service_norm"].str.startswith("handling in"), combined["qty"], 0.0)
    combined["handling_out_qty"] = np.where(combined["service_norm"].str.startswith("handling out"), combined["qty"], 0.0)
    combined["wrapping_qty"] = np.where(combined["service_norm"].str.contains("wrapping"), combined["qty"], 0.0)
    combined["putaway_qty"] = np.where(combined["service_norm"].str.contains("putaway"), combined["qty"], 0.0)
    daily = (
        combined.groupby("date")
        .agg(
            handling_in_qty=("handling_in_qty", "sum"),
            handling_out_qty=("handling_out_qty", "sum"),
            wrapping_qty=("wrapping_qty", "sum"),
            putaway_qty=("putaway_qty", "sum"),
        )
        .sort_index()
    )
    return daily


def build_daily_state(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    outbound = load_outbound_daily(root)
    packing = load_packing_daily(root)
    receipts = load_receipts_daily(REPORTING_DIR)
    receipts_group = load_receipts_group_daily(REPORTING_DIR)
    manip = load_manipulation_daily(REPORTING_DIR)

    hist_end = DATE_NOW.normalize()
    last_known = max(
        outbound.index.max() if not outbound.empty else hist_end,
        packing.index.max() if not packing.empty else hist_end,
        receipts.index.max() if not receipts.empty else hist_end,
        manip.index.max() if not manip.empty else hist_end,
    )
    idx = pd.date_range(min(x.index.min() for x in [outbound, packing, receipts, receipts_group, manip] if not x.empty), last_known, freq="D")
    daily = pd.DataFrame(index=idx)
    for frame in [outbound, packing, receipts, receipts_group, manip]:
        if frame.empty:
            continue
        covered_idx = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
        expanded = frame.reindex(covered_idx, fill_value=0.0)
        daily = daily.join(expanded, how="left")
    daily["net_flow_gross_kg"] = daily["inbound_gross_kg"].fillna(0.0) - daily["outbound_gross_kg"].fillna(0.0)
    daily["net_flow_cbm"] = daily["inbound_cbm"].fillna(0.0) - daily["outbound_cbm"].fillna(0.0)
    daily.index.name = "date"

    future_schedule = packing.loc[packing.index > hist_end].copy() if not packing.empty else pd.DataFrame()
    actual_daily = daily.loc[daily.index <= hist_end].copy()
    return actual_daily, future_schedule


def _future_index(series: pd.Series, mode: str, horizon: int) -> pd.DatetimeIndex:
    if mode == "daily":
        return pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.date_range(series.index.max() + pd.Timedelta(days=7), periods=horizon, freq="7D")


@dataclass
class TargetForecast:
    target: str
    mode: str
    selected_model: str
    forecast: pd.Series
    backtests: pd.DataFrame


def _forecast_target(
    series: pd.Series,
    mode: str,
    horizon: int,
    exog_full: pd.DataFrame | None = None,
    backtest_horizon: int | None = None,
) -> TargetForecast:
    s = series.dropna().astype(float)
    gap_days = max(0, (DATE_NOW.normalize() - pd.Timestamp(s.index.max()).normalize()).days)
    if mode == "daily":
        effective_horizon = horizon + gap_days
    else:
        effective_horizon = horizon + int(math.ceil(gap_days / 7))
    future_idx = _future_index(s, mode, effective_horizon)
    if exog_full is not None and not exog_full.empty:
        exog_full = exog_full.sort_index()
        exog_full = exog_full[~exog_full.index.duplicated(keep="last")]
        hist_exog, future_exog = align_known_exog(exog_full, future_idx, lag_periods=1)
        hist_exog = hist_exog.reindex(s.index).ffill().bfill()
        future_exog = future_exog[~future_exog.index.duplicated(keep="last")]
    else:
        hist_exog, future_exog = pd.DataFrame(), pd.DataFrame()

    baseline = baseline_as_result(s, effective_horizon, mode)
    internal = ridge_forecast(s, effective_horizon, mode)
    global_result = ridge_forecast(s, effective_horizon, mode, exog_hist=hist_exog, exog_future=future_exog) if not hist_exog.empty else None

    bt_h = backtest_horizon or (4 if mode == "weekly" else 10)
    metric_map = {
        "baseline": rolling_backtest(s, mode, bt_h, "baseline"),
        "internal": rolling_backtest(s, mode, bt_h, "internal"),
    }
    if not hist_exog.empty:
        metric_map["global"] = rolling_backtest(s, mode, bt_h, "global", exog_aligned=hist_exog)

    weights = compute_blend_weights(metric_map)
    blend_inputs = {"baseline": baseline, "internal": internal}
    if global_result is not None:
        blend_inputs["global"] = global_result
    blend = blend_forecasts(blend_inputs, weights)
    metric_map["blend"] = type(next(iter(metric_map.values())))("blend", np.nan, np.nan, np.nan, 0)

    score_rows = []
    actuals = {name: metrics for name, metrics in metric_map.items()}
    for name, metrics in actuals.items():
        score_rows.append(
            {
                "model": name,
                "wape": metrics.wape,
                "bias": metrics.bias,
                "mae": metrics.mae,
                "n_windows": metrics.n_windows,
                "weight": weights.get(name, 0.0),
            }
        )
    scores = pd.DataFrame(score_rows).sort_values(["wape", "mae"], na_position="last")

    if global_result is None:
        candidate_map = {"baseline": baseline, "internal": internal, "blend": blend}
    else:
        candidate_map = {"baseline": baseline, "internal": internal, "global": global_result, "blend": blend}

    best_model = scores.iloc[0]["model"] if not scores.empty else "blend"
    forecast = candidate_map[str(best_model)].forecast.copy()
    return TargetForecast(target=series.name or "target", mode=mode, selected_model=str(best_model), forecast=forecast, backtests=scores)


def _overlay_known_future(base_forecast: pd.Series, known_future: pd.Series) -> pd.Series:
    out = base_forecast.copy()
    if known_future is None or known_future.empty:
        return out
    overlay_idx = out.index.intersection(known_future.index)
    out.loc[overlay_idx] = known_future.loc[overlay_idx]
    return out


def build_weekly_features(actual_daily: pd.DataFrame, future_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow_cols = [
        "inbound_docs",
        "inbound_lines",
        "inbound_gross_kg",
        "outbound_docs",
        "outbound_orders",
        "outbound_gross_kg",
        "packing_pallets",
        "packing_gross_kg",
        "handling_in_qty",
        "handling_out_qty",
        "wrapping_qty",
    ]
    hist = actual_daily[flow_cols].resample("W-SUN").sum(min_count=1).fillna(0.0)
    all_daily = pd.concat([actual_daily[flow_cols], future_daily[flow_cols]], axis=0).sort_index()
    all_daily = all_daily[~all_daily.index.duplicated(keep="last")]
    fut = all_daily.resample("W-SUN").sum(min_count=1).fillna(0.0)
    return hist, fut


def future_points_from_daily(forecast_daily: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    rows = []
    valid_targets = [c for c in forecast_daily.columns if c != "date" and forecast_daily[c].notna().any()]
    for target in valid_targets:
        series = forecast_daily.set_index("date")[target]
        for horizon in horizons:
            idx = series.index.min() + pd.Timedelta(days=horizon - 1)
            if idx in series.index:
                rows.append({"target": target, "horizon_days": horizon, "date": idx, "forecast": float(series.loc[idx])})
    return pd.DataFrame(rows)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    inventory_summary, inventory_by_group = load_inventory_snapshot(REPORTING_DIR)
    actual_daily, future_schedule = build_daily_state(ROOT)
    actual_daily.to_csv(EXPORT_DIR / "warehouse_state_daily.csv", index=True, encoding="utf-8-sig")
    inventory_summary.to_csv(EXPORT_DIR / "inventory_snapshot_summary.csv", index=False, encoding="utf-8-sig")
    inventory_by_group.to_csv(EXPORT_DIR / "inventory_snapshot_by_group.csv", index=False, encoding="utf-8-sig")

    pack_series = actual_daily["packing_pallets"].rename("packing_pallets")
    pack_forecast_bundle = _forecast_target(pack_series, mode="daily", horizon=HORIZON_DAYS, backtest_horizon=10)
    pack_future = pack_forecast_bundle.forecast.rename("packing_pallets")
    known_pack = future_schedule["packing_pallets"].rename("packing_pallets") if "packing_pallets" in future_schedule.columns else pd.Series(dtype=float)
    pack_future = _overlay_known_future(pack_future, known_pack)

    pack_gross_series = actual_daily["packing_gross_kg"].rename("packing_gross_kg")
    pack_gross_bundle = _forecast_target(pack_gross_series, mode="daily", horizon=HORIZON_DAYS, backtest_horizon=10)
    pack_gross_future = pack_gross_bundle.forecast.rename("packing_gross_kg")
    known_pack_gross = future_schedule["packing_gross_kg"].rename("packing_gross_kg") if "packing_gross_kg" in future_schedule.columns else pd.Series(dtype=float)
    pack_gross_future = _overlay_known_future(pack_gross_future, known_pack_gross)

    outbound_exog = pd.concat([actual_daily[["packing_pallets", "packing_gross_kg"]], pd.DataFrame({"packing_pallets": pack_future, "packing_gross_kg": pack_gross_future})], axis=0)
    outbound_bundle = _forecast_target(actual_daily["outbound_gross_kg"].rename("outbound_gross_kg"), mode="daily", horizon=HORIZON_DAYS, exog_full=outbound_exog, backtest_horizon=10)
    inbound_bundle = _forecast_target(actual_daily["inbound_gross_kg"].rename("inbound_gross_kg"), mode="daily", horizon=HORIZON_DAYS, backtest_horizon=10)
    handling_in_bundle = _forecast_target(actual_daily["handling_in_qty"].rename("handling_in_qty"), mode="daily", horizon=HORIZON_DAYS, backtest_horizon=10)
    handling_out_bundle = _forecast_target(actual_daily["handling_out_qty"].rename("handling_out_qty"), mode="daily", horizon=HORIZON_DAYS, backtest_horizon=10)

    future_daily = pd.DataFrame(
        {
            "date": outbound_bundle.forecast.index,
            "packing_pallets": pack_future.reindex(outbound_bundle.forecast.index, fill_value=0.0).values,
            "packing_gross_kg": pack_gross_future.reindex(outbound_bundle.forecast.index, fill_value=0.0).values,
            "outbound_gross_kg": outbound_bundle.forecast.values,
            "inbound_gross_kg": inbound_bundle.forecast.reindex(outbound_bundle.forecast.index, fill_value=0.0).values,
            "handling_in_qty": handling_in_bundle.forecast.reindex(outbound_bundle.forecast.index, fill_value=0.0).values,
            "handling_out_qty": handling_out_bundle.forecast.reindex(outbound_bundle.forecast.index, fill_value=0.0).values,
        }
    )
    future_daily["inbound_docs"] = np.nan
    future_daily["inbound_lines"] = np.nan
    future_daily["outbound_docs"] = np.nan
    future_daily["outbound_orders"] = np.nan
    future_daily["wrapping_qty"] = np.nan
    future_daily["net_flow_gross_kg"] = future_daily["inbound_gross_kg"] - future_daily["outbound_gross_kg"]
    current_gross_kg = float(inventory_summary.loc[0, "inventory_gross_kg"])
    future_daily["gross_stock_proxy_kg"] = current_gross_kg + future_daily["net_flow_gross_kg"].cumsum()
    future_daily.to_csv(EXPORT_DIR / "warehouse_flow_forecast_daily.csv", index=False, encoding="utf-8-sig")

    invoice_weekly = load_invoice_weekly(REPORTING_DIR)
    weekly_hist_exog, weekly_future_exog = build_weekly_features(actual_daily, future_daily.set_index("date"))
    storage_target_col = "invoice_storage_eur_pallet"
    storage_shelf_col = "invoice_storage_shelf"
    weekly_exports = []
    backtest_exports = []

    for target_col in [storage_target_col, storage_shelf_col]:
        if target_col not in invoice_weekly.columns:
            continue
        target_series = invoice_weekly[target_col].astype(float)
        exog_full = pd.concat([weekly_hist_exog, weekly_future_exog.loc[weekly_future_exog.index > target_series.index.max()]], axis=0)
        bundle = _forecast_target(target_series.rename(target_col), mode="weekly", horizon=HORIZON_WEEKS, exog_full=exog_full, backtest_horizon=4)
        weekly_frame = pd.DataFrame({"week_end": bundle.forecast.index, "target": target_col, "forecast": bundle.forecast.values, "selected_model": bundle.selected_model})
        weekly_exports.append(weekly_frame)
        scores = bundle.backtests.copy()
        scores["target"] = target_col
        scores["mode"] = "weekly"
        backtest_exports.append(scores)

    for bundle in [pack_forecast_bundle, pack_gross_bundle, outbound_bundle, inbound_bundle, handling_in_bundle, handling_out_bundle]:
        scores = bundle.backtests.copy()
        scores["target"] = bundle.target
        scores["mode"] = bundle.mode
        backtest_exports.append(scores)

    if weekly_exports:
        pd.concat(weekly_exports, ignore_index=True).to_csv(EXPORT_DIR / "warehouse_occupancy_forecast_weekly.csv", index=False, encoding="utf-8-sig")
    pd.concat(backtest_exports, ignore_index=True).to_csv(EXPORT_DIR / "warehouse_forecast_backtests.csv", index=False, encoding="utf-8-sig")

    future_points_from_daily(future_daily, [5, 10, 15, 20, 30, 60, 90]).to_csv(
        EXPORT_DIR / "warehouse_flow_forecast_horizon_points.csv", index=False, encoding="utf-8-sig"
    )

    summary_lines = [
        "# Warehouse State Forecast",
        "",
        f"- Run date: {DATE_NOW.date()}",
        f"- Current inventory gross kg: {current_gross_kg:,.1f}",
        f"- Current inventory pallets: {int(inventory_summary.loc[0, 'inventory_pallets'])}",
        f"- Current inventory blocked qty: {float(inventory_summary.loc[0, 'inventory_blocked_qty']):,.1f}",
        f"- Historical daily flow rows: {len(actual_daily):,}",
        f"- Weekly occupancy observations: {len(invoice_weekly):,}",
        "",
        "## Approach",
        "",
        "- Inbound is built from historical receipt files and current-week receipt export.",
        "- Outbound is built from yearly issue files.",
        "- Packing schedule is used both as history and as known near-term future signal.",
        "- Occupancy proxy is based on weekly invoiced storage services.",
        "- Future gross stock proxy is current inventory gross kg plus forecast inbound minus forecast outbound.",
        "",
        "## Limits",
        "",
        "- Exact future stock by SKU is not backtestable from one current inventory snapshot alone.",
        "- Weekly storage invoice is the best historical proxy for warehouse occupancy in this data package.",
    ]
    (EXPORT_DIR / "WAREHOUSE_STATE_FORECAST.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
