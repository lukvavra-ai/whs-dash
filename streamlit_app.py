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

MODEL_DESCRIPTIONS = {
    "01 Seasonal baseline": "Jednoduchy sezonni benchmark podle podobnych dni v historii.",
    "02 Same weekday median": "Median poslednich srovnatelnych dnu stejneho weekdaye. Casto velmi stabilni pro provozni KPI.",
    "03 Rolling median 4 weeks": "Kratkodoby robustni trend z poslednich 4 tydnu bez slozitych driveru.",
    "04 Static YTD": "Historicka sezonnost upravena o tempo aktualniho roku.",
    "05 Dynamic rolling": "Historicka sezonnost upravena jen kratkodobym trendem poslednich tydnu.",
    "06 Calendar index": "Kalendarny model podle pozice v roce a typicke sily dne.",
    "07 Hybrid calendar x trend": "Kombinace kalendarniho indexu a kratkodobeho trendu. Dobry default pro operativu.",
    "08 Ridge internal": "Strojovy model jen z internich skladu a provoznich dat.",
    "09 Ridge with drivers": "Strojovy model s vybranymi drivere z prijmu, skladu, navazanych toku a world-state proxy.",
    "10 Smart blend": "Automaticky vybere nejlepsi kombinaci top modelu. Kdyz blend nepomaha, drzi se nejlepsiho modelu.",
}

METRIC_EXPLANATIONS = {
    "binhits": "Binhits jsou nejlepší rychlý obraz denní zátěže balení. Čím výš jsou, tím víc práce proteklo skladem po binových operacích.",
    "gross_tons": "GW ukazuje hmotnostní tlak. Je užitečné ho číst spolu s binhits, protože stejné binhits mohou mít jinou fyzickou náročnost.",
    "cartons_count": "Kartony ukazují jemnější balicí mix. Když rostou rychleji než palety, práce bývá detailnější a méně lineární.",
    "pallets_count": "Palety jsou hrubý obraz objemu balení. Dobře se hodí pro kapacitní pohled a porovnání s GW.",
    "vydejky_unique": "Výdejky ukazují počet unikátních dokladů. Často dobře vysvětlují provozní fragmentaci práce.",
    "trips_total": "Trips celkem jsou nejlepší základní KPI nakládek. Popisují, kolik reálných odjezdů sklad odbavil.",
    "trips_export": "Trips export zachycují exportní tlak. Jsou důležité hlavně pro vazbu na kontejnery a citlivost na externí svět.",
    "trips_europe": "Trips Evropa ukazují běžný evropský tok. Často mají jiný rytmus než export a jinou sezonnost.",
    "containers_count": "Kontejnery jsou nejcitlivější KPI na exportní mix, příjmy consumables a veřejné shipping indikátory.",
    "orders_nunique": "Objednávky jsou silný interní driver. Dobře fungují jako předstihový signál pro trips i část balení.",
}

STAFFING_METRIC_EXPLANATIONS = {
    "Binhits celkem": "Celkový forecast binhitů je základ staffing převodu. Z něj se dopočítávají hodiny i lidé.",
    "Binhits den": "Denní binhits jsou část práce, která připadá na denní bucket. Pomáhají rozlišit agenturní denní potřebu.",
    "Binhits noc": "Noční binhits ukazují tlak na noční část provozu. Jsou důležité hlavně pro agenturní noc.",
    "Potřebný headcount": "Celkový potřebný headcount je orientační počet lidí, který by měl pokrýt forecastovanou zátěž při aktuální produktivitě.",
    "Kmen ráno": "Kmen ráno je pevný ranní bucket. Model ho nepřehazuje volně, ale drží se reality posledních srovnatelných dnů.",
    "Kmen odpo": "Kmen odpo je pevný odpolední bucket. Slouží jako stabilní kostra směny, ne jako libovolná flexibilita.",
    "Agentura den": "Agentura den dopočítává zbylou denní potřebu po odečtení realistické kapacity kmene.",
    "Agentura noc": "Agentura noc dopočítává zbylou noční potřebu po odečtení toho, co reálně pokryje kmen a běžný noční mix.",
    "Denní směna": "Denní směna shrnuje všechny pracovníky potřebné pro denní část provozu bez ohledu na formu zaměstnání.",
    "Noční směna": "Noční směna shrnuje všechny pracovníky potřebné pro noční část provozu bez ohledu na formu zaměstnání.",
    "Placené hodiny": "Placené hodiny jsou mezikrok mezi forecastem práce a počtem lidí. Převádí výkon na potřebný čas.",
    "Produktivní workers": "Produktivní workers jsou užší odhad lidí, kteří skutečně pokrývají produktivní část práce.",
}

PUBLIC_DRIVER_EXPLANATIONS = {
    "Brent / diesel / TTF gas": "Energetické a transportní náklady. Neřídí sklad samy, ale pomáhají rozpoznat režim tlaku a operability.",
    "RWI / GSCPI / container stress": "Shipping a supply-chain vrstva. Nejdůležitější hlavně pro kontejnery a exportní chování.",
    "Eurostat industry / PPI": "Průmyslová poptávka a vstupní ceny. Fungují spíš jako střednědobý režimový signál než denní driver.",
    "ESAB / copper": "Firemní a průmyslová proxy vrstva. Sama o sobě forecast nespasí, ale může pomoct u změny trendového režimu.",
}

HOW_TO_READ_METRICS = {
    "WAPE": "WAPE je průměrná relativní chyba. Čím níž, tím líp. Prakticky ukazuje, jak daleko forecast bývá od reality v poměru k objemu.",
    "Bias": "Bias říká, jestli model systematicky přestřeluje nebo podstřeluje. Kladný bias znamená spíš nadhodnocení, záporný podhodnocení.",
    "MAE": "MAE je průměrná absolutní chyba v jednotkách KPI. Dobře se čte provozně, protože ukazuje typický rozdíl v kusech, trips nebo lidech.",
}

DRIVER_FAMILY_LABELS = {
    "self": "Stejny proces",
    "linked": "Navazany proces",
    "wh": "Sklad a prijmy",
    "ws": "Globalni signal",
}

TARGET_DRIVER_PROFILES = {
    "binhits": {
        "max_features": 9,
        "core": {
            "self": {"gross_tons", "cartons_count", "pallets_count", "vydejky_unique"},
            "linked": {"orders_nunique", "trips_total", "trips_europe", "trips_export", "containers_count", "gross_tons"},
            "wh": {
                "inbound_gross_kg",
                "inbound_lines",
                "inbound_docs",
                "inbound_consumables_gross_kg",
                "inbound_unknown_gross_kg",
                "outbound_orders",
                "outbound_docs",
                "outbound_gross_kg",
                "net_flow_gross_kg",
                "wrapping_qty",
                "putaway_qty",
            },
            "ws": {
                "industry_demand_index",
                "europe_demand_index",
                "de_prod_c25_13w_pct",
                "de_prod_c28_13w_pct",
                "de_prod_c29_13w_pct",
                "eu_gas_13w_pct",
                "cz_diesel_price_4w_pct",
                "welding_input_price_index",
                "energy_stress_index",
            },
        },
        "lags": {"self": {1, 2, 5}, "linked": {1, 2, 5, 10}, "wh": {1, 2, 5, 10}, "ws": {5, 10, 20}},
        "family_cap": {"self": 3, "linked": 2, "wh": 3, "ws": 1},
    },
    "trips_total": {
        "max_features": 9,
        "core": {
            "self": {"orders_nunique", "gross_tons", "trips_export", "trips_europe", "containers_count"},
            "linked": {"binhits", "cartons_count", "pallets_count", "vydejky_unique", "gross_tons"},
            "wh": {
                "outbound_docs",
                "outbound_orders",
                "outbound_gross_kg",
                "outbound_cbm",
                "outbound_load_units",
                "inbound_consumables_gross_kg",
                "inbound_gross_kg",
                "inbound_lines",
                "net_flow_gross_kg",
            },
            "ws": {
                "industry_demand_index",
                "europe_demand_index",
                "de_prod_c25",
                "de_prod_c25_13w_pct",
                "de_prod_c24_c25",
                "copper",
                "copper_13w_pct",
                "export_risk_index",
            },
        },
        "lags": {"self": {1, 2, 5}, "linked": {1, 2, 5}, "wh": {1, 2, 5, 10}, "ws": {5, 10, 20}},
        "family_cap": {"self": 3, "linked": 2, "wh": 3, "ws": 1},
    },
    "containers_count": {
        "max_features": 10,
        "core": {
            "self": {"trips_export", "trips_europe", "trips_total", "orders_nunique", "gross_tons"},
            "linked": {"binhits", "cartons_count", "pallets_count", "vydejky_unique", "gross_tons"},
            "wh": {
                "inbound_consumables_gross_kg",
                "inbound_docs",
                "inbound_lines",
                "outbound_gross_kg",
                "outbound_cbm",
                "outbound_orders",
                "outbound_docs",
                "net_flow_gross_kg",
                "inbound_equipment_gross_kg",
            },
            "ws": {
                "export_risk_index",
                "container_market_index",
                "container_stress_index",
                "rwi_north_trend",
                "rwi_total_trend",
                "gscpi_pub_safe_4w_delta",
                "de_prod_c25_13w_pct",
                "industry_demand_index",
                "copper_13w_pct",
                "cz_diesel_price_4w_pct",
            },
        },
        "lags": {"self": {1, 2, 5}, "linked": {1, 2, 5, 10}, "wh": {1, 2, 5, 10, 20}, "ws": {5, 10, 20}},
        "family_cap": {"self": 2, "linked": 2, "wh": 3, "ws": 3},
    },
}

DRIVER_LABELS = {
    "binhits": "Binhits",
    "gross_tons": "GW (t)",
    "cartons_count": "Kartony",
    "pallets_count": "Palety",
    "vydejky_unique": "Vydejky",
    "trips_total": "Trips celkem",
    "trips_export": "Trips export",
    "trips_europe": "Trips Evropa",
    "containers_count": "Kontejnery",
    "orders_nunique": "Objednavky",
    "packing_gross_kg": "Baleni kg",
    "packing_orders": "Baleni objednavky",
    "packing_lines": "Baleni radky",
    "packing_pallets": "Baleni palety",
    "outbound_gross_kg": "Vydeje kg",
    "outbound_docs": "Vydeje doklady",
    "outbound_lines": "Vydeje radky",
    "outbound_qty": "Vydeje kusy",
    "net_flow_kg": "Net flow kg",
    "inventory_total_gross_kg": "Stav skladu kg",
    "inbound_gross_kg": "Prijmy kg",
    "inbound_docs": "Prijmy doklady",
    "inbound_lines": "Prijmy radky",
    "inbound_consumables_gross_kg": "Prijmy consumables kg",
    "inbound_equipment_gross_kg": "Prijmy equipment kg",
    "inbound_other_gross_kg": "Prijmy other kg",
    "inbound_customs_gross_kg": "Prijmy customs kg",
    "inbound_unknown_gross_kg": "Prijmy unknown kg",
    "esab_close": "Akcie ESAB",
    "esab_close_4w_pct": "ESAB 4w zmena",
    "brent": "Brent",
    "brent_4w_pct": "Brent 4w zmena",
    "brent_13w_pct": "Brent 13w zmena",
    "vix": "VIX",
    "vix_4w_pct": "VIX 4w zmena",
    "vix_13w_pct": "VIX 13w zmena",
    "copper": "Copper",
    "copper_4w_pct": "Copper 4w zmena",
    "copper_13w_pct": "Copper 13w zmena",
    "eu_gas": "TTF plyn",
    "eu_gas_4w_pct": "TTF plyn 4w zmena",
    "eu_gas_13w_pct": "TTF plyn 13w zmena",
    "gscpi": "GSCPI",
    "gscpi_4w_delta": "GSCPI 4w delta",
    "gscpi_pub_safe": "GSCPI publ-safe",
    "gscpi_pub_safe_4w_delta": "GSCPI publ-safe 4w delta",
    "rwi_total_trend": "RWI throughput total",
    "rwi_total_trend_13w_pct": "RWI total 13w zmena",
    "rwi_north_trend": "RWI North Range",
    "rwi_north_trend_13w_pct": "RWI North Range 13w zmena",
    "eu_diesel_price": "EU diesel",
    "eu_diesel_price_4w_pct": "EU diesel 4w zmena",
    "eur_diesel_price": "Euro area diesel",
    "cz_diesel_price": "CZ diesel",
    "cz_diesel_price_4w_pct": "CZ diesel 4w zmena",
    "de_diesel_price": "DE diesel",
    "de_prod_c24_c25": "DE vyroba metals",
    "de_prod_c25": "DE vyroba fabricated metals",
    "de_prod_c28": "DE vyroba machinery",
    "de_prod_c29": "DE vyroba automotive",
    "ea_prod_mig_ing": "EA vyroba intermediate goods",
    "ea_prod_mig_cag": "EA vyroba capital goods",
    "ea_prod_c25": "EA vyroba fabricated metals",
    "de_ppi_c24_c25": "DE PPI metals",
    "ea_ppi_mig_ing": "EA PPI intermediate goods",
    "shipping_risk_news_7d_pct": "Shipping risk news 7d zmena",
    "energy_conflict_news_7d_pct": "Energy conflict news 7d zmena",
    "europe_industry_news_7d_pct": "Europe industry news 7d zmena",
    "industry_demand_index": "Industry demand index",
    "welding_input_price_index": "Welding input price index",
    "energy_stress_index": "Energy stress",
    "container_market_index": "Container market",
    "europe_demand_index": "Europe demand",
    "container_stress_index": "Container stress",
    "geo_event_index": "Geo event risk",
    "export_risk_index": "Export risk",
    "local_operability_index": "Local operability",
    "overall_world_risk_index": "Overall world risk",
}

FAST_BACKTEST_POINTS = 14
DRIVER_TUNING_POINTS = 6

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


@dataclass
class ExogBundle:
    warehouse_daily: Optional[pd.DataFrame]
    world_state_daily: Optional[pd.DataFrame]


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


def _load_optional_daily_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        return _coerce_date(frame)
    return None


def _load_world_state_daily(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    date_col = "week_start" if "week_start" in frame.columns else ("index" if "index" in frame.columns else frame.columns[0])
    frame = frame.rename(columns={date_col: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    value_cols = [col for col in frame.columns if col != "date" and pd.api.types.is_numeric_dtype(frame[col])]
    if not value_cols:
        return None
    daily_idx = pd.date_range(frame["date"].min(), frame["date"].max() + pd.Timedelta(days=6), freq="D")
    out = frame.set_index("date")[value_cols].reindex(daily_idx).ffill().bfill().reset_index().rename(columns={"index": "date"})
    return _coerce_date(out)


@st.cache_data(show_spinner=False)
def load_exog_bundle(base_dir_str: str) -> ExogBundle:
    base_dir = Path(base_dir_str)
    warehouse_daily = _load_optional_daily_csv(base_dir / "warehouse_state_exports" / "warehouse_state_daily.csv")
    world_state_daily = _load_world_state_daily(base_dir / "world_state_feature_weekly.csv")
    return ExogBundle(warehouse_daily=warehouse_daily, world_state_daily=world_state_daily)


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


def _lagged_feature_block(frame: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    base = frame.astype(float).sort_index()
    blocks = {
        f"{col}_lag{int(lag)}": base[col].shift(int(lag))
        for col in base.columns
        for lag in lags
    }
    return pd.DataFrame(blocks, index=base.index)


def _driver_feature_meta(feature: str) -> tuple[str, str, Optional[int]]:
    prefix, base, _ = _driver_feature_parts(feature)
    lag_num: Optional[int] = None
    text = str(feature)
    if "_lag" in text:
        _, lag_part = text.rsplit("_lag", 1)
        if lag_part.isdigit():
            lag_num = int(lag_part)
    return prefix, base, lag_num


def _target_driver_profile(metric: str) -> dict:
    return TARGET_DRIVER_PROFILES.get(metric, {})


def _feature_allowed_for_target(feature: str, metric: str) -> bool:
    profile = _target_driver_profile(metric)
    if not profile:
        return True
    prefix, base, lag_num = _driver_feature_meta(feature)
    allowed_core = profile.get("core", {}).get(prefix)
    if allowed_core and base not in allowed_core:
        return False
    allowed_lags = profile.get("lags", {}).get(prefix)
    if allowed_lags and lag_num is not None and lag_num not in allowed_lags:
        return False
    return True


def _feature_target_bonus(feature: str, metric: str) -> float:
    profile = _target_driver_profile(metric)
    if not profile:
        return 0.0
    prefix, base, lag_num = _driver_feature_meta(feature)
    bonus = 0.0
    if base in profile.get("core", {}).get(prefix, set()):
        bonus += 0.05
    if lag_num in profile.get("lags", {}).get(prefix, set()):
        bonus += 0.03
    if prefix == "ws" and lag_num is not None and lag_num >= 10:
        bonus += 0.01
    return bonus


def _select_driver_features(target_metric: str, target_series: pd.Series, exog: pd.DataFrame, max_features: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    if exog is None or exog.empty:
        return pd.DataFrame(index=target_series.index), pd.DataFrame(columns=["feature", "score", "corr_full", "corr_recent", "n_obs"])

    target = target_series.astype(float).sort_index()
    aligned = exog.reindex(target.index).replace([np.inf, -np.inf], np.nan)
    stats_rows: List[Dict[str, float]] = []
    recent_idx = target.index[-min(180, len(target)) :]
    profile = _target_driver_profile(target_metric)
    max_features = int(profile.get("max_features", max_features))

    for col in aligned.columns:
        if not _feature_allowed_for_target(col, target_metric):
            continue
        x = pd.to_numeric(aligned[col], errors="coerce")
        mask = x.notna() & target.notna()
        if int(mask.sum()) < 90:
            continue
        if x[mask].nunique(dropna=True) < 2 or target[mask].nunique(dropna=True) < 2:
            continue
        corr_full = float(x[mask].corr(target[mask]))
        recent_x = x.reindex(recent_idx)
        recent_y = target.reindex(recent_idx)
        recent_mask = recent_x.notna() & recent_y.notna()
        if int(recent_mask.sum()) >= 40 and recent_x[recent_mask].nunique(dropna=True) >= 2 and recent_y[recent_mask].nunique(dropna=True) >= 2:
            corr_recent = float(recent_x[recent_mask].corr(recent_y[recent_mask]))
        else:
            corr_recent = corr_full
        score = 0.45 * abs(corr_full) + 0.55 * abs(corr_recent) + _feature_target_bonus(col, target_metric)
        if pd.isna(score) or score < 0.05:
            continue
        stats_rows.append(
            {
                "feature": col,
                "score": float(score),
                "corr_full": float(corr_full),
                "corr_recent": float(corr_recent),
                "n_obs": int(mask.sum()),
            }
        )

    if not stats_rows:
        return pd.DataFrame(index=target.index), pd.DataFrame(columns=["feature", "score", "corr_full", "corr_recent", "n_obs"])

    stats = pd.DataFrame(stats_rows).sort_values(["score", "n_obs"], ascending=[False, False]).reset_index(drop=True)
    selected_cols: List[str] = []
    family_counts: Dict[str, int] = {}
    for row in stats.itertuples():
        prefix, _, _ = _driver_feature_meta(str(row.feature))
        family_cap = profile.get("family_cap", {}).get(prefix)
        if family_cap is not None and family_counts.get(prefix, 0) >= int(family_cap):
            continue
        candidate = aligned[row.feature].ffill().bfill()
        too_close = False
        for picked in selected_cols:
            picked_series = aligned[picked].ffill().bfill()
            corr = candidate.corr(picked_series)
            if pd.notna(corr) and abs(float(corr)) > 0.96:
                too_close = True
                break
        if too_close:
            continue
        selected_cols.append(str(row.feature))
        family_counts[prefix] = family_counts.get(prefix, 0) + 1
        if len(selected_cols) >= max_features:
            break

    if not selected_cols:
        return pd.DataFrame(index=target.index), pd.DataFrame(columns=["feature", "score", "corr_full", "corr_recent", "n_obs"])

    selected = aligned[selected_cols].ffill().bfill().fillna(0.0)
    selected_stats = stats[stats["feature"].isin(selected_cols)].copy()
    selected_stats["selected_rank"] = range(1, len(selected_stats) + 1)
    selected_stats = selected_stats[["selected_rank", "feature", "score", "corr_full", "corr_recent", "n_obs"]]
    return selected, selected_stats


def _build_exog_features(
    source_df: pd.DataFrame,
    linked_df: Optional[pd.DataFrame],
    warehouse_df: Optional[pd.DataFrame],
    world_state_daily: Optional[pd.DataFrame],
    target_metric: str,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    self_metrics = [col for col in available_metrics(source_df) if col != target_metric]
    self_frame = _daily_metric_frame(source_df, self_metrics)
    if not self_frame.empty:
        parts.append(_lagged_feature_block(self_frame.add_prefix("self_"), [1, 2, 5]))

    if linked_df is not None and not linked_df.empty:
        linked_metrics = available_metrics(linked_df)
        linked_frame = _daily_metric_frame(linked_df, linked_metrics)
        if not linked_frame.empty:
            parts.append(_lagged_feature_block(linked_frame.add_prefix("linked_"), [1, 2, 5, 10]))

    if warehouse_df is not None and not warehouse_df.empty:
        warehouse_exclude = {
            target_metric,
            "outbound_qty",
            "outbound_load_units",
            "outbound_picks",
            "outbound_full_picks",
            "outbound_partial_picks",
            "packing_lines",
            "packing_orders",
            "packing_volume_dm3",
            "packing_net_kg",
            "inbound_articles",
            "inbound_avizo",
            "wrapping_qty",
            "putaway_qty",
        }
        warehouse_metrics = [col for col in available_metrics(warehouse_df) if col not in warehouse_exclude]
        warehouse_frame = _daily_metric_frame(warehouse_df, warehouse_metrics)
        if not warehouse_frame.empty:
            parts.append(_lagged_feature_block(warehouse_frame.add_prefix("wh_"), [1, 2, 5, 10, 20]))

    if world_state_daily is not None and not world_state_daily.empty:
        world_cols = available_metrics(world_state_daily)
        world_frame = _daily_metric_frame(world_state_daily, world_cols)
        if not world_frame.empty:
            parts.append(_lagged_feature_block(world_frame.add_prefix("ws_"), [1, 5, 10, 20]))

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
    driver_alpha: float = 12.0,
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
        forecast = ridge_forecast(train_series, 1, "daily", exog_hist=hist, exog_future=fut, alpha=float(driver_alpha)).forecast
        return max(0.0, float(forecast.iloc[0]))
    raise ValueError(f"Unknown model: {model}")


def _inverse_wape_weights(score_df: pd.DataFrame, models: Sequence[str]) -> Dict[str, float]:
    subset = score_df[score_df["model"].isin(list(models))].copy()
    metric_map = {}
    for row in subset.itertuples():
        metric_map[row.model] = type("Metrics", (), {"wape": row.wape})()
    return compute_blend_weights(metric_map)


def _smart_blend_weights(score_df: pd.DataFrame, backtest: pd.DataFrame) -> Dict[str, float]:
    if score_df is None or score_df.empty or backtest is None or backtest.empty:
        return {}

    ranked = (
        score_df[score_df["model"] != "10 Smart blend"]
        .dropna(subset=["wape"])
        .sort_values(["wape", "mae"], na_position="last")
        .reset_index(drop=True)
    )
    ranked = ranked[ranked["model"].isin(backtest.columns)].reset_index(drop=True)
    if ranked.empty:
        return {}

    actual = backtest.set_index("date")["actual"]
    best_model = str(ranked.loc[0, "model"])
    best_wape = float(ranked.loc[0, "wape"])
    best_choice = {"weights": {best_model: 1.0}, "wape": best_wape}

    bt_idx = backtest.set_index("date")
    max_top_n = min(4, len(ranked))
    for top_n in range(2, max_top_n + 1):
        models = ranked["model"].tolist()[:top_n]
        weights = _inverse_wape_weights(ranked, models)
        if not weights:
            continue
        pred = pd.Series(0.0, index=actual.index)
        for model, weight in weights.items():
            pred = pred + bt_idx[model].astype(float) * float(weight)
        trial_wape, _, _, _ = _metric_summary(actual, pred)
        if pd.notna(trial_wape) and float(trial_wape) + 1e-12 < float(best_choice["wape"]):
            best_choice = {"weights": weights, "wape": float(trial_wape)}

    return {str(model): float(weight) for model, weight in best_choice["weights"].items()}


def _driver_feature_limit_grid(metric: str, default_max_features: int = 12) -> List[int]:
    profile = _target_driver_profile(metric)
    top = int(profile.get("max_features", default_max_features))
    candidates = sorted({min(5, top), top})
    return [value for value in candidates if value > 0]


def _driver_alpha_grid(metric: str) -> List[float]:
    if metric == "containers_count":
        return [24.0, 48.0, 96.0]
    if metric == "trips_total":
        return [16.0, 32.0, 64.0]
    return [8.0, 16.0, 32.0]


def _tune_driver_setup(
    metric: str,
    series: pd.Series,
    exog_candidates: pd.DataFrame,
    regular_raw: pd.DataFrame,
    eval_dates: Sequence[pd.Timestamp],
    lookback_weeks: int,
    include_weekend: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    empty_info = pd.DataFrame(columns=["selected_rank", "feature", "score", "corr_full", "corr_recent", "n_obs"])
    if exog_candidates is None or exog_candidates.empty:
        return pd.DataFrame(index=series.index), empty_info, 12.0

    tuning_dates = list(eval_dates)[-min(DRIVER_TUNING_POINTS, len(eval_dates)) :] if eval_dates else []
    actual = pd.Series({pd.Timestamp(day): float(series.loc[day]) for day in tuning_dates}, dtype=float)
    best = None
    for feature_limit in _driver_feature_limit_grid(metric):
        exog_full, driver_info = _select_driver_features(metric, series, exog_candidates, max_features=feature_limit)
        if exog_full.empty:
            continue
        exog_shifted = exog_full.copy()
        for alpha in _driver_alpha_grid(metric):
            preds = {}
            for target_date in tuning_dates:
                preds[pd.Timestamp(target_date)] = _predict_model_one_step(
                    model="09 Ridge with drivers",
                    raw_df=regular_raw,
                    series=series,
                    exog_shifted=exog_shifted,
                    metric=metric,
                    target_date=pd.Timestamp(target_date),
                    lookback_weeks=lookback_weeks,
                    include_weekend=include_weekend,
                    driver_alpha=float(alpha),
                )
            pred = pd.Series(preds, dtype=float)
            wape, _, mae, _ = _metric_summary(actual, pred.reindex(actual.index))
            if pd.isna(wape):
                continue
            candidate = {
                "wape": float(wape),
                "mae": float(mae) if pd.notna(mae) else np.inf,
                "alpha": float(alpha),
                "exog_full": exog_full,
                "driver_info": driver_info,
            }
            if best is None or candidate["wape"] < best["wape"] - 1e-9 or (
                abs(candidate["wape"] - best["wape"]) <= 1e-9 and candidate["mae"] < best["mae"]
            ):
                best = candidate

    if best is None:
        exog_full, driver_info = _select_driver_features(metric, series, exog_candidates, max_features=12)
        return exog_full, driver_info if driver_info is not None else empty_info, 12.0
    return best["exog_full"], best["driver_info"], float(best["alpha"])


@st.cache_data(show_spinner=False, max_entries=64)
def compute_model_suite(
    raw_df: pd.DataFrame,
    linked_df: Optional[pd.DataFrame],
    warehouse_df: Optional[pd.DataFrame],
    world_state_daily: Optional[pd.DataFrame],
    metric: str,
    horizon_days: int,
    lookback_weeks: int,
    include_weekend: bool,
) -> Dict[str, object]:
    series = _daily_series(raw_df, metric)
    if series.empty:
        return {"future": pd.DataFrame(), "backtest": pd.DataFrame(), "scores": pd.DataFrame(), "weights": pd.DataFrame(), "drivers": pd.DataFrame()}

    horizon_days = max(int(horizon_days), 1)
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    min_train = min(max(90, len(series) // 2), max(len(series) - 21, 30))
    eval_dates = list(series.index[min_train:]) if len(series) > min_train else []
    if len(eval_dates) > FAST_BACKTEST_POINTS:
        eval_dates = eval_dates[-FAST_BACKTEST_POINTS:]

    regular_raw = _regularize_daily(raw_df)
    exog_candidates = _build_exog_features(raw_df, linked_df, warehouse_df, world_state_daily, metric, series.index)
    exog_full, driver_info, driver_alpha = _tune_driver_setup(metric, series, exog_candidates, regular_raw, eval_dates, lookback_weeks, include_weekend)
    exog_shifted = exog_full.copy() if not exog_full.empty else pd.DataFrame(index=series.index)

    future = pd.DataFrame(index=future_idx)
    future["01 Seasonal baseline"] = baseline_as_result(series, horizon_days, "daily").forecast.reindex(future_idx).values
    future["02 Same weekday median"] = _recursive_forecast(series, future_idx, _predict_same_weekday).values
    future["03 Rolling median 4 weeks"] = _recursive_forecast(series, future_idx, _predict_rolling_median).values
    future["04 Static YTD"] = _legacy_forecast_series(raw_df, metric, future_idx, "static_ytd", lookback_weeks, include_weekend).values
    future["05 Dynamic rolling"] = _legacy_forecast_series(raw_df, metric, future_idx, "dynamic_rolling", lookback_weeks, include_weekend).values
    future["06 Calendar index"] = _legacy_forecast_series(raw_df, metric, future_idx, "calendar_index", lookback_weeks, include_weekend).values
    future["07 Hybrid calendar x trend"] = _legacy_forecast_series(raw_df, metric, future_idx, "hybrid", lookback_weeks, include_weekend).values
    future["08 Ridge internal"] = ridge_forecast(series, horizon_days, "daily").forecast.reindex(future_idx).values

    hist_exog, future_exog = align_known_exog(exog_full, future_idx, lag_periods=0) if not exog_full.empty else (pd.DataFrame(), pd.DataFrame())
    hist_exog = hist_exog.reindex(series.index).ffill().bfill() if not hist_exog.empty else hist_exog
    if hist_exog is not None and not hist_exog.empty:
        future["09 Ridge with drivers"] = ridge_forecast(
            series,
            horizon_days,
            "daily",
            exog_hist=hist_exog,
            exog_future=future_exog,
            alpha=float(driver_alpha),
        ).forecast.reindex(future_idx).values
    else:
        future["09 Ridge with drivers"] = future["08 Ridge internal"].values
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
                driver_alpha=float(driver_alpha),
            )
        backtest_rows.append(row)
    backtest = pd.DataFrame(backtest_rows)

    score_rows: List[Dict[str, object]] = []
    if not backtest.empty:
        actual = backtest.set_index("date")["actual"]
        for model in ADVANCED_MODELS[:-1]:
            score_rows.append(_backtest_row(model, actual, backtest.set_index("date")[model]))

    scores = pd.DataFrame(score_rows).sort_values(["wape", "mae"], na_position="last").reset_index(drop=True) if score_rows else pd.DataFrame()
    weights = _smart_blend_weights(scores, backtest) if not scores.empty and not backtest.empty else {}

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
    return {"future": future, "backtest": backtest, "scores": scores, "weights": weight_df, "drivers": driver_info}


def _future_dates_for_window(last_actual_date: pd.Timestamp, eval_year: int, week_start: int, week_end: int, include_weekend: bool) -> List[pd.Timestamp]:
    return [day for day in _window_dates_for_year(eval_year, week_start, week_end, include_weekend) if day > last_actual_date]


def _window_dates_for_year(eval_year: int, week_start: int, week_end: int, include_weekend: bool) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    for week in range(int(week_start), int(week_end) + 1):
        for weekday in range(1, 8):
            if not include_weekend and weekday > 5:
                continue
            try:
                day = pd.Timestamp(dt.date.fromisocalendar(int(eval_year), int(week), int(weekday)))
            except ValueError:
                continue
            dates.append(day)
    return sorted(dates)


def _date_option_label(value: str) -> str:
    day = pd.Timestamp(value)
    iso = day.isocalendar()
    return f"{day:%Y-%m-%d} | KT{int(iso.week):02d} {WEEKDAY_SHORT.get(int(iso.weekday), '?')}"


def _drop_selected_dates(df: pd.DataFrame, drop_dates: Sequence[str]) -> pd.DataFrame:
    if df.empty or not drop_dates:
        return df.copy()
    frame = df.copy()
    frame["_drop_day"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    to_drop = {pd.Timestamp(value).normalize() for value in drop_dates}
    frame = frame[~frame["_drop_day"].isin(to_drop)].copy()
    return frame.drop(columns=["_drop_day"], errors="ignore")


def _operational_window_forecast_df(
    regular_raw_df: pd.DataFrame,
    metric: str,
    eval_year: int,
    week_start: int,
    week_end: int,
    trend_mode: str,
    lookback_weeks: int,
    include_weekend: bool,
    manual_adj_pct: float,
) -> pd.DataFrame:
    window_dates = _window_dates_for_year(eval_year, week_start, week_end, include_weekend)
    if regular_raw_df.empty or metric not in regular_raw_df.columns or not window_dates:
        return pd.DataFrame(columns=["date", "forecast", "iso_week", "weekday", "x"])

    window_idx = pd.DatetimeIndex(window_dates)
    forecast = pd.Series(
        [
            _legacy_point_forecast(
                regular_raw_df,
                metric=metric,
                target_date=pd.Timestamp(day),
                mode=trend_mode,
                lookback_weeks=lookback_weeks,
                include_weekend=include_weekend,
            )
            for day in window_idx
        ],
        index=window_idx,
        dtype=float,
    )
    forecast = _apply_manual_adjustment(forecast, manual_adj_pct)

    forecast_df = forecast.reset_index().rename(columns={"index": "date", 0: "forecast"})
    forecast_df["iso_week"] = forecast_df["date"].dt.isocalendar().week.astype(int)
    forecast_df["weekday"] = forecast_df["date"].dt.isocalendar().day.astype(int)
    forecast_df["x"] = forecast_df.apply(lambda row: _xkey(row["iso_week"], row["weekday"]), axis=1)
    return forecast_df


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


def _quality_band(wape: float) -> str:
    if pd.isna(wape):
        return "Bez hodnoceni"
    if wape <= 0.12:
        return "Vyborne"
    if wape <= 0.20:
        return "Silne"
    if wape <= 0.30:
        return "Pouzitelne"
    return "Opatrne"


def _render_how_to_read_backtest() -> None:
    with st.expander("Jak číst přesnost a doporučení", expanded=False):
        st.markdown(
            "\n".join(
                [
                    f"- `WAPE`: {HOW_TO_READ_METRICS['WAPE']}",
                    f"- `Bias`: {HOW_TO_READ_METRICS['Bias']}",
                    f"- `MAE`: {HOW_TO_READ_METRICS['MAE']}",
                    "- `Nejlepší backtest` je vítěz na posledních rolling oknech, ne marketingový odhad.",
                    "- `Smart blend` zůstane čistě u vítěze, pokud míchání modelů nepřinese reálné zlepšení.",
                ]
            )
        )


def _render_public_driver_explainer() -> None:
    with st.expander("Co znamenají veřejné indexy v modelu", expanded=False):
        st.markdown(
            "\n".join(
                [
                    f"- `Brent / diesel / TTF gas`: {PUBLIC_DRIVER_EXPLANATIONS['Brent / diesel / TTF gas']}",
                    f"- `RWI / GSCPI / container stress`: {PUBLIC_DRIVER_EXPLANATIONS['RWI / GSCPI / container stress']}",
                    f"- `Eurostat industry / PPI`: {PUBLIC_DRIVER_EXPLANATIONS['Eurostat industry / PPI']}",
                    f"- `ESAB / copper`: {PUBLIC_DRIVER_EXPLANATIONS['ESAB / copper']}",
                    "- Tyto indexy jsou doplňková vrstva. Největší váhu mají tehdy, když opravdu zlepšují backtest.",
                ]
            )
        )


def _metric_explanation(metric: str) -> str:
    return METRIC_EXPLANATIONS.get(metric, "Vybraná metrika zachycuje část provozního tlaku skladu a je užitečné ji číst spolu s ostatními KPI.")


def _staffing_metric_explanation(metric_label: str) -> str:
    return STAFFING_METRIC_EXPLANATIONS.get(metric_label, "Tahle staffing metrika převádí forecast práce do srozumitelné kapacitní potřeby.")


def _render_metric_story(metric: str, source_label: str) -> None:
    st.caption(f"{source_label}: {_metric_explanation(metric)}")


def _driver_feature_parts(feature: str) -> tuple[str, str, str]:
    text = str(feature)
    prefix = text.split("_", 1)[0] if "_" in text else "self"
    lag_label = ""
    base = text
    if "_lag" in text:
        base, lag_part = text.rsplit("_lag", 1)
        if lag_part.isdigit():
            lag_label = f"lag {int(lag_part)} d"
    if "_" in base:
        _, base = base.split("_", 1)
    return prefix, base, lag_label


def _driver_feature_label(feature: str) -> str:
    _, base, lag_label = _driver_feature_parts(feature)
    if base in DRIVER_LABELS:
        label = DRIVER_LABELS[base]
    else:
        label = (
            str(base)
            .replace("_pct", " pct")
            .replace("_delta", " delta")
            .replace("_", " ")
            .strip()
            .title()
        )
    return f"{label} ({lag_label})" if lag_label else label


def _driver_feature_family(feature: str) -> str:
    prefix, _, _ = _driver_feature_parts(feature)
    return DRIVER_FAMILY_LABELS.get(prefix, prefix)


def _driver_table_for_display(driver_info: pd.DataFrame) -> pd.DataFrame:
    if driver_info is None or driver_info.empty:
        return pd.DataFrame()
    table = driver_info.copy()
    table["family"] = table["feature"].map(_driver_feature_family)
    table["driver"] = table["feature"].map(_driver_feature_label)
    table["score"] = table["score"].map(lambda value: round(float(value), 3))
    table["corr_full"] = table["corr_full"].map(lambda value: round(float(value), 3))
    table["corr_recent"] = table["corr_recent"].map(lambda value: round(float(value), 3))
    return table[["selected_rank", "family", "driver", "score", "corr_full", "corr_recent", "n_obs"]]


def _render_model_story(scores: pd.DataFrame, active_model: str, driver_info: Optional[pd.DataFrame] = None) -> None:
    if scores is None or scores.empty:
        st.info("Backtest zatim nema dost bodu pro spolehlive srovnani modelu.")
        return

    ranked = scores.sort_values(["wape", "mae"], na_position="last").reset_index(drop=True)
    best = ranked.iloc[0]
    picked = _score_lookup(scores, active_model)
    picked_wape = float(picked["wape"]) if picked is not None else np.nan
    best_wape = float(best["wape"]) if pd.notna(best["wape"]) else np.nan
    gap = picked_wape - best_wape if picked is not None and pd.notna(best_wape) else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nejlepsi backtest", str(best["model"]))
    c2.metric("WAPE viteze", _format_pct(best_wape))
    c3.metric("Vybrany model", active_model)
    c4.metric("Rozdil vs vitez", _format_pct(gap) if pd.notna(gap) else "n/a")
    st.caption(MODEL_DESCRIPTIONS.get(active_model, ""))
    _render_how_to_read_backtest()

    if picked is None:
        st.caption("Vybrany model zatim nema dost backtest bodu.")
    elif active_model == str(best["model"]):
        st.success(f"Vybrany model je v tomhle rezu aktualne nejpresnejsi. Kvalita: {_quality_band(picked_wape)}.")
    elif pd.notna(gap) and gap <= 0.02:
        st.info(
            f"Vybrany model je velmi blizko vitezi. WAPE je horsi jen o {_format_pct(gap)}. "
            f"Kvalita: {_quality_band(picked_wape)}."
        )
    else:
        st.warning(
            f"Vybrany model prohrava proti vitezi o {_format_pct(gap)} WAPE. "
            f"Pokud chces maximalni presnost, ber spis `{best['model']}`."
        )

    if driver_info is not None and not driver_info.empty:
        driver_table = _driver_table_for_display(driver_info)
        if not driver_table.empty:
            st.caption(
                "Model 09 nebere vsechny drivery naslepo. Pouziva jen promene, ktere maji dost bodu a stabilni vztah k cili."
            )
            st.dataframe(driver_table, use_container_width=True, hide_index=True)
            st.caption("`family` říká, odkud driver pochází: stejný proces, navázaný proces, sklad/příjmy nebo veřejný svět.")


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
    st.caption(
        "WAPE = typická relativní chyba, Bias = systematické nadhodnocení nebo podhodnocení, MAE = typická chyba v jednotkách vybrané metriky."
    )

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
    exog_bundle: ExogBundle,
    metric_labels: Dict[str, str],
    default_metrics: List[str],
    shift_format,
    prefix: str,
    include_weekend_toggle: bool,
) -> None:
    st.subheader(tab_name)
    with st.expander("Jak číst tenhle pohled", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- Horní část porovnává stejné týdny a dny mezi roky, takže rychle uvidíš sezonnost a odchylku aktuálního roku.",
                    "- `Trend mód` se používá jen pro provozní forecast v tomhle okně; detailní srovnání modelů je na kartě `Predikce dopředu`.",
                    "- Když vynecháš problematické dny z aktuálního roku, schovají se z grafu i z výpočtu forecastu.",
                    "- Na téhle kartě je cílem rychlé operativní čtení, ne maximálně komplexní model.",
                ]
            )
        )

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

    current_year_mask = df["iso_year"].astype(int).eq(eval_year) & df["iso_week"].between(week_start, week_end, inclusive="both")
    if not include_weekend:
        current_year_mask = current_year_mask & df["weekday"].astype(int).between(1, 5)
    current_year_days = (
        pd.to_datetime(df.loc[current_year_mask, "date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .sort_values()
        .drop_duplicates()
    )
    ignore_options = [day.strftime("%Y-%m-%d") for day in current_year_days]
    ignore_dates: List[str] = []
    with st.expander("Vynechat problematicke dny z aktualniho roku", expanded=False):
        if ignore_options:
            ignore_dates = st.multiselect(
                "Dny k vynechani z grafu i forecastu",
                options=ignore_options,
                default=[],
                key=f"{prefix}_ignore_days",
                format_func=_date_option_label,
            )
            st.caption("Vybrane dny schovam z aktualniho roku a vyradim je i z vypoctu forecastu.")
        else:
            st.caption("V aktualnim roce v tomhle vyrezu zatim nemam zadne dny k vynechani.")

    df_model = _drop_selected_dates(df, ignore_dates)
    if df_model.empty:
        st.warning("Po vynechani dnu nezustala pro tenhle vyrez zadna data.")
        return
    if ignore_dates:
        st.caption(f"Z aktualniho roku je docasne vynechano {len(ignore_dates)} dnu.")

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
        st.caption(MODEL_DESCRIPTIONS.get(TREND_TO_ADVANCED_MODEL[trend_mode], ""))
    with fc2:
        lookback_weeks = int(st.slider("Trend okno (týdny)", 2, 26, 8, key=f"{prefix}_lookback", disabled=not show_forecast))
    with fc3:
        manual_adj = int(st.slider("Korekce predikce (%)", -30, 30, 0, key=f"{prefix}_adj", disabled=not show_forecast))

    driver_notes = []
    if exog_bundle.warehouse_daily is not None and not exog_bundle.warehouse_daily.empty:
        driver_notes.append("Příjmy / handling / net flow")
    if exog_bundle.world_state_daily is not None and not exog_bundle.world_state_daily.empty:
        driver_notes.append("ESAB / energie / container stress")
    if driver_notes:
        st.caption("Rozšířené drivery pro model 09: " + " + ".join(driver_notes))

    selected_model_name = TREND_TO_ADVANCED_MODEL[trend_mode]
    grids: Dict[str, pd.DataFrame] = {}
    suites: Dict[str, Dict[str, object]] = {}
    forecast_frames: Dict[str, pd.DataFrame] = {}

    regular_model_df = _regularize_daily(df_model)
    last_actual_date = pd.to_datetime(df_model["date"]).max()
    future_window_dates = _future_dates_for_window(last_actual_date, eval_year, week_start, week_end, include_weekend)
    max_horizon = max((future_window_dates[-1] - last_actual_date).days, 1) if future_window_dates else 1

    for metric in order:
        st.markdown(f"#### {metric_labels.get(metric, metric)}")
        _render_metric_story(metric, tab_name)
        grid = _week_day_grid(df_model[df_model["iso_year"].isin(years_sel)], metric, week_start, week_end, include_weekend)
        grids[metric] = grid
        if grid.empty:
            st.info("V tomto výřezu nejsou data.")
            continue

        suite = compute_model_suite(
            raw_df=df_model,
            linked_df=linked_daily_df,
            warehouse_df=exog_bundle.warehouse_daily,
            world_state_daily=exog_bundle.world_state_daily,
            metric=metric,
            horizon_days=max_horizon,
            lookback_weeks=lookback_weeks,
            include_weekend=include_weekend,
        )
        suites[metric] = suite

        forecast_df = pd.DataFrame(columns=["date", "forecast", "iso_week", "weekday", "x"])
        if show_forecast:
            forecast_df = _operational_window_forecast_df(
                regular_raw_df=regular_model_df,
                metric=metric,
                eval_year=eval_year,
                week_start=week_start,
                week_end=week_end,
                trend_mode=trend_mode,
                lookback_weeks=lookback_weeks,
                include_weekend=include_weekend,
                manual_adj_pct=manual_adj,
            ).dropna(subset=["forecast"])
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

    _render_model_story(
        detail_suite.get("scores", pd.DataFrame()),
        selected_model_name,
        None,
    )

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


def tab_baleni(src: Sources, exog_bundle: ExogBundle) -> None:
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
        exog_bundle=exog_bundle,
        metric_labels=metric_labels,
        default_metrics=["binhits", "gross_tons", "pallets_count", "cartons_count"],
        shift_format=lambda item: "Celkem" if item == "all" else ("Denní" if item == "day" else ("Noční" if item == "night" else item)),
        prefix="pk",
        include_weekend_toggle=True,
    )


def tab_nakladky(src: Sources, exog_bundle: ExogBundle) -> None:
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
        exog_bundle=exog_bundle,
        metric_labels=metric_labels,
        default_metrics=["trips_total", "gross_tons", "containers_count"],
        shift_format=lambda item: "Celkem" if item == "all" else ("Ranní" if item == "morning" else ("Odpolední" if item == "afternoon" else item)),
        prefix="ld",
        include_weekend_toggle=False,
    )


def _render_prediction_models(src: Sources, exog_bundle: ExogBundle) -> None:
    source_options = {
        "Balení (Kompletace)": ("packed", src.packed_daily, src.loaded_daily),
        "Nakládky (Výdeje)": ("loaded", src.loaded_daily, src.packed_daily),
    }
    choice = st.selectbox("Zdroj", list(source_options.keys()), key="pred_source")
    source_key, df, linked_df = source_options[choice]
    with st.expander("Jak číst tuto kartu", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- Tahle karta slouží k poctivému srovnání modelů nad jednou vybranou metrikou.",
                    "- Forecast se počítá jen pro pracovní dny, aby se nepletly víkendové nuly s reálnou provozní predikcí.",
                    "- `Vybraný model` je tvoje aktivní volba; `Nejlepší backtest` říká, co opravdu vyhrálo na posledních historických oknech.",
                    "- Pokud veřejné indexy nepomáhají, model je sice vidí, ale backtest nedovolí, aby uměle přebily lepší jednoduchý model.",
                ]
            )
        )

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
        horizon = int(st.slider("Horizont (pracovní dny)", 5, 120, 30, key="pred_horizon"))
    with c4:
        manual_adj = int(st.slider("Korekce (%)", -30, 30, 0, key="pred_adj"))
    st.caption(MODEL_DESCRIPTIONS.get(model_name, ""))
    _render_metric_story(metric, choice)

    lookback_weeks = int(st.slider("Trend okno pro trendové modely", 2, 26, 8, key="pred_lookback"))
    include_weekend = False
    st.caption("Na kartě Predikce se zobrazují jen pracovní dny.")
    driver_notes = []
    if exog_bundle.warehouse_daily is not None and not exog_bundle.warehouse_daily.empty:
        driver_notes.append("Příjmy / handling / net flow")
    if exog_bundle.world_state_daily is not None and not exog_bundle.world_state_daily.empty:
        driver_notes.append("ESAB / energie / container stress")
    if driver_notes:
        st.caption("Model 09 používá: " + " + ".join(driver_notes))
    if exog_bundle.world_state_daily is not None and not exog_bundle.world_state_daily.empty:
        _render_public_driver_explainer()

    effective_horizon = int(max(horizon + 12, round(horizon * 1.6)))

    with st.spinner("Počítám forecast a backtest modelů..."):
        suite = compute_model_suite(
            df,
            linked_df,
            exog_bundle.warehouse_daily,
            exog_bundle.world_state_daily,
            metric,
            effective_horizon,
            lookback_weeks,
            include_weekend,
        )
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
    st.caption("Součet 10 dní je rychlý kapacitní pohled. Průměr den pomáhá číst běžnou denní zátěž bez toho, aby tě mátly jednotlivé špičky.")

    _render_model_story(scores, model_name, suite.get("drivers", pd.DataFrame()) if model_name in {"09 Ridge with drivers", "10 Smart blend"} else None)

    fig = px.line(plot_df, x="date", y="value", color="series", markers=True, title=f"{metric_labels.get(metric, metric)} - historie + predikce")
    fig.update_traces(connectgaps=False)
    st.plotly_chart(fig, use_container_width=True)

    if model_name == "10 Smart blend" and not weights.empty:
        st.markdown("#### Váhy Smart blend")
        show_weights = weights.copy()
        show_weights["weight"] = show_weights["weight"].map(lambda value: f"{value:.1%}")
        if len(weights) == 1:
            st.caption("Smart blend v tomhle řezu nenašel lepší kombinaci, takže drží čistě nejpřesnější model.")
        else:
            st.caption("Smart blend míchá jen top modely, pokud jejich kombinace v backtestu opravdu zlepšila přesnost.")
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
    with st.expander("Jak číst staffing forecast", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- Staffing karta převádí forecast práce do lidí a hodin. Není to samostatný svět, stojí na forecastu binhitů a historické produktivitě.",
                    "- `Kmen` je držený v pevných bucketech `ráno` a `odpo`, aby model zůstal blízko realitě a nehýbal kmenem nereálně.",
                    "- `Agentura` dopočítává zbytek potřeby po odečtení realistické kapacity kmene.",
                    "- Víkendy jsou zobrazením odfiltrované, aby karty ukazovaly pracovní plán, ne nulové víkendové body.",
                ]
            )
        )

    forecast = bundle.forecast.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast = forecast[forecast["date"].dt.weekday < 5].copy()
    actuals = bundle.actuals.copy() if bundle.actuals is not None else pd.DataFrame()
    if not actuals.empty:
        actuals["date"] = pd.to_datetime(actuals["date"])
        actuals = actuals[actuals["date"].dt.weekday < 5].copy()
    horizon = bundle.horizon.copy() if bundle.horizon is not None else pd.DataFrame()
    if not horizon.empty:
        horizon["date"] = pd.to_datetime(horizon["date"])
        horizon = horizon[horizon["date"].dt.weekday < 5].copy()

    metric_map_all = {
        "Binhits celkem": ("binhits", "forecast_binhits"),
        "Binhits den": ("packed_day_binhits", "forecast_day_binhits"),
        "Binhits noc": ("packed_night_binhits", "forecast_night_binhits"),
        "Potřebný headcount": ("att_headcount", "required_headcount_ceiling"),
        "Kmen ráno": ("att_kmen_early_headcount", "required_kmen_early"),
        "Kmen odpo": ("att_kmen_late_headcount", "required_kmen_late"),
        "Agentura den": ("att_agency_day_headcount", "required_agency_day"),
        "Agentura noc": ("att_agency_night_headcount", "required_agency_night"),
        "Denní směna": ("day_att_headcount", "required_day_shift_workers"),
        "Noční směna": ("night_att_headcount", "required_night_shift_workers"),
        "Placené hodiny": ("att_hours", "required_paid_hours"),
        "Produktivní workers": ("prod_workers", "required_productive_workers_ceiling"),
    }
    metric_map = {
        label: cols
        for label, cols in metric_map_all.items()
        if cols[1] in forecast.columns and (actuals.empty or cols[0] in actuals.columns or cols[0] == "binhits")
    }
    if not metric_map:
        st.warning("Staffing exporty nemají očekávané sloupce pro zobrazení grafu.")
        return

    c1, c2 = st.columns([2, 3])
    with c1:
        metric_label = st.selectbox("Staffing metrika", list(metric_map.keys()), key="staff_metric")
    with c2:
        horizon_days = int(st.slider("Horizont staffing grafu (pracovní dny)", 5, 90, 30, key="staff_horizon"))

    actual_col, forecast_col = metric_map[metric_label]
    st.caption(_staffing_metric_explanation(metric_label))

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

    st.caption("Kmen je držený v pevných bucketech `ráno` a `odpo`. Forecast počítá zvlášť denní a noční binhits; agentura dopočítává jen zbylou flexibilitu pro den a noc. Víkendy jsou ze zobrazení skryté.")

    card_horizons = [5, 10, 20, 30]
    cols = st.columns(len(card_horizons))
    for col, horizon_day in zip(cols, card_horizons):
        if len(forecast) < horizon_day:
            col.metric(f"{horizon_day} dní", "n/a")
            continue
        val = float(forecast.iloc[horizon_day - 1][forecast_col]) if forecast_col in forecast.columns else np.nan
        col.metric(f"{horizon_day} dní", _format_num(val, 1))

    if bundle.driver_backtests is not None and not bundle.driver_backtests.empty:
        st.markdown("#### Backtest driver modelů")
        st.caption("Tahle tabulka ukazuje, jak dobře forecastujeme samotné provozní drivere, ze kterých staffing vychází.")
        bt = bundle.driver_backtests.copy()
        for col in ["wape", "bias"]:
            bt[col] = bt[col].map(_format_pct)
        bt["mae"] = bt["mae"].map(lambda value: _format_num(value, 2))
        st.dataframe(bt, use_container_width=True, hide_index=True)

    if bundle.ratio_backtests is not None and not bundle.ratio_backtests.empty:
        st.markdown("#### Backtest staffing převodu")
        st.caption("Tahle tabulka říká, jak přesně se daří převést známý výkon na lidi a hodiny. Odděluje chybu forecastu práce od chyby staffing převodu.")
        ratio = bundle.ratio_backtests.copy()
        ratio["wape"] = ratio["wape"].map(_format_pct)
        ratio["mae"] = ratio["mae"].map(lambda value: _format_num(value, 2))
        st.dataframe(ratio, use_container_width=True, hide_index=True)

    if not horizon.empty:
        st.markdown("#### Horizon points")
        st.caption("Body 5/10/20/30 dní jsou rychlé orientační checkpointy pro plánování. Ber je jako pracovní orientaci, ne jako fixní rozpis směn.")
        st.dataframe(horizon, use_container_width=True, hide_index=True)

    st.markdown("#### Staffing forecast tabulka")
    preferred_cols = [
        "date",
        "weekday",
        "forecast_day_binhits",
        "forecast_night_binhits",
        "forecast_binhits",
        "required_headcount_ceiling",
        "capacity_gap_headcount",
        "kmen_early_capacity_cap",
        "kmen_late_capacity_cap",
        "required_kmen_early",
        "required_kmen_late",
        "required_agency_day",
        "required_agency_night",
        "required_kmen",
        "required_agency",
        "required_day_shift_workers",
        "required_night_shift_workers",
    ]
    available_cols = [col for col in preferred_cols if col in forecast.columns]
    if available_cols:
        st.dataframe(
            forecast[available_cols].head(horizon_days),
            use_container_width=True,
            hide_index=True,
        )
    missing_cols = [col for col in preferred_cols if col not in forecast.columns]
    if missing_cols:
        st.caption("Některé nové staffing sloupce v exportu chybí: " + ", ".join(missing_cols[:6]) + ("..." if len(missing_cols) > 6 else ""))


def tab_predikce(src: Sources, staffing_bundle: StaffingBundle, exog_bundle: ExogBundle) -> None:
    st.subheader("Predikce dopředu 📈")
    st.caption("`Modely výkonu` porovnávají forecasty KPI jako binhits, trips a kontejnery. `Staffing` převádí forecast práce do lidí, hodin a směnových bucketů.")
    sub1, sub2 = st.tabs(["Modely výkonu", "Staffing"])

    with sub1:
        _render_prediction_models(src, exog_bundle)

    with sub2:
        _render_staffing_tab(staffing_bundle)


def main() -> None:
    st.title("Warehouse Dashboard - Balení & Nakládky")

    with st.sidebar:
        st.header("Data")
        base = st.text_input("Složka s KPI CSV", value=".")
        base_dir = Path(base).resolve()
        st.caption("Očekává: packed/loaded KPI CSV a volitelně staffing_forecast_exports + warehouse_state_exports + world_state_feature_weekly.csv")

    src, missing = load_sources(str(base_dir))
    staffing_bundle = load_staffing_bundle(str(base_dir))
    exog_bundle = load_exog_bundle(str(base_dir))

    if missing:
        st.warning("Chybí některé KPI soubory: " + ", ".join(missing))
    driver_notes = []
    if exog_bundle.warehouse_daily is not None and not exog_bundle.warehouse_daily.empty:
        driver_notes.append("Příjmy / handling / net flow")
    if exog_bundle.world_state_daily is not None and not exog_bundle.world_state_daily.empty:
        driver_notes.append("ESAB / energie / container stress")
    if driver_notes:
        st.caption("Driver vrstva aktivní: " + " + ".join(driver_notes))

    tab1, tab2, tab3 = st.tabs(["Balení", "Nakládky", "Predikce dopředu"])

    with tab1:
        tab_baleni(src, exog_bundle)
    with tab2:
        tab_nakladky(src, exog_bundle)
    with tab3:
        tab_predikce(src, staffing_bundle, exog_bundle)


if __name__ == "__main__":
    main()
