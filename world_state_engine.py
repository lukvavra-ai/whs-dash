from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import re

import numpy as np
import pandas as pd
import requests

from data_pipeline import safe_div, zscore


FACTOR_META = {
    "brent": {"label": "Brent", "unit": "USD/bbl", "source": "Yahoo Finance", "category": "energy"},
    "vix": {"label": "VIX", "unit": "index", "source": "Yahoo Finance / CBOE", "category": "risk"},
    "copper": {"label": "Copper", "unit": "USD/lb", "source": "Yahoo Finance", "category": "industry"},
    "eu_gas": {"label": "TTF gas", "unit": "EUR/MWh proxy", "source": "Yahoo Finance", "category": "energy"},
    "gscpi": {"label": "GSCPI", "unit": "index", "source": "New York Fed", "category": "supply_chain"},
    "esab_close": {"label": "ESAB stock", "unit": "USD", "source": "Yahoo Finance", "category": "company"},
    "rwi_total_trend": {"label": "RWI total throughput", "unit": "2015=100", "source": "RWI / ISL", "category": "shipping"},
    "rwi_north_trend": {"label": "RWI North Range throughput", "unit": "2015=100", "source": "RWI / ISL", "category": "shipping"},
    "eu_diesel_price": {"label": "EU diesel", "unit": "EUR/1000l", "source": "European Commission", "category": "transport"},
    "eur_diesel_price": {"label": "Euro area diesel", "unit": "EUR/1000l", "source": "European Commission", "category": "transport"},
    "cz_diesel_price": {"label": "CZ diesel", "unit": "EUR/1000l", "source": "European Commission", "category": "transport"},
    "de_diesel_price": {"label": "DE diesel", "unit": "EUR/1000l", "source": "European Commission", "category": "transport"},
    "de_prod_c24_c25": {"label": "DE prod metals", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "de_prod_c25": {"label": "DE prod fabricated metals", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "de_prod_c28": {"label": "DE prod machinery", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "de_prod_c29": {"label": "DE prod automotive", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "ea_prod_mig_ing": {"label": "EA prod intermediate goods", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "ea_prod_mig_cag": {"label": "EA prod capital goods", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "ea_prod_c25": {"label": "EA prod fabricated metals", "unit": "2021=100", "source": "Eurostat", "category": "industry"},
    "de_ppi_c24_c25": {"label": "DE PPI metals", "unit": "2021=100", "source": "Eurostat", "category": "price"},
    "ea_ppi_mig_ing": {"label": "EA PPI intermediate goods", "unit": "2021=100", "source": "Eurostat", "category": "price"},
    "custom_factor": {"label": "Custom factor", "unit": "value", "source": "Uploaded CSV", "category": "custom"},
}

NEWS_QUERIES = {
    "shipping_risk_news": {
        "label": "Shipping / trade risk",
        "query": '(("Red Sea" OR Suez OR Hormuz OR tariff OR sanctions OR "container shipping") AND (shipping OR vessel OR container OR port OR logistics))',
        "timespan": "12weeks",
    },
    "energy_conflict_news": {
        "label": "Energy / Middle East risk",
        "query": '((Iran OR "Middle East" OR Hormuz OR LNG OR gas OR oil) AND (attack OR strike OR disruption OR sanctions OR conflict))',
        "timespan": "12weeks",
    },
    "europe_industry_news": {
        "label": "Europe industry momentum",
        "query": '((Germany OR Europe OR Eurozone) AND (manufacturing OR industrial OR automotive OR factory) AND (orders OR slowdown OR recovery OR output))',
        "timespan": "12weeks",
    },
}

YAHOO_SERIES = {
    "brent": ("BZ=F", "brent"),
    "vix": ("^VIX", "vix"),
    "copper": ("HG=F", "copper"),
    "eu_gas": ("TTF=F", "eu_gas"),
    "esab_close": ("ESAB", "esab_close"),
}

EUROSTAT_SPECS = [
    ("STS_INPR_M", {"geo": "DE", "nace_r2": "C24_C25", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "de_prod_c24_c25"),
    ("STS_INPR_M", {"geo": "DE", "nace_r2": "C25", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "de_prod_c25"),
    ("STS_INPR_M", {"geo": "DE", "nace_r2": "C28", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "de_prod_c28"),
    ("STS_INPR_M", {"geo": "DE", "nace_r2": "C29", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "de_prod_c29"),
    ("STS_INPR_M", {"geo": "EA20", "nace_r2": "MIG_ING", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "ea_prod_mig_ing"),
    ("STS_INPR_M", {"geo": "EA20", "nace_r2": "MIG_CAG", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "ea_prod_mig_cag"),
    ("STS_INPR_M", {"geo": "EA20", "nace_r2": "C25", "s_adj": "SCA", "unit": "I21", "indic_bt": "PRD"}, "ea_prod_c25"),
    ("STS_INPP_M", {"geo": "DE", "nace_r2": "C24_C25", "s_adj": "NSA", "unit": "I21", "indic_bt": "PRC_PRR"}, "de_ppi_c24_c25"),
    ("STS_INPP_M", {"geo": "EA20", "nace_r2": "MIG_ING", "s_adj": "NSA", "unit": "I21", "indic_bt": "PRC_PRR"}, "ea_ppi_mig_ing"),
]

WEEKLY_METHODS = {
    "gscpi": "last",
    "rwi_total_original": "last",
    "rwi_total_adjusted": "last",
    "rwi_total_trend": "last",
    "rwi_north_original": "last",
    "rwi_north_adjusted": "last",
    "rwi_north_trend": "last",
    "eu_diesel_price": "last",
    "eur_diesel_price": "last",
    "cz_diesel_price": "last",
    "de_diesel_price": "last",
    "de_prod_c24_c25": "last",
    "de_prod_c25": "last",
    "de_prod_c28": "last",
    "de_prod_c29": "last",
    "ea_prod_mig_ing": "last",
    "ea_prod_mig_cag": "last",
    "ea_prod_c25": "last",
    "de_ppi_c24_c25": "last",
    "ea_ppi_mig_ing": "last",
}


def _http_get(url: str, timeout: int = 20, params: Optional[dict] = None) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Pardubice-WHS-WorldState/1.0)"}
    response = requests.get(url, timeout=timeout, headers=headers, params=params)
    response.raise_for_status()
    return response


def fmt_num(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:+.{digits}f} %".replace(".", ",")


def composite_index(frame: pd.DataFrame, cols: Iterable[str], signs: Optional[Iterable[int]] = None) -> pd.Series:
    requested = list(cols)
    if not requested:
        return pd.Series(dtype=float)
    sign_list = list(signs) if signs is not None else [1] * len(requested)
    selected: List[tuple[str, int]] = []
    for col, sign in zip(requested, sign_list):
        if col in frame.columns:
            selected.append((col, int(sign)))
    if not selected:
        return pd.Series(dtype=float)
    parts = []
    for col, sign in selected:
        parts.append(sign * zscore(pd.to_numeric(frame[col], errors="coerce")))
    return 100 + 10 * (sum(parts) / len(parts))


def _coerce_numeric_columns(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for col in out.columns:
        if col != date_col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=[date_col]).sort_values(date_col)


def fetch_yahoo_chart_series(symbol: str, value_name: str, range_window: str = "10y") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        response = _http_get(url, timeout=30, params={"interval": "1d", "range": range_window, "includeAdjustedClose": "true"})
        payload = response.json()
        result = payload.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame(columns=["date", value_name])
        node = result[0]
        timestamps = node.get("timestamp") or []
        close_values = (node.get("indicators", {}).get("quote", [{}])[0].get("close") or [])
        if not timestamps or not close_values:
            return pd.DataFrame(columns=["date", value_name])
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                value_name: pd.to_numeric(close_values, errors="coerce"),
            }
        )
        return out.dropna(subset=["date", value_name]).drop_duplicates("date").sort_values("date")
    except Exception:
        return pd.DataFrame(columns=["date", value_name])


def fetch_gscpi() -> pd.DataFrame:
    url = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
    try:
        response = _http_get(url, timeout=30)
        book = BytesIO(response.content)
        frame = pd.read_excel(book, sheet_name="GSCPI Monthly Data")
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        date_col = next((c for c in frame.columns if "date" in c or "month" in c or "period" in c), None)
        value_col = next((c for c in frame.columns if "gscpi" in c), None)
        if not date_col or not value_col:
            return pd.DataFrame(columns=["date", "gscpi"])
        out = frame[[date_col, value_col]].rename(columns={date_col: "date", value_col: "gscpi"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["gscpi"] = pd.to_numeric(out["gscpi"], errors="coerce")
        return out.dropna(subset=["date", "gscpi"]).sort_values("date")
    except Exception:
        return pd.DataFrame(columns=["date", "gscpi"])


def _resolve_rwi_xlsx_url() -> str:
    fallback = "https://www.rwi-essen.de/fileadmin/user_upload/RWI/Presse/containerumschlag-Index_260227.xlsx"
    try:
        landing = _http_get("https://www.rwi-essen.de/containerindex", timeout=30).text
        press_links = re.findall(r'href=["\']([^"\']*containerumschlag-index[^"\']+)["\']', landing, re.I)
        if press_links:
            first = press_links[0]
            if first.startswith("/"):
                first = f"https://www.rwi-essen.de{first}"
            press_page = _http_get(first, timeout=30).text
            match = re.search(r'href=["\']([^"\']*containerumschlag[^"\']+\.xlsx)["\']', press_page, re.I)
            if match:
                link = match.group(1)
                return f"https://www.rwi-essen.de{link}" if link.startswith("/") else link
    except Exception:
        pass
    return fallback


def fetch_rwi_container_index() -> pd.DataFrame:
    try:
        response = _http_get(_resolve_rwi_xlsx_url(), timeout=30)
        raw = pd.read_excel(BytesIO(response.content), sheet_name="Output", header=None)
        body = raw.iloc[6:, :7].copy()
        body.columns = [
            "date",
            "rwi_total_original",
            "rwi_total_adjusted",
            "rwi_total_trend",
            "rwi_north_original",
            "rwi_north_adjusted",
            "rwi_north_trend",
        ]
        return _coerce_numeric_columns(body)
    except Exception:
        return pd.DataFrame(columns=["date", "rwi_total_trend", "rwi_north_trend"])


def _resolve_oil_history_url() -> str:
    fallback = (
        "https://energy.ec.europa.eu/document/download/906e60ca-8b6a-44e7-8589-652854d2fd3f_en"
        "?filename=Weekly_Oil_Bulletin_Prices_History_maticni_4web.xlsx"
    )
    try:
        landing = _http_get("https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en", timeout=30).text
        match = re.search(r'href=["\']([^"\']*Weekly_Oil_Bulletin_Prices_History_maticni_4web\.xlsx[^"\']*)["\']', landing, re.I)
        if match:
            link = match.group(1)
            return f"https://energy.ec.europa.eu{link}" if link.startswith("/") else link
    except Exception:
        pass
    return fallback


def fetch_weekly_oil_bulletin_history() -> pd.DataFrame:
    try:
        response = _http_get(_resolve_oil_history_url(), timeout=60)
        raw = pd.read_excel(BytesIO(response.content), sheet_name="Prices with taxes", header=0)
        first_col = raw.columns[0]
        body = raw.rename(columns={first_col: "date"}).iloc[2:].copy()
        keep = ["date", "EU_price_with_tax_diesel", "EUR_price_with_tax_diesel", "CZ_price_with_tax_diesel", "DE_price_with_tax_diesel"]
        body = body[keep].rename(
            columns={
                "EU_price_with_tax_diesel": "eu_diesel_price",
                "EUR_price_with_tax_diesel": "eur_diesel_price",
                "CZ_price_with_tax_diesel": "cz_diesel_price",
                "DE_price_with_tax_diesel": "de_diesel_price",
            }
        )
        return _coerce_numeric_columns(body)
    except Exception:
        return pd.DataFrame(columns=["date", "eu_diesel_price", "eur_diesel_price", "cz_diesel_price", "de_diesel_price"])


def fetch_eurostat_series(dataset: str, params: Dict[str, str], value_name: str) -> pd.DataFrame:
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
    try:
        payload = _http_get(url, timeout=60, params=params).json()
        values = payload.get("value", {})
        if not values:
            return pd.DataFrame(columns=["date", value_name])
        time_index = payload["dimension"]["time"]["category"]["index"]
        inv_time = {int(pos): label for label, pos in time_index.items()}
        rows: List[Dict[str, object]] = []
        for key, value in values.items():
            label = inv_time.get(int(key))
            if label is None:
                continue
            if re.fullmatch(r"\d{4}-\d{2}", str(label)):
                dt = pd.to_datetime(f"{label}-01", errors="coerce")
            else:
                dt = pd.to_datetime(label, errors="coerce")
            rows.append({"date": dt, value_name: value})
        return _coerce_numeric_columns(pd.DataFrame(rows))
    except Exception:
        return pd.DataFrame(columns=["date", value_name])


def _parse_gdelt_timeline(payload: dict, value_name: str) -> pd.DataFrame:
    timeline = None
    for key in ("timeline", "timeline_data", "timeline_raw"):
        if isinstance(payload.get(key), list):
            timeline = payload[key]
            break
    if timeline is None:
        return pd.DataFrame(columns=["date", value_name])

    rows = []
    if timeline and isinstance(timeline[0], dict) and isinstance(timeline[0].get("data"), list):
        for group in timeline:
            for rec in group.get("data", []):
                if not isinstance(rec, dict):
                    continue
                date_parsed = pd.to_datetime(rec.get("date"), errors="coerce")
                value_val = pd.to_numeric(rec.get("norm", rec.get("value")), errors="coerce")
                if pd.isna(date_parsed) or pd.isna(value_val):
                    continue
                rows.append({"date": date_parsed.normalize(), value_name: value_val})
    else:
        for rec in timeline:
            if not isinstance(rec, dict):
                continue
            date_key = next((k for k in rec if "date" in str(k).lower()), None)
            value_key = next((k for k in rec if str(k).lower() in {"value", "count", "volume", "norm"}), None)
            if not date_key or not value_key:
                continue
            date_parsed = pd.to_datetime(rec[date_key], errors="coerce")
            value_val = pd.to_numeric(rec[value_key], errors="coerce")
            if pd.isna(date_parsed) or pd.isna(value_val):
                continue
            rows.append({"date": date_parsed.normalize(), value_name: value_val})

    if not rows:
        return pd.DataFrame(columns=["date", value_name])
    out = pd.DataFrame(rows).groupby("date")[value_name].sum(min_count=1).reset_index()
    return out.sort_values("date")


def fetch_gdelt_timeline(query: str, value_name: str, timespan: str = "12weeks") -> pd.DataFrame:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": query, "mode": "timelinevolraw", "format": "json", "timespan": timespan}
    try:
        response = _http_get(url, params=params, timeout=30)
        return _parse_gdelt_timeline(response.json(), value_name)
    except Exception:
        return pd.DataFrame(columns=["date", value_name])


def fetch_live_world_state() -> dict:
    factors = {key: fetch_yahoo_chart_series(symbol, value_name) for key, (symbol, value_name) in YAHOO_SERIES.items()}
    factors["gscpi"] = fetch_gscpi()
    factors["rwi"] = fetch_rwi_container_index()
    factors["diesel"] = fetch_weekly_oil_bulletin_history()
    for dataset, params, value_name in EUROSTAT_SPECS:
        factors[value_name] = fetch_eurostat_series(dataset, params, value_name)
    news = {key: fetch_gdelt_timeline(meta["query"], key, meta.get("timespan", "12weeks")) for key, meta in NEWS_QUERIES.items()}
    return {"factors": factors, "news": news}


def to_weekly_raw(frame: pd.DataFrame, value_col: str, method: str = "mean") -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[value_col])
    x = frame[["date", value_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna(subset=["date", value_col])
    if x.empty:
        return pd.DataFrame(columns=[value_col])
    x["week_start"] = x["date"] - pd.to_timedelta(x["date"].dt.weekday, unit="D")
    if method == "last":
        out = x.groupby("week_start")[value_col].last().to_frame(value_col)
    elif method == "sum":
        out = x.groupby("week_start")[value_col].sum(min_count=1).to_frame(value_col)
    else:
        out = x.groupby("week_start")[value_col].mean().to_frame(value_col)
    out.index = pd.to_datetime(out.index)
    out.index.name = "week_start"
    return out.sort_index()


def factors_to_weekly(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for frame in factors.values():
        if frame is None or frame.empty or "date" not in frame.columns:
            continue
        value_cols = [col for col in frame.columns if col != "date" and pd.api.types.is_numeric_dtype(frame[col])]
        for value_col in value_cols:
            method = WEEKLY_METHODS.get(value_col, "mean")
            weekly = to_weekly_raw(frame[["date", value_col]].copy(), value_col, method=method)
            if not weekly.empty:
                parts.append(weekly)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="W-MON")
    return out.reindex(full_idx).ffill().bfill()


def news_to_daily(news: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key, frame in news.items():
        if frame is None or frame.empty or key not in frame.columns:
            continue
        x = frame[["date", key]].copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x[key] = pd.to_numeric(x[key], errors="coerce")
        x = x.dropna(subset=["date"]).groupby("date")[key].sum(min_count=1).to_frame()
        parts.append(x)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(idx).fillna(0.0)


def derive_news_watch(news_daily: pd.DataFrame) -> pd.DataFrame:
    if news_daily is None or news_daily.empty:
        return pd.DataFrame()
    out = news_daily.copy()
    for col in news_daily.columns:
        out[f"{col}_7d"] = out[col].rolling(7, min_periods=1).sum()
        out[f"{col}_prev7"] = out[col].shift(7).rolling(7, min_periods=1).sum()
        out[f"{col}_7d_pct"] = (safe_div(out[f"{col}_7d"], out[f"{col}_prev7"]) - 1) * 100
    return out


def news_watch_to_weekly(news_watch: pd.DataFrame) -> pd.DataFrame:
    if news_watch is None or news_watch.empty:
        return pd.DataFrame()
    parts = []
    src = news_watch.reset_index().rename(columns={news_watch.index.name or "index": "date"})
    for col in news_watch.columns:
        weekly = to_weekly_raw(src[["date", col]].copy(), col, method="last")
        if not weekly.empty:
            parts.append(weekly)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="W-MON")
    return out.reindex(full_idx).ffill().bfill()


def derive_factor_features(raw_weekly: pd.DataFrame) -> pd.DataFrame:
    if raw_weekly is None or raw_weekly.empty:
        return pd.DataFrame()
    raw = raw_weekly.sort_index().ffill().bfill()
    blocks: Dict[str, pd.Series] = {}
    for col in raw.columns:
        series = pd.to_numeric(raw[col], errors="coerce")
        blocks[col] = series
        blocks[f"{col}_4w_pct"] = series.pct_change(4) * 100
        blocks[f"{col}_13w_pct"] = series.pct_change(13) * 100
        blocks[f"{col}_z"] = zscore(series)
    feat = pd.DataFrame(blocks, index=raw.index)
    if "gscpi" in raw.columns:
        feat["gscpi_4w_delta"] = raw["gscpi"].diff(4)
        feat["gscpi_pub_safe"] = raw["gscpi"].shift(5)
        feat["gscpi_pub_safe_4w_delta"] = feat["gscpi_pub_safe"].diff(4)
    return feat.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def derive_world_indices(feature_weekly: pd.DataFrame, internal_weekly: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if feature_weekly is None or feature_weekly.empty:
        return pd.DataFrame()
    data = feature_weekly.copy()
    if internal_weekly is not None and not internal_weekly.empty:
        data = data.join(internal_weekly, how="left")

    idx = pd.DataFrame(index=data.index)
    idx["industry_demand_index"] = composite_index(
        data,
        ["de_prod_c25_13w_pct", "de_prod_c28_13w_pct", "de_prod_c29_13w_pct", "ea_prod_mig_ing_13w_pct", "ea_prod_c25_13w_pct", "europe_industry_news_7d_pct"],
        signs=[1, 1, 1, 1, 1, 1],
    )
    idx["welding_input_price_index"] = composite_index(
        data,
        ["copper_4w_pct", "brent_4w_pct", "eu_gas_4w_pct", "de_ppi_c24_c25_13w_pct", "ea_ppi_mig_ing_13w_pct", "cz_diesel_price_4w_pct"],
        signs=[1, 1, 1, 1, 1, 1],
    )
    idx["energy_stress_index"] = composite_index(
        data,
        ["brent_4w_pct", "eu_gas_4w_pct", "eu_diesel_price_4w_pct", "cz_diesel_price_4w_pct", "vix_4w_pct", "energy_conflict_news_7d_pct"],
        signs=[1, 1, 1, 1, 1, 1],
    )
    idx["container_market_index"] = composite_index(
        data,
        ["rwi_total_trend_13w_pct", "rwi_north_trend_13w_pct", "shipping_risk_news_7d_pct", "loaded_container_share"],
        signs=[1, 1, 1, 1],
    )
    idx["container_stress_index"] = composite_index(
        data,
        ["gscpi_pub_safe_4w_delta", "eu_diesel_price_4w_pct", "vix_4w_pct", "shipping_risk_news_7d_pct"],
        signs=[1, 1, 1, 1],
    )
    idx["geo_event_index"] = composite_index(
        data,
        ["vix_4w_pct", "brent_4w_pct", "eu_gas_4w_pct", "energy_conflict_news_7d_pct"],
        signs=[1, 1, 1, 1],
    )
    idx["europe_demand_index"] = composite_index(
        idx.join(data, how="left"),
        ["industry_demand_index", "esab_close_13w_pct", "copper_13w_pct", "europe_industry_news_7d_pct"],
        signs=[1, 1, 1, 1],
    )
    idx["export_risk_index"] = composite_index(
        idx.join(data, how="left"),
        ["container_stress_index", "container_market_index", "gscpi_pub_safe_4w_delta", "loaded_export_share", "loaded_container_share"],
        signs=[1, 1, 1, 1, 1],
    )
    idx["local_operability_index"] = composite_index(
        idx.join(data, how="left"),
        ["loaded_europe_share", "loaded_orders_per_trip", "packed_tons_per_pallet", "energy_stress_index", "cz_diesel_price_4w_pct"],
        signs=[1, 1, 1, -1, -1],
    )
    idx["overall_world_risk_index"] = composite_index(
        idx,
        ["energy_stress_index", "container_stress_index", "geo_event_index", "export_risk_index", "welding_input_price_index"],
        signs=[1, 1, 1, 1, 1],
    )
    return idx.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def build_world_alerts(raw_weekly: pd.DataFrame, news_watch: pd.DataFrame, internal_weekly: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    alerts = []
    if raw_weekly is not None and not raw_weekly.empty:
        latest = raw_weekly.ffill().iloc[-1]
        stamp = raw_weekly.index[-1]
        for factor, threshold, label, text in (
            ("brent", 8.0, "Energy", "Brent is rising and increases pressure on transport and energy."),
            ("eu_gas", 10.0, "TTF gas", "European gas is rising and can weaken industrial momentum."),
            ("cz_diesel_price", 8.0, "CZ diesel", "Czech diesel prices are rising and can pressure transport costs."),
            ("vix", 12.0, "Market stress", "Financial volatility is rising and increases uncertainty."),
        ):
            if factor in raw_weekly.columns:
                change = raw_weekly[factor].pct_change(4).iloc[-1] * 100
                if pd.notna(change) and change >= threshold:
                    alerts.append({"date": stamp, "severity": "high" if change >= threshold * 1.8 else "medium", "topic": label, "message": f"{text} 4w change {fmt_pct(change)}."})
        if "gscpi" in raw_weekly.columns:
            delta = raw_weekly["gscpi"].diff(4).iloc[-1]
            if pd.notna(delta) and delta > 0:
                alerts.append({"date": stamp, "severity": "medium", "topic": "Supply chain", "message": f"GSCPI increased by {fmt_num(delta, 2)} over 4 weeks."})
        if "rwi_north_trend" in raw_weekly.columns:
            delta = raw_weekly["rwi_north_trend"].pct_change(13).iloc[-1] * 100
            if pd.notna(delta) and abs(delta) >= 5:
                alerts.append({"date": stamp, "severity": "info", "topic": "North Range", "message": f"RWI North Range trend changed by {fmt_pct(delta)} over 13 weeks."})
        if "vix" in raw_weekly.columns and pd.notna(latest.get("vix")) and latest.get("vix") >= 24:
            alerts.append({"date": stamp, "severity": "high", "topic": "Geo-event risk", "message": f"VIX is {fmt_num(float(latest['vix']), 1)}."})
    if news_watch is not None and not news_watch.empty:
        last = news_watch.iloc[-1]
        stamp = news_watch.index[-1]
        for key, meta in NEWS_QUERIES.items():
            pct_col = f"{key}_7d_pct"
            qty_col = f"{key}_7d"
            pct = last.get(pct_col, np.nan)
            qty = last.get(qty_col, np.nan)
            if pd.notna(pct) and pd.notna(qty) and qty >= 5 and pct >= 30:
                alerts.append({"date": stamp, "severity": "high" if pct >= 80 else "medium", "topic": meta["label"], "message": f"News radar: {meta['label']} is {fmt_pct(pct)} versus previous 7 days."})
    if internal_weekly is not None and not internal_weekly.empty:
        latest = internal_weekly.ffill().iloc[-1]
        stamp = internal_weekly.index[-1]
        if "loaded_export_share" in latest and pd.notna(latest["loaded_export_share"]) and latest["loaded_export_share"] >= 0.18:
            alerts.append({"date": stamp, "severity": "info", "topic": "Export exposure", "message": f"Export share is {fmt_pct(float(latest['loaded_export_share']) * 100, 1)}."})
        if "loaded_container_share" in latest and pd.notna(latest["loaded_container_share"]) and latest["loaded_container_share"] >= 0.10:
            alerts.append({"date": stamp, "severity": "info", "topic": "Container exposure", "message": f"Container share is {fmt_pct(float(latest['loaded_container_share']) * 100, 1)}."})
    out = pd.DataFrame(alerts)
    if out.empty:
        return pd.DataFrame(columns=["date", "severity", "topic", "message"])
    severity_order = {"high": 0, "medium": 1, "info": 2}
    out["severity_rank"] = out["severity"].map(severity_order).fillna(3)
    return out.sort_values(["severity_rank", "date"], ascending=[True, False]).drop(columns=["severity_rank"]).reset_index(drop=True)


def load_cached_world_state(data_dir: Path) -> Optional[dict]:
    root = Path(data_dir)
    files = {
        "raw_weekly": root / "world_state_raw_weekly.csv",
        "feature_weekly": root / "world_state_feature_weekly.csv",
        "news_daily": root / "world_state_news_daily.csv",
        "alerts": root / "world_state_alerts.csv",
        "meta": root / "world_state_meta.json",
    }
    if not files["raw_weekly"].exists() or not files["feature_weekly"].exists():
        return None
    try:
        raw_weekly = pd.read_csv(files["raw_weekly"], parse_dates=["week_start"]).set_index("week_start")
        feature_weekly = pd.read_csv(files["feature_weekly"], parse_dates=["week_start"]).set_index("week_start")
        news_daily = pd.read_csv(files["news_daily"], parse_dates=["date"]).set_index("date") if files["news_daily"].exists() else pd.DataFrame()
        alerts = pd.read_csv(files["alerts"], parse_dates=["date"]) if files["alerts"].exists() else pd.DataFrame()
        meta = json.loads(files["meta"].read_text(encoding="utf-8")) if files["meta"].exists() else {}
        return {"raw_weekly": raw_weekly, "feature_weekly": feature_weekly, "news_daily": news_daily, "alerts": alerts, "meta": meta}
    except Exception:
        return None


def save_world_state_cache(
    data_dir: Path,
    raw_weekly: pd.DataFrame,
    feature_weekly: pd.DataFrame,
    news_daily: pd.DataFrame,
    alerts: pd.DataFrame,
    meta: Optional[dict] = None,
) -> None:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    (raw_weekly.reset_index() if not raw_weekly.empty else pd.DataFrame(columns=["week_start"])).to_csv(root / "world_state_raw_weekly.csv", index=False)
    (feature_weekly.reset_index() if not feature_weekly.empty else pd.DataFrame(columns=["week_start"])).to_csv(root / "world_state_feature_weekly.csv", index=False)
    (news_daily.reset_index() if not news_daily.empty else pd.DataFrame(columns=["date"])).to_csv(root / "world_state_news_daily.csv", index=False)
    (alerts if not alerts.empty else pd.DataFrame(columns=["date", "severity", "topic", "message"])).to_csv(root / "world_state_alerts.csv", index=False)
    payload = meta or {}
    payload.setdefault("updated_at", pd.Timestamp.utcnow().isoformat())
    (root / "world_state_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_with_cache(live: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
    if cached is None or cached.empty:
        return live
    if live is None or live.empty:
        return cached
    merged = cached.combine_first(live)
    merged.update(live)
    return merged.sort_index()


def refresh_and_cache_world_state(data_dir: Path, internal_weekly: Optional[pd.DataFrame] = None) -> dict:
    cached = load_cached_world_state(data_dir)
    live = fetch_live_world_state()
    live_raw = factors_to_weekly(live["factors"])
    live_news = news_to_daily(live["news"])

    cached_raw = cached["raw_weekly"] if cached else pd.DataFrame()
    cached_news = cached["news_daily"] if cached else pd.DataFrame()

    raw_weekly = _merge_with_cache(live_raw, cached_raw)
    news_daily = _merge_with_cache(live_news, cached_news)
    news_watch = derive_news_watch(news_daily)
    news_weekly = news_watch_to_weekly(news_watch)
    feature_weekly = derive_factor_features(raw_weekly)
    if not news_weekly.empty:
        feature_weekly = feature_weekly.join(news_weekly, how="left")
    world_indices = derive_world_indices(feature_weekly, internal_weekly)
    feature_weekly = feature_weekly.join(world_indices, how="left")
    alerts = build_world_alerts(raw_weekly, news_watch, internal_weekly)

    availability = {name: not frame.empty for name, frame in live["factors"].items()}
    news_availability = {name: not frame.empty for name, frame in live["news"].items()}
    meta = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "factor_rows": {name: int(len(frame)) for name, frame in live["factors"].items()},
        "news_rows": {name: int(len(frame)) for name, frame in live["news"].items()},
        "factor_live_available": availability,
        "news_live_available": news_availability,
        "used_cached_fallback": any(not ok for ok in availability.values()) or any(not ok for ok in news_availability.values()),
    }
    save_world_state_cache(Path(data_dir), raw_weekly, feature_weekly, news_daily, alerts, meta)
    return {"raw_weekly": raw_weekly, "feature_weekly": feature_weekly, "news_daily": news_daily, "alerts": alerts, "meta": meta}
