from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


FRED_SERIES = {
    "brent_usd_bbl": "DCOILBRENTEU",
    "copper_usd_t": "PCOPPUSDM",
    "vix_index": "VIXCLS",
    "eu_gas_usd_mmbtu": "PNGASEUUSDM",
}


def _http_get(url: str, timeout: int = 20) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; WarehouseForecast/1.0)"}
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response


def fetch_fred_series(series_id: str, value_name: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        response = _http_get(url)
        frame = pd.read_csv(BytesIO(response.content))
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        value_col = next((col for col in frame.columns if col != "date"), None)
        if value_col is None:
            return pd.DataFrame(columns=["date", value_name])
        out = frame[["date", value_col]].rename(columns={value_col: value_name})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
        return out.dropna(subset=["date", value_name]).sort_values("date")
    except Exception:
        return pd.DataFrame(columns=["date", value_name])


def fetch_esab_history() -> pd.DataFrame:
    urls = [
        "https://stooq.com/q/d/l/?s=esab.us&i=d",
        "https://stooq.com/q/d/?s=esab.us&i=d",
    ]
    for url in urls:
        try:
            response = _http_get(url)
            frame = pd.read_csv(BytesIO(response.content))
            cols = {str(col).strip().lower(): col for col in frame.columns}
            if "date" not in cols or "close" not in cols:
                continue
            out = frame[[cols["date"], cols["close"]]].rename(columns={cols["date"]: "date", cols["close"]: "esab_close"})
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["esab_close"] = pd.to_numeric(out["esab_close"], errors="coerce")
            out = out.dropna(subset=["date", "esab_close"]).sort_values("date")
            if len(out) >= 30:
                return out
        except Exception:
            continue
    return pd.DataFrame(columns=["date", "esab_close"])


def fetch_gscpi() -> pd.DataFrame:
    url = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
    try:
        response = _http_get(url, timeout=25)
        book = BytesIO(response.content)
        best = pd.DataFrame()
        for skiprows in range(0, 8):
            try:
                book.seek(0)
                frame = pd.read_excel(book, sheet_name=0, skiprows=skiprows)
            except Exception:
                continue
            frame.columns = [str(col).strip().lower() for col in frame.columns]
            date_col = next((col for col in frame.columns if "date" in col or "month" in col or "period" in col), None)
            value_col = next((col for col in frame.columns if "gscpi" in col), None)
            if not date_col or not value_col:
                continue
            candidate = frame[[date_col, value_col]].rename(columns={date_col: "date", value_col: "gscpi_index"})
            candidate["date"] = pd.to_datetime(candidate["date"], errors="coerce")
            candidate["gscpi_index"] = pd.to_numeric(candidate["gscpi_index"], errors="coerce")
            candidate = candidate.dropna(subset=["date", "gscpi_index"])
            if len(candidate) > len(best):
                best = candidate
        return best.sort_values("date") if not best.empty else pd.DataFrame(columns=["date", "gscpi_index"])
    except Exception:
        return pd.DataFrame(columns=["date", "gscpi_index"])


def _merge_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for _, frame in frames.items():
        if frame is None or frame.empty:
            continue
        value_col = next((col for col in frame.columns if col != "date"), None)
        if value_col is None:
            continue
        x = frame[["date", value_col]].copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        parts.append(x.dropna(subset=["date"]).set_index("date"))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(idx).ffill().bfill()


def _featureize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    for col in list(frame.columns):
        out[f"{col}_1d_pct"] = frame[col].pct_change(1) * 100
        out[f"{col}_5d_pct"] = frame[col].pct_change(5) * 100
        out[f"{col}_20d_pct"] = frame[col].pct_change(20) * 100
    return out.replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()


def load_or_refresh_market_signals(data_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = Path(data_dir) / "external_market_daily.csv"
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()
        except Exception:
            cached = pd.DataFrame()

    # Prefer the bundled local cache for app runtime speed and stability.
    # Live refresh should happen explicitly via maintenance script, not on page load.
    if not force_refresh and not cached.empty:
        return cached

    frames = {name: fetch_fred_series(series_id, name) for name, series_id in FRED_SERIES.items()}
    frames["esab_close"] = fetch_esab_history()
    frames["gscpi_index"] = fetch_gscpi()
    live = _featureize(_merge_frames(frames))

    if live.empty and not cached.empty:
        return cached
    if not live.empty:
        live.reset_index(names="date").to_csv(cache_path, index=False, encoding="utf-8")
        return live
    return pd.DataFrame()
