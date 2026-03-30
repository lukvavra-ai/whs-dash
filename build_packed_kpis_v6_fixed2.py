#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build daily + shift KPIs from Kompletace CSVs.

Key points (per user's spec):
- Time column: "Kompletováno" (optionally filter Status == "C" if present)
- Shifts (12H): day 07:00-19:00, night 19:00-07:00
- Palety/Kartony are counted as UNIQUE "Paleta číslo" per day (and per shift), not per row.
- Packaging classification:
    - Prefer mapping from TypObalu.csv (in C:\\Kompletace\\TypObalu.csv => same parent as input-dir)
      Expected columns: typ;EPL   where EPL contains "Karton" or "Paleta"
    - Fallback rule: BX or XCR => carton, else pallet
Outputs (into --out-dir, default ./kpi):
- packed_daily_kpis.csv
- packed_shift_kpis.csv
- packed_typobalu_audit.csv  (counts of unique Paleta číslo per typ/class/year)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def read_csv_robust(path: Path, sep: str = ";") -> pd.DataFrame:
    # Try common Czech encodings
    for enc in ("utf-8-sig", "cp1250", "latin1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except Exception:
            pass
    # last resort
    return pd.read_csv(path, sep=sep, encoding_errors="ignore", low_memory=False)


def parse_dt_series(s: pd.Series) -> pd.Series:
    # Czech date formats: "10.01.2026 0:28"
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def iso_parts(dts: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    iso = dts.dt.isocalendar()
    return iso["year"].astype(int), iso["week"].astype(int), iso["day"].astype(int)


def shift_12h_from_ts(ts: pd.Series) -> pd.Series:
    # day: [07:00, 19:00), night: otherwise
    t = ts.dt.time
    day_start = time(7, 0)
    day_end = time(19, 0)
    is_day = (t >= day_start) & (t < day_end)
    return np.where(is_day, "day", "night")


def normalize_typ(x: pd.Series) -> pd.Series:
    return (
        x.astype(str)
         .str.strip()
         .str.upper()
         .str.replace(r"\s+", "", regex=True)
    )


def load_typobalu_map(base_dir: Path) -> dict[str, str]:
    """
    Loads TypObalu.csv mapping.
    File is expected at base_dir / TypObalu.csv (e.g. C:\\Kompletace\\TypObalu.csv).
    Format: typ;EPL, where EPL contains "Karton"/"Paleta".
    Returns typ_code -> "BOX" or "PAL"
    """
    path = base_dir / "TypObalu.csv"
    if not path.exists():
        return {}
    df = read_csv_robust(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    if "typ" not in df.columns or "EPL" not in df.columns:
        return {}
    m: dict[str, str] = {}
    for _, r in df.iterrows():
        k = str(r["typ"]).strip().upper()
        v = str(r["EPL"]).strip().lower()
        if not k or k == "NAN":
            continue
        if "karton" in v:
            m[k] = "BOX"
        elif "palet" in v:
            m[k] = "PAL"
    return m


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.strip(): c for c in df.columns}
    for cand in candidates:
        for k, orig in cols.items():
            if k.lower() == cand.lower():
                return orig
    return None


def to_float_kg(x: pd.Series) -> pd.Series:
    # handles "906,12" etc.
    s = x.astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


@dataclass
class BuildResult:
    daily: pd.DataFrame
    shift: pd.DataFrame
    audit: pd.DataFrame


# ----------------------------
# Core build
# ----------------------------

def build_from_file(path: Path, typ_map: dict[str, str]) -> BuildResult:
    df = read_csv_robust(path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    ts_col = find_col(df, ["Kompletováno", "Kompletovano"])
    if ts_col is None:
        raise ValueError(f"Missing 'Kompletováno' in {path.name}")

    # optional status filter
    status_col = find_col(df, ["Status"])
    if status_col is not None:
        df = df[df[status_col].astype(str).str.upper().str.strip() == "C"].copy()

    ts = parse_dt_series(df[ts_col])
    df = df.loc[ts.notna()].copy()
    df["_ts"] = ts.loc[ts.notna()]
    df["date"] = df["_ts"].dt.date

    df["iso_year"], df["iso_week"], df["iso_weekday"] = iso_parts(df["_ts"])

    # shift
    df["shift"] = shift_12h_from_ts(df["_ts"])

    # gross tons: prefer "Brutto palety KG", else 0
    gross_col = find_col(df, ["Brutto palety KG", "Brutto palety KG "])
    if gross_col is None:
        # try other likely columns
        gross_col = find_col(df, ["Dílčí netto KG", "Dilci netto KG", "Dílčí netto kg", "Dilci netto kg"])
    if gross_col is not None:
        kg = to_float_kg(df[gross_col]).fillna(0.0)
        df["gross_tons"] = kg / 1000.0
    else:
        df["gross_tons"] = 0.0

    # binhits = rows
    df["binhits"] = 1

    # Paleta číslo (unique key for pallet/carton counting)
    palno_col = find_col(df, ["Paleta číslo", "Paleta cislo", "Paleta"])
    if palno_col is None:
        # fallback: SSCC might exist, but user asked paleta číslo; still keep something
        palno_col = find_col(df, ["SSCC"])
    if palno_col is None:
        raise ValueError(f"Missing 'Paleta číslo' (or SSCC fallback) in {path.name}")

    df["_palno"] = df[palno_col].astype(str).str.strip()

    # Typ obalu classification
    typ_col = find_col(df, ["Typ obalu", "Typ obalu ", "Typ", "TypObalu"])
    if typ_col is not None:
        typ = normalize_typ(df[typ_col])
        if typ_map:
            cls = typ.map(lambda x: typ_map.get(str(x).upper().strip(), "PAL"))
            is_carton = cls.eq("BOX")
        else:
            is_carton = typ.isin(["BX", "XCR"])
        df["_typ"] = typ
        df["_class"] = np.where(is_carton, "Karton", "Paleta")
    else:
        df["_typ"] = ""
        df["_class"] = "Paleta"

    # Unique pallet/carton counting: unique Paleta číslo per (date,shift) etc.
    # For cartons_count: count unique palno where class == Karton
    # For pallets_count: count unique palno where class == Paleta
    uniq = df.dropna(subset=["_palno"]).drop_duplicates(subset=["date", "shift", "_palno"])

    cart_uniq = uniq[uniq["_class"] == "Karton"]
    pal_uniq = uniq[uniq["_class"] == "Paleta"]

    cart_counts_shift = (
        cart_uniq.groupby(["date", "iso_year", "iso_week", "iso_weekday", "shift"], dropna=False)["_palno"]
        .nunique()
        .rename("cartons_count")
        .reset_index()
    )
    pal_counts_shift = (
        pal_uniq.groupby(["date", "iso_year", "iso_week", "iso_weekday", "shift"], dropna=False)["_palno"]
        .nunique()
        .rename("pallets_count")
        .reset_index()
    )

    # Shift KPIs (sum rows for binhits, sum gross_tons, unique counts merged)
    shift_kpis = (
        df.groupby(["date", "iso_year", "iso_week", "iso_weekday", "shift"], dropna=False)
          .agg(binhits=("binhits", "sum"), gross_tons=("gross_tons", "sum"))
          .reset_index()
    )
    shift_kpis = shift_kpis.merge(cart_counts_shift, on=["date","iso_year","iso_week","iso_weekday","shift"], how="left")
    shift_kpis = shift_kpis.merge(pal_counts_shift, on=["date","iso_year","iso_week","iso_weekday","shift"], how="left")
    shift_kpis["cartons_count"] = shift_kpis["cartons_count"].fillna(0).astype(int)
    shift_kpis["pallets_count"] = shift_kpis["pallets_count"].fillna(0).astype(int)

    # Daily KPIs (combine both shifts)
    daily_kpis = (
        df.groupby(["date", "iso_year", "iso_week", "iso_weekday"], dropna=False)
          .agg(binhits=("binhits", "sum"), gross_tons=("gross_tons", "sum"))
          .reset_index()
    )
    cart_counts_daily = (
        cart_uniq.drop_duplicates(subset=["date", "_palno"])
                .groupby(["date","iso_year","iso_week","iso_weekday"], dropna=False)["_palno"]
                .nunique()
                .rename("cartons_count")
                .reset_index()
    )
    pal_counts_daily = (
        pal_uniq.drop_duplicates(subset=["date", "_palno"])
               .groupby(["date","iso_year","iso_week","iso_weekday"], dropna=False)["_palno"]
               .nunique()
               .rename("pallets_count")
               .reset_index()
    )
    daily_kpis = daily_kpis.merge(cart_counts_daily, on=["date","iso_year","iso_week","iso_weekday"], how="left")
    daily_kpis = daily_kpis.merge(pal_counts_daily, on=["date","iso_year","iso_week","iso_weekday"], how="left")
    daily_kpis["cartons_count"] = daily_kpis["cartons_count"].fillna(0).astype(int)
    daily_kpis["pallets_count"] = daily_kpis["pallets_count"].fillna(0).astype(int)

    # Vydejky (unikátní) – if "Výdej číslo" exists, count unique per day
    vydej_col = find_col(df, ["Výdej číslo", "Vydej cislo", "Vydej číslo "])
    if vydej_col is not None:
        daily_vydej = (
            df.groupby(["date","iso_year","iso_week","iso_weekday"], dropna=False)[vydej_col]
              .nunique()
              .rename("vydejky_unique")
              .reset_index()
        )
        shift_vydej = (
            df.groupby(["date","iso_year","iso_week","iso_weekday","shift"], dropna=False)[vydej_col]
              .nunique()
              .rename("vydejky_unique")
              .reset_index()
        )
        daily_kpis = daily_kpis.merge(daily_vydej, on=["date","iso_year","iso_week","iso_weekday"], how="left")
        shift_kpis = shift_kpis.merge(shift_vydej, on=["date","iso_year","iso_week","iso_weekday","shift"], how="left")
        daily_kpis["vydejky_unique"] = daily_kpis["vydejky_unique"].fillna(0).astype(int)
        shift_kpis["vydejky_unique"] = shift_kpis["vydejky_unique"].fillna(0).astype(int)
    else:
        daily_kpis["vydejky_unique"] = 0
        shift_kpis["vydejky_unique"] = 0

    # Audit: unique Paleta číslo by typ/class/year
    aud_base = uniq.drop_duplicates(subset=["iso_year","_palno"])
    audit = (
        aud_base.groupby(["iso_year", "_typ", "_class"], dropna=False)["_palno"]
                .nunique()
                .reset_index()
                .rename(columns={"_typ": "typ_obalu", "_class": "class", "_palno": "unique_paleta_cislo"})
    )

    return BuildResult(daily=daily_kpis, shift=shift_kpis, audit=audit)


def build_all(input_dir: Path, years: list[int], shift_scheme: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # TypObalu.csv is in parent folder (C:\\Kompletace\\TypObalu.csv) when input_dir is C:\\Kompletace\\Kompletace
    typ_map = load_typobalu_map(input_dir.parent)

    all_daily = []
    all_shift = []
    all_audit = []

    for y in years:
        f = input_dir / f"{y}.csv"
        if not f.exists():
            print(f"[WARN] Missing {f}")
            continue
        res = build_from_file(f, typ_map)
        all_daily.append(res.daily)
        all_shift.append(res.shift)
        all_audit.append(res.audit)

    if not all_daily:
        raise SystemExit("No input files processed.")

    df_daily = pd.concat(all_daily, ignore_index=True)
    df_shift = pd.concat(all_shift, ignore_index=True)
    df_audit = pd.concat(all_audit, ignore_index=True)

    # Ensure types
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_shift["date"] = pd.to_datetime(df_shift["date"])

    # Sort
    df_daily = df_daily.sort_values(["date"]).reset_index(drop=True)
    df_shift = df_shift.sort_values(["date","shift"]).reset_index(drop=True)

    df_daily.to_csv(out_dir / "packed_daily_kpis.csv", index=False, encoding="utf-8-sig")
    df_shift.to_csv(out_dir / "packed_shift_kpis.csv", index=False, encoding="utf-8-sig")
    df_audit.to_csv(out_dir / "packed_typobalu_audit.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote: {out_dir / 'packed_daily_kpis.csv'}")
    print(f"[OK] Wrote: {out_dir / 'packed_shift_kpis.csv'}")
    print(f"[OK] Wrote: {out_dir / 'packed_typobalu_audit.csv'}")
    if not typ_map:
        print("[INFO] TypObalu.csv not found or unreadable — using fallback BX/XCR => carton.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder with yearly Kompletace CSVs (e.g. C:\\Kompletace\\Kompletace)")
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--shift-scheme", default="12H", choices=["12H"], help="Only 12H supported here (07-19 / 19-07).")
    ap.add_argument("--out-dir", default="kpi", help="Output folder (default ./kpi)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    build_all(input_dir=input_dir, years=args.years, shift_scheme=args.shift_scheme, out_dir=out_dir)


if __name__ == "__main__":
    main()
