#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Warehouse Dashboard – Balení & Nakládky", layout="wide")


EXPECTED_FILES = {
    "packed_daily": "packed_daily_kpis.csv",
    "packed_shift": "packed_shift_kpis.csv",
    "loaded_daily": "loaded_daily_kpis.csv",
    "loaded_shift": "loaded_shift_kpis.csv",
}


@dataclass
class Sources:
    packed_daily: Optional[pd.DataFrame]
    packed_shift: Optional[pd.DataFrame]
    loaded_daily: Optional[pd.DataFrame]
    loaded_shift: Optional[pd.DataFrame]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def _coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "shift_start" in df.columns:
        df["shift_start"] = pd.to_datetime(df["shift_start"], errors="coerce")
    return df


def _weekday_short(wd: int) -> str:
    # 1..7
    return {1: "Po", 2: "Út", 3: "St", 4: "Čt", 5: "Pá", 6: "So", 7: "Ne"}.get(int(wd), str(wd))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}

    if "iso_weekday" in df.columns and "weekday" not in df.columns:
        rename_map["iso_weekday"] = "weekday"

    if "vydejky_unique" in df.columns and "orders_nunique" not in df.columns:
        rename_map["vydejky_unique"] = "orders_nunique"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "weekday" in df.columns:
        df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce")

    if "weekday" in df.columns and "weekday_name" not in df.columns:
        weekday_map = {1: "Po", 2: "Út", 3: "St", 4: "Čt", 5: "Pá", 6: "So", 7: "Ne"}
        df["weekday_name"] = df["weekday"].map(weekday_map)

    return df


def load_sources(base_dir: Path) -> Tuple[Sources, List[str]]:
    missing: List[str] = []

    def try_load(key: str) -> Optional[pd.DataFrame]:
        fn = EXPECTED_FILES[key]
        p = base_dir / fn
        if not p.exists():
            missing.append(fn)
            return None
        df = _read_csv(p)
        df = _coerce_date(df)
        df = _normalize_columns(df)
        return df

    src = Sources(
        packed_daily=try_load("packed_daily"),
        packed_shift=try_load("packed_shift"),
        loaded_daily=try_load("loaded_daily"),
        loaded_shift=try_load("loaded_shift"),
    )
    return src, missing


def available_metrics(df: pd.DataFrame) -> List[str]:
    skip = {"date", "shift_start", "shift", "shift_name", "iso_year", "iso_week", "weekday", "weekday_name"}
    out = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _xkey(iso_week: int, weekday: int) -> str:
    return f"KT{int(iso_week):02d} {_weekday_short(int(weekday))}"


def _week_day_grid(
    df: pd.DataFrame,
    metric: str,
    week_start: int,
    week_end: int,
    include_weekend: bool,
) -> pd.DataFrame:
    d = df.copy()
    d = d[d["iso_week"].between(week_start, week_end, inclusive="both")].copy()
    if not include_weekend:
        d = d[d["weekday"].between(1, 5, inclusive="both")].copy()

    g = d.groupby(["iso_year", "iso_week", "weekday"], dropna=False)[metric].sum().reset_index()
    g["x"] = g.apply(lambda r: _xkey(r["iso_week"], r["weekday"]), axis=1)

    xs = []
    for w in range(int(week_start), int(week_end) + 1):
        for wd in range(1, 8):
            if (not include_weekend) and wd > 5:
                continue
            xs.append(_xkey(w, wd))
    g["x"] = pd.Categorical(g["x"], categories=xs, ordered=True)
    g = g.sort_values(["x", "iso_year"]).reset_index(drop=True)
    return g


def forecast_for_weeks(
    df: pd.DataFrame,
    metric: str,
    eval_year: int,
    week_start: int,
    week_end: int,
    include_weekend: bool,
    lookback_weeks: int,
    mode: str,
    manual_adj_pct: float,
) -> pd.DataFrame:
    """
    Returns forecast points for eval_year for the selected iso_week range.

    Modes:
      - dynamic_rolling: baseline(iso_week, weekday) * rolling trend factor (Po–Pá)
      - static_ytd:      baseline(iso_week, weekday) * YTD strength factor (Po–Pá)
      - calendar_index:  (standard_day_hist * YTD strength) * index(iso_week, weekday)
      - hybrid:          calendar_index * short-term adjustment (dynamic/static)
    """
    d = df.copy()
    d = d[d["date"].notna()].copy()

    train_years = sorted([y for y in d["iso_year"].dropna().unique().astype(int).tolist() if int(y) < int(eval_year)])
    if not train_years:
        return pd.DataFrame()

    baseline = baseline_from_years(d, metric, train_years)

    df_eval = d[d["iso_year"] == eval_year].copy()
    if df_eval.empty:
        return pd.DataFrame()

    last_date = df_eval["date"].max()

    def _wk_filter(dd: pd.DataFrame, include_we: bool) -> pd.DataFrame:
        if include_we:
            return dd
        return dd[dd["weekday"].astype(int) <= 5].copy()

    def _po_pa(dd: pd.DataFrame) -> pd.DataFrame:
        return dd[dd["weekday"].astype(int).between(1, 5)].copy()

    def baseline_value(iso_week: int, weekday: int) -> float:
        v = baseline[(baseline["iso_week"] == iso_week) & (baseline["weekday"] == weekday)][metric]
        return float(v.mean()) if not v.empty else 0.0

    static_factor = 1.0
    ytd_eval = _po_pa(df_eval[df_eval["date"] <= last_date].copy())
    current_ytd = float(ytd_eval[metric].sum())

    merged = ytd_eval[["iso_week", "weekday", metric]].merge(
        baseline, on=["iso_week", "weekday"], how="left", suffixes=("", "_base")
    )
    expected_ytd = float(merged[f"{metric}_base"].fillna(0.0).sum())
    static_factor = (current_ytd / expected_ytd) if expected_ytd > 0 else 1.0

    def standard_day_for_year(y: int) -> float:
        dd = d[d["iso_year"] == y].copy()
        dd = _po_pa(dd)
        vals = dd[metric].astype(float)
        vals = vals[vals > 0]
        if vals.empty:
            return 0.0
        return float(vals.median())

    std_by_year = {int(y): standard_day_for_year(int(y)) for y in train_years}
    std_by_year = {y: v for y, v in std_by_year.items() if v > 0}

    if std_by_year:
        ys = sorted(std_by_year.keys())
        weights = {y: (i + 1) for i, y in enumerate(ys)}
        wsum = sum(weights.values())
        standard_day_hist = sum(std_by_year[y] * weights[y] for y in ys) / wsum
    else:
        standard_day_hist = 0.0

    idx_rows = []
    if standard_day_hist > 0 and std_by_year:
        for y in train_years:
            std = std_by_year.get(int(y), 0.0)
            if std <= 0:
                continue
            dd = d[d["iso_year"] == int(y)].copy()
            dd = dd.groupby(["iso_week", "weekday"], dropna=False).agg(val=(metric, "sum")).reset_index()
            dd["idx"] = dd["val"].astype(float) / float(std)
            dd = dd.replace([np.inf, -np.inf], np.nan).dropna(subset=["idx"])
            idx_rows.append(dd[["iso_week", "weekday", "idx"]])
        if idx_rows:
            idx_df = pd.concat(idx_rows, ignore_index=True)
            idx_profile = idx_df.groupby(["iso_week", "weekday"], dropna=False)["idx"].mean().reset_index()
        else:
            idx_profile = pd.DataFrame(columns=["iso_week", "weekday", "idx"])
    else:
        idx_profile = pd.DataFrame(columns=["iso_week", "weekday", "idx"])

    def idx_value(iso_week: int, weekday: int) -> float:
        v = idx_profile[(idx_profile["iso_week"] == iso_week) & (idx_profile["weekday"] == weekday)]["idx"]
        return float(v.mean()) if not v.empty else 0.0

    strength_factor = 1.0
    if current_ytd > 0:
        pairs = ytd_eval[["iso_week", "weekday"]].drop_duplicates()
        hist_sums = []
        for y in train_years:
            dd = d[d["iso_year"] == int(y)].copy()
            dd = _po_pa(dd)
            dd = dd.merge(pairs, on=["iso_week", "weekday"], how="inner")
            hist_sums.append(float(dd[metric].sum()))
        hist_mean = float(np.mean([s for s in hist_sums if s > 0])) if hist_sums else 0.0
        strength_factor = (current_ytd / hist_mean) if hist_mean > 0 else 1.0

    calendar_base = standard_day_hist * strength_factor if standard_day_hist > 0 else 0.0

    rows = []
    for w in range(int(week_start), int(week_end) + 1):
        for wd in range(1, 8):
            if (not include_weekend) and wd > 5:
                continue

            try:
                dt = datetime.date.fromisocalendar(int(eval_year), int(w), int(wd))
            except ValueError:
                continue
            day = pd.to_datetime(dt)

            manual_factor = 1.0 + (manual_adj_pct / 100.0)

            if mode == "dynamic_rolling":
                base = baseline_value(w, wd)
                factor = dynamic_trend_factor(df_eval, baseline, metric, day, lookback_weeks)
                yhat = base * factor * manual_factor

            elif mode == "static_ytd":
                base = baseline_value(w, wd)
                factor = static_factor
                yhat = base * factor * manual_factor

            elif mode == "calendar_index":
                base = idx_value(w, wd)
                factor = strength_factor
                yhat = calendar_base * base * manual_factor

            elif mode == "hybrid":
                idxv = idx_value(w, wd)
                yhat = calendar_base * idxv

                dyn = dynamic_trend_factor(df_eval, baseline, metric, day, lookback_weeks)
                adj = (dyn / static_factor) if static_factor > 0 else dyn
                yhat = yhat * adj * manual_factor
                factor = adj

                base = idxv

            else:
                base = baseline_value(w, wd)
                factor = static_factor
                yhat = base * factor * manual_factor

            rows.append(
                {
                    "iso_year": int(eval_year),
                    "iso_week": int(w),
                    "weekday": int(wd),
                    "x": _xkey(w, wd),
                    "forecast": float(yhat),
                    "baseline": float(base),
                    "trend_factor": float(factor),
                    "manual_adj_pct": float(manual_adj_pct),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    xs = []
    for w in range(int(week_start), int(week_end) + 1):
        for wd in range(1, 8):
            if (not include_weekend) and wd > 5:
                continue
            xs.append(_xkey(w, wd))
    out["x"] = pd.Categorical(out["x"], categories=xs, ordered=True)
    out = out.sort_values(["x"]).reset_index(drop=True)
    return out


def filter_range(df: pd.DataFrame, date_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    d = df.copy()
    d = d[d[date_col].notna()]
    d = d[(d[date_col] >= start) & (d[date_col] <= end)]
    return d


def weekly_totals(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["iso_year", "iso_week"], dropna=False)[metric].sum().reset_index()
    g = g.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    return g


def baseline_from_years(df: pd.DataFrame, metric: str, train_years: List[int]) -> pd.DataFrame:
    d = df[df["iso_year"].isin(train_years)].copy()
    b = d.groupby(["iso_week", "weekday"], dropna=False)[metric].mean().reset_index()
    return b


def dynamic_trend_factor(df_eval: pd.DataFrame, baseline: pd.DataFrame, metric: str, dt: pd.Timestamp, lookback_weeks: int) -> float:
    if lookback_weeks <= 0:
        return 1.0

    start = dt - pd.Timedelta(days=7 * lookback_weeks)
    end = dt - pd.Timedelta(days=1)
    hist = df_eval[(df_eval["date"] >= start) & (df_eval["date"] <= end)].copy()
    if hist.empty:
        return 1.0

    current_sum = float(hist[metric].sum())

    merged = hist[["iso_week", "weekday", metric]].merge(
        baseline, on=["iso_week", "weekday"], how="left", suffixes=("", "_base")
    )
    expected_sum = float(merged[f"{metric}_base"].fillna(0.0).sum())

    if expected_sum <= 0:
        return 1.0
    return current_sum / expected_sum


def forecast_dates(
    df: pd.DataFrame,
    metric: str,
    eval_year: int,
    horizon_days: int,
    lookback_weeks: int,
    mode: str,
    manual_adj_pct: float,
) -> Tuple[pd.DataFrame, float]:
    """
    mode:
      - "static_ytd": one factor for whole horizon based on YTD vs baseline(YTD)
      - "dynamic_rolling": factor computed per day from previous N weeks
    """
    d = df.copy()
    d = d[d["date"].notna()].copy()

    train_years = sorted([y for y in d["iso_year"].dropna().unique().tolist() if int(y) != int(eval_year)])
    if not train_years:
        return pd.DataFrame(), 1.0

    df_eval = d[d["iso_year"] == eval_year].copy()
    if df_eval.empty:
        return pd.DataFrame(), 1.0

    last_date = df_eval["date"].max()
    baseline = baseline_from_years(d, metric, train_years)

    def baseline_value(iso_week: int, weekday: int) -> float:
        v = baseline[(baseline["iso_week"] == iso_week) & (baseline["weekday"] == weekday)][metric]
        return float(v.mean()) if not v.empty else 0.0

    static_factor = 1.0
    if mode == "static_ytd":
        ytd = df_eval[df_eval["date"] <= last_date].copy()
        current_ytd = float(ytd[metric].sum())
        merged = ytd[["iso_week", "weekday", metric]].merge(
            baseline, on=["iso_week", "weekday"], how="left", suffixes=("", "_base")
        )
        expected_ytd = float(merged[f"{metric}_base"].fillna(0.0).sum())
        static_factor = (current_ytd / expected_ytd) if expected_ytd > 0 else 1.0

    rows = []
    start = last_date + pd.Timedelta(days=1)
    for i in range(int(horizon_days)):
        day = start + pd.Timedelta(days=i)
        iso = day.isocalendar()
        wk = int(iso.week)
        wd = int(iso.weekday)
        base = baseline_value(wk, wd)

        if mode == "dynamic_rolling":
            factor = dynamic_trend_factor(df_eval, baseline, metric, day, lookback_weeks)
        else:
            factor = static_factor

        manual_factor = 1.0 + (manual_adj_pct / 100.0)
        yhat = base * factor * manual_factor
        rows.append(
            {
                "date": day,
                "iso_week": wk,
                "weekday": wd,
                "baseline": base,
                "trend_factor": factor,
                "manual_adj_pct": manual_adj_pct,
                "forecast": yhat,
            }
        )

    out = pd.DataFrame(rows)
    return out, float(static_factor)


def backtest_year(
    df: pd.DataFrame,
    metric: str,
    eval_year: int,
    lookback_weeks: int,
    mode: str,
    manual_adj_pct: float,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    d = df.copy()
    d = d[d["date"].notna()].copy()

    train_years = sorted([y for y in d["iso_year"].dropna().unique().tolist() if int(y) < int(eval_year)])
    if not train_years:
        return pd.DataFrame()

    df_eval = d[d["iso_year"] == eval_year].copy()
    if df_eval.empty:
        return pd.DataFrame()

    if start_date is None:
        start_date = df_eval["date"].min()
    if end_date is None:
        end_date = df_eval["date"].max()

    df_eval = df_eval[(df_eval["date"] >= start_date) & (df_eval["date"] <= end_date)].copy()
    baseline = baseline_from_years(d, metric, train_years)

    static_factor = 1.0
    if mode == "static_ytd":
        current_sum = float(df_eval[metric].sum())
        merged = df_eval[["iso_week", "weekday", metric]].merge(
            baseline, on=["iso_week", "weekday"], how="left", suffixes=("", "_base")
        )
        expected_sum = float(merged[f"{metric}_base"].fillna(0.0).sum())
        static_factor = (current_sum / expected_sum) if expected_sum > 0 else 1.0

    preds = []
    for _, r in df_eval.sort_values("date").iterrows():
        day = pd.to_datetime(r["date"])
        wk = int(r["iso_week"])
        wd = int(r["weekday"])
        base_v = baseline[(baseline["iso_week"] == wk) & (baseline["weekday"] == wd)][metric]
        base = float(base_v.mean()) if not base_v.empty else 0.0

        if mode == "dynamic_rolling":
            factor = dynamic_trend_factor(df_eval, baseline, metric, day, lookback_weeks)
        else:
            factor = static_factor

        manual_factor = 1.0 + (manual_adj_pct / 100.0)
        preds.append(
            {
                "date": day,
                "forecast": base * factor * manual_factor,
                "trend_factor": factor,
                "baseline": base,
                "manual_adj_pct": manual_adj_pct,
            }
        )

    out = pd.DataFrame(preds)
    out = out.merge(df_eval[["date", metric]], on="date", how="left")
    out["error"] = out[metric] - out["forecast"]
    out["ape"] = np.where(out[metric] > 0, np.abs(out["error"]) / out[metric], np.nan)
    return out


def plot_multi_year_lines_weekly(df: pd.DataFrame, metric: str, anchor_week: int, window: int, title: str) -> None:
    weeks = list(range(anchor_week, min(53, anchor_week + window) + 1))
    w = df[df["iso_week"].isin(weeks)].copy()
    if w.empty:
        st.info("V tomhle rozsahu nejsou data.")
        return
    wt = weekly_totals(w, metric)
    fig = px.line(wt, x="iso_week", y=metric, color="iso_year", markers=True, title=title)
    fig.update_traces(connectgaps=False)
    st.plotly_chart(fig, width="stretch")


def tab_balení(src: Sources) -> None:
    st.subheader("Balení (Kompletace) 📦")

    if src.packed_daily is None or src.packed_daily.empty:
        st.error("Chybí packed_daily_kpis.csv")
        return

    shift_col = None
    if src.packed_shift is not None and not src.packed_shift.empty:
        if "shift_name" in src.packed_shift.columns:
            shift_col = "shift_name"
        elif "shift" in src.packed_shift.columns:
            shift_col = "shift"

    shift_opts = ["all"]
    if shift_col:
        shift_opts += sorted(src.packed_shift[shift_col].dropna().unique().astype(str).tolist())

    shift_sel = st.selectbox(
        "Směna",
        shift_opts,
        index=0,
        key="pk_shift_sel",
        format_func=lambda x: "Celkem" if x == "all" else ("Denní" if x == "day" else ("Noční" if x == "night" else x)),
    )

    if shift_sel == "all":
        df = src.packed_daily.copy()
    else:
        df = src.packed_shift.copy()
        df = df[df[shift_col].astype(str) == shift_sel].copy()

        group_cols = [c for c in ["date", "iso_year", "iso_week", "weekday", "weekday_name"] if c in df.columns]
        df = df.groupby(group_cols, dropna=False).sum(numeric_only=True).reset_index()

    if "date" not in df.columns:
        st.error("V packed_daily_kpis.csv chybí 'date'.")
        return

    metrics_all = available_metrics(df)
    if not metrics_all:
        st.error("Nenašel jsem žádné číselné metriky.")
        return

    years_all = sorted(df["iso_year"].dropna().unique().astype(int).tolist())
    if not years_all:
        st.error("Nenalezeny roky (iso_year) v datech.")
        return

    eval_year = max(years_all)

    metric_labels = {
        "binhits": "Binhits",
        "gross_tons": "GW (t)",
        "cartons_count": "Kartony",
        "pallets_count": "Palety",
        "orders_nunique": "Výdejky (unikátní)",
    }

    default_metrics = [m for m in ["binhits", "gross_tons", "pallets_count", "cartons_count", "orders_nunique"] if m in metrics_all]
    if not default_metrics:
        default_metrics = metrics_all[:1]

    st.markdown("### Pohled: vybrané KT → Po–Ne (roky vedle sebe + predikce)")

    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
    with c1:
        metrics_sel = st.multiselect(
            "Metriky v grafech (zobrazí se pod sebou)",
            options=metrics_all,
            default=default_metrics,
            key="wk_metrics_sel",
            format_func=lambda m: metric_labels.get(m, m),
        )
    with c2:
        week_start = st.number_input("KT od", min_value=1, max_value=53, value=6, step=1, key="wk_start")
    with c3:
        week_end = st.number_input("KT do", min_value=1, max_value=53, value=10, step=1, key="wk_end")
    with c4:
        include_weekend = st.checkbox(
            "Víkendy?",
            value=False,
            key="wk_weekend",
            help="Kompletace víkend občas má (protažení). Predikce se dá zobrazit i na víkendové dny.",
        )

    c5, c6 = st.columns([4, 3])
    with c5:
        years_sel = st.multiselect(
            "Roky v grafu",
            options=years_all,
            default=years_all,
            key="wk_years_sel",
            help="Zúží graf, ať není přeplácaný. Doporučení: srovnej 2–3 roky + aktuál.",
        )
    with c6:
        show_forecast = st.checkbox(f"Zobrazit predikci ({eval_year})", value=True, key="wk_show_fc")

    if week_end < week_start:
        st.warning("KT do je menší než KT od – prohodím je.")
        week_start, week_end = week_end, week_start

    if not metrics_sel:
        st.info("Vyber aspoň jednu metriku.")
        return

    if "wk_metrics_order" not in st.session_state:
        st.session_state["wk_metrics_order"] = list(metrics_sel)

    order = [m for m in st.session_state["wk_metrics_order"] if m in metrics_sel]
    for mtr in metrics_sel:
        if mtr not in order:
            order.append(mtr)
    st.session_state["wk_metrics_order"] = order
    metrics_ordered = order

    st.markdown("#### Pořadí metrik")
    cpo1, cpo2, cpo3, cpo4 = st.columns([3, 1, 1, 3])
    with cpo1:
        picked = st.selectbox("Vybraná metrika", metrics_ordered, format_func=lambda m: metric_labels.get(m, m), key="wk_order_pick")
    with cpo2:
        if st.button("⬆️ nahoru", key="wk_order_up"):
            i = metrics_ordered.index(picked)
            if i > 0:
                metrics_ordered[i - 1], metrics_ordered[i] = metrics_ordered[i], metrics_ordered[i - 1]
                st.session_state["wk_metrics_order"] = metrics_ordered
    with cpo3:
        if st.button("⬇️ dolů", key="wk_order_down"):
            i = metrics_ordered.index(picked)
            if i < len(metrics_ordered) - 1:
                metrics_ordered[i + 1], metrics_ordered[i] = metrics_ordered[i], metrics_ordered[i + 1]
                st.session_state["wk_metrics_order"] = metrics_ordered
    with cpo4:
        st.caption("Tip: vyber metriku a posuň ji nahoru/dolů. Pořadí se použije pro grafy pod sebou.")

    if eval_year not in years_sel:
        years_sel = years_sel + [eval_year]

    c7, c8, c9 = st.columns([2, 3, 2])
    with c7:
        trend_mode = st.selectbox(
            "Trend mód",
            ["hybrid", "calendar_index", "dynamic_rolling", "static_ytd"],
            index=0,
            key="wk_fc_mode",
            format_func=lambda x: (
                "Hybrid (kalendář × krátkodobý trend)"
                if x == "hybrid"
                else (
                    "Kalendář (index KT×den + síla roku)"
                    if x == "calendar_index"
                    else ("Dynamický (rolling týdny)" if x == "dynamic_rolling" else "Statický (okno/YTD)")
                )
            ),
            disabled=not show_forecast,
        )
    with c8:
        lookback_weeks = st.slider("Trend okno (týdny)", 2, 26, 8, key="wk_fc_weeks", disabled=not show_forecast)
    with c9:
        manual_adj = st.slider("Korekce predikce (%)", -30, 30, 0, key="wk_fc_adj", disabled=not show_forecast)

    grids: Dict[str, pd.DataFrame] = {}
    fcs: Dict[str, pd.DataFrame] = {}

    for metric in metrics_ordered:
        st.markdown(f"#### {metric_labels.get(metric, metric)}")

        grid = _week_day_grid(df[df["iso_year"].isin(years_sel)], metric, int(week_start), int(week_end), include_weekend)
        grids[metric] = grid

        if grid.empty:
            st.info(f"Pro metriku {metric_labels.get(metric, metric)} nejsou data v tomto rozsahu.")
            continue

        fc = pd.DataFrame()
        forecast_label = f"Predikce {eval_year}"
        if show_forecast:
            fc = forecast_for_weeks(
                df,
                metric=metric,
                eval_year=eval_year,
                week_start=int(week_start),
                week_end=int(week_end),
                include_weekend=include_weekend,
                lookback_weeks=int(lookback_weeks),
                mode=trend_mode,
                manual_adj_pct=float(manual_adj),
            )
        fcs[metric] = fc

        long_actual = grid.rename(columns={metric: "value"}).copy()
        long_actual["series"] = long_actual["iso_year"].astype(int).astype(str)
        long = long_actual[["x", "value", "series"]].copy()

        if show_forecast and (not fc.empty):
            fc_long = fc.rename(columns={"forecast": "value"}).copy()
            fc_long["series"] = forecast_label
            long = pd.concat([long, fc_long[["x", "value", "series"]]], ignore_index=True)

        fig = px.line(long, x="x", y="value", color="series", markers=True, title=None)
        fig.update_traces(connectgaps=False)
        fig.update_layout(xaxis_title="KT + den", yaxis_title=metric_labels.get(metric, metric), legend_title_text="Rok / predikce")

        for tr in fig.data:
            name = str(tr.name)
            if name == forecast_label:
                tr.update(line=dict(width=4, dash="dash"))
            elif name == str(eval_year):
                tr.update(line=dict(width=4))
            else:
                tr.update(opacity=0.45, line=dict(width=2))

        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Detail (tabulka + odchylky)")

    detail_metric = st.selectbox(
        "Detail metrika",
        options=metrics_ordered,
        index=0,
        key="wk_detail_metric",
        format_func=lambda m: metric_labels.get(m, m),
        help="Tabulka a odchylky se vždy vztahují k této metrice.",
    )

    grid = grids.get(detail_metric, pd.DataFrame())
    fc = fcs.get(detail_metric, pd.DataFrame())
    forecast_label = f"Predikce {eval_year}"

    if grid is None or grid.empty:
        st.info("Pro detail metriku nejsou data v tomto rozsahu.")
        return

    pivot = grid.pivot_table(index="x", columns="iso_year", values=detail_metric, aggfunc="sum", fill_value=0).reset_index()

    years_sel_int = [int(y) for y in years_sel]
    keep_years_existing = [y for y in years_sel_int if y in pivot.columns]
    cols = ["x"] + keep_years_existing
    pivot = pivot[cols]

    if show_forecast and (fc is not None) and (not fc.empty):
        p_fc = fc.pivot_table(index="x", values="forecast", aggfunc="sum", fill_value=0).rename(columns={"forecast": forecast_label}).reset_index()
        pivot = pivot.merge(p_fc, on="x", how="left")

        if eval_year in pivot.columns:
            pivot["Δ (2026 - predikce)"] = pivot[eval_year] - pivot[forecast_label]
            pivot["Δ %"] = np.where(pivot[forecast_label] != 0, pivot["Δ (2026 - predikce)"] / pivot[forecast_label], np.nan)

    st.dataframe(pivot, width="stretch")

    if show_forecast and (fc is not None) and (not fc.empty) and (eval_year in keep_years_existing):
        st.markdown("#### Odchylka 2026 vs predikce (v tomhle výřezu)")
        actual_2026 = grid[grid["iso_year"] == eval_year][["x", detail_metric]].rename(columns={detail_metric: "actual"})
        fc_x = fc[["x", "forecast"]].rename(columns={"forecast": "forecast"})
        cmp = actual_2026.merge(fc_x, on="x", how="inner")
        if not cmp.empty:
            cmp["diff"] = cmp["actual"] - cmp["forecast"]
            cmp["diff_pct"] = np.where(cmp["forecast"] != 0, cmp["diff"] / cmp["forecast"], np.nan)
            mape = float(np.nanmean(np.abs(cmp["diff_pct"]))) if cmp["diff_pct"].notna().any() else np.nan

            c10, c11 = st.columns([1, 3])
            with c10:
                st.metric("MAPE", f"{mape:.1%}" if not np.isnan(mape) else "n/a")
            with c11:
                worst = cmp.sort_values(by="diff_pct", key=lambda s: np.abs(s), ascending=False).head(5)
                worst_show = worst[["x", "actual", "forecast", "diff", "diff_pct"]].copy()
                worst_show["diff_pct"] = worst_show["diff_pct"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
                st.write("Největší odchylky (top 5):")
                st.dataframe(worst_show, width="stretch")
        else:
            st.info("Nemám překryv (2026 × predikce) v tomto výřezu.")


def tab_nakladky(src: Sources) -> None:
    st.subheader("Nakládky (Výdeje) 🚚")

    if src.loaded_daily is None or src.loaded_daily.empty:
        st.error("Chybí loaded_daily_kpis.csv")
        return

    shift_opts = ["all"]
    if src.loaded_shift is not None and not src.loaded_shift.empty and "shift_name" in src.loaded_shift.columns:
        shift_opts += sorted(src.loaded_shift["shift_name"].dropna().unique().astype(str).tolist())

    shift_sel = st.selectbox(
        "Směna",
        shift_opts,
        index=0,
        key="ld_shift_sel",
        format_func=lambda x: "Celkem" if x == "all" else ("Ranní" if x == "morning" else ("Odpolední" if x == "afternoon" else x)),
    )

    if shift_sel == "all":
        df = src.loaded_daily.copy()
    else:
        df = src.loaded_shift.copy()
        df = df[df["shift_name"].astype(str) == shift_sel].copy()
        df = df.groupby(["date", "iso_year", "iso_week", "weekday", "weekday_name"], dropna=False).sum(numeric_only=True).reset_index()

    if "date" not in df.columns:
        st.error("V loaded_daily_kpis.csv chybí 'date'.")
        return

    preferred = ["trips_total", "trips_export", "trips_europe", "containers_count", "gross_tons"]
    metrics_all = [m for m in preferred if m in df.columns]

    missing = [m for m in preferred if m not in df.columns]
    if missing:
        st.warning(
            "Chybí metriky v loaded_daily_kpis.csv: "
            + ", ".join(missing)
            + ". Pokud čekáš, že tam mají být, přegeneruj KPI skriptem build_loaded_kpis_from_vydeje_v3.py."
        )

    if not metrics_all:
        st.error("Nenalezl jsem očekávané metriky pro nakládky.")
        return

    years_all = sorted(df["iso_year"].dropna().unique().astype(int).tolist())
    eval_year = max(years_all)

    metric_labels = {
        "trips_total": "Trips celkem",
        "trips_export": "Trips export",
        "trips_europe": "Trips Evropa",
        "containers_count": "Kontejnery",
        "gross_tons": "GW (t)",
    }

    default_metrics = [m for m in ["trips_total", "gross_tons", "containers_count"] if m in metrics_all]
    if not default_metrics:
        default_metrics = metrics_all[:1]

    st.markdown("### Pohled: vybrané KT → Po–Pá (roky vedle sebe + predikce)")

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        metrics_sel = st.multiselect(
            "Metriky v grafech (zobrazí se pod sebou)",
            options=metrics_all,
            default=default_metrics,
            key="ld_metrics_sel",
            format_func=lambda m: metric_labels.get(m, m),
        )
    with c2:
        week_start = st.number_input("KT od", 1, 53, 6, key="ld_start")
    with c3:
        week_end = st.number_input("KT do", 1, 53, 10, key="ld_end")

    c4, c5 = st.columns([4, 3])
    with c4:
        years_sel = st.multiselect(
            "Roky v grafu",
            options=years_all,
            default=years_all,
            key="ld_years_sel",
        )
    with c5:
        show_forecast = st.checkbox(f"Zobrazit predikci ({eval_year})", value=True, key="ld_show_fc")

    if week_end < week_start:
        week_start, week_end = week_end, week_start

    if not metrics_sel:
        st.info("Vyber aspoň jednu metriku.")
        return

    if "ld_metrics_order" not in st.session_state:
        st.session_state["ld_metrics_order"] = list(metrics_sel)

    order = [m for m in st.session_state["ld_metrics_order"] if m in metrics_sel]
    for mtr in metrics_sel:
        if mtr not in order:
            order.append(mtr)
    st.session_state["ld_metrics_order"] = order
    metrics_ordered = order

    st.markdown("#### Pořadí metrik")
    cpo1, cpo2, cpo3 = st.columns([3, 1, 1])
    with cpo1:
        picked = st.selectbox(
            "Vybraná metrika",
            metrics_ordered,
            format_func=lambda m: metric_labels.get(m, m),
            key="ld_order_pick",
        )
    with cpo2:
        if st.button("⬆️", key="ld_up"):
            i = metrics_ordered.index(picked)
            if i > 0:
                metrics_ordered[i - 1], metrics_ordered[i] = metrics_ordered[i], metrics_ordered[i - 1]
                st.session_state["ld_metrics_order"] = metrics_ordered
    with cpo3:
        if st.button("⬇️", key="ld_down"):
            i = metrics_ordered.index(picked)
            if i < len(metrics_ordered) - 1:
                metrics_ordered[i + 1], metrics_ordered[i] = metrics_ordered[i], metrics_ordered[i + 1]
                st.session_state["ld_metrics_order"] = metrics_ordered

    if eval_year not in years_sel:
        years_sel = years_sel + [eval_year]

    c6, c7, c8 = st.columns([2, 3, 2])
    with c6:
        trend_mode = st.selectbox(
            "Trend mód",
            ["hybrid", "calendar_index", "dynamic_rolling", "static_ytd"],
            index=0,
            key="ld_fc_mode",
            disabled=not show_forecast,
            format_func=lambda x: (
                "Hybrid (kalendář × krátkodobý trend)"
                if x == "hybrid"
                else (
                    "Kalendář (index KT×den + síla roku)"
                    if x == "calendar_index"
                    else ("Dynamický (rolling týdny)" if x == "dynamic_rolling" else "Statický (okno/YTD)")
                )
            ),
        )
    with c7:
        lookback_weeks = st.slider("Trend okno (týdny)", 2, 26, 8, key="ld_fc_weeks", disabled=not show_forecast)
    with c8:
        manual_adj = st.slider("Korekce predikce (%)", -30, 30, 0, key="ld_fc_adj", disabled=not show_forecast)

    grids, fcs = {}, {}
    include_weekend = False

    for metric in metrics_ordered:
        st.markdown(f"#### {metric_labels.get(metric, metric)}")

        grid = _week_day_grid(df[df["iso_year"].isin(years_sel)], metric, int(week_start), int(week_end), include_weekend)
        grids[metric] = grid

        if grid.empty:
            st.info("Bez dat v tomto výřezu.")
            continue

        fc = pd.DataFrame()
        forecast_label = f"Predikce {eval_year}"
        if show_forecast:
            fc = forecast_for_weeks(
                df,
                metric=metric,
                eval_year=eval_year,
                week_start=int(week_start),
                week_end=int(week_end),
                include_weekend=include_weekend,
                lookback_weeks=int(lookback_weeks),
                mode=trend_mode,
                manual_adj_pct=float(manual_adj),
            )
        fcs[metric] = fc

        long_actual = grid.rename(columns={metric: "value"}).copy()
        long_actual["series"] = long_actual["iso_year"].astype(int).astype(str)
        long = long_actual[["x", "value", "series"]]

        if show_forecast and (not fc.empty):
            fc_long = fc.rename(columns={"forecast": "value"}).copy()
            fc_long["series"] = forecast_label
            long = pd.concat([long, fc_long[["x", "value", "series"]]], ignore_index=True)

        fig = px.line(long, x="x", y="value", color="series", markers=True)
        fig.update_traces(connectgaps=False)
        for tr in fig.data:
            name = str(tr.name)
            if name == forecast_label:
                tr.update(line=dict(width=4, dash="dash"))
            elif name == str(eval_year):
                tr.update(line=dict(width=4))
            else:
                tr.update(opacity=0.45, line=dict(width=2))

        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Detail (tabulka + odchylky)")

    detail_metric = st.selectbox(
        "Detail metrika",
        options=metrics_ordered,
        index=0,
        key="ld_detail_metric",
        format_func=lambda m: metric_labels.get(m, m),
    )

    grid = grids.get(detail_metric, pd.DataFrame())
    fc = fcs.get(detail_metric, pd.DataFrame())
    forecast_label = f"Predikce {eval_year}"

    if grid.empty:
        st.info("Pro detail metriku nejsou data.")
        return

    pivot = grid.pivot_table(index="x", columns="iso_year", values=detail_metric, aggfunc="sum", fill_value=0).reset_index()
    years_sel_int = [int(y) for y in years_sel]
    keep_years_existing = [y for y in years_sel_int if y in pivot.columns]
    pivot = pivot[["x"] + keep_years_existing]

    if show_forecast and (not fc.empty):
        p_fc = fc.pivot_table(index="x", values="forecast", aggfunc="sum", fill_value=0).rename(columns={"forecast": forecast_label}).reset_index()
        pivot = pivot.merge(p_fc, on="x", how="left")

        if eval_year in pivot.columns:
            pivot["Δ (2026 - predikce)"] = pivot[eval_year] - pivot[forecast_label]
            pivot["Δ %"] = np.where(pivot[forecast_label] != 0, pivot["Δ (2026 - predikce)"] / pivot[forecast_label], np.nan)

    st.dataframe(pivot, width="stretch")

    if show_forecast and (not fc.empty) and (eval_year in keep_years_existing):
        st.markdown("#### Odchylka 2026 vs predikce (v tomhle výřezu)")
        actual_2026 = grid[grid["iso_year"] == eval_year][["x", detail_metric]].rename(columns={detail_metric: "actual"})
        fc_x = fc[["x", "forecast"]].rename(columns={"forecast": "forecast"})
        cmp = actual_2026.merge(fc_x, on="x", how="inner")
        if not cmp.empty:
            cmp["diff"] = cmp["actual"] - cmp["forecast"]
            cmp["diff_pct"] = np.where(cmp["forecast"] != 0, cmp["diff"] / cmp["forecast"], np.nan)
            mape = float(np.nanmean(np.abs(cmp["diff_pct"]))) if cmp["diff_pct"].notna().any() else np.nan

            c10, c11 = st.columns([1, 3])
            with c10:
                st.metric("MAPE", f"{mape:.1%}" if not np.isnan(mape) else "n/a")
            with c11:
                worst = cmp.sort_values(by="diff_pct", key=lambda s: np.abs(s), ascending=False).head(5)
                worst_show = worst[["x", "actual", "forecast", "diff", "diff_pct"]].copy()
                worst_show["diff_pct"] = worst_show["diff_pct"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
                st.write("Největší odchylky (top 5):")
                st.dataframe(worst_show, width="stretch")
        else:
            st.info("Nemám překryv (2026 × predikce) v tomto výřezu.")


def tab_predikce(src: Sources) -> None:
    st.subheader("Predikce (dopředu) 📈")

    options: Dict[str, Optional[pd.DataFrame]] = {
        "Balení (Kompletace) – daily": src.packed_daily,
        "Nakládky (Výdeje) – daily": src.loaded_daily,
    }
    choice = st.selectbox("Zdroj", list(options.keys()), key="pred_source")
    df = options[choice]
    if df is None or df.empty:
        st.warning("Chybí data pro predikci.")
        return

    metrics = available_metrics(df)
    if not metrics:
        st.error("Nenašel jsem žádné číselné metriky.")
        return

    metric = st.selectbox("Metrika", metrics, index=(metrics.index("binhits") if "binhits" in metrics else 0), key="pred_metric")
    years = sorted(df["iso_year"].dropna().unique().astype(int).tolist())
    eval_year = max(years)

    c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
    with c1:
        horizon = st.slider("Horizont (dní dopředu)", 3, 14, 5, key="pred_h_days")
    with c2:
        mode = st.selectbox(
            "Trend mód",
            ["hybrid", "calendar_index", "dynamic_rolling", "static_ytd"],
            index=0,
            key="pred_mode",
            format_func=lambda x: (
                "Hybrid (kalendář × krátkodobý trend)"
                if x == "hybrid"
                else (
                    "Kalendář (index KT×den + síla roku)"
                    if x == "calendar_index"
                    else ("Dynamický (rolling týdny)" if x == "dynamic_rolling" else "Statický (okno/YTD)")
                )
            ),
        )
    with c3:
        lookback_weeks = st.slider("Trend okno (týdny)", 2, 26, 8, key="pred_lb_weeks")
    with c4:
        manual_adj = st.slider("Korekce predikce (%)", -30, 30, 0, key="pred_adj")

    fc, static_factor = forecast_dates(
        df,
        metric,
        eval_year=eval_year,
        horizon_days=horizon,
        lookback_weeks=lookback_weeks,
        mode=mode,
        manual_adj_pct=float(manual_adj),
    )
    if fc.empty:
        st.info("Není dost dat pro predikci.")
        return

    df_eval = df[df["iso_year"] == eval_year].copy().sort_values("date")
    tail = df_eval.tail(60)[["date", metric]].rename(columns={metric: "value"}).assign(series="Historie")
    fut = fc[["date", "forecast"]].rename(columns={"forecast": "value"}).assign(series="Predikce")

    plot_df = pd.concat([tail, fut], ignore_index=True)
    fig = px.line(plot_df, x="date", y="value", color="series", markers=True, title=f"{metric} – {eval_year}: historie + predikce")
    fig.update_traces(connectgaps=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Forecast tabulka")
    st.dataframe(fc, width="stretch")


def main() -> None:
    st.title("Warehouse Dashboard – Balení & Nakládky")

    with st.sidebar:
        st.header("Data")
        base = st.text_input("Složka s KPI CSV", value=".")
        base_dir = Path(base).resolve()
        st.caption("Očekává: packed_daily_kpis.csv, packed_shift_kpis.csv, loaded_daily_kpis.csv, loaded_shift_kpis.csv")

    src, missing = load_sources(base_dir)

    if missing:
        st.warning("Chybí některé KPI soubory. Dashboard poběží, ale část pohledů bude prázdná.\n\n**Chybí:** " + ", ".join(missing))

    tab1, tab2, tab3 = st.tabs(["Balení", "Nakládky", "Predikce dopředu"])

    with tab1:
        tab_balení(src)

    with tab2:
        tab_nakladky(src)

    with tab3:
        tab_predikce(src)


if __name__ == "__main__":
    main()
