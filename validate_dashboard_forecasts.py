from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit_app as app  # noqa: E402


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fmt_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1%}"


def _fmt_num(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def _suite_summary(name: str, suite: Dict[str, object]) -> Dict[str, object]:
    scores = suite["scores"].copy()
    _check(not scores.empty, f"{name}: missing backtest scores")
    best = scores.sort_values(["wape", "mae"], na_position="last").iloc[0]
    smart = scores[scores["model"] == "10 Smart blend"]
    smart_wape = float(smart.iloc[0]["wape"]) if not smart.empty else float("nan")
    best_wape = float(best["wape"])
    _check(len(suite["future"]) > 0, f"{name}: future forecast is empty")
    _check(best_wape < 0.55, f"{name}: best WAPE too high ({best_wape:.3f})")
    _check(
        pd.isna(smart_wape) or smart_wape <= best_wape + 1e-9,
        f"{name}: smart blend is worse than best model ({smart_wape:.3f} > {best_wape:.3f})",
    )
    return {
        "case": name,
        "best_model": str(best["model"]),
        "best_wape": best_wape,
        "smart_blend_wape": smart_wape,
        "drivers_selected": int(len(suite.get("drivers", pd.DataFrame()))),
    }


def main() -> None:
    src, missing = app.load_sources(str(BASE_DIR))
    exog = app.load_exog_bundle(str(BASE_DIR))

    _check(not missing, f"Missing source files: {missing}")
    _check(src.packed_daily is not None and not src.packed_daily.empty, "packed_daily missing")
    _check(src.loaded_daily is not None and not src.loaded_daily.empty, "loaded_daily missing")
    _check(exog.warehouse_daily is not None and not exog.warehouse_daily.empty, "warehouse exog missing")
    _check(exog.world_state_daily is not None and not exog.world_state_daily.empty, "world-state exog missing")

    required_wh_cols = {
        "inbound_consumables_gross_kg",
        "inbound_equipment_gross_kg",
        "inbound_unknown_gross_kg",
    }
    _check(
        required_wh_cols.issubset(set(exog.warehouse_daily.columns)),
        "warehouse_state_daily.csv is missing grouped receipts columns",
    )

    cases = [
        ("baleni_binhits", src.packed_daily, src.loaded_daily, "binhits"),
        ("baleni_gross_tons", src.packed_daily, src.loaded_daily, "gross_tons"),
        ("nakladky_trips_total", src.loaded_daily, src.packed_daily, "trips_total"),
        ("nakladky_containers", src.loaded_daily, src.packed_daily, "containers_count"),
    ]

    results: List[Dict[str, object]] = []
    for name, daily_df, linked_df, metric in cases:
        suite = app.compute_model_suite(
            daily_df,
            linked_df,
            exog.warehouse_daily,
            exog.world_state_daily,
            metric,
            40,
            8,
            False,
        )
        results.append(_suite_summary(name, suite))

    regular = app._regularize_daily(src.packed_daily)
    op_fc = app._operational_window_forecast_df(regular, "binhits", 2026, 6, 14, "hybrid", 8, False, 0)
    _check(not op_fc.empty, "operational full-window forecast is empty")
    _check(op_fc["weekday"].between(1, 5).all(), "operational forecast contains weekends despite weekend-off mode")
    _check(len(op_fc) == 45, f"unexpected operational forecast length: {len(op_fc)}")

    mask = src.packed_daily["iso_year"].astype(int).eq(2026) & src.packed_daily["iso_week"].between(6, 14, inclusive="both")
    mask = mask & src.packed_daily["weekday"].astype(int).between(1, 5)
    ignore_options = (
        pd.to_datetime(src.packed_daily.loc[mask, "date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .sort_values()
        .drop_duplicates()
    )
    _check(len(ignore_options) >= 30, "too few current-year dates available for the ignore-days control")
    sample_drop = [day.strftime("%Y-%m-%d") for day in ignore_options[:2]]
    dropped = app._drop_selected_dates(src.packed_daily, sample_drop)
    original_slice = src.packed_daily.loc[mask].copy()
    dropped_slice = dropped[
        dropped["iso_year"].astype(int).eq(2026)
        & dropped["iso_week"].between(6, 14, inclusive="both")
        & dropped["weekday"].astype(int).between(1, 5)
    ].copy()
    _check(len(original_slice) - len(dropped_slice) >= 2, "ignore-days drop logic did not remove selected dates")

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "operational_window_rows": int(len(op_fc)),
        "ignore_day_options": int(len(ignore_options)),
        "sample_drop_days": sample_drop,
    }

    report_path = BASE_DIR / "forecast_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("VALIDATION OK")
    print(f"report: {report_path}")
    for row in results:
        print(
            " | ".join(
                [
                    str(row["case"]),
                    f"best={row['best_model']}",
                    f"best_wape={_fmt_pct(float(row['best_wape']))}",
                    f"smart_blend={_fmt_pct(float(row['smart_blend_wape']))}",
                    f"drivers={row['drivers_selected']}",
                ]
            )
        )
    print(f"operational_window_rows={len(op_fc)}")
    print(f"ignore_day_options={len(ignore_options)}")
    print(f"sample_drop_days={sample_drop}")


if __name__ == "__main__":
    main()
