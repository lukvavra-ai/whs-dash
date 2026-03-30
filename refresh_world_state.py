from __future__ import annotations

from pathlib import Path
import traceback

from data_pipeline import build_internal_weekly, find_data_dir, load_local_data
from world_state_engine import refresh_and_cache_world_state


def main() -> int:
    data_dir = find_data_dir(Path(__file__).resolve().parent)
    try:
        local_data = load_local_data(data_dir)
        internal_weekly = build_internal_weekly(local_data)
        result = refresh_and_cache_world_state(data_dir, internal_weekly)
        meta = result.get("meta", {})
        print(f"World-state cache updated: {meta.get('updated_at', '')}")
        print(f"Data dir: {data_dir}")
        for name, rows in meta.get("factor_rows", {}).items():
            live_ok = meta.get("factor_live_available", {}).get(name, False)
            print(f"factor {name}: {rows} rows, live={'yes' if live_ok else 'no / cache fallback'}")
        for name, rows in meta.get("news_rows", {}).items():
            live_ok = meta.get("news_live_available", {}).get(name, False)
            print(f"news {name}: {rows} rows, live={'yes' if live_ok else 'no / cache fallback'}")
        print(f"Used cached fallback: {meta.get('used_cached_fallback', False)}")
        return 0
    except Exception as exc:
        print("ERROR during refresh_world_state.py")
        print(exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
