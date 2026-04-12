Upload bundle for GitHub / deploy

This is the minimal runtime bundle.

Included:
- streamlit_app.py
- forecast_models.py
- external_signals.py
- external_market_daily.csv
- requirements.txt
- loaded_daily_kpis.csv
- loaded_shift_kpis.csv
- packed_daily_kpis.csv
- packed_shift_kpis.csv

Not included on purpose:
- raw Kompletace / Vydeje / Prijmy data
- rebuild scripts
- yearly receipts
- audit and analysis exports

Reason:
- smaller repo
- faster deploy
- less startup noise

Upload the CONTENTS of this folder into repository root, not the `Upload` folder itself.