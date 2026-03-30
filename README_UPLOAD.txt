WHS DASH upload balicek
Vytvoreno: 2026-03-30

Co obsahuje:
- aktualni `streamlit_app.py`
- forecast runtime soubory a data
- `staffing_forecast_exports`
- `warehouse_state_exports`
- `analysis_exports` se shrnutim driveru a validace

Jak nahrat:
1. Otevri root repa `whs-dash`.
2. Nahraj OBSAH teto slozky do rootu repa.
3. Nenahravej slozku `UPLOAD_WHS_DASH_2026-03-30` jako podslozku.
4. Pokud uz v repu existuji stejne soubory nebo slozky, prepis je.

Nejdulzitejsi runtime soubory:
- `streamlit_app.py`
- `forecasting.py`
- `packed_daily_kpis.csv`
- `packed_shift_kpis.csv`
- `loaded_daily_kpis.csv`
- `loaded_shift_kpis.csv`
- `world_state_feature_weekly.csv`
- `staffing_forecast_exports`
- `warehouse_state_exports`

Kontrola po nasazeni:
- karta `Predikce dopredu` ma vysvetleni modelu a metrik
- karta `Staffing` ma vysvetleni kapacitni logiky
- appka ukazuje nove public drivere a poctive backtest vysledky
