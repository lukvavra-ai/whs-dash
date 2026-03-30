# Staffing Forecast

- Run date: 2026-03-28
- FTE model: baseline
- Last packed actual: 2026-03-28
- Last staffing actual: 2026-03-28

## Staffing Logic

- Historie lidi se bere z workeru, kteri meli nejakou cinnost v `TimeManagement/Data`.
- Celkovy forecast je ukotveny pres `ESAB placene hodiny / 8 = FTE`.
- Rozpad na kmen/agenturu a cinnosti bere prioritu z rucne vyplneneho `staffing_manual_history.csv`.
- Pokud manualni rozpad chybi, fallback je na auto-suggest z aktivit. Nakladka je v auto-suggestu nulova a ma se dopsat rucne.

## Manual Input

- Vyplnuj `staffing_manual_history.csv` jen pro minulost.
- Kmen: morning 06-14, afternoon 14-22.
- Agentura: day 07-19, night 19-07.
- `helper_esab_paid_hours` a `helper_esab_fte_8h` slouzi jako kotva reality.