# Staffing Manual Input

Soubor `staffing_manual_history.csv` slouzi jako rucne vyplnitelna historie staffing rozpadů pro ESAB.

Vyplnuj pouze minulost. Budoucnost pocita forecast.

Logika poli:
- `kmen_*_morning` = kmen 06:00-14:00
- `kmen_*_afternoon` = kmen 14:00-22:00
- `agency_*_day` = agentura 07:00-19:00
- `agency_*_night` = agentura 19:00-07:00

Kategorie:
- `inbound` = vykladka / prijem / zaskladneni
- `loading` = nakladky
- `pick` = pick / kompletace
- `other` = ostatni cinnosti

Pomocne sloupce `helper_*` a `suggested_*` jsou generovane automaticky. Nejsou povinne pro editaci.

Forecast je ukotveny pres `helper_esab_paid_hours` a `helper_esab_fte_8h`.