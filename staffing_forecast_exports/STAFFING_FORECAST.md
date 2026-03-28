# Staffing Forecast

- Run date: 2026-03-27
- Binhits forecast model: internal
- Binhits last actual date: 2026-03-27
- Attendance last actual date: 2026-03-21
- Median binhits per paid hour: 2.76
- Median binhits per productive hour: 7.37
- Median paid hours per person: 9.15

## Staffing Logic

- Forecast daily binhits.
- Convert binhits to paid and productive hours using recent weekday productivity.
- Convert required hours to people using recent hours-per-person.
- Split total people into kmen and agentura using recent attendance mix.
- Split productive workers into day and night using recent shift shares from TimeManagement/Data.

## Notes

- Staffing backtest is conditional on actual binhits for the day.
- End-to-end planning error is therefore staffing-ratio error plus binhits forecast error.