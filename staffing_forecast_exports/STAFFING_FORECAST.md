# Staffing Forecast

- Run date: 2026-03-27
- Day binhits forecast model: internal
- Night binhits forecast model: internal
- Binhits last actual date: 2026-03-27
- Attendance last actual date: 2026-03-21
- Median day binhits per paid hour: 2.71
- Median night binhits per paid hour: 2.65

## Staffing Logic

- Forecast day and night binhits separately.
- Convert shift binhits to paid hours with recent weekday productivity for the same shift.
- Hold `kmen` in fixed buckets: early (06-14 proxy) and late (14-22 proxy).
- Let agency absorb flex separately for day and night based on remaining required hours.
- Keep a total productive-workers estimate as a secondary view for operations.

## Notes

- `required_kmen_*` is anchored to recent positive observed weekday capacity, not free reallocation.
- End-to-end planning error is staffing-ratio error plus shift-level binhits forecast error.