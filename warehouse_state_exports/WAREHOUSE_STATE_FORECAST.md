# Warehouse State Forecast

- Run date: 2026-03-27
- Current inventory gross kg: 9,204,503.9
- Current inventory pallets: 25262
- Current inventory blocked qty: 18,146.0
- Historical daily flow rows: 1,545
- Weekly occupancy observations: 60

## Approach

- Inbound is built from historical receipt files and current-week receipt export.
- Outbound is built from yearly issue files.
- Packing schedule is used both as history and as known near-term future signal.
- Occupancy proxy is based on weekly invoiced storage services.
- Future gross stock proxy is current inventory gross kg plus forecast inbound minus forecast outbound.

## Limits

- Exact future stock by SKU is not backtestable from one current inventory snapshot alone.
- Weekly storage invoice is the best historical proxy for warehouse occupancy in this data package.