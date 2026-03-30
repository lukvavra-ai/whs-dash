# Forecast Driver Summary

Datum: 2026-03-30

## Co bylo znovu přepočítáno

- `packed_daily_kpis.csv` a `packed_shift_kpis.csv` byly přegenerované z nových dat ve složce `Kompletace`.
- `warehouse_state_exports/warehouse_state_daily.csv` byl přestavěný po opravě builderu tak, aby četl jen roční CSV.
- `world_state_feature_weekly.csv` bylo osvěžené z nového world-state enginu.

## Veřejné indexy, které teď opravdu běží

- Yahoo Finance: `Brent`, `VIX`, `Copper`, `TTF gas`, `ESAB`
- `GSCPI` od New York Fed
- `RWI / ISL` container throughput
- `EU / Euro area / CZ / DE diesel` z Weekly Oil Bulletin
- `Eurostat` průmyslová výroba a producer prices

Poznámka:

- `GDELT` news radar zůstává aktuálně prázdný, takže veřejná vrstva stojí hlavně na tvrdých indexech, ne na news sentimentu.

## Co nejlépe kopíruje minulost

### Binhits

Nejsilnější stejnotýdenní signály:

- `wh_packing_lines`: `0.979`
- `packed_gross_tons`: `0.956`
- `wh_packing_gross_kg`: `0.904`
- `warehouse_pressure_index`: `0.862`
- `loaded_orders_nunique`: `0.729`

Nejlepší veřejné indexy:

- `eu_gas_13w_pct` lag `1`: `0.367`
- `de_prod_c25_13w_pct` lag `3`: `-0.304`
- `de_diesel_price_4w_pct` lag `1`: `0.271`

Praktický závěr:

- pro `Binhits` pořád vítězí interní flow skladu a vlastní packing signály
- veřejná vrstva funguje spíš jako režimový korektor

### Trips total

Nejsilnější stejnotýdenní signály:

- `loaded_trips_europe`: `0.951`
- `wh_outbound_load_units`: `0.943`
- `loaded_gross_tons`: `0.926`
- `wh_outbound_gross_kg`: `0.915`
- `loaded_orders_nunique`: `0.895`

Nejlepší předstihové signály:

- `wh_outbound_orders` lag `1`: `0.649`
- `wh_outbound_docs` lag `1`: `0.649`
- `loaded_orders_nunique` lag `1`: `0.639`
- `wh_inbound_consumables_gross_kg` lag `1`: `0.560`

Nejlepší veřejné indexy:

- `de_prod_c25` lag `1`: `-0.351`
- `de_prod_c25_13w_pct` lag `3`: `-0.346`
- `copper` lag `1`: `0.335`

Praktický závěr:

- pro `Trips total` jsou nejcennější výdejové a order signály
- z příjmů dává smysl hlavně `inbound_consumables_gross_kg`
- veřejné indexy pomáhají, ale jsou slabší než interní provoz

### Containers

Nejsilnější stejnotýdenní signály:

- `loaded_trips_export`: `0.821`
- `loaded_container_share`: `0.708`
- `loaded_gross_tons`: `0.570`
- `wh_outbound_gross_kg`: `0.553`
- `wh_outbound_cbm`: `0.543`

Nejlepší předstihové signály:

- `wh_inbound_consumables_gross_kg` lag `2`: `0.358`
- `loaded_orders_nunique` lag `1`: `0.354`
- `wh_inbound_consumables_gross_kg` lag `1`: `0.350`

Nejlepší veřejné indexy:

- `export_risk_index` lag `0`: `0.393`
- `container_market_index` lag `0`: `0.357`
- `de_prod_c25_13w_pct` lag `3`: `-0.266`
- `export_risk_index` lag `4`: `0.240`

Praktický závěr:

- právě u `Containers` dává veřejná shipping a export vrstva největší smysl
- nejlepší kombinace je interní exportní mix + inbound consumables + container/public regime indexy

## Backtest modelů v dashboardu

Výsledek validace z `forecast_validation_report.json`:

- `Balení / Binhits`: nejlepší `02 Same weekday median`, WAPE `14.6 %`
- `Balení / GW`: nejlepší `10 Smart blend`, WAPE `18.8 %`
- `Nakládky / Trips total`: nejlepší `02 Same weekday median`, WAPE `19.8 %`
- `Nakládky / Containers`: nejlepší `02 Same weekday median`, WAPE `38.0 %`

Důležitý závěr:

- `09 Ridge with drivers` už teď vidí receipts, sklad i veřejné indexy
- ale na současných datech zatím neporáží nejlepší jednoduché modely
- proto je správně používat externí vrstvu jako doplněk a ne ji násilně tlačit jako hlavní forecast driver

## Co z toho plyne pro další ladění

1. `Binhits`: stavět hlavně na `orders`, `packing flow`, `gross tons`, receipts a kalendáři
2. `Trips total`: držet interní výdejové drivere jako hlavní vrstvu
3. `Containers`: dál rozvíjet `export_risk_index`, `container_market_index` a receipts `consumables`
4. veřejné indexy ponechat v modelu, ale nečekat, že samy přebijí interní provozní signály

## Výstupy

- `analysis_exports/driver_correlation_scan.csv`
- `analysis_exports/driver_correlation_best_lag.csv`
- `forecast_validation_report.json`
