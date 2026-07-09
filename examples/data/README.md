# Example datasets

Small, public datasets bundled so the example notebooks run offline and reproduce
the same numbers for everyone. Nothing here is proprietary.

## market_regimes.csv

Daily US market and rates series, used by `01_finance_market_regimes.ipynb`.

| Column | FRED series | Meaning |
| --- | --- | --- |
| `vix` | `VIXCLS` | CBOE Volatility Index |
| `dgs10` | `DGS10` | 10-year Treasury constant maturity yield |
| `term_spread` | `T10Y2Y` | 10-year minus 2-year Treasury yield |
| `oil` | `DCOILWTICO` | WTI crude oil spot price |

Source: FRED (Federal Reserve Bank of St. Louis), retrieved from the public
`fredgraph.csv` endpoint, which needs no API key. The series are aligned on common
trading days and cover 1990 to mid-2026. FRED data are free to use and
redistribute; the underlying VIX and WTI marks originate with CBOE and the EIA
respectively and are redistributed here for non-commercial demonstration.

To refresh the snapshot, the exact fetch code is shown in the notebook's data
section.

## gb_electricity_demand.csv

Daily Great Britain electricity demand, used by `02_energy_seasonal_demand.ipynb`.

| Column | Meaning |
| --- | --- |
| `demand_mean` | Mean national demand across the day, MW |
| `demand_peak` | Peak national demand in the day, MW |
| `wind` | Mean embedded (distribution-connected) wind generation, MW |
| `solar` | Mean embedded solar generation, MW |

Source: NESO (National Energy System Operator), Historic Demand Data, retrieved
from the public CKAN API at `api.neso.energy`, which needs no API key. NESO
publish one half-hourly file per year; the 48 daily settlement periods are
aggregated to the daily figures above, covering 2016 to 2023. The data are
published under the NESO Open Data Licence and are free to reuse with attribution.

To refresh the snapshot, the exact fetch code is shown in the notebook's data
section.
