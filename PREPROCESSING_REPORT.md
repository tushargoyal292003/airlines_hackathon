# TFT-DCP Preprocessing Report

What the pipeline does to raw BTS / NOAA / ASPM data before it reaches the model, and what gets eliminated at each stage.

## Pipeline order (Phase 1A in `main.py` → `DataPipeline.run()`)

| Step | Module / method | Purpose |
|---|---|---|
| 1 | `BTSProcessor.load()` | Load BTS CSVs, hub-filter while streaming |
| 2 | `BTSProcessor.filter_hub_connections()` | Tag inbound/outbound legs for DFW |
| 3 | `AnomalyDetector.detect()` | Remove physically-impossible records |
| 4 | `_clean_flights()` | Drop cancelled/diverted + rows missing critical fields |
| 5 | `AirportFeatureExtractor.extract()` | Derive airport-level features |
| 6 | `NOAAProcessor.load/process()` + `_merge_weather()` | Join hourly weather |
| 6B | `ASPMProcessor.load/process()` + `_merge_aspm()` | Join FAA city-pair delay/taxi stats |
| 7 | `_compute_derived_features()` | Engineer time / chain / rolling features |
| 8 | `_build_flight_chains()` | Group by tail + date into chains |
| 9 | `_encode_features()` | James-Stein encode categoricals + Min-Max scale numerics |

Phase 1B (`ProxyEngineer.run`) then builds A→DFW→B proxy sequences (duty, MCT, WOCL).

---

## What is eliminated and why

### 1. Hub filter at load time ([preprocessor.py:115](data/preprocessor.py#L115))
Rows where neither `Origin == DFW` nor `Dest == DFW` are dropped while streaming chunks. Typical reduction on full BTS: **~95–98% of raw rows removed** (BTS has all US flights; we only need DFW-touching ones).

### 2. Deduplication ([preprocessor.py:144](data/preprocessor.py#L144))
Dedup key = `(Tail_Number, FlightDate, CRSDepTime, Origin, Dest)`. Because BTS publishes both Reporting-Carrier and Marketing-Carrier views, the same flight appears twice. Expect **~50% removed** after dedup.

### 3. Anomaly detection ([preprocessor.py:591](data/preprocessor.py#L591))
Rows flagged as data errors are removed **unless** they are genuine extreme delays (≥ 180 min → kept per paper §3.1.3).

Error rules:
- `TaxiOut > 200 min` (physically impossible)
- `Distance <= 0`
- `DepDelayMinutes < 0 AND ArrDelay > 180` (contradictory — likely swapped fields)
- `DepDelayMinutes > 1440` (>24 h → date-wrap error)

Typically **< 0.5% removed**.

### 4. `_clean_flights` ([preprocessor.py:854](data/preprocessor.py#L854))
- `Cancelled == 1` → removed
- `Diverted == 1` → removed
- Any row missing **any** of: `DepDelay`, `Tail_Number`, `Origin`, `Dest`, `CRSDepTime` → removed
- Missing `CarrierDelay/WeatherDelay/NASDelay/SecurityDelay/LateAircraftDelay` → **filled with 0**, not dropped

Typical reduction: **1–3%** (cancel/divert rate at DFW).

### 5. Flight-chain filter ([dataset.py:117-118](data/dataset.py#L117-L118))
In the `FlightChainDataset`, chains with fewer than 2 flights are skipped — a single-leg aircraft-day gives no propagation context. Visible as the gap between `total_records` and `chain_count × avg_chain_len` in the manifest.

### 6. NOAA weather ([preprocessor.py:212](data/preprocessor.py#L212))
Rows with unparseable `DATE` are dropped (`dropna(subset=["datetime"])`). Weather with missing `airport_code` is dropped during processing ([preprocessor.py:932](data/preprocessor.py#L932)). Weather is then **left-joined** onto flights on `(Origin, dep_hour_key)` — flights that don't find a matching hour keep the row but get NaN weather (later filled to 0 by the dataset).

### 7. ASPM ([preprocessor.py:545](data/preprocessor.py#L545), [974](data/preprocessor.py#L974))
Rows missing `Departure / Arrival / aspm_year / aspm_month` are dropped. Merge onto flights is **left-join** on `(Origin, Dest, Year, Month, Dep Hour)` — unmatched flights keep the row, ASPM columns become NaN.

### 8. Encoding + scaling ([preprocessor.py:1111](data/preprocessor.py#L1111))
No rows removed. Transformations:
- `Origin, Dest, Reporting_Airline, Tail_Number, OriginState, DestState` → James-Stein encoded to floats (target = `DepDelay`).
- **Raw strings are preserved in `Origin_str` / `Dest_str`** for downstream hub filtering (proxy engineering, pair risk scoring).
- All other numerics → Min-Max scaled to [0, 1] (excluding `DepDelay`, `DepDelayMinutes`, `chain_id`, `chain_position`).
- Remaining NaNs in numeric columns → filled with 0 before scaling.

### 9. Year-based split ([main.py](main.py))
After preprocessing:
- **Train**: `Year ∈ {2019, 2022, 2023}`
- **Val**: `Year = 2024`
- **Test**: `Year = 2025`

Any row with `Year` outside these sets is excluded from training/eval (but remains in `processed_flights.parquet`).

---

## Summary of typical record counts

From your last run:
- Raw BTS (60 months, all US flights) → **stream-filtered to DFW**
- After dedup / clean / anomaly: **2,910,363** records
- Flight chains: **959,584**
- Extreme delays retained (> 180 min): **54,034 (1.9%)**

---

## Where data is *not* eliminated but silently altered

These are worth knowing about because they affect model behavior without showing up in row counts:

1. **Delay-cause NaN → 0** ([preprocessor.py:872](data/preprocessor.py#L872)) — treats "no cause reported" as "no contribution from that cause." Acceptable for BTS (they only populate these on delayed flights) but means the model cannot distinguish "missing" from "zero."
2. **Weather NaN → 0 at dataset time** ([dataset.py:84](data/dataset.py#L84)) — unmatched weather hours become a zero vector, which is a valid scaled value; the model treats missing weather the same as exactly-average weather.
3. **James-Stein target leakage risk** — fit on full training data with `DepDelay` as target. If encoder is refit on train+val, it leaks. Current code fits once inside `_encode_features`; confirm it's called on train slice only if you retrain per split.
4. **Chain truncation** ([dataset.py:137](data/dataset.py#L137)) — any chain longer than `seq_len=14` is left-truncated (oldest legs dropped). Rare at DFW but affects red-eye turns.

---

## Recommendations

- Log the **pre-filter** row count per stage and save to the manifest so the elimination funnel is reproducible run-to-run.
- Consider keeping `Cancelled == 1` rows as a separate labeled class rather than dropping — cancellation prediction is operationally valuable even if it's not the current target.
- Add a NaN-vs-zero indicator column for weather so the model can distinguish "unknown weather" from "calm weather."
