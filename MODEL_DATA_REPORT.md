# TFT-DCP — What Data Goes In, How the Model Uses It

## 1. Input tensors per sample

The `FlightChainDataset` ([data/dataset.py](data/dataset.py)) emits one sample per **target flight**, carrying the history of its aircraft's prior legs that day. Every training example is a dict:

| Tensor | Shape | Source columns | Role |
|---|---|---|---|
| `dynamic` | (seq_len=14, num_dynamic) | `flight_features` + `weather_features` | Time-varying per leg |
| `static` | (num_static,) | `Origin, Dest, CRSDepTime, CRSArrTime, Distance, Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos, Reporting_Airline` | Fixed per target flight |
| `chain_delays` | (seq_len,) | `prev_arr_delay` | Delay carried over from each prior leg |
| `turnaround_times` | (seq_len,) | `turnaround_minutes` | Ground time between consecutive legs |
| `mask` | (seq_len,) | built at sample time | 0 for pad positions, 1 for real legs |
| `target` | scalar | `DepDelay` (minutes) | Regression label |

`seq_len = 14` means up to 14 preceding legs of the same aircraft on the same day (`chain_id = Tail_Number + FlightDate`). Shorter chains are left-padded with zeros.

---

## 2. Dynamic features (flown through the TCN)

Three groups concatenated per time step:

**Flight features** (per-leg identity, schedule, realized taxi/airborne, propagation state)
- Route: `Origin`, `Dest`, `Reporting_Airline`, `OriginState`, `DestState`, `Flight_Number_Reporting_Airline` (all James-Stein encoded to floats)
- Schedule: `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance`
- Ground: `TaxiOut` (scheduled), `actual_taxi_out`, `taxi_out_excess`
- Airborne: `actual_airborne`, `airborne_excess`
- Temporal cyclicals: `Month_sin/cos`, `DayOfWeek_sin/cos`
- Chain state: `prev_arr_delay`, `turnaround_minutes`, `chain_position`, `chain_length`
- Prior delay causes: `prev_carrierdelay`, `prev_weatherdelay`, `prev_nasdelay`, `prev_securitydelay`, `prev_lateaircraftdelay`

**Airport features** (origin airport operational state, ASPM city-pair stats)
- Computed: `op_density`, `hist_peak_capacity`, `utilization_rate`, `cum_delay_past_hour`
- ASPM: `aspm_flight_count`, `aspm_pct_ontime_{gate_dep,airport_dep,gate_arr}`, `aspm_{edct_count,avg_edct}`, `aspm_{gate_dep,taxi_out,airport_dep,airborne,taxi_in,block,gate_arr}_delay`, `aspm_avg_taxi_out`

**Weather features** (NOAA LCD hourly, joined on `Origin × dep_hour`)
- `HourlyWindSpeed`, `HourlyVisibility`, `HourlyPrecipitation`, `HourlyDryBulbTemperature`, `HourlyRelativeHumidity`, `cloud_cover`, `wx_severity`

All normalized to [0, 1] via Min-Max (except raw label + calendar raws).

---

## 3. Static features (flown through the GRN)

Ten columns describing the target flight's identity:
`Origin, Dest, CRSDepTime, CRSArrTime, Distance, Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos, Reporting_Airline`

These are also normalized/encoded. The GRN produces a single context vector that is added to the TCN outputs to condition the sequence encoding on the target leg.

---

## 4. How the model uses it ([model/tft_dcp.py](model/tft_dcp.py))

The forward pass wires the tensors through 6 stages:

### Stage 1 — TCN dynamic encoder ([model/tcn.py](model/tcn.py))
`dynamic (B, 14, D)` → dilated causal convolutions → per-step embeddings `h_dynamic (B, 14, hidden)` and a mask-pooled `h_global (B, hidden)`. The last valid position is extracted as `h_current` — "representation of the target leg given its chain history."

### Stage 2 — GRN static encoder ([model/grn.py](model/grn.py))
`static (B, 10)` → Gated Residual Network → `h_static (B, hidden)`. Then:
```
h_current ← h_current + GRN(h_static)
h_global  ← h_global  + h_static
```
Static context is injected into both the current-leg and chain-global views.

### Stage 3 — Historical retrieval ([model/historical_retrieval.py](model/historical_retrieval.py))
A running FIFO bank of past `h_current` embeddings (size 50k) acts as nonparametric memory. For each new sample: cosine-similarity top-K=5, weighted average → `h_f (B, hidden)` — "what happened on flights that looked like this one."

### Stage 4 — MS-CA-EFM fusion ([model/ms_ca_efm.py](model/ms_ca_efm.py))
Multi-Source Channel-Attention Feature Mixer combines the three hidden vectors:
```
h_fused = MS-CA-EFM(h_current, h_f, h_global)
```
Channel attention learns which feature dimensions matter in each source.

### Stage 5 — Dynamic chain propagation ([model/propagation.py](model/propagation.py))
Takes `chain_delays` + `turnaround_times` + `mask` through a learnable-β decay:
```
y_prop ≈ Σ_k exp(−β · turnaround_k) · chain_delays_k
```
β is a trainable scalar (softplus-bounded). Output: scalar `y_prop` and embedding `h_prop (B, hidden)`. This is the deterministic "how much of the upstream delay propagates through the turn" signal.

### Stage 6 — Prediction head
```
h_final = concat(h_fused, h_prop)   # (B, 2·hidden)
ŷ = MLP(h_final)                    # (B,) → predicted DepDelay in minutes
```

Forward returns `{prediction, h_current, h_global, h_f, y_prop, β}`. After each batch the main embedding is written back into the history bank (`update_history`).

---

## 5. What the model does NOT see directly

- **Target value `DepDelay`** — supervised label, only in the loss.
- **Raw calendar fields** (`Year_raw`, `Month_raw`, `DayOfWeek_raw`) — kept only for train/val/test splitting and seasonal reporting.
- **String route identifiers** (`Origin_str`, `Dest_str`) — kept only for downstream hub filtering and pair scoring.
- **`Cancelled`, `Diverted`** — rows where these = 1 were dropped during cleaning.
- **Delay-cause columns on the target leg** (`CarrierDelay`, `WeatherDelay`, …) — would leak the label; only the *previous* leg's causes are features.

---

## 6. Dimensionality sanity check

From `config.py` defaults:
- `num_dynamic_features ≈ 40` (flight 20 + airport 14 + weather 7 + cyclicals 4 overlap resolved)
- `num_static_features = 10`
- `hidden_dim = 128`
- TCN channels `[64, 128, 128]`, kernel 3, dropout 0.1
- History DB 50,000 × 128 = ~25 MB

These are set before `TFTDCP(...)` is built; exact counts print in the preprocessing manifest and again at training startup.

---

## 7. Output contract

Per target flight → scalar predicted `DepDelay`. Aggregated by `PairRiskScorer` ([risk_scorer.py](risk_scorer.py)) into A→DFW→B rows:
```
airport_a · airport_b · month · ml_risk_score · wocl_multiplier ·
final_score · duty_flag · mct_violation · wocl_flag
```
`final_score = ml_risk × wocl_multiplier`, zeroed when `mct_violation = 1`.
