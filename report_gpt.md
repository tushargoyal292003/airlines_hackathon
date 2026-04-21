# TFT-DCP Hackathon Context Pack (`report_gpt.md`)

This document is a full context handoff for LLM-based report generation. It consolidates the project docs and the latest code/output updates.

Primary source docs merged into this context:
- `FULL_PROJECT_EXPLAINER.md`
- `DELAY_TO_RISK_REPORT.md`
- `PREPROCESSING_REPORT.md`
- `MODEL_DATA_REPORT.md`

Additional current-session updates included:
- Baseline-vs-ours checkpoint comparison in `results/checkpoint_comparison.csv`
- Leakage/risk evaluation findings
- Score semantics refactor (`risk_score`, `is_feasible`, `display_score`)
- New feasibility policy (persistent MCT violation rate)
- Refreshed `proxy_sequences.parquet` and pair-risk outputs

---

## 1) Problem Statement and Deliverable

American Airlines needs to avoid risky pilot sequences of the form:
- `A -> DFW -> B`

Goal is not only flight-delay regression, but operational ranking of risky `(A,B)` pairings through DFW.

Core deliverable:
- Ranked pair-level risk outputs (`scored_pairs.csv`, `flagged_pairs.csv`, seasonal variants)

---

## 2) Current Build Snapshot (As Of 2026-04-14)

### Model performance (held-out 2025)
From `results/metrics.json`:
- MAE: `21.64`
- RMSE: `58.24`
- R2: `0.2767`
- learned beta: `0.6931`
- n_flights: `410,761`

Seasonal flight-level metrics:
- Winter: MAE `19.27`, RMSE `56.88`, R2 `0.2459`, extreme `%` `1.68`
- Spring: MAE `22.06`, RMSE `55.64`, R2 `0.3059`, extreme `%` `2.21`
- Summer: MAE `25.71`, RMSE `65.15`, R2 `0.2789`, extreme `%` `2.89`
- Fall: MAE `19.02`, RMSE `53.98`, R2 `0.2545`, extreme `%` `1.88`

### Baseline comparison (existing baseline checkpoints)
From `results/checkpoint_comparison.csv`:
- LightGBM: MAE `11.55`, RMSE `33.09`, R2 `0.7942`
- XGBoost: MAE `12.10`, RMSE `34.11`, R2 `0.7813`
- TFT-baseline: MAE `20.68`, RMSE `58.51`, R2 `0.2699`
- TFT-DCP (ours): MAE `21.64`, RMSE `58.24`, R2 `0.2767`

Interpretation:
- For pure flight-level delay regression, tree baselines are stronger on these metrics.
- TFT-DCP remains the architecture integrated with chain propagation and pair-risk logic.

### Pair-risk output status (after policy update)
From refreshed `results/pair_risk_scores_full.csv`:
- total pair rows: `33,670`
- feasible rows (`is_feasible=1`): `33,427`
- infeasible rows (`is_feasible=0`): `243` (`0.72%`)
- flagged (feasible and `risk_score>=0.6`): `37`

Seasonal flagged counts (current regenerated outputs):
- Winter: `143`
- Spring: `45`
- Summer: `18`
- Fall: `136`

---

## 3) Data Sources and Scope

### BTS On-Time Performance
Fields used:
- schedule + actual times
- origin/destination/carrier/tail number
- delay causes
- cancellation/diversion flags
- taxi/air times

Time window used in project:
- 2019 + 2022/2023/2024/2025

### NOAA LCD hourly weather
Used for airport-hour weather context:
- wind speed, visibility, precipitation, temperature, humidity, cloud cover, weather severity

### FAA ASPM city-pair stats
Used for monthly/hourly route-level operational priors:
- on-time percentages
- EDCT stats
- gate/taxi/airport/airborne/block delay metrics

---

## 4) End-to-End Pipeline

Orchestration entrypoint:
- `main.py`

High-level stages:
1. Preprocess (`DataPipeline.run`) -> `processed_flights.parquet`
2. Proxy engineering (`ProxyEngineer.run`) -> `proxy_sequences.parquet`
3. Train TFT-DCP -> `checkpoints/best_model.pt`
4. Evaluate on 2025 + risk scoring -> `results/*`
5. Optional baseline and ablation runs

---

## 5) Preprocessing Details (What Gets Filtered / Transformed)

### Core sequence
1. Hub-filter to DFW-touching flights at load time
2. Deduplicate flights
3. Anomaly filter (retain genuine extremes)
4. Drop canceled/diverted and missing critical fields
5. Derive airport congestion/utilization signals
6. Merge NOAA hourly weather
7. Merge ASPM city-pair features
8. Derive taxi/airborne excess features
9. Build aircraft-day flight chains
10. Encode categoricals + scale numerics
11. Preserve raw columns for splitting/proxy logic (`*_raw`, `Origin_str`, `Dest_str`)

### Typical scale after processing
- `processed_flights.parquet`: ~2.9M rows

### Important subtle points
- Weather/ASPM are left-joined (rows kept even when enrichment missing)
- Extreme-delay examples are intentionally retained
- `chain_id = Tail_Number + FlightDate`
- Dataset samples are target-leg + up to 14 preceding legs of the same chain

---

## 6) Feature Design Summary

### Static (target-flight identity context)
Typical static cols:
- `Origin, Dest, CRSDepTime, CRSArrTime, Distance, Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos, Reporting_Airline`

### Dynamic sequence features
- flight behavior/history
- airport operational state
- weather
- chain propagation state (`prev_arr_delay`, `turnaround_minutes`, prior-leg delay causes)

### Dedicated propagation tensors
- `chain_delays`
- `turnaround_times`
- `mask`

---

## 7) Model Architecture (TFT-DCP)

Main modules:
1. TCN encoder for dynamic sequence features
2. GRN for static context
3. Historical retrieval memory (top-K similar historical embeddings)
4. MS-CA-EFM fusion across current/retrieved/global chain views
5. Delay propagation module with learnable decay parameter `beta`
6. Prediction head

Model output during scoring:
- `prediction` (delay minutes)
- `y_prop` (propagation component)

---

## 8) Split Strategy

Year-based split for temporal realism:
- Train: 2019, 2022, 2023
- Val: 2024
- Test: 2025

Why:
- avoids random split leakage across near-identical time contexts
- simulates true forward generalization

---

## 9) From Flight Predictions to Pair Risk

### Flight-level prediction table
`results/flight_predictions.csv` has:
- `origin, dest, year, month`
- `pred_delay, actual_delay, propagated_delay`

### Pair construction
- inbound stats from flights where `dest == DFW` (airport A side)
- outbound stats from flights where `origin == DFW` (airport B side)
- cross-join A × B

### ML risk composition
Sub-risks:
- delay_risk
- propagation_risk
- variance_risk
- extreme_risk

Weighted score:
- `ml_risk_score = 0.35*delay + 0.30*prop + 0.20*variance + 0.15*extreme`

---

## 10) Regulatory Proxy Layer and Policy Evolution

### Historical behavior (old policy)
Old logic used effectively:
- pair infeasible if **any** MCT violation ever observed (`max`)
- worst connection minute retained (`min`)
- final score hard-zeroed for infeasible pairs

Impact observed:
- infeasible fraction became extremely high in old outputs
- score semantics became confusing (`0` looked like "no risk" but often meant "disqualified")

### Current policy (implemented in this session)
Implemented in `data/proxy_engineering.py` and consumed in `risk_scorer.py`:
- compute per-pair counts across enumerated sequences:
  - `n_sequences`
  - `n_mct_violations`
  - `mct_violation_rate`
  - `n_wocl`
  - `wocl_exposure_rate`
- mark infeasible only when violations are persistent:
  - `mct_violation = 1 if mct_violation_rate >= 0.50 else 0`
- use exposure-aware WOCL multiplier:
  - `wocl_multiplier = 1 + 0.35 * wocl_exposure_rate` (capped at 1.35)

This changed proxy feasibility statistics substantially:
- in regenerated `proxy_sequences.parquet`, infeasible proxy pairs dropped to ~`1.1%`

---

## 11) Score Semantics Refactor (Implemented)

To remove ambiguity, scorer now outputs:
- `risk_score`: continuous risk signal (`ml_risk_score * wocl_multiplier`)
- `is_feasible`: binary feasibility (`mct_violation == 0`)
- `display_score`: `risk_score` for feasible rows, `NaN` otherwise
- `final_score`: backward-compat alias of `display_score`

Flagging policy now:
- only feasible rows are eligible
- `risk_score >= 0.6` => `AVOID`
- `risk_score >= 0.8` => `CRITICAL`

---

## 12) Current Output Contracts

### `results/pair_risk_scores_full.csv`
Detailed debug table now includes:
- core risk columns (`ml_risk_score`, `risk_score`, `display_score`, `is_feasible`, `final_score`)
- proxy diagnostics (`mct_violation`, `mct_violation_rate`, `wocl_flag`, `wocl_exposure_rate`, `n_sequences`, `avg_conn_mins`)
- component sub-risks (`delay_risk`, `propagation_risk`, `variance_risk`, `extreme_risk`)

### `results/scored_pairs.csv`
Compact scored contract with same core semantics.

### `results/flagged_pairs.csv`
Feasible high-risk pairs only with recommendation labels.

### Seasonal outputs
- `pair_risk_scores_{winter,spring,summer,fall}.csv`
- `flagged_pairs_{winter,spring,summer,fall}.csv`

---

## 13) Visualization Data Mapping (for downstream LLM / plotting)

### DFW risk map
Use:
- `results/pair_risk_scores_full.csv`
Columns:
- `airport_b`, `risk_score`, `is_feasible`

Recommended map value:
- line thickness by aggregated `risk_score` over `airport_b` (e.g., max or mean over airport_a)
- render infeasible separately using `is_feasible`

### Seasonal heatmap (airport × season)
Use:
- seasonal `pair_risk_scores_*.csv`
Columns:
- `airport_b`, `risk_score`, `is_feasible`

### Top-10 risk breakdown
Use:
- `results/pair_risk_scores_full.csv`
Columns:
- `delay_risk, propagation_risk, variance_risk, extreme_risk, wocl_multiplier, ml_risk_score, risk_score`

---

## 14) Known Modeling / Evaluation Risks

### Metric alignment risk
Because end-goal is ranking risky pairs, pure flight-level MAE/RMSE/R2 can overweight common low-delay cases.

Recommendation:
- include ranking-aware metrics (Precision@K, Recall@K, NDCG@K, lift)
- evaluate tail-event capture for top-ranked pairs

### Leakage risks identified in code path
Potential leakage vectors observed in current pipeline design include:
- target-aware encoding/scaling being fit before temporal split
- features with potential hindsight composition

Practical implication:
- baseline comparisons (especially tree models) should be interpreted cautiously until leakage-safe re-pipeline is enforced.

---

## 15) Repro / Refresh Commands

### Run evaluate (must run from repo root so checkpoint path resolves)
```bash
cd /home/dal851386/tft_dcp
/home/dal851386/miniconda3/envs/airlines/bin/python main.py \
  --mode evaluate \
  --processed-dir ./data/processed \
  --results-dir ./results
```

### Regenerate proxies only (new policy) from processed flights
(used in this session)
- run `ProxyEngineer.run` over `data/processed/processed_flights.parquet`
- write `data/processed/proxy_sequences.parquet`

### Rebuild pair-risk outputs from existing flight predictions
(used in this session to avoid full model rerun)
- load `results/flight_predictions.csv`
- load new `data/processed/proxy_sequences.parquet`
- aggregate/export with updated `PairRiskScorer`

---

## 16) File/Module Map

Core code:
- `main.py`
- `data/preprocessor.py`
- `data/proxy_engineering.py`
- `data/dataset.py`
- `model/tft_dcp.py`
- `model/{tcn,grn,historical_retrieval,ms_ca_efm,propagation}.py`
- `train.py`
- `risk_scorer.py`
- `baselines.py`, `experiments.py`
- `visualize.py`

Core artifacts:
- `data/processed/processed_flights.parquet`
- `data/processed/proxy_sequences.parquet`
- `checkpoints/best_model.pt`
- `results/metrics.json`
- `results/flight_predictions.csv`
- `results/pair_risk_scores_full.csv`
- `results/scored_pairs.csv`
- `results/flagged_pairs.csv`
- seasonal pair/flagged outputs
- `results/checkpoint_comparison.csv`

---

## 17) Suggested Prompt Starter For Report LLM

Use this context pack and ask the report LLM to produce:
1. Executive summary (problem, approach, outcomes)
2. Data and preprocessing methodology
3. Model architecture and rationale
4. Evaluation and baseline interpretation
5. Risk-scoring policy and latest feasibility refactor
6. Seasonal insights and operational recommendations
7. Risks/limitations and next-phase roadmap

Also instruct the LLM to:
- treat `risk_score` and `is_feasible` as primary semantics
- mention that old docs may reference hard-gated `final_score` behavior
- use current output counts from this report where conflicts exist

---

## 18) What Changed In This Session (Audit Trail)

Implemented code updates:
- `risk_scorer.py`:
  - explicit risk/feasibility/display semantics
  - compatibility-aware proxy merge
  - richer export schemas
- `main.py`:
  - seasonal flagged logic now uses feasible + `risk_score`
- `data/proxy_engineering.py`:
  - persistent-violation feasibility policy (`MCT_INFEASIBLE_RATE=0.50`)
  - sequence-count/rate aggregation fields
  - exposure-based WOCL multiplier

Regenerated artifacts:
- `data/processed/proxy_sequences.parquet`
- `results/pair_risk_scores_full.csv`
- `results/scored_pairs.csv`
- `results/flagged_pairs.csv`
- seasonal pair/flagged files

---

## 19) Final Positioning Summary

The project is a chain-aware, regulation-augmented risk ranking system:
- predicts flight delay behavior on held-out future year data
- translates predictions into A->DFW->B pair risk
- overlays operational constraints and fatigue proxies
- now exposes clean semantics (`risk_score` vs feasibility) suitable for decision support and visualization

