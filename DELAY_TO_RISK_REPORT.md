# From Delay Prediction to A→DFW→B Risk Pairings — Full Code Walkthrough

End-to-end trace of what happens after training ends: how per-flight delay predictions turn into ranked airport-pair recommendations.

---

## Stage 0 — Inputs at inference time

Two independent artifacts:

| Artifact | Produced by | Contents |
|---|---|---|
| **`processed_flights.parquet`** | Phase 1A (`DataPipeline.run`) | 2.9M rows × ~85 columns: normalized features, raw `DepDelay`, calendar raws, `Origin_str`/`Dest_str`, raw HHMM times |
| **`proxy_sequences.parquet`** | Phase 1B (`ProxyEngineer.run`) | ~40k rows: one per unique `(airport_a, airport_b)` pair, with `duty_flag`, `mct_violation`, `wocl_flag`, `wocl_multiplier`, `avg_duty_mins`, `avg_conn_mins` |
| **`checkpoints/best.pt`** | `train.py` | Trained `TFTDCP` weights |

---

## Stage 1 — Restrict to the test year

[main.py:296-299](main.py#L296-L299):

```python
year_col = "Year_raw" if "Year_raw" in df.columns else "Year"
test_df = df[df["Year"].isin(config.data.test_years)].copy()   # test_years = [2025]
```

Only 2025 rows enter the evaluator. This is the "true future" held-out set (train = 2019/2022/2023, val = 2024).

---

## Stage 2 — Build test dataloader

[main.py:302-317](main.py#L302-L317) wraps `test_df` in a `FlightChainDataset` identical to training's. Each sample still carries:
- `dynamic`, `static`, `chain_delays`, `turnarounds`, `mask` (what the model consumes)
- `origin`, `dest` (raw strings, for later grouping)
- `year`, `month` (raw ints, for seasonal slicing)
- `target` = actual `DepDelay` in minutes

---

## Stage 3 — Per-flight delay prediction

[risk_scorer.py:32-69](risk_scorer.py#L32-L69) → `score_from_dataloader`:

```python
for batch in dataloader:
    output = self.model(dynamic, static, chain_delays, turnarounds, mask)
    preds   = output["prediction"].cpu().numpy()    # (B,) in minutes
    y_props = output["y_prop"].cpu().numpy()        # propagation component
    rows.append({
        "origin": origins[i], "dest": dests[i],
        "year": year, "month": month,
        "pred_delay": float(preds[i]),
        "actual_delay": float(targets[i]),
        "propagated_delay": float(y_props[i]),
    })
```

Output: `flight_preds` DataFrame — one row per 2025 flight, with predicted and actual delay side by side.

**Flight-level metrics** ([main.py:324-335](main.py#L324-L335)):
- **MAE** = mean(|actual − predicted|), in minutes
- **RMSE** = sqrt(mean((actual − predicted)²))
- **R²** = 1 − SS_res / SS_tot

These are logged overall and per season (winter/spring/summer/fall, [main.py:338-361](main.py#L338-L361)).

---

## Stage 4 — Split flights into inbound and outbound legs at DFW

[risk_scorer.py:81-82](risk_scorer.py#L81-L82):

```python
inbound  = flight_preds[flight_preds["dest"]   == hub]   # A → DFW
outbound = flight_preds[flight_preds["origin"] == hub]   # DFW → B
```

The A→DFW→B structure is reconstructed from these two subsets.

---

## Stage 5 — Aggregate delays to per-airport statistics

### Airport A (inbound to DFW): [risk_scorer.py:84-89](risk_scorer.py#L84-L89)
For each origin airport A, compute over all 2025 flights from A → DFW:
```python
avg_delay_a     = mean(pred_delay)
std_delay_a     = std(pred_delay)
extreme_pct_a   = % of flights with pred_delay > 180 min
n_flights_a     = count
```

### Airport B (outbound from DFW): [risk_scorer.py:91-96](risk_scorer.py#L91-L96)
For each destination B, compute over all DFW → B flights:
```python
avg_delay_b     = mean(pred_delay)
std_delay_b     = std(pred_delay)
avg_prop_delay  = mean(propagated_delay)    # y_prop from model
n_flights_b     = count
```

So each airport collapses to a single summary row.

---

## Stage 6 — Cross-join airport A × airport B

[risk_scorer.py:99-101](risk_scorer.py#L99-L101):
```python
a_stats["_key"] = 1
b_stats["_key"] = 1
pairs = a_stats.merge(b_stats, on="_key").drop(columns="_key")
```

If there are ~150 inbound origins and ~150 outbound destinations, that's ~22,500 `(A, B)` pair rows. Each carries both airports' delay statistics.

---

## Stage 7 — Compose the **ML risk score**

[risk_scorer.py:104-125](risk_scorer.py#L104-L125) — four sub-risks, each normalized to [0, 1]:

| Sub-risk | Formula | Meaning |
|---|---|---|
| `delay_risk` | `(avg_delay_a + avg_delay_b) / 2 / P99(combined)` | Typical delay magnitude |
| `propagation_risk` | `avg_prop_delay / P99` | Upstream-delay carry-through |
| `variance_risk` | `(std_a + std_b) / (2 · max_std)` | Schedule unreliability |
| `extreme_risk` | `extreme_pct_a / max(extreme_pct_a)` | Long-tail blow-ups |

Weighted sum per paper:
```python
ml_risk_score = 0.35·delay_risk + 0.30·propagation_risk
              + 0.20·variance_risk + 0.15·extreme_risk
```

This is **purely data-driven** — no regulatory/human-factors knowledge yet.

---

## Stage 8 — Merge regulatory proxy features

[risk_scorer.py:128-147](risk_scorer.py#L128-L147):

```python
proxy_agg = proxy_df.groupby(["airport_a","airport_b"]).agg(
    duty_flag       = max,     # any day of 2019-2025 with duty > 14h
    mct_violation   = max,     # any day where DFW connection < 45 min
    wocl_flag       = max,     # any leg fell in 2-6 AM circadian low
    wocl_multiplier = max,     # 1.35 if WOCL, else 1.0
    avg_duty_mins   = mean,
    avg_conn_mins   = mean,
).reset_index()
pairs = pairs.merge(proxy_agg, on=["airport_a","airport_b"], how="left")
```

Missing proxies default to 0 / 1.0. Now each pair row holds both the ML-learned risk and the rule-based regulatory flags.

---

## Stage 9 — Final pair score

[risk_scorer.py:150-154](risk_scorer.py#L150-L154):
```python
final_score = ml_risk_score × wocl_multiplier × (1 − mct_violation)
```

Three intuitions baked in:
1. **MCT violation is disqualifying** — `(1 − 1) = 0` zeroes the score (legally infeasible turn).
2. **WOCL exposure amplifies** — 35% risk boost when the crew touches the 2-6 AM circadian trough.
3. **Duty-flag is informational**, not a multiplier (logged, not applied — can surface in reports).

Sorted descending → worst pairs at top.

---

## Stage 10 — Export contracts

[risk_scorer.py:158-189](risk_scorer.py#L158-L189) writes three CSVs to `./results/`:

| File | Rows | Purpose |
|---|---|---|
| **`pair_risk_scores_full.csv`** | all ~22k pairs | Full debug with every intermediate column |
| **`scored_pairs.csv`** | all pairs | Output-contract columns only: `airport_a, airport_b, ml_risk_score, wocl_multiplier, final_score, duty_flag, mct_violation, wocl_flag` |
| **`flagged_pairs.csv`** | pairs with `final_score ≥ 0.6` | Plus `recommendation ∈ {AVOID, CRITICAL}` where CRITICAL = `final_score ≥ 0.8` |

Per-season variants are also produced ([main.py:375-385](main.py#L375-L385)):
- `pair_risk_scores_winter.csv`, `..._spring.csv`, `..._summer.csv`, `..._fall.csv`
- `flagged_pairs_<season>.csv`

Each re-runs Stages 5-9 on the season-filtered slice of `flight_preds`.

---

## Stage 11 — Metrics manifest

[main.py:360-372](main.py#L360-L372) writes `results/metrics.json`:
```json
{
  "test_years": [2025],
  "overall": {"MAE": ..., "RMSE": ..., "R2": ..., "beta": ..., "n_flights": ...},
  "seasonal": {
    "winter": {"n": ..., "MAE": ..., "RMSE": ..., "R2": ..., "extreme_pct": ..., "months": [12,1,2]},
    ...
  }
}
```

`beta` is the **learned propagation-decay parameter** (softplus of `DelayPropagationModule.beta`). Paper-reported range is 0.73–0.89; big deviations hint at data mismatch.

---

## End-to-end flow diagram

```
processed_flights.parquet
      ↓ filter Year==2025
test_df (398k 2025 rows)
      ↓ FlightChainDataset
test_loader (batches of seq_len=14 chains)
      ↓ TFTDCP.forward (per batch)
flight_preds        ← (origin, dest, year, month, pred_delay, actual_delay, propagated_delay)
      ↓ split at DFW
inbound (→DFW)  outbound (DFW→)
      ↓ groupby origin  ↓ groupby dest
a_stats           b_stats                 proxy_sequences.parquet
      ↓ cross-join                               ↓ groupby (airport_a, airport_b) max
pairs (≈22k rows)  ←——————————————————————  proxy_agg
      ↓ 4 sub-risks → weighted sum
ml_risk_score
      ↓ × wocl_multiplier × (1 − mct_violation)
final_score
      ↓ sort desc, threshold at 0.6 / 0.8
scored_pairs.csv, flagged_pairs.csv (+ per-season)
```

---

## Where the three layers meet

| Layer | Source | Kind | Effect on final |
|---|---|---|---|
| **ML risk** (delay/prop/var/extreme) | Model predictions on 2025 | Continuous, data-driven | Base score |
| **WOCL** | Rule on scheduled HHMM | Boolean → multiplier 1.0 or 1.35 | Amplifier |
| **MCT** | Rule on `dep_b − arr_a < 45 min` | Boolean 0/1 | Hard gate (zeroes score) |

`duty_flag` rides along for reporting but doesn't change `final_score`. That's a deliberate choice — duty is typically managed at the roster level rather than the pair level, but the column is there if you want to turn it into a second multiplier.
