# The TFT-DCP Pilot-Sequence Risk Project — Full Explainer

Written for someone who understands basic ML terms (features, training, loss, MAE) but hasn't seen these specific architectures or data preprocessing patterns before. By the end, you should be able to defend any line of this project in a judge Q&A.

---

## Part 1 — The problem, restated plainly

American Airlines assigns pilots to "sequences" of flights. In this hackathon, a sequence is exactly **three legs**: a pilot flies from airport **A** into DFW, then from DFW out to airport **B**. If A and B are both weather-prone, or the layover at DFW is tight, or the whole shift runs past legal duty limits, or it lands the crew in the 2-6 AM fatigue window, the combination (A → DFW → B) is **operationally risky**.

We are asked to rank every possible (A, B) combination and surface the worst ones so American can avoid building pilot rosters around them.

We do **not** need to predict "will this specific flight on this specific day be delayed." That's the *means*. The *end* is a ranked list of pair combinations.

Our deliverable is a CSV of ~33,000 (A, B) pairs sorted by a **final risk score**, with the worst 27 flagged CRITICAL or AVOID.

---

## Part 2 — The data sources

### 2.1 BTS On-Time Performance (Bureau of Transportation Statistics)

The backbone of the project. A U.S. government dataset where every scheduled commercial flight reports:
- Scheduled and actual departure/arrival times
- Origin, destination, carrier, tail number (specific aircraft ID)
- Delay causes broken out (carrier delay, weather delay, NAS delay, security delay, late-aircraft delay)
- Cancellation / diversion flags
- Taxi-out and airborne times

We downloaded **60 months** of CSVs spanning 2019 + 2022-2025 (skipping 2020-2021 pandemic distortion). Raw size: ~15 GB uncompressed.

### 2.2 NOAA Local Climatological Data (LCD)

Hourly weather observations at weather stations colocated with airports. Per hour we get wind speed, visibility, precipitation, temperature, humidity, cloud cover. We pulled 736 airport-year CSVs covering every origin that appears in BTS.

**Why hourly, not daily?** A 3-hour thunderstorm window can delay every 4pm departure while leaving noon departures untouched. Daily averages hide this.

### 2.3 FAA ASPM (Aviation System Performance Metrics)

FAA's city-pair delay and taxi-time statistics. Monthly rollups per origin-destination-hour combination, covering queuing delay, EDCT (Expect Departure Clearance Time — ground stops), gate delay, taxi-out delay, airport departure delay, airborne delay, taxi-in delay, block delay, gate arrival delay.

ASPM files arrive as **.xls that are actually HTML tables**. A custom parser (`ASPMProcessor`) uses `pandas.read_html` instead of `read_excel`.

**Why ASPM on top of BTS?** BTS tells you *what* happened on a specific flight. ASPM tells you *the historical tendency* of a route. A flight out of PHX→DFW at 3pm in July has an ASPM-derived "typical taxi-out delay of 12 min"; this baseline helps the model calibrate expectations.

---

## Part 3 — Data preprocessing, step by step

This is where a data analytics hackathon actually gets won. Model architecture matters less than getting the data into a sane state.

### 3.1 Step 1 — Hub-filtered streaming load

BTS contains every U.S. flight. We only care about flights touching DFW. But loading 60 months of all-U.S. CSVs into memory (~15 GB) would OOM.

Solution: **stream each CSV in 100k-row chunks**, inside each chunk keep only rows where `Origin == "DFW"` OR `Dest == "DFW"`, discard the rest. Memory usage stays under 2 GB no matter how much raw data we process.

Typical reduction: 95-98% of raw rows dropped immediately.

### 3.2 Step 2 — Deduplication

BTS actually publishes each flight **twice** — once from the reporting carrier's perspective, once from the marketing carrier's. For codeshares (American sells a seat on a Delta-operated flight, for instance) both carriers file the same flight with slightly different metadata.

Dedup key: `(Tail_Number, FlightDate, CRSDepTime, Origin, Dest)`. That uniquely identifies an aircraft-day-departure. Drops ~50% of rows.

### 3.3 Step 3 — Anomaly detection

BTS has keyboard-entry errors. We remove rows where:
- TaxiOut > 200 minutes (physically impossible — no one sits on a taxiway for 3.5 hours)
- Distance ≤ 0 (obviously broken)
- Departed early (DepDelay < 0) but arrived 3+ hours late (contradictory — likely a field swap)
- DepDelay > 24 hours (almost certainly a date-rollover clerical error)

**Critical exception**: rows that represent *genuine* extreme delays (≥ 180 min) are **never removed**, even if they trigger the anomaly rules. The whole point of the model is to predict these rare-but-costly events; throwing them out because they look "weird" would defeat the exercise.

This is a small but principled decision — the paper explicitly calls it out and we preserved the intent.

### 3.4 Step 4 — Cancelled/Diverted + missing-critical filtering

- `Cancelled == 1` → dropped (the flight never happened, there is no "delay" to predict)
- `Diverted == 1` → dropped (different operational phenomenon, would pollute the regression target)
- Missing `DepDelay`, `Tail_Number`, `Origin`, `Dest`, or `CRSDepTime` → dropped (these are structural — no model can impute them sensibly)
- Missing delay-cause breakdowns (CarrierDelay, WeatherDelay, etc.) → **filled with 0**, not dropped. BTS only populates those columns when a flight was actually delayed. "Blank" means "no delay from that cause," which is a legitimate value.

### 3.5 Step 5 — Airport feature extraction

For each flight we compute derived "how busy is the airport right now?" features:
- `op_density`: how many flights are scheduled from this origin in the same departure hour
- `hist_peak_capacity`: the historical max of `op_density` at this airport
- `utilization_rate`: `op_density / hist_peak_capacity` — are we near peak or well below it?
- `cum_delay_past_hour`: rolling sum of delays from flights that have already departed this origin in the past hour

This is the "gate-level congestion" signal. A flight out of DFW at 5 PM on a Friday has a very different prior than the same flight at 11 AM on a Tuesday.

### 3.6 Step 6 — Weather merge

NOAA hourly observations joined onto each flight on **(airport_code, scheduled_departure_hour)**. A flight scheduled to leave JFK at 1447 local time picks up JFK's weather from the 14:00 observation.

Merge type is **left join**: if a flight has no matching weather hour (station offline, data gap), the flight row is kept with NaN weather. Those NaNs get filled to 0 at dataset-construction time, which in the scaled [0, 1] space means "exactly average."

### 3.7 Step 7 — ASPM merge

Monthly ASPM statistics joined on **(Origin, Destination, Year, Month, Departure Hour)**. Adds 14 city-pair features like `aspm_gate_dep_delay`, `aspm_avg_taxi_out`, `aspm_pct_ontime_gate_arr`.

Why a 5-dimensional merge key? ASPM is month × city-pair × hour granularity; using only (Origin, Dest) would collapse 12 months of Phoenix-Dallas into one number. The hourly dimension matters because morning and evening DFW operations are structurally different.

### 3.8 Step 8 — Derived features

Things computed from raw BTS fields:
- `actual_taxi_out = WheelsOff − DepTime` — real wheels-up minus real pushback
- `taxi_out_excess = actual_taxi_out − TaxiOut` — how much we overran the scheduled taxi time
- `actual_airborne = WheelsOn − WheelsOff`
- `airborne_excess = actual_airborne − AirTime`

"Excess" features encode congestion that the airline didn't anticipate when building the schedule.

### 3.9 Step 9 — Flight chain construction

This is the **single most important preprocessing step**, because it's what turns individual flights into sequences the model can reason about.

Define a **chain_id** = `Tail_Number + "_" + FlightDate`. Every flight operated by the same aircraft on the same calendar day is part of the same chain.

Why this matters: airlines don't fly each leg fresh. A single aircraft might fly DFW → AUS → DFW → OKC → DFW → ABQ in one day. A delay on leg 1 propagates through every subsequent leg of that chain, because the same physical airplane has to be at the next origin by the scheduled pushback.

We sort each chain by departure time and assign `chain_position` (1st leg, 2nd leg, ...) and `chain_length`. For every leg after the first, we compute:
- `prev_arr_delay`: how late the inbound leg of this aircraft landed
- `turnaround_minutes`: time between inbound arrival and outbound scheduled departure
- `prev_carrierdelay`, `prev_weatherdelay`, `prev_nasdelay`, `prev_securitydelay`, `prev_lateaircraftdelay`: **why** the previous leg was late, broken out by cause

The model will use this chain history as input: "given the last 14 legs this aircraft flew today and how late each one was, and why, how late will the next leg be?"

### 3.10 Step 10 — Feature encoding

Two different kinds of features need two different transformations.

**Categorical variables** (`Origin`, `Dest`, `Reporting_Airline`, `Tail_Number`, `OriginState`, `DestState`) can't be fed to a neural network as strings. Options:

- **One-hot encoding**: creates one column per airport code. With 300+ airports × 2 positions, that's 600+ columns of mostly zeros. Bloats model size.
- **Label encoding** (airport "ABQ" = 1, "ATL" = 2, ...): compact but meaningless — model thinks 2 > 1, which has no semantic truth.
- **James-Stein encoding** (what we chose): replaces each category with the **mean target value** for that category, regularized toward the global mean. So "ORD" gets replaced by the average DepDelay across all ORD flights, nudged toward the overall average based on sample size.

James-Stein is "target-aware" — the model starts with a sensible numerical value for each airport that already correlates with delay. It makes the network easier to train on sparse categories.

**Numeric variables** get Min-Max scaled to [0, 1] so no single feature (e.g., Distance ranging 100-4000 miles) dominates the gradient updates.

Columns we explicitly do **not** scale:
- `DepDelay`, `DepDelayMinutes` (the label — keep in raw minutes so MAE is interpretable)
- `chain_id`, `chain_position` (identifiers, not features)
- `Year_raw`, `Month_raw`, `DayOfWeek_raw` (kept for splitting train/val/test by year)
- `CRSDepTime_raw`, `CRSArrTime_raw`, `DepTime_raw`, `ArrTime_raw` (kept for proxy engineering's duty/MCT/WOCL calculations, which need real HHMM times)
- `Month_sin/cos`, `DayOfWeek_sin/cos` (cyclical encoding — see next subsection, already in [-1, 1])

### 3.11 Step 11 — Cyclical time encoding

Month and day-of-week are categorical-but-ordered AND **they wrap around**. December (12) is adjacent to January (1), but to a naive model 12 and 1 look maximally far apart.

Fix: represent each cyclical variable as **two columns of a unit circle**:
```
Month_sin = sin(2π · Month / 12)
Month_cos = cos(2π · Month / 12)
```
December and January end up at almost-identical (sin, cos) coordinates. The model now sees "this flight is in the winter band" instead of "this flight's month is the number 12."

Same trick for DayOfWeek (period = 7).

### 3.12 Step 12 — Year-based split

Because we want to prove generalization to a **true future** year, we split by calendar year, not randomly:
- **Train**: 2019, 2022, 2023
- **Validation**: 2024
- **Test (held-out)**: 2025

Random train/test splits would let information leak across the time boundary (the model would see near-neighbor flights in train that co-occurred with test flights). Year-based splits force the model to extrapolate forward, which is what airlines actually need.

---

## Part 4 — Proxy feature engineering (the regulatory layer)

Four flags that capture rules no ML model can learn from data alone — because they're regulatory or physiological constants, not statistical patterns.

### 4.1 Duty time flag (FAA 14-hour rule)

Federal Aviation Regulations (FAR 117) limit pilot duty time to 14 hours per shift. If the A → DFW → B sequence (from inbound departure time at A to outbound arrival time at B) exceeds 840 minutes, `duty_flag = 1`.

### 4.2 MCT violation (Minimum Connection Time)

DFW's operational minimum for a legal aircraft-to-aircraft turn is **45 minutes**. If the scheduled DFW layover (outbound departure − inbound arrival) is less than 45 minutes, `mct_violation = 1`. Such a pair is essentially infeasible — there isn't enough time for refueling, catering, crew change.

### 4.3 WOCL exposure (Window of Circadian Low)

Sleep-physiology research shows pilot alertness bottoms out between **2 AM and 6 AM local time** — the "WOCL." Any sequence whose legs depart or arrive in this window gets `wocl_flag = 1` and a **1.35× risk multiplier**.

### 4.4 How these are computed — the scaling issue we had to solve

Proxy features are derived from raw HHMM departure/arrival times. But preprocessing Min-Max scales everything, turning `CRSDepTime = 1430` (2:30 PM) into `0.604`. That destroys the HHMM arithmetic we need.

Fix: explicitly preserve `CRSDepTime_raw` and `CRSArrTime_raw` columns before scaling. The proxy engineer reads the `_raw` columns; the neural network reads the scaled ones.

### 4.5 Day-chunked enumeration

Naively, to get every valid A → DFW → B pair across 5 years, you'd cross-join 1.45M inbound flights with 1.45M outbound flights on `FlightDate`. That produces 1.1 **billion** rows and OOMs.

We process **one calendar day at a time** (~800 inbound × 800 outbound = 640k rows per day, trivial memory), compute the proxy features, then aggregate to (airport_a, airport_b) worst-case within that day. Finally concatenate 1,826 daily aggregates and take the overall worst case per pair.

Result: 40,438 unique (A, B) pairs with their regulatory flags, saved as `proxy_sequences.parquet`.

---

## Part 5 — The model (TFT-DCP)

TFT-DCP = **Temporal Fusion Transformer with Dynamic Chain Propagation**. A paper-proposed architecture specifically for flight delay prediction. It has five components plus a prediction head.

### 5.1 What the model sees per training example

One training example is **one target flight** plus up to 14 preceding legs in its chain. Concretely:

| Tensor | Shape | Meaning |
|---|---|---|
| `dynamic` | (14, ~45) | 14 prior legs × 45 time-varying features (flight + weather + airport state) |
| `static` | (10,) | 10 features describing the target flight itself |
| `chain_delays` | (14,) | how late each prior leg was |
| `turnaround_times` | (14,) | ground minutes between consecutive prior legs |
| `mask` | (14,) | 1 for real legs, 0 for padding (chains < 14 legs) |
| `target` | scalar | the DepDelay (in minutes) we're trying to predict |

Short chains are left-padded with zeros. Chains longer than 14 are left-truncated (we keep the most recent 14 legs).

### 5.2 Module 1 — TCN Encoder (handles dynamic features)

**TCN** = Temporal Convolutional Network. It's an alternative to LSTMs/GRUs for sequence modeling, with two nice properties:
- **Dilated convolutions** give it a large receptive field with few layers. A dilation-1 then dilation-2 then dilation-4 stack sees 14 time steps with only 3 conv layers.
- **Causal**: each output position only sees inputs up to that position, so it can't cheat by peeking at the future.

Input: `dynamic` tensor `(batch, 14, ~45)`.
Output: per-step embeddings `h_dynamic (batch, 14, 128)` and a mask-pooled chain-global summary `h_global (batch, 128)`.

Intuition: the TCN reads the flight chain left-to-right like a miniature language model, building up a representation of the "story so far" of this aircraft's day.

The last valid position's embedding becomes `h_current` — the contextualized representation of the **target leg** given its chain history.

### 5.3 Module 2 — GRN (handles static features)

**GRN** = Gated Residual Network. A feed-forward block with a skip connection and a learnable gate:
```
y = gate · transform(x) + (1 − gate) · x
```
The gate lets the network decide, per-sample, whether to transform the input or pass it through unchanged. That's useful when some static features (e.g., Distance) are informative for one flight and irrelevant for another.

Input: `static (batch, 10)`.
Output: `h_static (batch, 128)`.

`h_static` is then added into both `h_current` and `h_global` — the sequence representation gets **contextualized by the target flight's identity**.

### 5.4 Module 3 — Historical Retrieval

The model keeps a **memory bank** of up to 50,000 past `h_current` embeddings from previously-seen flights. For each new sample:

1. Compute cosine similarity between the new `h_current` and every vector in the bank.
2. Take the top-K=5 most similar past flights.
3. Compute a weighted average of their embeddings, weighted by similarity — call it `h_f`.

Intuition: "Here's a flight that looks like ours — specifically, one where the aircraft had similar taxi patterns, weather, prior-leg delays, and airline operational context. What happened to that flight?"

This is especially useful for rare extreme delays. A fresh-gradient neural network struggles to learn patterns that appear once per 1,000 flights. Retrieval augments it with explicit lookups.

After each batch, the bank is updated (FIFO) with the current batch's `h_current`. By the end of training, the bank has seen a representative cross-section of the whole dataset.

### 5.5 Module 4 — MS-CA-EFM Fusion

**MS-CA-EFM** = Multi-Source Channel-Attention Explicit Feature Mixer. Takes three hidden vectors:
- `h_current` — the target leg given its chain history
- `h_f` — similar past flights
- `h_global` — chain-level summary

And fuses them. "Channel attention" means it learns which of the 128 hidden-dimensions are important for each source. Some dimensions might carry weather signal, others carry congestion signal — attention lets the network weight them differently depending on which source vector they came from.

Output: `h_fused (batch, 128)`.

Intuition: "Blend what this specific chain tells me with what similar historical flights tell me with what the whole chain summary tells me, and figure out which feature channels to trust from each source."

### 5.6 Module 5 — Dynamic Chain Propagation

This is the module the architecture is named for. It models **how delay physically carries through a flight chain** via a learnable decay:

```
y_prop ≈ Σ_k exp(−β · turnaround_k) · chain_delays_k
```

Reading this: "Sum up every prior leg's delay, but discount older/longer-turnaround legs by an exponential factor." A short turnaround (30 min gate time) passes delay through almost fully; a long turnaround (3 hours) gives the airline time to recover, so delay decays.

**β is a single scalar parameter the model learns during training.** The paper reports β in the range 0.73-0.89 across their datasets; we learned β ≈ 0.69, slightly lower, meaning our model finds delay decays a bit slower in our 5-year DFW sample than in the paper's dataset.

Output: `y_prop` (scalar delay prediction from propagation alone) and `h_prop (batch, 128)` (hidden representation of the chain state).

This module is **interpretable**: `y_prop` has a direct operational meaning ("minutes of delay expected from chain propagation"), separate from whatever the rest of the network predicts. An operations center could look at `y_prop` alone to ask "how much of tomorrow's DFW-ABQ delay is just because the plane was late coming in?"

### 5.7 Prediction head

Two hidden vectors feed the final MLP:
- `h_fused` — everything the fusion module learned (data-driven risk)
- `h_prop` — the propagation module's chain-decay representation

Concatenated to `h_final (batch, 256)`, passed through a 3-layer MLP with ReLU + dropout, outputs a scalar: the predicted DepDelay in minutes.

### 5.8 Loss and training

- **Loss**: Mean Squared Error between predicted and actual DepDelay. MSE (vs MAE) penalizes big misses more — appropriate because large delays are the operationally costly events.
- **Optimizer**: Adam, learning rate 0.001, weight decay 1e-5, dropout 0.1.
- **Batch size**: 128 per GPU × 4 GPUs = effective batch 512 (DistributedDataParallel).
- **Early stopping**: patience of 10 epochs on validation MAE. We stopped at epoch 51; best weights were from epoch 41.

### 5.9 Why not just a random forest?

Fair question. Random forests and gradient boosting (XGBoost, LightGBM) are strong flight-delay baselines. TFT-DCP adds value when:

1. The chain structure matters (legs depend on each other) — trees don't naturally model sequences.
2. The static/dynamic split is meaningful (some features describe the target, others describe its history) — TFT-style architectures handle this cleanly.
3. Rare extreme events need explicit retrieval — trees memorize training data but don't have an attention-based "find similar past example" mechanism.

We include LSTM baseline code for comparison in `baselines.py` but didn't run full benchmarks for this hackathon due to time constraints.

---

## Part 6 — From delay predictions to pair risk scores

Model predicts delays per flight. We need ranks per pair. The bridge:

### 6.1 Collect per-flight predictions on 2025

Run the trained model on every 2025 flight, write `flight_predictions.csv` with:
- `origin`, `dest`, `month`
- `pred_delay` (model output)
- `actual_delay` (ground truth)
- `propagated_delay` (the `y_prop` scalar from the chain module)

### 6.2 Split at DFW

- **Inbound** = rows where `dest == "DFW"` → these represent the A-leg of sequences
- **Outbound** = rows where `origin == "DFW"` → these represent the B-leg

### 6.3 Aggregate to airport level

For each origin airport A, compute across all 2025 inbound DFW flights from A:
- `avg_delay_a` — mean predicted delay
- `std_delay_a` — standard deviation (schedule unreliability)
- `extreme_pct_a` — % of flights with predicted delay > 180 min
- `n_flights_a` — sample count

Same for each destination airport B on the outbound side (plus `avg_prop_delay` from the propagation module).

### 6.4 Cross-join A × B

Every origin × every destination = ~22,500 candidate pair rows. Each row carries both airports' statistics.

### 6.5 Build ML risk score (0-1)

Four sub-scores, each normalized to [0, 1] using 99th-percentile clipping:

| Sub-score | Formula | Meaning |
|---|---|---|
| `delay_risk` | mean delay / P99 | "Do pairs tend to run late?" |
| `propagation_risk` | mean propagation / P99 | "Does the chain-decay model expect carry-through?" |
| `variance_risk` | (std_a + std_b) / (2 × max_std) | "Is the schedule reliable?" |
| `extreme_risk` | extreme_pct_a / max | "Any long-tail blowups?" |

Weighted sum (weights from the paper):
```
ml_risk_score = 0.35 · delay_risk
              + 0.30 · propagation_risk
              + 0.20 · variance_risk
              + 0.15 · extreme_risk
```

### 6.6 Apply regulatory layer

Merge `proxy_sequences.parquet` on `(airport_a, airport_b)`. Each pair now carries `duty_flag`, `mct_violation`, `wocl_flag`, `wocl_multiplier`.

### 6.7 Final score

```
final_score = ml_risk_score × wocl_multiplier × (1 − mct_violation)
```

Three pieces of operational knowledge baked in:
1. **MCT violation is disqualifying** — if the pair requires an infeasible turn, the final score is 0 regardless of how good or bad the data-driven risk was. You simply can't staff this pair.
2. **WOCL exposure amplifies** — crews in the circadian-low window get a 1.35× risk multiplier because fatigue compounds whatever the data already predicts.
3. **Duty flag is informational**, not a multiplier. It's logged for human review rather than applied. Duty is typically managed at the roster level, above the pair level.

Sort descending. Pairs with `final_score ≥ 0.8` → **CRITICAL**. Pairs with `final_score ≥ 0.6` → **AVOID**. Everything else → unflagged but ranked.

### 6.8 Seasonal breakdown

Everything above is also recomputed per season (winter = Dec/Jan/Feb, spring = Mar/Apr/May, summer = Jun/Jul/Aug, fall = Sep/Oct/Nov). Produces `pair_risk_scores_{season}.csv` and `flagged_pairs_{season}.csv` — the per-season deliverables directly addressing the challenge's seasonality question.

---

## Part 7 — Reading the results

### 7.1 Flight-level metrics (on held-out 2025)

| Metric | Value | How to read it |
|---|---|---|
| MAE | 21.6 min | On average, predictions are off by 21.6 minutes |
| RMSE | 58.2 min | Penalizes big misses; long-tail flights hurt this |
| R² | 0.28 | Model explains 28% of variance in actual delays |

**Is R² = 0.28 good?** For flight delay regression on a future year the model has never seen, yes — published benchmarks for this task land in the 0.2-0.4 range. The remaining variance is genuinely unpredictable from the data we have (we don't see crew rosters, real-time ATC ground stops, ground-crew availability, passenger connection problems).

### 7.2 Seasonal gradient

| Season | N flights | MAE | Extreme % |
|---|---|---|---|
| Winter (Dec-Feb) | 94,816 | 19.3 | 1.7% |
| Spring (Mar-May) | 104,275 | 22.1 | 2.2% |
| Summer (Jun-Aug) | 110,265 | **25.7** | **2.9%** |
| Fall (Sep-Nov) | 101,405 | 19.0 | 1.9% |

**Summer is the hardest season** — thunderstorms, peak traffic, more extreme events. Winter is second-hardest (snow/ice). Shoulder seasons (spring, fall) are easiest. This matches every airline operations person's intuition.

### 7.3 Pair rankings

33,670 ranked pairs total. 27 flagged overall (0.08% — the challenge asked us to find the worst of the worst). Seasonal flags: 3 winter, 21 spring, 24 summer, 56 fall.

Top-10 flagged pairs were dominated by regional-to-regional sequences (MDT Harrisburg, RDM Redmond as origins) — small spokes with long-haul tails. Structurally these are the riskiest because the aircraft's day starts from a weather-exposed regional, turns at DFW, then heads to a long-haul destination (Anchorage, Honolulu). Any delay at A gets amplified through the full sequence.

### 7.4 Learned β

β = 0.69. Slightly below the paper's 0.73-0.89 band. Interpretation: in our 5-year DFW sample, delay propagates through turnarounds a bit **more strongly** than in the paper's dataset. That's consistent with DFW being a true hub-and-spoke operation where aircraft turnarounds are tightly scheduled with little slack.

---

## Part 8 — What we did not have access to

Honest limitations:

1. **Crew rosters**: we don't see who's actually flying. We can identify *risky combinations* but not *risky pilots* (new-hires, reserve callouts, fatigued crews coming off trips).
2. **TFMS / ATC ground stops**: FAA-imposed ground stops are in the FAA's TFMS feed, not in BTS. Some of our model's residual error is absorbing these.
3. **TAF forecasts**: we use historical NOAA (observed) weather. Pilots schedule against forecasted weather. Adding TAF terminal-area forecasts would close that gap.
4. **Aircraft type**: BTS has tail number but not aircraft type. A 737 vs an A319 have different crew qualifications and turn times.
5. **Gate / runway configuration**: DFW's runway config changes with wind direction, affecting taxi times. Not in our data.

All of these would be Phase-2 data adds in a production version.

---

## Part 9 — The 5 questions the challenge explicitly asks

### Q1. What features are important, what other data would you want?

**Using**: 60-month BTS + hourly NOAA + monthly ASPM. Inside, we have flight identity (Origin/Dest/Carrier/TailNumber), scheduled/actual times, prior-leg delays + causes, chain position/length, airport operational density, weather severity, city-pair ASPM rollups, cyclical calendar encoding.

**Would add**: crew rosters, TAF forecasts, real-time TFMS, aircraft type, gate/runway config.

### Q2. What type of model would work best?

Chain-aware sequence models. We used TFT-DCP because the flight-chain structure is sequential, static and dynamic features have different roles, and extreme delays benefit from explicit retrieval. Simpler alternatives: gradient-boosted trees (strong baseline) or LSTMs.

### Q3. How to handle seasonality?

Three layers:
- Year-based train/val/test split (not random) so the model generalizes forward in time.
- Cyclical encoding of Month and DayOfWeek (sin/cos pairs) so the model sees December and January as adjacent.
- Per-season evaluation and per-season deliverable CSVs.

### Q4. Sparsity of severe weather events?

Extreme delays (>180 min) are 1.9% of training data. Three mitigations:
- Keep them explicitly during anomaly filtering (never drop a real extreme, even if it looks "weird").
- Historical retrieval module pattern-matches against rare historical examples, so the model doesn't rely solely on gradient flow through 1.9%-minority cases.
- `wx_severity` aggregate weather-severity feature combines multiple NOAA dimensions into one stronger signal.

### Q5. What accuracy metrics?

Flight-level: MAE and RMSE (both in interpretable minutes), R² for variance explanation, extreme-delay rate per split.

Pair-level: the deliverable is a **ranking**, not a regression. We evaluate it by reviewing top-K flagged pairs for operational plausibility. In production you'd measure it with a **top-K recall against human-audited risky pairs** or by backtesting "did avoiding these pairs reduce historical cancellation rates?"

---

## Part 10 — Code map

Where each concept lives in the repo, so you can point judges at specific files if asked.

| Concept | File | Key function/class |
|---|---|---|
| Data pipeline driver | `data/preprocessor.py` | `DataPipeline.run()` |
| BTS loader (hub-filtered streaming) | `data/preprocessor.py` | `BTSProcessor.load()` |
| NOAA loader | `data/preprocessor.py` | `NOAAProcessor` |
| ASPM HTML-as-xls parser | `data/preprocessor.py` | `ASPMProcessor` |
| Anomaly rules | `data/preprocessor.py` | `AnomalyDetector.detect()` |
| Feature encoding | `data/preprocessor.py` | `_encode_features()` |
| Proxy feature engineering | `data/proxy_engineering.py` | `ProxyEngineer.run()` |
| Flight chain dataset | `data/dataset.py` | `FlightChainDataset` |
| Model orchestrator | `model/tft_dcp.py` | `TFTDCP.forward()` |
| TCN encoder | `model/tcn.py` | `TCNEncoder` |
| GRN | `model/grn.py` | `GatedResidualNetwork` |
| Historical retrieval | `model/historical_retrieval.py` | `HistoricalRetrievalModule` |
| MS-CA-EFM fusion | `model/ms_ca_efm.py` | `MSCAEFM` |
| Chain delay propagation | `model/propagation.py` | `DelayPropagationModule` |
| Training loop | `train.py` | `Trainer.train()` |
| Evaluation + scoring | `main.py` → `risk_scorer.py` | `evaluate()`, `PairRiskScorer` |
| Outputs | `results/` | `scored_pairs.csv`, `flagged_pairs.csv`, `pair_risk_scores_{season}.csv` |

---

## The one-line elevator pitch

**"We built a chain-aware neural network that learns how flight delays propagate through an aircraft's day, combined it with FAA regulatory flags (duty time, minimum connection, circadian low), and used it to rank every A→DFW→B pilot sequence in 2025 by operational risk — surfacing 27 CRITICAL combinations the airline should avoid scheduling around."**
