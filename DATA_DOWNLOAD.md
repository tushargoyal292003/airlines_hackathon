# TFT-DCP — Data Sources & Gap Analysis

## Problem
Identify (A, B) airport pairs that should **not** share a pilot sequence through
DFW — i.e., routes where the cascading-delay risk of a A→DFW→B pairing is high
enough that the airline should avoid assigning them to the same crew.

## Current Data Sources

| Source | Granularity | Coverage | What It Gives Us |
|---|---|---|---|
| **BTS On-Time** | Flight × Day | 2019, 2022-2024 (48 mo) | Flight-level actuals: `DepDelay`, `ArrDelay`, taxi times, tail numbers, delay-cause splits (carrier, weather, NAS, security, late-aircraft), cancellations |
| **NOAA LCD** | Airport × Hour × Day | 2019, 2022-2025, 147 airports | Hourly weather at origin/destination: wind, visibility, precipitation, temperature, present-weather type, sky condition |
| **FAA ASPM City-Pair** | Route × Month × Hour | 2019, 2022-2025, 2,176 DFW-connected city-pairs | Seasonal/structural route risk: EDCT counts, gate departure delay, taxi-out delay, airport departure delay, airborne delay, block delay, on-time %, per (A→DFW) and (DFW→B) direction |

### How They Compose

```
BTS (day-specific outcome)  ─┐
                              ├── Day-specific signals
NOAA (day-specific weather) ─┘

                              ┌── Seasonal/structural baseline
ASPM (route × month × hour) ─┘
```

BTS and NOAA carry the **day-specific** risk; ASPM carries the **structural
seasonality** of each route × time-of-day. Together they cover:
- *"Is today going to be bad?"* (BTS + NOAA)
- *"Is this route chronically bad at this time of year/day?"* (ASPM)

## Data We Are Missing

These are gaps — worth listing in the hackathon final report so reviewers
understand what would strengthen the model if it were available.

### Tier 1 — Materially Changes the Problem

1. **Crew / pilot pairing data**
   - The problem is framed as *pilot* sequence risk, but we have no crew
     identifiers. We are using **tail number** (aircraft) as a proxy for the
     A→DFW→B sequence.
   - Aircraft routing and crew routing overlap heavily at hubs but are not
     identical — crews often "change iron" between legs, and duty-time rules
     (FAR 117) impose cascading constraints the model cannot see.
   - **Impact:** We're predicting *aircraft-level* delay propagation risk. To
     predict *crew-level* risk we'd need proprietary airline crew-pairing
     data (likely from the hackathon sponsor).

2. **FAR 117 / Crew duty-time state**
   - If a crew is approaching max duty hours, even a small DFW delay can
     trigger a crew rollover that propagates into a cancelation — an
     operationally different outcome than a simple delay.
   - Without this we cannot distinguish "90-minute delay" from "90-minute
     delay that triggered a crew rest illegal → flight cancelled."

### Tier 2 — Would Meaningfully Improve Predictions

3. **FAA TFMS / GDP historical logs**
   - ASPM gives us `Arrivals With EDCT` count, but not the GDP start/end
     times, scope airports, or program rate. TFMS logs (FAA SWIM or OIS)
     provide this.
   - **Impact:** Currently we can detect *that* a GDP affected a route;
     we can't condition on the *severity* of the program.

4. **Gate assignment / turnaround ground ops**
   - The A→DFW→B problem hinges on the turnaround at DFW. We compute
     `turnaround_minutes` from scheduled times, but actual gate availability,
     gate-change events, and ground crew delays are invisible.
   - **Impact:** Would improve the delay-propagation module β learning.

5. **Daily-granularity ASPM**
   - Current ASPM data is monthly. We have no way to distinguish
     "DFW→LAX at 5pm on July 3rd" from "DFW→LAX at 5pm on July 27th."
     Day-specific signal comes from NOAA/BTS instead.
   - **Impact:** Would sharpen the "structural vs. day-specific" split.
     Not critical for seasonal trends, which is the current use.

### Tier 3 — Nice-to-Have

6. **TAF forecasts (not just LCD observed weather)**
   - LCD is observed weather — great for training, but for forward-looking
     inference we'd want the TAF that was available at schedule-planning
     time.

7. **Aircraft type / equipment**
   - BTS has tail number but not equipment type. Wide-bodies vs. regional
     jets have very different minimum turn times and weather thresholds.

8. **Runway configuration**
   - Affects per-hour capacity. Weakly captured by ASPM's `Airport Dep Delay`
     aggregate, but not directly observable.

9. **Maintenance / MEL status**
   - Tail-specific maintenance issues (Minimum Equipment List items) cause
     delays that look random from the model's perspective.

## What We Have Is Enough

Despite the gaps, the current three-source combination covers the strongest
signals:

- **Delay propagation**: BTS `LateAircraftDelay` + chain-level propagation
  module with learnable β decay
- **Weather**: NOAA hourly at origin airport
- **Route structural risk**: ASPM city-pair × month × hour
- **Congestion**: BTS-derived hourly operational density + ASPM airport
  departure delay

The model should produce meaningful A→DFW→B pair risk rankings. The
**aircraft-level → pilot-level** extrapolation is the single biggest caveat to
flag in the final write-up.

## Pipeline Status

- [x] BTS loading + chain construction (48 months, 2019 + 2022-24)
- [x] NOAA LCD loading + weather merge (147 airports)
- [x] ASPM city-pair parser (HTML/.xls format, 255,542 rows, 5 years, 2,176 city-pairs)
- [x] ASPM merged on (Origin, Dest, Year, Month, Dep Hour) into flight features
- [x] Features exposed as dynamic inputs to TCN encoder
- [x] Multi-GPU (4x RTX 6000 Ada) DDP training ready

## Commands to Run

```bash
# Full preprocessing with all three sources
python main.py --mode preprocess \
    --bts-dir ./data/data_bts/raw/bts \
    --noaa-dir ./data/data/raw/noaa \
    --aspm-dir ./data/data_aspm

# Training (auto-detects 4 GPUs)
python main.py --mode train --epochs 100 --batch-size 256

# Evaluation + pair risk scoring
python main.py --mode evaluate
```
