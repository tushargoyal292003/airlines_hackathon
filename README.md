# TFT-DCP: Flight Delay Propagation Prediction

Implementation of the **Temporal Fusion Transformer with Dynamic Chain Propagation** (TFT-DCP) 
from Guo et al. (2025), adapted for the American Airlines hackathon problem:  
**Identify airport pairs (A, B) that should not be in the same pilot sequence through DFW.**

## Architecture

```
                    ┌─────────────────────┐
                    │   Dynamic Features  │
                    │  (flight + weather)  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │    TCN Encoder       │  ── h_dynamic, h_global
                    │  (dilated causal)    │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │ Historical │     │   Static    │     │   Delay     │
    │ Retrieval  │     │  GRN Embed  │     │ Propagation │
    │ (top-k)    │     │             │     │ (learnable β)│
    └─────┬─────┘     └──────┬──────┘     └──────┬──────┘
          │                   │                   │
          └───────────┬───────┘                   │
                      │                           │
              ┌───────▼───────┐                   │
              │   MS-CA-EFM   │                   │
              │  (channel     │                   │
              │   attention)  │                   │
              └───────┬───────┘                   │
                      │                           │
                      └─────────┬─────────────────┘
                                │
                      ┌─────────▼─────────┐
                      │  Prediction Head  │  ── delay (min)
                      └─────────┬─────────┘
                                │
                      ┌─────────▼─────────┐
                      │  Pair Risk Scorer  │  ── A→DFW→B risk
                      └───────────────────┘
```

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place BTS on-time data CSVs in `./data/raw/bts/` and NOAA LCD CSVs in `./data/raw/noaa/`.

- BTS: https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp
- NOAA: https://www.ncdc.noaa.gov/cdo-web/datasets/LCD/stations

### 3. Run Pipeline
```bash
# Full pipeline
python main.py --mode all --hub DFW

# Or step by step
python main.py --mode preprocess --bts-dir ./data/raw/bts --noaa-dir ./data/raw/noaa
python main.py --mode train --epochs 100 --batch-size 256
python main.py --mode evaluate
```

### 4. Multi-GPU Training (4x A6000)
The training script auto-detects GPUs and uses DistributedDataParallel.
Effective batch size = batch_size × num_gpus = 256 × 4 = 1024.

## Output
- `results/pair_risk_scores.csv` — risk score for every (A, B) pair
- `results/flagged_pairs.csv` — high-risk pairs to avoid in pilot sequences
- `checkpoints/best_model.pt` — trained model weights
- `logs/training_history.json` — training metrics per epoch

## Paper Reference
Guo, J.; Li, J.; Yuan, J.; Yang, Y.; Ren, Z. "A Data-Driven Framework for Flight Delay 
Propagation Forecasting During Extreme Weather." *Mathematics* 2025, 13, 3551.
