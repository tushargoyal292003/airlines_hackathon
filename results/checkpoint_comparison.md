# Checkpoint Comparison (Baselines vs TFT-DCP)

| Rank (MAE) | Model | MAE | RMSE | R2 | ΔMAE vs TFT-base | ΔRMSE vs TFT-base | ΔR2 vs TFT-base |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM | 11.55 | 33.09 | 0.7942 | -9.13 | -25.42 | +0.5243 |
| 2 | XGBoost | 12.10 | 34.11 | 0.7813 | -8.58 | -24.40 | +0.5114 |
| 3 | TFT-baseline | 20.68 | 58.51 | 0.2699 | +0.00 | +0.00 | +0.0000 |
| 4 | TFT-DCP (ours) | 21.64 | 58.24 | 0.2767 | +0.96 | -0.27 | +0.0068 |
| 5 | LSTM | 25.06 | 62.87 | 0.1571 | +4.38 | +4.36 | -0.1128 |
| 6 | TCN | 25.45 | 62.57 | 0.1651 | +4.77 | +4.06 | -0.1048 |
| 7 | Informer | 27.33 | 64.00 | 0.1264 | +6.65 | +5.49 | -0.1435 |
| 8 | HA | 30.12 | 73.01 | -0.0020 | +9.44 | +14.50 | -0.2719 |

Notes: negative ΔMAE/ΔRMSE is better; positive ΔR2 is better.

## Source Files
- LightGBM: `checkpoints_baselines/baselines/lightgbm_metrics.json`
- XGBoost: `checkpoints_baselines/baselines/xgboost_metrics.json`
- TFT-baseline: `checkpoints_baselines/baselines/tft_baseline_metrics.json`
- TFT-DCP (ours): `results/metrics.json`
- LSTM: `checkpoints_baselines/baselines/lstm_metrics.json`
- TCN: `checkpoints_baselines/baselines/tcn_metrics.json`
- Informer: `checkpoints_baselines/baselines/informer_metrics.json`
- HA: `checkpoints_baselines/baselines/ha_metrics.json`
