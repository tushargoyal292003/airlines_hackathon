"""
TFT-DCP: Main Entry Point
Orchestrates: preprocess → proxy engineering → train → evaluate → score pairs

Usage:
  python main.py --mode all           # Run everything
  python main.py --mode preprocess    # Step 1: Process raw data + build proxies
  python main.py --mode train         # Step 2: Train TFT-DCP model
  python main.py --mode evaluate      # Step 3: Evaluate + score pairs
  python main.py --mode baselines     # Step 4: Run baseline comparisons

Directory layout (matches DataConfig defaults):
  data/raw/bts/       ← On_Time_*.csv  (BTS on-time performance)
  data/raw/noaa/      ← {AIRPORT}_{StationID}_{Year}.csv  (NOAA LCD)
  data/raw/aspm/      ← {Year}-A.xls, {Year}-D.xls  (FAA ASPM)
  data/processed/     ← written by preprocess step, read by train/evaluate
  results/            ← written by evaluate/baselines steps
"""
import argparse
import json
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from sklearn.metrics import average_precision_score
except Exception:
    average_precision_score = None

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.preprocessor import DataPipeline, get_feature_groups
from data.dataset import FlightChainDataset, flight_collate_fn
from data.proxy_engineering import ProxyEngineer
from model import TFTDCP
from train import train_distributed, train_single_gpu, Trainer
from risk_scorer import PairRiskScorer
from causal_features import (
    CAUSAL_FEATURES, BLACKLIST, drop_blacklist,
    build_route_priors, attach_priors,
)


# ══════════════════════════════════════════════════════════════
# PATH HELPERS  — every file path goes through config, never hardcoded
# ══════════════════════════════════════════════════════════════

def processed_dir(config: Config) -> Path:
    """Directory where preprocessor writes and train/evaluate reads from."""
    p = Path(config.data.processed_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(config: Config) -> Path:
    """Directory where evaluate/baselines write outputs."""
    p = Path(config.data.results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════════════════
# PHASE 1: PREPROCESS
# ══════════════════════════════════════════════════════════════

def preprocess(config: Config, force: bool = False):
    """Run data pipeline + proxy engineering.

    If processed_flights.parquet already exists and `force=False`, skip the
    expensive Phase 1A load/merge and reuse it.  Set `force=True` (or delete
    the parquet) to rerun preprocessing from scratch.
    """
    flights_path = processed_dir(config) / "processed_flights.parquet"

    if flights_path.exists() and not force:
        print("\n" + "=" * 70)
        print(f"PHASE 1A: SKIPPED — reusing cached {flights_path.name}")
        print("=" * 70)
        df = pd.read_parquet(flights_path)
        print(f"  Loaded {len(df):,} records × {len(df.columns)} cols")
    else:
        print("\n" + "=" * 70)
        print("PHASE 1A: DATA PREPROCESSING")
        print("=" * 70)
        pipeline = DataPipeline(config.data)
        df = pipeline.run(save=True)          # saves processed_flights.parquet
                                              # to config.data.processed_dir

    # ── Framing A (causal): drop leaky cols, attach leak-safe route priors ──
    print("\n  [causal] dropping blacklist + attaching historical priors")
    year_col = "Year_raw" if "Year_raw" in df.columns else "Year"
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    prior_years = list(config.data.train_years) + list(config.data.val_years)
    priors = build_route_priors(df, prior_years=prior_years, year_col=year_col)
    df = drop_blacklist(df)
    df = attach_priors(df, priors)
    # rewrite parquet so evaluate/baselines see the causal columns
    df.to_parquet(flights_path, index=False)
    print(f"  [causal] priors attached from years {prior_years} → re-saved parquet")

    feature_groups = get_feature_groups(df)
    print("\n--- Feature Summary ---")
    for group, cols in feature_groups.items():
        print(f"  {group} ({len(cols)}): {cols}")
    print(f"\n  Total records: {len(df):,}")
    print(f"  Flight chains: {df['chain_id'].nunique():,}")
    if "DepDelay" in df.columns:
        extreme = (df["DepDelay"] > config.data.extreme_delay_threshold).sum()
        print(f"  Extreme delays (>{config.data.extreme_delay_threshold}min): "
              f"{extreme:,} ({extreme / len(df) * 100:.1f}%)")

    # ── Proxy Engineering (Architecture Layer 2) ──
    print("\n" + "=" * 70)
    print("PHASE 1B: PROXY ENGINEERING")
    print("=" * 70)

    proxy_eng = ProxyEngineer(hub=config.data.hub_airport)
    proxy_df = proxy_eng.run(df)

    if len(proxy_df) > 0:
        proxy_path = processed_dir(config) / "proxy_sequences.parquet"
        proxy_df.to_parquet(proxy_path, index=False)
        print(f"\n  Saved {len(proxy_df):,} proxy sequences → {proxy_path}")

    return df, proxy_df


# ══════════════════════════════════════════════════════════════
# PHASE 2: TRAIN
# ══════════════════════════════════════════════════════════════

def _resolve_feature_cols(df: pd.DataFrame):
    """Determine static, dynamic, and weather column lists — CAUSAL ONLY.

    Framing A: no actuals, no same-day chain observations. All leaky columns
    are removed via BLACKLIST; historical priors replace same-day propagation.
    """
    feature_groups = get_feature_groups(df)

    static_cols = ["Origin", "Dest", "CRSDepTime", "CRSArrTime", "Distance",
                   "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
                   "Reporting_Airline"]
    static_cols = [c for c in static_cols if c in df.columns and c not in BLACKLIST]

    dynamic_cols = feature_groups["flight"] + feature_groups["airport"]
    dynamic_cols = [
        c for c in dynamic_cols
        if c not in static_cols and c in df.columns and c not in BLACKLIST
    ]

    weather_cols = [c for c in feature_groups["weather"]
                    if c in df.columns and c not in BLACKLIST]

    # Append the leak-safe priors as static features (one value per row, time-invariant)
    prior_cols = [c for c in CAUSAL_FEATURES if c.endswith("_prior") and c in df.columns]
    for c in prior_cols:
        if c not in static_cols:
            static_cols.append(c)

    return static_cols, dynamic_cols, weather_cols


def train_model(config: Config, df: pd.DataFrame = None):
    """Train the TFT-DCP model.

    If df is None (i.e. called with --mode train), loads from the parquet
    written by the preprocess step.  Path comes from config.data.processed_dir
    so it always stays in sync.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 70)

    pdir = processed_dir(config)

    if df is None:
        data_path = pdir / "processed_flights.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No processed data found at:\n"
                f"  {data_path}\n\n"
                f"Run the preprocess step first:\n"
                f"  python main.py --mode preprocess \\\n"
                f"    --bts-dir  {config.data.bts_data_dir} \\\n"
                f"    --noaa-dir {config.data.noaa_data_dir} \\\n"
                f"    --aspm-dir {config.data.aspm_data_dir}"
            )
        df = pd.read_parquet(data_path)
        print(f"  Loaded {len(df):,} records from {data_path}")

    static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)

    config.model.num_static_features = len(static_cols)
    config.model.num_dynamic_features = len(dynamic_cols) + len(weather_cols)

    print(f"\n  Static features  ({len(static_cols)}): {static_cols}")
    print(f"  Dynamic features ({len(dynamic_cols)}): {dynamic_cols}")
    print(f"  Weather features ({len(weather_cols)}): {weather_cols}")
    print(f"  Total dynamic input dim: {config.model.num_dynamic_features}")

    # Temporal split — year-based (2025 held out for final test)
    year_col = "Year_raw" if "Year_raw" in df.columns else "Year"
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df["Year"] = df[year_col]  # alias so downstream .isin() works unchanged
    train_df = df[df["Year"].isin(config.data.train_years)]
    val_df   = df[df["Year"].isin(config.data.val_years)]
    test_df  = df[df["Year"].isin(config.data.test_years)]
    print(f"\n  Train years {config.data.train_years}: {len(train_df):,} rows")
    print(f"  Val   years {config.data.val_years}: {len(val_df):,} rows")
    print(f"  Test  years {config.data.test_years}: {len(test_df):,} rows (held out, not used for training)")

    ds_kwargs = dict(
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
        weather_cols=weather_cols,
        seq_len=config.model.sequence_length,
        hub=config.data.hub_airport,
    )

    train_dataset = FlightChainDataset(train_df, **ds_kwargs)
    val_dataset = FlightChainDataset(val_df, **ds_kwargs)

    num_gpus = torch.cuda.device_count()
    print(f"\n  GPUs detected: {num_gpus}")

    if num_gpus > 1 and config.train.distributed:
        print(f"  Launching DDP on {num_gpus} GPUs")
        mp.spawn(
            train_distributed,
            args=(num_gpus, config, train_dataset, val_dataset, config.train.resume_from),
            nprocs=num_gpus,
            join=True,
        )
    else:
        train_single_gpu(config, train_dataset, val_dataset, config.train.resume_from)

    # Save feature column metadata so evaluate() can reconstruct the model
    # without needing the df in memory
    meta = {
        "static_cols": static_cols,
        "dynamic_cols": dynamic_cols,
        "weather_cols": weather_cols,
        "num_dynamic": config.model.num_dynamic_features,
        "num_static": config.model.num_static_features,
    }
    meta_path = pdir / "feature_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved feature metadata → {meta_path}")


# ══════════════════════════════════════════════════════════════
# PHASE 3: EVALUATE + SCORE PAIRS
# ══════════════════════════════════════════════════════════════

def evaluate(config: Config, df: pd.DataFrame = None, proxy_df: pd.DataFrame = None):
    """Evaluate model and generate airport pair risk scores."""
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION & RISK SCORING")
    print("=" * 70)

    pdir = processed_dir(config)
    rdir = results_dir(config)

    # ── Load flight data ──────────────────────────────────────────────────
    if df is None:
        data_path = pdir / "processed_flights.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No processed data at {data_path}. Run preprocess first."
            )
        df = pd.read_parquet(data_path)
        print(f"  Loaded {len(df):,} flight records from {data_path}")

    # ── Load proxy sequences ──────────────────────────────────────────────
    if proxy_df is None:
        proxy_path = pdir / "proxy_sequences.parquet"
        if proxy_path.exists():
            proxy_df = pd.read_parquet(proxy_path)
            print(f"  Loaded {len(proxy_df):,} proxy sequences from {proxy_path}")
        else:
            print("  No proxy sequences found — scoring without regulatory flags")
            proxy_df = pd.DataFrame()

    # ── Load feature metadata ─────────────────────────────────────────────
    meta_path = pdir / "feature_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        static_cols  = meta["static_cols"]
        dynamic_cols = meta["dynamic_cols"]
        weather_cols = meta["weather_cols"]
        num_dynamic  = meta["num_dynamic"]
        num_static   = meta["num_static"]
    else:
        static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)
        num_dynamic = len(dynamic_cols) + len(weather_cols)
        num_static  = len(static_cols)

    # ── Load best checkpoint ──────────────────────────────────────────────
    ckpt_path = Path(config.train.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run train first:\n"
            f"  python main.py --mode train"
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TFTDCP(
        num_dynamic_features=num_dynamic,
        num_static_features=num_static,
        hidden_dim=config.model.hidden_dim,
        tcn_channels=config.model.tcn_num_channels,
        grn_hidden_dim=config.model.grn_hidden_dim,
        top_k_retrieval=config.model.top_k_retrieval,
        retrieval_alpha=config.model.retrieval_alpha,
        history_db_size=config.model.history_db_size,
        channel_reduction_ratio=config.model.channel_reduction_ratio,
        beta_init=config.model.beta_init,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded model from epoch {checkpoint['epoch']}")

    beta = torch.nn.functional.softplus(model.delay_propagation.beta).item()
    print(f"  Learned β: {beta:.4f} (paper range: 0.73–0.89)")

    # ── Restrict to held-out test year(s): 2025 ──────────────────────────
    from torch.utils.data import DataLoader
    year_col = "Year_raw" if "Year_raw" in df.columns else "Year"
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df["Year"] = df[year_col]  # alias so downstream .isin() works unchanged
    test_df = df[df["Year"].isin(config.data.test_years)].copy()
    print(f"\n  Test set: {len(test_df):,} rows from years {config.data.test_years}")

    test_dataset = FlightChainDataset(
        test_df,
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
        weather_cols=weather_cols,
        seq_len=config.model.sequence_length,
        hub=config.data.hub_airport,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=flight_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = PairRiskScorer(model, device=device)
    flight_preds = scorer.score_from_dataloader(test_loader)

    # ── Metrics ───────────────────────────────────────────────────────────
    y_true = flight_preds["actual_delay"].values
    y_pred = flight_preds["pred_delay"].values
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    print(f"\n  ── Flight-Level Metrics (Test = {config.data.test_years}) ──")
    print(f"  MAE:  {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")
    print(f"  R²:   {r2:.4f}")

    # Tail-focused regression metrics on delayed subsets
    tail_thresholds = [15, 60, 180]
    tail_metrics = {}
    print("\n  ── Tail-Focused Regression Metrics ──")
    print(f"  {'Subset':<14} {'N':>9}  {'Pct':>7}  {'MAE':>7}  {'RMSE':>7}")
    for t in tail_thresholds:
        m = y_true > t
        n_t = int(m.sum())
        if n_t == 0:
            tail_metrics[f"gt_{t}"] = {
                "threshold_minutes": int(t),
                "n": 0,
                "pct_of_test": 0.0,
                "MAE": None,
                "RMSE": None,
            }
            print(f"  >{t:<12} {0:>9,}  {0.00:>6.2f}%  {'-':>7}  {'-':>7}")
            continue
        yt = y_true[m]
        yp = y_pred[m]
        t_mae = float(np.mean(np.abs(yt - yp)))
        t_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        pct = float(100.0 * n_t / max(len(y_true), 1))
        tail_metrics[f"gt_{t}"] = {
            "threshold_minutes": int(t),
            "n": n_t,
            "pct_of_test": round(pct, 2),
            "MAE": round(t_mae, 2),
            "RMSE": round(t_rmse, 2),
        }
        print(f"  >{t:<12} {n_t:>9,}  {pct:>6.2f}%  {t_mae:>7.2f}  {t_rmse:>7.2f}")

    # Extreme-event classification for actual delay > 180 minutes
    extreme_thr = 180
    y_true_ext = (y_true > extreme_thr).astype(np.int32)
    y_pred_ext = (y_pred > extreme_thr).astype(np.int32)

    tp = int(((y_true_ext == 1) & (y_pred_ext == 1)).sum())
    fp = int(((y_true_ext == 0) & (y_pred_ext == 1)).sum())
    fn = int(((y_true_ext == 1) & (y_pred_ext == 0)).sum())
    tn = int(((y_true_ext == 0) & (y_pred_ext == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    prevalence = float(y_true_ext.mean() * 100.0)

    auprc = None
    if average_precision_score is not None and y_true_ext.min() != y_true_ext.max():
        try:
            auprc = float(average_precision_score(y_true_ext, y_pred))
        except Exception:
            auprc = None

    print("\n  ── Extreme-Event Classification (DepDelay > 180 min) ──")
    print(f"  Positives: {int(y_true_ext.sum()):,}/{len(y_true_ext):,} ({prevalence:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    if auprc is not None:
        print(f"  AUPRC:     {auprc:.4f}")
    else:
        print("  AUPRC:     unavailable")

    # ── Seasonal breakdown on the test year ───────────────────────────────
    seasonal_metrics = {}
    print(f"\n  ── Seasonal Breakdown (Test = {config.data.test_years}) ──")
    print(f"  {'Season':<8} {'N':>8}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'ExtremePct':>11}")
    for season, months in config.data.seasons.items():
        m = flight_preds["month"].isin(months)
        if m.sum() == 0:
            continue
        yt = flight_preds.loc[m, "actual_delay"].values
        yp = flight_preds.loc[m, "pred_delay"].values
        s_mae = float(np.mean(np.abs(yt - yp)))
        s_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        s_ss_res = float(np.sum((yt - yp) ** 2))
        s_ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        s_r2 = 1 - s_ss_res / max(s_ss_tot, 1e-8)
        s_extreme = float((yt > 180).mean() * 100)
        seasonal_metrics[season] = {
            "n": int(m.sum()),
            "MAE": round(s_mae, 2),
            "RMSE": round(s_rmse, 2),
            "R2": round(s_r2, 4),
            "extreme_pct": round(s_extreme, 2),
            "months": months,
        }
        print(f"  {season:<8} {m.sum():>8,}  {s_mae:>7.2f}  {s_rmse:>7.2f}  {s_r2:>7.4f}  {s_extreme:>10.2f}%")

    # ── Aggregate pair risks + export ─────────────────────────────────────
    pair_risks = scorer.aggregate_pair_risks(
        flight_preds,
        hub=config.data.hub_airport,
        proxy_df=proxy_df,
    )
    scorer.export(pair_risks, output_dir=str(rdir))

    flight_preds.to_csv(rdir / "flight_predictions.csv", index=False)

    metrics = {
        "test_years": list(config.data.test_years),
        "overall": {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "beta": round(beta, 4),
            "n_flights": int(len(flight_preds)),
        },
        "tail_regression": tail_metrics,
        "extreme_event_classification": {
            "event_threshold_minutes": int(extreme_thr),
            "prediction_threshold_minutes": int(extreme_thr),
            "n": int(len(y_true_ext)),
            "n_positive": int(y_true_ext.sum()),
            "prevalence_pct": round(prevalence, 2),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "auprc": (round(float(auprc), 4) if auprc is not None else None),
        },
        "seasonal": seasonal_metrics,
    }
    with open(rdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Per-season pair risk CSVs (seasonal A→DFW→B recommendations) ────
    for season, months in config.data.seasons.items():
        sub = flight_preds[flight_preds["month"].isin(months)]
        if len(sub) == 0:
            continue
        s_pairs = scorer.aggregate_pair_risks(
            sub, hub=config.data.hub_airport, proxy_df=proxy_df,
        )
        s_pairs.to_csv(rdir / f"pair_risk_scores_{season}.csv", index=False)
        if "is_feasible" in s_pairs.columns and "risk_score" in s_pairs.columns:
            s_flagged = s_pairs[
                (s_pairs["is_feasible"] == 1) & (s_pairs["risk_score"] >= 0.6)
            ]
        else:
            # Backward compatibility for older scorer outputs.
            s_flagged = s_pairs[s_pairs["final_score"] >= 0.6]
        s_flagged.to_csv(rdir / f"flagged_pairs_{season}.csv", index=False)
        print(f"  {season}: {len(s_pairs):,} pairs, {len(s_flagged):,} flagged")

    print(f"\n  All results saved to {rdir}/")


# ══════════════════════════════════════════════════════════════
# PHASE 4: BASELINES
# ══════════════════════════════════════════════════════════════

def run_baselines(config: Config, df: pd.DataFrame = None):
    """Run baseline models for report comparison."""
    print("\n" + "=" * 70)
    print("PHASE 4: BASELINE COMPARISON")
    print("=" * 70)

    pdir = processed_dir(config)
    rdir = results_dir(config)

    if df is None:
        data_path = pdir / "processed_flights.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No processed data at {data_path}. Run preprocess first."
            )
        df = pd.read_parquet(data_path)

    static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)
    num_dynamic = len(dynamic_cols) + len(weather_cols)
    num_static  = len(static_cols)

    year_col = "Year_raw" if "Year_raw" in df.columns else "Year"
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df["Year"] = df[year_col]  # alias so downstream .isin() works unchanged
    train_df = df[df["Year"].isin(config.data.train_years)]
    test_df  = df[df["Year"].isin(config.data.test_years)]

    ds_kwargs = dict(
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
        weather_cols=weather_cols,
        seq_len=config.model.sequence_length,
        hub=config.data.hub_airport,
    )

    from torch.utils.data import DataLoader
    train_ds = FlightChainDataset(train_df, **ds_kwargs)
    test_ds  = FlightChainDataset(test_df,  **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size,
                              shuffle=True,  num_workers=4, collate_fn=flight_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config.train.batch_size,
                              shuffle=False, num_workers=4, collate_fn=flight_collate_fn)

    all_feature_cols = static_cols + dynamic_cols + weather_cols
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    from experiments import run_benchmark_comparison, run_ablation_study

    run_benchmark_comparison(
        train_loader=train_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        train_df=train_df,
        test_df=test_df,
        num_dynamic=num_dynamic,
        num_static=num_static,
        feature_cols=all_feature_cols,
        config=config,
    )

    run_ablation_study(
        train_loader=train_loader,
        test_loader=test_loader,
        num_dynamic=num_dynamic,
        num_static=num_static,
        config=config,
    )

    from visualize import generate_all_figures
    generate_all_figures()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TFT-DCP Flight Delay Prediction")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "evaluate", "baselines", "all"],
        default="all",
    )
    parser.add_argument("--bts-dir",  type=str, default="./data/raw/bts")
    parser.add_argument("--noaa-dir", type=str, default="./data/raw/noaa")
    parser.add_argument("--aspm-dir", type=str, default="./data/raw/aspm")
    parser.add_argument("--processed-dir", type=str, default="./data/processed")
    parser.add_argument("--results-dir",   type=str, default="./results")
    parser.add_argument("--hub",       type=str,  default="DFW")
    parser.add_argument("--epochs",    type=int,  default=100)
    parser.add_argument("--batch-size",type=int,  default=128)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Re-run Phase 1A even if processed_flights.parquet exists")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    config = Config(
        data=DataConfig(
            bts_data_dir=args.bts_dir,
            noaa_data_dir=args.noaa_dir,
            aspm_data_dir=args.aspm_dir,
            processed_dir=args.processed_dir,
            results_dir=args.results_dir,
            hub_airport=args.hub,
        ),
        model=ModelConfig(),
        train=TrainConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=args.resume,
        ),
    )

    df = None
    proxy_df = None

    if args.mode in ("preprocess", "all"):
        df, proxy_df = preprocess(config, force=args.force_preprocess)

    if args.mode in ("train", "all"):
        train_model(config, df)          # df=None when --mode train → loads from disk

    if args.mode in ("evaluate", "all"):
        evaluate(config, df, proxy_df)   # df=None when --mode evaluate → loads from disk

    if args.mode in ("baselines", "all"):
        run_baselines(config, df)


if __name__ == "__main__":
    main()
