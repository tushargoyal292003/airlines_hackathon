"""
TFT-DCP: Main Entry Point
Orchestrates: preprocess → proxy engineering → train → evaluate → score pairs

Usage:
  python main.py --mode all           # Run everything
  python main.py --mode preprocess    # Step 1: Process raw data + build proxies
  python main.py --mode train         # Step 2: Train TFT-DCP model
  python main.py --mode evaluate      # Step 3: Evaluate + score pairs
  python main.py --mode baselines     # Step 4: Run baseline comparisons
"""
import argparse
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.preprocessor import DataPipeline, get_feature_groups
from data.dataset import FlightChainDataset, flight_collate_fn
from data.proxy_engineering import ProxyEngineer
from model import TFTDCP
from train import train_distributed, train_single_gpu, Trainer
from risk_scorer import PairRiskScorer


PROCESSED_DIR = Path("./data/processed")
RESULTS_DIR = Path("./results")


# ══════════════════════════════════════════════════════════════
# PHASE 1: PREPROCESS
# ══════════════════════════════════════════════════════════════

def preprocess(config: Config):
    """Run data pipeline + proxy engineering."""
    print("\n" + "=" * 70)
    print("PHASE 1A: DATA PREPROCESSING")
    print("=" * 70)

    pipeline = DataPipeline(config.data)
    df = pipeline.run(save=True)

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
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        proxy_df.to_parquet(PROCESSED_DIR / "proxy_sequences.parquet", index=False)
        print(f"\n  Saved {len(proxy_df):,} proxy sequences")

    return df, proxy_df


# ══════════════════════════════════════════════════════════════
# PHASE 2: TRAIN
# ══════════════════════════════════════════════════════════════

def _resolve_feature_cols(df: pd.DataFrame):
    """Determine static, dynamic, and weather column lists from the data."""
    feature_groups = get_feature_groups(df)

    static_cols = ["Origin", "Dest", "CRSDepTime", "CRSArrTime",
                   "Distance", "DayOfWeek", "Month", "Reporting_Airline"]
    static_cols = [c for c in static_cols if c in df.columns]

    dynamic_cols = feature_groups["flight"] + feature_groups["airport"]
    dynamic_cols = [c for c in dynamic_cols if c not in static_cols and c in df.columns]

    weather_cols = [c for c in feature_groups["weather"] if c in df.columns]

    return static_cols, dynamic_cols, weather_cols


def train_model(config: Config, df: pd.DataFrame = None):
    """Train the TFT-DCP model."""
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 70)

    if df is None:
        data_path = PROCESSED_DIR / "processed_flights.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No processed data at {data_path}. "
                "Run: python main.py --mode preprocess "
                "--bts-dir ./data/data_bts/raw/bts "
                "--noaa-dir ./data/data_noaa/raw/noaa "
                "--aspm-dir ./data/data_aspm"
            )
        df = pd.read_parquet(data_path)
        print(f"  Loaded {len(df):,} records")

    static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)

    config.model.num_static_features = len(static_cols)
    config.model.num_dynamic_features = len(dynamic_cols) + len(weather_cols)

    print(f"\n  Static features  ({len(static_cols)}): {static_cols}")
    print(f"  Dynamic features ({len(dynamic_cols)}): {dynamic_cols}")
    print(f"  Weather features ({len(weather_cols)}): {weather_cols}")
    print(f"  Total dynamic input dim: {config.model.num_dynamic_features}")

    # Temporal split
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    train_df = df[df["Month"].isin(config.data.train_months)]
    val_df = df[df["Month"].isin(config.data.val_months)]

    print(f"\n  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

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
            args=(num_gpus, config, train_dataset, val_dataset),
            nprocs=num_gpus,
            join=True,
        )
    else:
        trainer = train_single_gpu(config, train_dataset, val_dataset)

    # Save feature column lists for evaluation phase
    import json
    meta = {
        "static_cols": static_cols,
        "dynamic_cols": dynamic_cols,
        "weather_cols": weather_cols,
        "num_dynamic": config.model.num_dynamic_features,
        "num_static": config.model.num_static_features,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved feature metadata to {PROCESSED_DIR / 'feature_meta.json'}")


# ══════════════════════════════════════════════════════════════
# PHASE 3: EVALUATE + SCORE PAIRS
# ══════════════════════════════════════════════════════════════

def evaluate(config: Config, df: pd.DataFrame = None, proxy_df: pd.DataFrame = None):
    """Evaluate model and generate airport pair risk scores."""
    import json

    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION & RISK SCORING")
    print("=" * 70)

    # Load data
    if df is None:
        data_path = PROCESSED_DIR / "processed_flights.parquet"
        df = pd.read_parquet(data_path)
        print(f"  Loaded {len(df):,} flight records")

    if proxy_df is None:
        proxy_path = PROCESSED_DIR / "proxy_sequences.parquet"
        if proxy_path.exists():
            proxy_df = pd.read_parquet(proxy_path)
            print(f"  Loaded {len(proxy_df):,} proxy sequences")
        else:
            print("  No proxy data found — scoring without regulatory flags")
            proxy_df = pd.DataFrame()

    # Load feature metadata
    meta_path = PROCESSED_DIR / "feature_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        static_cols = meta["static_cols"]
        dynamic_cols = meta["dynamic_cols"]
        weather_cols = meta["weather_cols"]
        num_dynamic = meta["num_dynamic"]
        num_static = meta["num_static"]
    else:
        static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)
        num_dynamic = len(dynamic_cols) + len(weather_cols)
        num_static = len(static_cols)

    # Load best model
    ckpt_path = Path(config.train.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

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

    # Learned beta
    beta = torch.nn.functional.softplus(model.delay_propagation.beta).item()
    print(f"  Learned β: {beta:.4f} (paper range: 0.73–0.89)")

    # Build test dataset
    test_dataset = FlightChainDataset(
        df, static_cols=static_cols, dynamic_cols=dynamic_cols,
        weather_cols=weather_cols, seq_len=config.model.sequence_length,
        hub=config.data.hub_airport,
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=config.train.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=flight_collate_fn,
    )

    # Score flights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = PairRiskScorer(model, device=device)
    flight_preds = scorer.score_from_dataloader(test_loader)

    # Compute per-flight metrics (MAE, RMSE, R²)
    y_true = flight_preds["actual_delay"].values
    y_pred = flight_preds["pred_delay"].values
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    print(f"\n  ── Flight-Level Metrics ──")
    print(f"  MAE:  {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")
    print(f"  R²:   {r2:.4f}")

    # Aggregate to pair risk scores + apply proxy constraints
    pair_risks = scorer.aggregate_pair_risks(
        flight_preds, hub=config.data.hub_airport, proxy_df=proxy_df,
    )

    # Export (Architecture Layer 4 output contract)
    RESULTS_DIR.mkdir(exist_ok=True)
    scorer.export(pair_risks, output_dir=str(RESULTS_DIR))

    # Save flight-level predictions for visualization
    flight_preds.to_csv(RESULTS_DIR / "flight_predictions.csv", index=False)

    # Save metrics summary
    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4), "beta": round(beta, 4)}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  All results saved to {RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════
# PHASE 4: BASELINES (for report)
# ══════════════════════════════════════════════════════════════

def run_baselines(config: Config, df: pd.DataFrame = None):
    """Run baseline models for report comparison."""
    print("\n" + "=" * 70)
    print("PHASE 4: BASELINE COMPARISON")
    print("=" * 70)

    if df is None:
        data_path = PROCESSED_DIR / "processed_flights.parquet"
        df = pd.read_parquet(data_path)

    static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)
    num_dynamic = len(dynamic_cols) + len(weather_cols)
    num_static = len(static_cols)

    # Build datasets
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    train_df = df[df["Month"].isin(config.data.train_months)]
    test_df = df[df["Month"].isin(config.data.val_months)]

    ds_kwargs = dict(
        static_cols=static_cols, dynamic_cols=dynamic_cols,
        weather_cols=weather_cols, seq_len=config.model.sequence_length,
        hub=config.data.hub_airport,
    )

    train_ds = FlightChainDataset(train_df, **ds_kwargs)
    test_ds = FlightChainDataset(test_df, **ds_kwargs)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size,
                              shuffle=True, num_workers=4, collate_fn=flight_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size,
                             shuffle=False, num_workers=4, collate_fn=flight_collate_fn)

    # All tabular features for GBM baselines
    all_feature_cols = static_cols + dynamic_cols + weather_cols
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    from experiments import run_benchmark_comparison, run_ablation_study

    RESULTS_DIR.mkdir(exist_ok=True)

    # Benchmark comparison (Table 2)
    benchmark_results = run_benchmark_comparison(
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

    # Ablation study (Table 4)
    ablation_results = run_ablation_study(
        train_loader=train_loader,
        test_loader=test_loader,
        num_dynamic=num_dynamic,
        num_static=num_static,
        config=config,
    )

    # Generate figures
    from visualize import generate_all_figures
    generate_all_figures()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TFT-DCP Flight Delay Prediction")
    parser.add_argument("--mode", choices=["preprocess", "train", "evaluate", "baselines", "all"],
                        default="all")
    parser.add_argument("--bts-dir", type=str, default="./data/data_bts/raw/bts")
    parser.add_argument("--noaa-dir", type=str, default="./data/data_noaa/raw/noaa")
    parser.add_argument("--aspm-dir", type=str, default="./data/data_aspm")
    parser.add_argument("--hub", type=str, default="DFW")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    config = Config(
        data=DataConfig(
            bts_data_dir=args.bts_dir,
            noaa_data_dir=args.noaa_dir,
            aspm_data_dir=args.aspm_dir,
            hub_airport=args.hub,
        ),
        model=ModelConfig(),
        train=TrainConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        ),
    )

    df = None
    proxy_df = None

    if args.mode in ("preprocess", "all"):
        df, proxy_df = preprocess(config)

    if args.mode in ("train", "all"):
        train_model(config, df)

    if args.mode in ("evaluate", "all"):
        evaluate(config, df, proxy_df)

    if args.mode in ("baselines", "all"):
        run_baselines(config, df)


if __name__ == "__main__":
    main()
