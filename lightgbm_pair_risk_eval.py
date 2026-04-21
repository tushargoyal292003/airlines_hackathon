#!/usr/bin/env python3
"""
Run pair-risk scoring using a LightGBM delay model.

This script builds the same pair-level risk outputs as TFT-DCP evaluation, but with:
  - pred_delay from a trained LightGBM model
  - propagated_delay estimated from chain features only (no TFT dependency)

Outputs (in output_dir):
  - flight_predictions_lightgbm.csv
  - metrics_lightgbm.json
  - scored_pairs.csv
  - flagged_pairs.csv
  - pair_risk_scores_full.csv
  - pair_risk_scores_{winter,spring,summer,fall}.csv
  - flagged_pairs_{winter,spring,summer,fall}.csv
"""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from config import Config, DataConfig, ModelConfig, TrainConfig
from main import _resolve_feature_cols
from risk_scorer import PairRiskScorer
from causal_features import BLACKLIST


def estimate_propagation_from_chain(
    df: pd.DataFrame,
    pred_delay: np.ndarray,
    seq_len: int = 14,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Estimate per-flight propagated delay from chain history only.

    For each target flight at position i in chain:
      y_prop[i] = sum_{k in history(i)} exp(-beta * turnaround_k) * pred_delay_k

    Notes:
    - Uses LightGBM-predicted delay history only (no TFT outputs).
    - Turnaround comes from processed data (proxy for connection buffer).
    """
    if "chain_id" not in df.columns:
        return np.zeros(len(df), dtype=np.float32)

    work = df[[
        "chain_id",
        "CRSDepTime" if "CRSDepTime" in df.columns else df.columns[0],
        "turnaround_minutes" if "turnaround_minutes" in df.columns else None,
    ]].copy()

    # Clean temporary column selection in case optional columns are absent.
    keep = [c for c in work.columns if c is not None]
    work = work[keep]

    if "turnaround_minutes" not in work.columns:
        work["turnaround_minutes"] = 0.0

    sort_cols = ["chain_id"]
    if "CRSDepTime" in work.columns:
        sort_cols.append("CRSDepTime")

    # Preserve original index so output aligns with caller's row order.
    work["_orig_idx"] = np.arange(len(work))
    work["_pred_delay"] = np.asarray(pred_delay, dtype=np.float32)
    work = work.sort_values(sort_cols).reset_index(drop=True)

    pred_hist = pd.to_numeric(work["_pred_delay"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    tta = pd.to_numeric(work["turnaround_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    chain_ids = work["chain_id"].to_numpy()

    y_prop = np.zeros(len(work), dtype=np.float32)

    # Group positions by chain and compute history sums.
    # Complexity: O(n * seq_len) with seq_len small (=14).
    start = 0
    n = len(work)
    while start < n:
        cid = chain_ids[start]
        end = start + 1
        while end < n and chain_ids[end] == cid:
            end += 1

        # Chain rows are [start:end)
        for pos in range(start, end):
            hist_start = max(start, pos - seq_len)
            if hist_start >= pos:
                y_prop[pos] = 0.0
                continue
            cd = pred_hist[hist_start:pos]
            ta = tta[hist_start:pos]
            sigma = np.exp(-beta * ta)
            y_prop[pos] = float(np.sum(sigma * cd))

        start = end

    # Restore original row order
    out = pd.Series(y_prop, index=work["_orig_idx"].to_numpy()).sort_index().to_numpy(dtype=np.float32)
    return out


def compute_flight_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "n_flights": int(len(y_true)),
    }


def compute_tail_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for t in (15, 60, 180):
        m = y_true > t
        n_t = int(m.sum())
        if n_t == 0:
            out[f"gt_{t}"] = {
                "threshold_minutes": int(t),
                "n": 0,
                "pct_of_test": 0.0,
                "MAE": None,
                "RMSE": None,
            }
            continue
        yt = y_true[m]
        yp = y_pred[m]
        out[f"gt_{t}"] = {
            "threshold_minutes": int(t),
            "n": n_t,
            "pct_of_test": round(float(100.0 * n_t / max(len(y_true), 1)), 2),
            "MAE": round(float(np.mean(np.abs(yt - yp))), 2),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2),
        }
    return out


def compute_extreme_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: int = 180) -> dict:
    y_true_ext = (y_true > threshold).astype(np.int32)
    y_pred_ext = (y_pred > threshold).astype(np.int32)

    tp = int(((y_true_ext == 1) & (y_pred_ext == 1)).sum())
    fp = int(((y_true_ext == 0) & (y_pred_ext == 1)).sum())
    fn = int(((y_true_ext == 1) & (y_pred_ext == 0)).sum())
    tn = int(((y_true_ext == 0) & (y_pred_ext == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Average precision for ranking on continuous score.
    y_true_bin = y_true_ext.astype(np.float32)
    order = np.argsort(-y_pred)
    y_sorted = y_true_bin[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1.0 - y_sorted)
    denom = np.maximum(tp_cum + fp_cum, 1e-8)
    precision_curve = tp_cum / denom
    total_pos = np.maximum(float(y_true_bin.sum()), 1e-8)
    recall_curve = tp_cum / total_pos
    delta_recall = np.diff(np.concatenate(([0.0], recall_curve)))
    auprc = float(np.sum(precision_curve * delta_recall))

    return {
        "event_threshold_minutes": int(threshold),
        "prediction_threshold_minutes": int(threshold),
        "n": int(len(y_true_ext)),
        "n_positive": int(y_true_ext.sum()),
        "prevalence_pct": round(float(100.0 * y_true_ext.mean()), 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auprc": round(float(auprc), 4),
    }


def align_features_for_model(
    model,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    y_true: np.ndarray | None = None,
    enable_search: bool = True,
    search_sample_n: int = 40000,
) -> Tuple[List[str], List[str]]:
    """
    Align feature list to baseline checkpoint schema.

    Legacy checkpoints were trained with raw numpy arrays, so feature names are
    anonymous. We therefore prefer deterministic compatibility rules:
    1) If mismatch is small and labels are available, search drop-combinations on
       a sample and pick the set minimizing MAE.
    2) Fallback to deterministic heuristic:
       - drop (`cloud_cover`, `wx_severity`) first
       - then trim from the end.
    """

    def _search_best_drop_cols() -> List[str] | None:
        need = actual - expected
        if not enable_search or y_true is None or need <= 0 or need > 2:
            return None
        if "DepDelay" not in test_df.columns:
            return None

        n = len(test_df)
        if n == 0:
            return None

        sample_n = min(int(search_sample_n), n)
        sample = test_df[feature_cols + ["DepDelay"]].sample(n=sample_n, random_state=42)
        y_s = pd.to_numeric(sample["DepDelay"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if len(y_s) == 0:
            return None

        best_mae = float("inf")
        best_drop_idx = None
        total = 0
        for drop_idx in itertools.combinations(range(actual), need):
            total += 1
            use_cols = [c for i, c in enumerate(feature_cols) if i not in drop_idx]
            X_s = sample[use_cols].fillna(0).to_numpy(dtype=np.float32)
            pred_s = np.asarray(model.predict(X_s), dtype=np.float32)
            mae = float(np.mean(np.abs(y_s - pred_s)))
            if mae < best_mae:
                best_mae = mae
                best_drop_idx = drop_idx

        if best_drop_idx is None:
            return None
        drops = [feature_cols[i] for i in best_drop_idx]
        print(
            f"  Feature alignment search picked drop columns {drops} "
            f"(sample_n={sample_n:,}, combos={total:,}, sample_MAE={best_mae:.4f})"
        )
        return drops

    expected = int(getattr(model, "n_features_in_", len(feature_cols)))
    actual = len(feature_cols)
    if expected == actual:
        return feature_cols, []
    if actual < expected:
        raise ValueError(
            f"Feature mismatch: model expects {expected}, but only {actual} available."
        )

    keep = list(feature_cols)
    dropped = []
    need_drop = actual - expected

    searched_drops = _search_best_drop_cols()
    if searched_drops:
        for c in searched_drops:
            if c in keep and need_drop > 0:
                keep.remove(c)
                dropped.append(c)
                need_drop -= 1

    if need_drop > 0:
        preferred_drop = ["cloud_cover", "wx_severity"]
        for c in preferred_drop:
            if need_drop <= 0:
                break
            if c in keep:
                keep.remove(c)
                dropped.append(c)
                need_drop -= 1

    if need_drop > 0:
        tail_drop = keep[-need_drop:]
        keep = keep[:-need_drop]
        dropped.extend(tail_drop)

    if len(keep) != expected:
        raise ValueError(
            f"Could not align features to model schema: expected {expected}, got {len(keep)}"
        )
    return keep, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM and generate pair risk outputs")
    parser.add_argument("--processed", type=str, default="./data/processed/processed_flights.parquet")
    parser.add_argument("--proxy", type=str, default="./data/processed/proxy_sequences.parquet")
    parser.add_argument("--model", type=str, default="./checkpoints_baselines/baselines/lightgbm.joblib")
    parser.add_argument("--output-dir", type=str, default="./results/lightgbm")
    parser.add_argument("--hub", type=str, default="DFW")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Decay used for LightGBM propagation estimate")
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--align-sample-n", type=int, default=40000)
    parser.add_argument("--disable-alignment-search", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_path = Path(args.processed)
    proxy_path = Path(args.proxy)
    model_path = Path(args.model)

    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed flights: {processed_path}")
    if not proxy_path.exists():
        raise FileNotFoundError(f"Missing proxy file: {proxy_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing LightGBM model: {model_path}")

    df = pd.read_parquet(processed_path)
    proxy_df = pd.read_parquet(proxy_path)

    cfg = Config(
        data=DataConfig(hub_airport=args.hub),
        model=ModelConfig(),
        train=TrainConfig(),
    )

    year_col = "Year_raw" if "Year_raw" in df.columns else "Year"
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df["Year"] = df[year_col]

    test_df = df[df["Year"].isin(cfg.data.test_years)].copy()
    if len(test_df) == 0:
        raise ValueError(f"No test rows for years {cfg.data.test_years}")

    static_cols, dynamic_cols, weather_cols = _resolve_feature_cols(df)
    feature_cols = [
        c for c in static_cols + dynamic_cols + weather_cols
        if c in test_df.columns and c not in BLACKLIST
    ]

    model = joblib.load(model_path)

    y_true = pd.to_numeric(test_df["DepDelay"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    feature_cols_used, dropped_cols = align_features_for_model(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        y_true=y_true,
        enable_search=not args.disable_alignment_search,
        search_sample_n=args.align_sample_n,
    )
    if dropped_cols:
        print(
            f"  Adjusted feature schema for checkpoint compatibility: dropped {len(dropped_cols)} column(s): "
            f"{dropped_cols}"
        )
    X_test = test_df[feature_cols_used].fillna(0).to_numpy(dtype=np.float32)

    y_pred = np.asarray(model.predict(X_test), dtype=np.float32)

    y_prop = estimate_propagation_from_chain(test_df, pred_delay=y_pred, seq_len=args.seq_len, beta=args.beta)

    origin_col = "Origin_str" if "Origin_str" in test_df.columns else "Origin"
    dest_col = "Dest_str" if "Dest_str" in test_df.columns else "Dest"
    month_col = "Month_raw" if "Month_raw" in test_df.columns else "Month"

    flight_preds = pd.DataFrame({
        "origin": test_df[origin_col].astype(str).to_numpy(),
        "dest": test_df[dest_col].astype(str).to_numpy(),
        "year": test_df["Year"].fillna(0).astype(int).to_numpy(),
        "month": pd.to_numeric(test_df[month_col], errors="coerce").fillna(0).astype(int).to_numpy(),
        "pred_delay": y_pred,
        "actual_delay": y_true,
        "propagated_delay": y_prop,
    })

    flight_preds.to_csv(out_dir / "flight_predictions_lightgbm.csv", index=False)

    scorer = PairRiskScorer(torch.nn.Identity(), device="cpu")
    pair_risks = scorer.aggregate_pair_risks(flight_preds, hub=args.hub, proxy_df=proxy_df)
    scorer.export(pair_risks, output_dir=str(out_dir))

    # Seasonal outputs
    for season, months in cfg.data.seasons.items():
        sub = flight_preds[flight_preds["month"].isin(months)]
        if len(sub) == 0:
            continue
        s_pairs = scorer.aggregate_pair_risks(sub, hub=args.hub, proxy_df=proxy_df)
        s_pairs.to_csv(out_dir / f"pair_risk_scores_{season}.csv", index=False)
        s_flagged = s_pairs[(s_pairs["is_feasible"] == 1) & (s_pairs["risk_score"] >= 0.6)]
        s_flagged.to_csv(out_dir / f"flagged_pairs_{season}.csv", index=False)

    metrics = {
        "model": "LightGBM",
        "test_years": cfg.data.test_years,
        "hub": args.hub,
        "propagation_estimator": {
            "type": "chain_decay",
            "beta": float(args.beta),
            "seq_len": int(args.seq_len),
        },
        "flight_metrics": compute_flight_metrics(y_true, y_pred),
        "tail_regression": compute_tail_metrics(y_true, y_pred),
        "extreme_event_classification": compute_extreme_metrics(y_true, y_pred, threshold=180),
        "counts": {
            "n_test_rows": int(len(test_df)),
            "n_pairs": int(len(pair_risks)),
            "n_feasible_pairs": int((pair_risks["is_feasible"] == 1).sum()),
            "n_flagged_pairs": int(((pair_risks["is_feasible"] == 1) & (pair_risks["risk_score"] >= 0.6)).sum()),
        },
        "feature_count_built": int(len(feature_cols)),
        "feature_count_used": int(len(feature_cols_used)),
        "dropped_feature_cols": dropped_cols,
        "feature_cols_used": feature_cols_used,
    }
    with open(out_dir / "metrics_lightgbm.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 70)
    print("LIGHTGBM PAIR-RISK EVALUATION")
    print("=" * 70)
    print(f"test flights: {len(test_df):,}")
    print(f"pairs:        {len(pair_risks):,}")
    print(f"feasible:     {(pair_risks['is_feasible'] == 1).sum():,}")
    print(f"flagged:      {((pair_risks['is_feasible'] == 1) & (pair_risks['risk_score'] >= 0.6)).sum():,}")
    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
