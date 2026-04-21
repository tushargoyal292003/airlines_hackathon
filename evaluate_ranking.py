"""
Ranking evaluation: oracle (actuals) vs predicted pair ranking.

Spearman ρ / Kendall τ / Precision@K / NDCG@K — the right metrics for a
pair-risk ranking task where operational utility is ordinal.

Usage:
    python evaluate_ranking.py --preds results/pair_risk_scores_full.csv
    python evaluate_ranking.py --preds results/lightgbm/pair_risk_scores_full.csv \
        --flight-preds results/lightgbm/flight_predictions_lightgbm.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def build_oracle_ranking(flight_preds: pd.DataFrame, hub: str = "DFW") -> pd.DataFrame:
    """Same aggregation as predicted scorer, but using actual_delay."""
    df = flight_preds.copy()
    df["origin"] = df["origin"].astype(str)
    df["dest"] = df["dest"].astype(str)
    inb = df[df["dest"] == hub]
    outb = df[df["origin"] == hub]

    def agg(part, key, suf):
        return part.groupby(key).agg(
            **{
                f"p90_{suf}": ("actual_delay", lambda x: float(np.percentile(x, 90))),
                f"std_{suf}": ("actual_delay", "std"),
                f"ext_{suf}": ("actual_delay", lambda x: float((x > 180).mean())),
            }
        ).reset_index().rename(columns={key: f"airport_{suf}"})

    a = agg(inb, "origin", "a"); b = agg(outb, "dest", "b")
    a["_k"] = 1; b["_k"] = 1
    pairs = a.merge(b, on="_k").drop(columns="_k")
    pairs["oracle_score"] = (
        0.5 * (pairs["p90_a"] + pairs["p90_b"]) / max(1, (pairs["p90_a"] + pairs["p90_b"]).quantile(0.95))
        + 0.3 * (pairs["std_a"].fillna(0) + pairs["std_b"].fillna(0)) / max(1, (pairs["std_a"].fillna(0) + pairs["std_b"].fillna(0)).quantile(0.95))
        + 0.2 * (pairs["ext_a"] + pairs["ext_b"]) / max(1e-6, (pairs["ext_a"] + pairs["ext_b"]).quantile(0.95))
    )
    return pairs[["airport_a", "airport_b", "oracle_score"]]


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    order_pred = np.argsort(-y_pred)[:k]
    order_true = np.argsort(-y_true)[:k]
    dcg = np.sum(y_true[order_pred] / np.log2(np.arange(k) + 2))
    idcg = np.sum(y_true[order_true] / np.log2(np.arange(k) + 2))
    return float(dcg / idcg) if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pair_risk_scores CSV with airport_a, airport_b, risk_score/final_score")
    ap.add_argument("--flight-preds", required=True, help="flight_predictions CSV with origin,dest,actual_delay,pred_delay")
    ap.add_argument("--hub", default="DFW")
    ap.add_argument("--score-col", default=None, help="auto-detect risk_score or final_score if omitted")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred_pairs = pd.read_csv(args.preds)
    score_col = args.score_col or (
        "risk_score" if "risk_score" in pred_pairs.columns else "final_score"
    )
    if score_col not in pred_pairs.columns:
        raise ValueError(f"Neither risk_score nor final_score in {args.preds}")

    flight_preds = pd.read_csv(args.flight_preds)
    oracle = build_oracle_ranking(flight_preds, hub=args.hub)

    merged = pred_pairs.merge(oracle, on=["airport_a", "airport_b"], how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping (airport_a, airport_b) between preds and oracle")

    y_pred = merged[score_col].to_numpy(dtype=np.float64)
    y_true = merged["oracle_score"].to_numpy(dtype=np.float64)

    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)

    results = {
        "n_pairs": int(len(merged)),
        "score_column": score_col,
        "spearman_rho": round(float(rho), 4),
        "kendall_tau": round(float(tau), 4),
    }
    for k in (10, 20, 50, 100):
        if k > len(merged):
            continue
        top_pred = set(merged.sort_values(score_col, ascending=False).head(k).index)
        top_true = set(merged.sort_values("oracle_score", ascending=False).head(k).index)
        results[f"precision_at_{k}"] = round(len(top_pred & top_true) / k, 4)
        results[f"ndcg_at_{k}"] = round(ndcg_at_k(y_true, y_pred, k), 4)

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
