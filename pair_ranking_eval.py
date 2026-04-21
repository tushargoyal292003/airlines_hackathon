#!/usr/bin/env python3
"""
Pair Ranking Evaluation for A->DFW->B Risk Scoring.

Builds pair-level ground truth from ACTUAL delays on held-out flights,
aligns it with predicted pair risk scores, and reports ranking metrics:
  - Precision@K
  - Recall@K
  - NDCG@K
  - Spearman / Kendall rank correlation

Outputs:
  - pair_ranking_aligned.csv
  - pair_ranking_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import ndcg_score
except Exception:  # pragma: no cover
    ndcg_score = None


def parse_ks(raw: str) -> List[int]:
    ks = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError(f"K values must be > 0, got {k}")
        ks.append(k)
    if not ks:
        raise ValueError("No valid K values parsed")
    return sorted(set(ks))


def _safe_q99(values: pd.Series) -> float:
    q = float(values.quantile(0.99)) if len(values) else 0.0
    return max(q, 1.0)


def _safe_max(values: pd.Series) -> float:
    if len(values) == 0:
        return 1.0
    return max(float(values.max()), 1.0)


def build_true_pair_scores(
    flight_preds: pd.DataFrame,
    hub: str,
    extreme_threshold: float,
    truth_mode: str,
) -> pd.DataFrame:
    """Build pair-level truth scores from actual_delay on held-out flights."""
    required_cols = {"origin", "dest", "actual_delay"}
    missing = required_cols - set(flight_preds.columns)
    if missing:
        raise ValueError(f"flight_predictions is missing columns: {sorted(missing)}")

    inbound = flight_preds[flight_preds["dest"] == hub].copy()
    outbound = flight_preds[flight_preds["origin"] == hub].copy()

    if len(inbound) == 0 or len(outbound) == 0:
        raise ValueError(
            f"No inbound/outbound rows for hub={hub}. "
            f"inbound={len(inbound)}, outbound={len(outbound)}"
        )

    a_stats = inbound.groupby("origin").agg(
        avg_actual_delay_a=("actual_delay", "mean"),
        std_actual_delay_a=("actual_delay", "std"),
        extreme_pct_actual_a=(
            "actual_delay", lambda x: float((x > extreme_threshold).mean() * 100.0)
        ),
        n_actual_flights_a=("actual_delay", "count"),
    ).reset_index().rename(columns={"origin": "airport_a"})

    b_agg = {
        "avg_actual_delay_b": ("actual_delay", "mean"),
        "std_actual_delay_b": ("actual_delay", "std"),
        "n_actual_flights_b": ("actual_delay", "count"),
    }
    use_prop = truth_mode == "actual_plus_model_propagation" and "propagated_delay" in outbound.columns
    if use_prop:
        b_agg["avg_model_prop_b"] = ("propagated_delay", "mean")

    b_stats = outbound.groupby("dest").agg(**b_agg).reset_index().rename(columns={"dest": "airport_b"})

    a_stats["_k"] = 1
    b_stats["_k"] = 1
    pairs = a_stats.merge(b_stats, on="_k").drop(columns="_k")

    pairs["true_avg_delay_combined"] = (
        pairs["avg_actual_delay_a"] + pairs["avg_actual_delay_b"]
    ) / 2.0

    q99_delay = _safe_q99(pairs["true_avg_delay_combined"])
    pairs["true_delay_risk"] = (
        pairs["true_avg_delay_combined"] / q99_delay
    ).clip(lower=0.0, upper=1.0)

    max_std = max(
        float(pairs[["std_actual_delay_a", "std_actual_delay_b"]].max().max()),
        1.0,
    )
    pairs["true_variance_risk"] = (
        (
            pairs["std_actual_delay_a"].fillna(0.0)
            + pairs["std_actual_delay_b"].fillna(0.0)
        )
        / (2.0 * max_std)
    ).clip(upper=1.0)

    max_extreme = _safe_max(pairs["extreme_pct_actual_a"])
    pairs["true_extreme_risk"] = (
        pairs["extreme_pct_actual_a"] / max_extreme
    ).clip(upper=1.0)

    if use_prop:
        q99_prop = _safe_q99(pairs["avg_model_prop_b"])
        pairs["true_propagation_risk"] = (
            pairs["avg_model_prop_b"] / q99_prop
        ).clip(lower=0.0, upper=1.0)
        pairs["true_ml_risk_score"] = (
            0.35 * pairs["true_delay_risk"]
            + 0.30 * pairs["true_propagation_risk"]
            + 0.20 * pairs["true_variance_risk"]
            + 0.15 * pairs["true_extreme_risk"]
        )
    else:
        # Renormalize non-propagation weights from the main scorer: 0.35, 0.20, 0.15
        w_delay, w_var, w_ext = 0.35, 0.20, 0.15
        denom = w_delay + w_var + w_ext
        pairs["true_propagation_risk"] = np.nan
        pairs["true_ml_risk_score"] = (
            (w_delay / denom) * pairs["true_delay_risk"]
            + (w_var / denom) * pairs["true_variance_risk"]
            + (w_ext / denom) * pairs["true_extreme_risk"]
        )

    return pairs


def _kendall_corr(a: pd.Series, b: pd.Series) -> float:
    try:
        # pandas can compute kendall correlation directly.
        return float(a.corr(b, method="kendall"))
    except Exception:
        return float("nan")


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    # NDCG expects non-negative relevance values.
    if y_true.size and float(np.nanmin(y_true)) < 0:
        y_true = y_true - float(np.nanmin(y_true))

    if ndcg_score is not None:
        return float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))

    # Fallback implementation (non-ideal tie behavior, but deterministic).
    order = np.argsort(-y_score)
    top = order[:k]
    gains = y_true[top]
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_order = np.argsort(-y_true)
    ideal_top = ideal_order[:k]
    ideal_gains = y_true[ideal_top]
    idcg = float(np.sum(ideal_gains * discounts))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_rankings(
    aligned: pd.DataFrame,
    ks: Iterable[int],
    top_pct: float,
) -> Dict:
    n = len(aligned)
    if n == 0:
        raise ValueError("No aligned pairs to evaluate")

    aligned = aligned.copy()
    aligned["pred_rank"] = aligned["pred_risk_score"].rank(method="first", ascending=False)
    aligned["true_rank"] = aligned["true_risk_score"].rank(method="first", ascending=False)

    n_true_top = max(1, int(math.ceil(top_pct * n)))
    true_top_ids = set(aligned.nsmallest(n_true_top, "true_rank")["pair_id"].tolist())

    y_true = aligned["true_risk_score"].to_numpy(dtype=float)
    y_score = aligned["pred_risk_score"].to_numpy(dtype=float)

    ranking = {
        "n_pairs_evaluated": int(n),
        "top_pct": float(top_pct),
        "n_true_top_pairs": int(n_true_top),
        "spearman": float(aligned["pred_rank"].corr(aligned["true_rank"], method="spearman")),
        "kendall": _kendall_corr(aligned["pred_rank"], aligned["true_rank"]),
        "at_k": {},
    }

    # Evaluate K metrics against "true top-pct" positives.
    order = aligned.sort_values("pred_risk_score", ascending=False)
    for k in ks:
        kk = min(int(k), n)
        pred_top_ids = set(order.head(kk)["pair_id"].tolist())
        tp = len(pred_top_ids & true_top_ids)
        precision = tp / kk if kk > 0 else float("nan")
        recall = tp / n_true_top if n_true_top > 0 else float("nan")
        ndcg = _ndcg_at_k(y_true, y_score, kk)
        ranking["at_k"][str(kk)] = {
            "k": int(kk),
            "tp": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "ndcg": float(ndcg),
        }

    return ranking, aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pair ranking quality against actual-delay-derived ground truth")
    parser.add_argument("--predicted-pairs", type=str, default="results/pair_risk_scores_full.csv")
    parser.add_argument("--flight-predictions", type=str, default="results/flight_predictions.csv")
    parser.add_argument("--output-dir", type=str, default="results/pair_ranking_eval")
    parser.add_argument("--hub", type=str, default="DFW")
    parser.add_argument("--extreme-threshold", type=float, default=180.0)
    parser.add_argument("--top-pct", type=float, default=0.05,
                        help="Top percentile of true-risk pairs treated as positives for Precision/Recall@K")
    parser.add_argument("--ks", type=str, default="10,25,50,100,250,500")
    parser.add_argument(
        "--truth-mode",
        type=str,
        choices=["actual_only", "actual_plus_model_propagation"],
        default="actual_only",
        help="Ground-truth score construction mode",
    )
    parser.add_argument("--include-infeasible", action="store_true",
                        help="Evaluate on all pairs; default evaluates feasible pairs only")
    parser.add_argument("--min-inbound-flights", type=int, default=1)
    parser.add_argument("--min-outbound-flights", type=int, default=1)
    args = parser.parse_args()

    if not (0 < args.top_pct <= 1.0):
        raise ValueError("--top-pct must be in (0, 1]")

    ks = parse_ks(args.ks)

    pred_path = Path(args.predicted_pairs)
    flights_path = Path(args.flight_predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predicted = pd.read_csv(pred_path)
    flights = pd.read_csv(flights_path)

    required_pred_cols = {"airport_a", "airport_b"}
    missing_pred = required_pred_cols - set(predicted.columns)
    if missing_pred:
        raise ValueError(f"predicted pairs file missing columns: {sorted(missing_pred)}")

    if "pred_risk_score" not in predicted.columns:
        if "risk_score" in predicted.columns:
            predicted = predicted.rename(columns={"risk_score": "pred_risk_score"})
        elif "final_score" in predicted.columns:
            predicted = predicted.rename(columns={"final_score": "pred_risk_score"})
        else:
            raise ValueError("predicted pairs file needs risk_score or final_score")

    if "is_feasible" not in predicted.columns:
        if "mct_violation" in predicted.columns:
            predicted["is_feasible"] = (predicted["mct_violation"] == 0).astype(int)
        else:
            predicted["is_feasible"] = 1

    truth_pairs = build_true_pair_scores(
        flights,
        hub=args.hub,
        extreme_threshold=args.extreme_threshold,
        truth_mode=args.truth_mode,
    )

    # Minimum support filters to reduce noisy pairs with tiny flight counts.
    truth_pairs = truth_pairs[
        (truth_pairs["n_actual_flights_a"] >= args.min_inbound_flights)
        & (truth_pairs["n_actual_flights_b"] >= args.min_outbound_flights)
    ].copy()

    merged = predicted.merge(
        truth_pairs,
        on=["airport_a", "airport_b"],
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No overlapping pairs between predicted and ground-truth tables")

    # Apply the same regulatory multiplier to truth-side score where available,
    # so ranking is comparable to operational scoring semantics.
    if "wocl_multiplier" not in merged.columns:
        merged["wocl_multiplier"] = 1.0

    merged["true_risk_score"] = (
        merged["true_ml_risk_score"] * merged["wocl_multiplier"].fillna(1.0)
    )

    if not args.include_infeasible:
        merged = merged[merged["is_feasible"] == 1].copy()

    merged["pair_id"] = merged["airport_a"].astype(str) + "->" + merged["airport_b"].astype(str)

    ranking_metrics, aligned = evaluate_rankings(merged, ks=ks, top_pct=args.top_pct)

    # Save aligned table with ranks and all score components.
    aligned_out = aligned.copy()
    aligned_out["pred_rank"] = aligned_out["pred_risk_score"].rank(method="first", ascending=False).astype(int)
    aligned_out["true_rank"] = aligned_out["true_risk_score"].rank(method="first", ascending=False).astype(int)
    aligned_out = aligned_out.sort_values("pred_rank")

    aligned_path = out_dir / "pair_ranking_aligned.csv"
    metrics_path = out_dir / "pair_ranking_metrics.json"

    aligned_out.to_csv(aligned_path, index=False)

    payload = {
        "inputs": {
            "predicted_pairs": str(pred_path),
            "flight_predictions": str(flights_path),
            "hub": args.hub,
            "extreme_threshold": args.extreme_threshold,
            "top_pct": args.top_pct,
            "ks": ks,
            "truth_mode": args.truth_mode,
            "include_infeasible": bool(args.include_infeasible),
            "min_inbound_flights": int(args.min_inbound_flights),
            "min_outbound_flights": int(args.min_outbound_flights),
        },
        "summary": {
            "n_pairs_after_join": int(len(merged)),
            "n_pairs_after_filter": int(len(aligned_out)),
            "pred_risk_min": float(aligned_out["pred_risk_score"].min()),
            "pred_risk_max": float(aligned_out["pred_risk_score"].max()),
            "true_risk_min": float(aligned_out["true_risk_score"].min()),
            "true_risk_max": float(aligned_out["true_risk_score"].max()),
        },
        "ranking_metrics": ranking_metrics,
        "outputs": {
            "aligned_csv": str(aligned_path),
            "metrics_json": str(metrics_path),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 70)
    print("PAIR RANKING EVALUATION")
    print("=" * 70)
    print(f"pairs evaluated: {ranking_metrics['n_pairs_evaluated']:,}")
    print(f"spearman: {ranking_metrics['spearman']:.4f}")
    print(f"kendall:  {ranking_metrics['kendall']:.4f}")
    print("\n@K metrics")
    for k in sorted(ranking_metrics["at_k"], key=lambda x: int(x)):
        m = ranking_metrics["at_k"][k]
        print(
            f"  K={m['k']:>4}: precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} ndcg={m['ndcg']:.4f} tp={m['tp']}"
        )

    print(f"\nSaved: {aligned_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
