"""
Airport Pair Risk Scorer — Architecture Layer 4 Output Contract

Properly maps model predictions back to (airport_a, airport_b) pairs.
Applies proxy features (duty_flag, mct_violation, wocl_multiplier) as
post-prediction constraints:
  final_score = ml_risk × wocl_multiplier, zeroed if mct_violation = 1

Output contract:
  airport_a · airport_b · month · ml_risk_score · wocl_multiplier ·
  final_score · duty_flag · mct_violation · wocl_flag
"""
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from model import TFTDCP


class PairRiskScorer:
    """Computes risk scores for A→DFW→B airport pairs."""

    def __init__(self, model: TFTDCP, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_from_dataloader(self, dataloader) -> pd.DataFrame:
        """
        Run model on dataloader, collect predictions aligned with origin/dest metadata.
        Returns one row per scored flight.
        """
        rows = []

        for batch in dataloader:
            dynamic = batch["dynamic"].to(self.device)
            static = batch["static"].to(self.device)
            chain_delays = batch["chain_delays"].to(self.device)
            turnarounds = batch["turnaround_times"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["target"]

            output = self.model(dynamic, static, chain_delays, turnarounds, mask)
            preds = output["prediction"].cpu().numpy()
            y_props = output["y_prop"].cpu().numpy()

            origins = batch["origin"]   # list of strings from custom collate
            dests = batch["dest"]       # list of strings from custom collate

            for i in range(len(preds)):
                rows.append({
                    "origin": origins[i],
                    "dest": dests[i],
                    "pred_delay": float(preds[i]),
                    "actual_delay": float(targets[i]),
                    "propagated_delay": float(y_props[i]),
                })

        flight_preds = pd.DataFrame(rows)
        print(f"  Scored {len(flight_preds):,} flights")
        return flight_preds

    def aggregate_pair_risks(
        self,
        flight_preds: pd.DataFrame,
        hub: str = "DFW",
        proxy_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Aggregate flight-level predictions to (airport_a, airport_b) pair risk scores.
        Merges with proxy features if provided.
        """
        inbound = flight_preds[flight_preds["dest"] == hub]
        outbound = flight_preds[flight_preds["origin"] == hub]

        a_stats = inbound.groupby("origin").agg(
            avg_delay_a=("pred_delay", "mean"),
            std_delay_a=("pred_delay", "std"),
            extreme_pct_a=("pred_delay", lambda x: (x > 180).mean() * 100),
            n_flights_a=("pred_delay", "count"),
        ).reset_index().rename(columns={"origin": "airport_a"})

        b_stats = outbound.groupby("dest").agg(
            avg_delay_b=("pred_delay", "mean"),
            std_delay_b=("pred_delay", "std"),
            avg_prop_delay=("propagated_delay", "mean"),
            n_flights_b=("pred_delay", "count"),
        ).reset_index().rename(columns={"dest": "airport_b"})

        # Cross-join A × B
        a_stats["_key"] = 1
        b_stats["_key"] = 1
        pairs = a_stats.merge(b_stats, on="_key").drop(columns="_key")

        # Composite ML risk (0-1)
        pairs["avg_delay_combined"] = (pairs["avg_delay_a"] + pairs["avg_delay_b"]) / 2
        max_delay = max(pairs["avg_delay_combined"].quantile(0.99), 1)
        pairs["delay_risk"] = (pairs["avg_delay_combined"] / max_delay).clip(upper=1)

        max_prop = max(pairs["avg_prop_delay"].quantile(0.99), 1)
        pairs["propagation_risk"] = (pairs["avg_prop_delay"] / max_prop).clip(upper=1)

        max_std = max(pairs[["std_delay_a", "std_delay_b"]].max().max(), 1)
        pairs["variance_risk"] = (
            (pairs["std_delay_a"].fillna(0) + pairs["std_delay_b"].fillna(0)) / (2 * max_std)
        ).clip(upper=1)

        pairs["extreme_risk"] = (
            pairs["extreme_pct_a"] / max(pairs["extreme_pct_a"].max(), 1)
        ).clip(upper=1)

        pairs["ml_risk_score"] = (
            0.35 * pairs["delay_risk"] +
            0.30 * pairs["propagation_risk"] +
            0.20 * pairs["variance_risk"] +
            0.15 * pairs["extreme_risk"]
        ).round(4)

        # Merge proxy features
        if proxy_df is not None and len(proxy_df) > 0:
            proxy_agg = proxy_df.groupby(["airport_a", "airport_b"]).agg(
                duty_flag=("duty_flag", "max"),
                mct_violation=("mct_violation", "max"),
                wocl_flag=("wocl_flag", "max"),
                wocl_multiplier=("wocl_multiplier", "max"),
                avg_duty_mins=("duty_time_mins", "mean"),
                avg_conn_mins=("dfw_conn_mins", "mean"),
            ).reset_index()
            pairs = pairs.merge(proxy_agg, on=["airport_a", "airport_b"], how="left")
        else:
            pairs["duty_flag"] = 0
            pairs["mct_violation"] = 0
            pairs["wocl_flag"] = 0
            pairs["wocl_multiplier"] = 1.0

        pairs["duty_flag"] = pairs["duty_flag"].fillna(0).astype(int)
        pairs["mct_violation"] = pairs["mct_violation"].fillna(0).astype(int)
        pairs["wocl_flag"] = pairs["wocl_flag"].fillna(0).astype(int)
        pairs["wocl_multiplier"] = pairs["wocl_multiplier"].fillna(1.0)

        # final_score = ml_risk × wocl_multiplier, zeroed if mct_violation
        pairs["final_score"] = (
            pairs["ml_risk_score"] *
            pairs["wocl_multiplier"] *
            (1 - pairs["mct_violation"])
        ).round(4)

        return pairs.sort_values("final_score", ascending=False)

    def export(self, pairs: pd.DataFrame, output_dir: str = "./results") -> pd.DataFrame:
        """Export scored pairs matching the PostgreSQL output contract."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        contract_cols = [
            "airport_a", "airport_b", "ml_risk_score", "wocl_multiplier",
            "final_score", "duty_flag", "mct_violation", "wocl_flag",
        ]
        available = [c for c in contract_cols if c in pairs.columns]
        scored = pairs[available].copy()

        pairs.to_csv(output_dir / "pair_risk_scores_full.csv", index=False)
        scored.to_csv(output_dir / "scored_pairs.csv", index=False)

        flagged = pairs[pairs["final_score"] >= 0.6].copy()
        flagged["recommendation"] = "AVOID"
        flagged.loc[flagged["final_score"] >= 0.8, "recommendation"] = "CRITICAL"

        flag_cols = [c for c in ["airport_a", "airport_b", "final_score",
                     "recommendation", "ml_risk_score", "duty_flag",
                     "mct_violation", "wocl_flag"] if c in flagged.columns]
        flagged[flag_cols].to_csv(output_dir / "flagged_pairs.csv", index=False)

        print(f"\n  scored_pairs.csv — {len(scored):,} pairs")
        print(f"  flagged_pairs.csv — {len(flagged):,} high-risk pairs")

        if len(flagged) > 0:
            print(f"\n  TOP 10 RISKIEST PAIRS:")
            print(flagged[flag_cols].head(10).to_string(index=False))

        return scored
