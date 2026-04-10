"""
Proxy Engineering — Layer 2 of Architecture
Derives 4 regulatory proxy features from BTS scheduled times:
  1. Enumerate all valid (A,B) pairs through DFW
  2. Duty time flag (FAA 14-hr rule)
  3. MCT violation (45-min DFW minimum connection)
  4. WOCL exposure (2AM-6AM local time)

These are deterministic features computed OUTSIDE the ML model.
"""
import pandas as pd
import numpy as np
from typing import Tuple


class ProxyEngineer:
    """Computes regulatory proxy features for A→DFW→B sequences."""

    MCT_MINUTES = 45        # Minimum Connection Time at DFW
    DUTY_LIMIT_MINS = 840   # 14 hours in minutes
    DUTY_WARN_MINS = 720    # 12 hours soft warning
    WOCL_START = 2           # Window Of Circadian Low start (2 AM)
    WOCL_END = 6             # WOCL end (6 AM)
    WOCL_MULTIPLIER = 1.35  # Risk multiplier for WOCL exposure

    def __init__(self, hub: str = "DFW"):
        self.hub = hub

    def build_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enumerate all valid A→DFW→B combinations on the same calendar day.
        Each row = one hypothetical pilot sequence.
        """
        hub = self.hub

        # Split into inbound (→DFW) and outbound (DFW→)
        inbound = df[df["Dest"] == hub].copy()
        outbound = df[df["Origin"] == hub].copy()

        if len(inbound) == 0 or len(outbound) == 0:
            print(f"  WARNING: No inbound or outbound flights to {hub}")
            return pd.DataFrame()

        # Parse times as minutes since midnight
        for subset, prefix in [(inbound, "in"), (outbound, "out")]:
            for col in ["CRSDepTime", "CRSArrTime", "DepTime", "ArrTime"]:
                if col in subset.columns:
                    raw = pd.to_numeric(subset[col], errors="coerce")
                    h = (raw // 100).astype(float)
                    m = (raw % 100).astype(float)
                    subset[f"{prefix}_{col}_mins"] = h * 60 + m

        # Rename to avoid collisions
        in_cols = {c: f"in_{c}" for c in inbound.columns if c != "FlightDate"}
        out_cols = {c: f"out_{c}" for c in outbound.columns if c != "FlightDate"}
        inbound = inbound.rename(columns=in_cols)
        outbound = outbound.rename(columns=out_cols)

        # Join: same calendar day, outbound departs after inbound arrives
        sequences = inbound.merge(outbound, on="FlightDate", how="inner")

        # Filter: outbound scheduled departure > inbound scheduled arrival + MCT
        dep_b = sequences.get("out_CRSDepTime_mins", sequences.get("out_out_CRSDepTime_mins"))
        arr_a = sequences.get("in_CRSArrTime_mins", sequences.get("in_in_CRSArrTime_mins"))

        # Handle column naming (might be double-prefixed depending on rename)
        if dep_b is None:
            for c in sequences.columns:
                if "CRSDepTime_mins" in c and "out" in c:
                    dep_b = sequences[c]
                    break
        if arr_a is None:
            for c in sequences.columns:
                if "CRSArrTime_mins" in c and "in" in c:
                    arr_a = sequences[c]
                    break

        if dep_b is not None and arr_a is not None:
            sequences["dfw_conn_mins"] = dep_b - arr_a
            # Only keep sequences where connection is physically possible (> 0 min)
            sequences = sequences[sequences["dfw_conn_mins"] > 0].copy()
        else:
            print("  WARNING: Could not compute DFW connection time — check column names")
            sequences["dfw_conn_mins"] = 60  # default

        print(f"  Enumerated {len(sequences):,} valid A→{hub}→B sequences")
        return sequences

    def compute_proxies(self, sequences: pd.DataFrame) -> pd.DataFrame:
        """Compute all 4 proxy features on the sequence table."""
        seq = sequences.copy()

        # --- Proxy 1: Extract airport pairs ---
        origin_col = [c for c in seq.columns if "Origin" in c and "in_" in c]
        dest_col = [c for c in seq.columns if "Dest" in c and "out_" in c]
        if origin_col:
            seq["airport_a"] = seq[origin_col[0]]
        if dest_col:
            seq["airport_b"] = seq[dest_col[0]]

        # --- Proxy 2: Duty time (FAA 14-hr rule) ---
        dep_a_col = [c for c in seq.columns if "CRSDepTime_mins" in c and "in" in c]
        arr_b_col = [c for c in seq.columns if "CRSArrTime_mins" in c and "out" in c]

        if dep_a_col and arr_b_col:
            dep_a = seq[dep_a_col[0]]
            arr_b = seq[arr_b_col[0]]
            seq["duty_time_mins"] = arr_b - dep_a
            # Handle overnight (negative = next day)
            seq.loc[seq["duty_time_mins"] < 0, "duty_time_mins"] += 1440
        else:
            seq["duty_time_mins"] = 0

        seq["duty_flag"] = (seq["duty_time_mins"] > self.DUTY_LIMIT_MINS).astype(int)
        seq["duty_warn"] = (seq["duty_time_mins"] > self.DUTY_WARN_MINS).astype(int)

        # --- Proxy 3: MCT violation ---
        seq["mct_violation"] = (seq["dfw_conn_mins"] < self.MCT_MINUTES).astype(int)

        # --- Proxy 4: WOCL exposure ---
        wocl_start = self.WOCL_START * 60  # 120 min
        wocl_end = self.WOCL_END * 60  # 360 min

        seq["wocl_flag"] = 0
        for time_col in seq.columns:
            if "CRSDepTime_mins" in time_col or "CRSArrTime_mins" in time_col:
                vals = seq[time_col]
                in_wocl = (vals >= wocl_start) & (vals <= wocl_end)
                seq.loc[in_wocl, "wocl_flag"] = 1

        seq["wocl_multiplier"] = np.where(seq["wocl_flag"] == 1, self.WOCL_MULTIPLIER, 1.0)

        # Summary
        n = len(seq)
        print(f"  Proxy summary ({n:,} sequences):")
        print(f"    Duty flag (>14hr): {seq['duty_flag'].sum():,} ({seq['duty_flag'].mean()*100:.1f}%)")
        print(f"    MCT violation (<45min): {seq['mct_violation'].sum():,} ({seq['mct_violation'].mean()*100:.1f}%)")
        print(f"    WOCL exposure: {seq['wocl_flag'].sum():,} ({seq['wocl_flag'].mean()*100:.1f}%)")

        return seq

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full proxy pipeline: enumerate pairs → compute proxies."""
        sequences = self.build_sequences(df)
        if len(sequences) == 0:
            return sequences
        return self.compute_proxies(sequences)
