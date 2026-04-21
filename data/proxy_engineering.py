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
    # Pair-level infeasibility policy: a pair is infeasible only when
    # MCT violations are persistent, not when they occur once.
    MCT_INFEASIBLE_RATE = 0.50
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

        # Split into inbound (→DFW) and outbound (DFW→).
        # Prefer raw string columns (preserved by preprocessor) since
        # Origin/Dest may have been James-Stein-encoded to floats.
        origin_col = "Origin_str" if "Origin_str" in df.columns else "Origin"
        dest_col   = "Dest_str"   if "Dest_str"   in df.columns else "Dest"

        # Slim to only the columns proxy engineering needs — otherwise the
        # inbound × outbound merge carries 75 cols × ~1B rows and OOMs.
        keep = [origin_col, dest_col, "FlightDate",
                "CRSDepTime", "CRSArrTime", "DepTime", "ArrTime"]
        keep = [c for c in keep if c in df.columns]

        inbound = df.loc[df[dest_col] == hub, keep].copy()
        outbound = df.loc[df[origin_col] == hub, keep].copy()

        if len(inbound) == 0 or len(outbound) == 0:
            print(f"  WARNING: No inbound or outbound flights to {hub}")
            return pd.DataFrame()

        # Dedupe: proxy features depend on (airport, date, scheduled times)
        # only — collapse duplicate schedule rows so the merge stays bounded.
        inbound = inbound.drop_duplicates(
            subset=[origin_col, "FlightDate", "CRSDepTime", "CRSArrTime"]
        )
        outbound = outbound.drop_duplicates(
            subset=[dest_col, "FlightDate", "CRSDepTime", "CRSArrTime"]
        )
        print(f"  After slim+dedupe: {len(inbound):,} inbound × {len(outbound):,} outbound")

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
        """
        Full proxy pipeline.  Enumerating every inbound×outbound combo on the
        same day blows up to ~1B rows at DFW scale.  Instead we process one
        day at a time, compute proxies, and aggregate to (airport_a, airport_b)
        worst-case per day — concatenated across days this stays bounded.
        """
        hub = self.hub
        origin_col = "Origin_str" if "Origin_str" in df.columns else "Origin"
        dest_col   = "Dest_str"   if "Dest_str"   in df.columns else "Dest"

        dep_col = "CRSDepTime_raw" if "CRSDepTime_raw" in df.columns else "CRSDepTime"
        arr_col = "CRSArrTime_raw" if "CRSArrTime_raw" in df.columns else "CRSArrTime"
        keep = [c for c in [origin_col, dest_col, "FlightDate",
                            dep_col, arr_col] if c in df.columns]
        inbound_all  = df.loc[df[dest_col] == hub, keep]
        outbound_all = df.loc[df[origin_col] == hub, keep]

        if len(inbound_all) == 0 or len(outbound_all) == 0:
            print(f"  WARNING: No inbound or outbound flights to {hub}")
            return pd.DataFrame()

        dates = sorted(set(inbound_all["FlightDate"].unique()) &
                       set(outbound_all["FlightDate"].unique()))
        print(f"  Processing {len(dates):,} days of A→{hub}→B pairs")

        in_grp  = {d: g for d, g in inbound_all.groupby("FlightDate", sort=False)}
        out_grp = {d: g for d, g in outbound_all.groupby("FlightDate", sort=False)}

        daily_parts = []
        for i, d in enumerate(dates):
            inb = in_grp[d]
            out = out_grp[d]

            inb_a  = (pd.to_numeric(inb[arr_col], errors="coerce") // 100) * 60 + \
                     (pd.to_numeric(inb[arr_col], errors="coerce") % 100)
            inb_d  = (pd.to_numeric(inb[dep_col], errors="coerce") // 100) * 60 + \
                     (pd.to_numeric(inb[dep_col], errors="coerce") % 100)
            out_d  = (pd.to_numeric(out[dep_col], errors="coerce") // 100) * 60 + \
                     (pd.to_numeric(out[dep_col], errors="coerce") % 100)
            out_a  = (pd.to_numeric(out[arr_col], errors="coerce") // 100) * 60 + \
                     (pd.to_numeric(out[arr_col], errors="coerce") % 100)

            inb_small = pd.DataFrame({
                "airport_a": inb[origin_col].values,
                "arr_a_mins": inb_a.values,
                "dep_a_mins": inb_d.values,
            }).dropna()
            out_small = pd.DataFrame({
                "airport_b": out[dest_col].values,
                "dep_b_mins": out_d.values,
                "arr_b_mins": out_a.values,
            }).dropna()

            inb_small["_k"] = 1
            out_small["_k"] = 1
            seq = inb_small.merge(out_small, on="_k").drop(columns="_k")

            seq["dfw_conn_mins"] = seq["dep_b_mins"] - seq["arr_a_mins"]
            seq = seq[seq["dfw_conn_mins"] > 0]
            if len(seq) == 0:
                continue

            duty = seq["arr_b_mins"] - seq["dep_a_mins"]
            duty = duty.where(duty >= 0, duty + 1440)
            seq["duty_time_mins"] = duty
            seq["duty_flag"]     = (duty > self.DUTY_LIMIT_MINS).astype(int)
            seq["mct_violation"] = (seq["dfw_conn_mins"] < self.MCT_MINUTES).astype(int)

            wstart, wend = self.WOCL_START * 60, self.WOCL_END * 60
            in_wocl = (
                seq["dep_a_mins"].between(wstart, wend) |
                seq["arr_a_mins"].between(wstart, wend) |
                seq["dep_b_mins"].between(wstart, wend) |
                seq["arr_b_mins"].between(wstart, wend)
            )
            seq["wocl_flag"] = in_wocl.astype(int)
            seq["wocl_multiplier"] = np.where(in_wocl, self.WOCL_MULTIPLIER, 1.0)

            agg = seq.groupby(["airport_a", "airport_b"], sort=False).agg(
                duty_flag=("duty_flag", "max"),
                n_sequences=("dfw_conn_mins", "size"),
                n_mct_violations=("mct_violation", "sum"),
                n_wocl=("wocl_flag", "sum"),
                duty_time_mins=("duty_time_mins", "max"),
                conn_min_mins=("dfw_conn_mins", "min"),
                conn_sum_mins=("dfw_conn_mins", "sum"),
            ).reset_index()
            daily_parts.append(agg)

            if (i + 1) % 200 == 0:
                print(f"    processed {i+1:,}/{len(dates):,} days")

        if not daily_parts:
            return pd.DataFrame()

        all_days = pd.concat(daily_parts, ignore_index=True)
        final = all_days.groupby(["airport_a", "airport_b"], sort=False).agg(
            duty_flag=("duty_flag", "max"),
            n_sequences=("n_sequences", "sum"),
            n_mct_violations=("n_mct_violations", "sum"),
            n_wocl=("n_wocl", "sum"),
            duty_time_mins=("duty_time_mins", "max"),
            conn_min_mins=("conn_min_mins", "min"),
            conn_sum_mins=("conn_sum_mins", "sum"),
        ).reset_index()

        final["n_sequences"] = final["n_sequences"].clip(lower=1)
        final["mct_violation_rate"] = final["n_mct_violations"] / final["n_sequences"]
        final["wocl_exposure_rate"] = final["n_wocl"] / final["n_sequences"]
        final["avg_duty_mins"] = final["duty_time_mins"]
        final["avg_conn_mins"] = final["conn_sum_mins"] / final["n_sequences"]

        # Feasibility now depends on persistent MCT violations.
        final["mct_violation"] = (
            final["mct_violation_rate"] >= self.MCT_INFEASIBLE_RATE
        ).astype(int)
        final["wocl_flag"] = (final["wocl_exposure_rate"] > 0).astype(int)
        final["wocl_multiplier"] = (
            1.0 + (self.WOCL_MULTIPLIER - 1.0) * final["wocl_exposure_rate"]
        ).clip(lower=1.0, upper=self.WOCL_MULTIPLIER)

        n = len(final)
        print(f"\n  Aggregated to {n:,} unique (airport_a, airport_b) pairs")
        print(f"    Duty flag (>14hr): {int(final['duty_flag'].sum()):,} ({final['duty_flag'].mean()*100:.1f}%)")
        print(
            "    MCT infeasible (violation rate "
            f">= {self.MCT_INFEASIBLE_RATE:.2f}): "
            f"{int(final['mct_violation'].sum()):,} ({final['mct_violation'].mean()*100:.1f}%)"
        )
        print(f"    WOCL exposure: {int(final['wocl_flag'].sum()):,} ({final['wocl_flag'].mean()*100:.1f}%)")
        return final
