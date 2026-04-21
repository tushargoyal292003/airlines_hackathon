"""
Framing A — Pre-season strategic ranking.

Produces a causal feature set usable on Dec 31 of year Y-1 to rank pair risk
for year Y. No actuals, no same-day chain observations, no wheels/taxi/airtime.
Propagation is preserved via HISTORICAL priors computed on strictly prior years.

Usage:
    from causal_features import (
        build_route_priors, attach_priors, CAUSAL_FEATURES, BLACKLIST
    )

    # prior_years = years strictly before the target split's test year
    priors = build_route_priors(df_all, prior_years=[2019, 2022, 2023])
    train_df = attach_priors(train_df, priors)
    val_df   = attach_priors(val_df, priors)
    test_df  = attach_priors(test_df, priors)

    X_train = train_df[CAUSAL_FEATURES].fillna(0).values
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable


# ──────────────────────────────────────────────────────────────
# Feature policy
# ──────────────────────────────────────────────────────────────

# Known-before-departure inputs only. Keep additions here; never add an `actual_*`.
CAUSAL_FEATURES = [
    # schedule
    "CRSDepTime_raw", "CRSArrTime_raw", "CRSElapsedTime", "Distance",
    # encoded identities (James-Stein or ordinal; both OK if fit on prior years)
    "Origin", "Dest", "Reporting_Airline",
    # calendar (cyclical + raw year)
    "Year", "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    # weather FORECAST proxies (NOAA obs used as a stand-in for forecast at training time)
    "TMAX", "TMIN", "PRCP", "AWND", "SNOW",
    # leak-safe priors (attached by attach_priors)
    "route_avg_delay_prior",
    "route_p90_delay_prior",
    "route_std_delay_prior",
    "route_extreme_rate_prior",
    "route_prop_delay_prior",     # historical mean of prop_delay feature
    "route_buffer_exceeded_prior",
    "origin_avg_delay_prior",
    "dest_avg_delay_prior",
    "dep_hour_avg_delay_prior",
]

# Hard blacklist — remove from both LightGBM X and TFT-DCP dynamic features.
BLACKLIST = {
    "actual_taxi_out", "taxi_out_excess", "actual_airborne", "airborne_excess",
    "TaxiOut", "AirTime", "WheelsOff", "WheelsOn",
    "DepTime", "ArrTime", "DepTime_raw", "ArrTime_raw",
    "ArrDelay", "ArrDelayMinutes", "DepDelayMinutes",
    "prev_arr_delay", "prev_dep_delay",   # same-day chain observations
    "prop_delay", "prop_weight",          # same-day propagation, replaced by route_prop_delay_prior
    "chain_cumulative_delay", "chain_delay_diff", "chain_max_delay",
    "turnaround_buffer", "buffer_exceeded",
    "prop_delay_2hop", "prop_delay_3hop", "pos_x_prev_delay",
    "cum_delay_past_hour",                # operational real-time signal, not available pre-season
}


def drop_blacklist(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any leaky columns the preprocessor may have produced."""
    cols = [c for c in df.columns if c in BLACKLIST]
    if cols:
        df = df.drop(columns=cols)
    return df


# ──────────────────────────────────────────────────────────────
# Leak-safe propagation: historical same-day chain prop, then average per route
# ──────────────────────────────────────────────────────────────

def _compute_same_day_propagation(df: pd.DataFrame, beta: float = 0.8) -> pd.DataFrame:
    """
    Compute prop_delay on a *historical* slice ONLY.
    Caller must filter df to prior years before calling this.
    """
    df = df.sort_values(["chain_id", "CRSDepTime_raw"]).copy()
    # predecessor arrival delay within the same chain
    df["prev_arr_delay"] = (
        df.groupby("chain_id")["ArrDelay"].shift(1)
        if "ArrDelay" in df.columns
        else df.groupby("chain_id")["DepDelay"].shift(1)
    )
    # turnaround_minutes may or may not exist; approximate from CRS times
    if "turnaround_minutes" not in df.columns:
        prev_crs_arr = df.groupby("chain_id")["CRSArrTime_raw"].shift(1)
        df["turnaround_minutes"] = _hhmm_diff_minutes(prev_crs_arr, df["CRSDepTime_raw"])
    dt_hours = df["turnaround_minutes"].fillna(60).clip(lower=5) / 60.0
    decay = np.exp(-beta * dt_hours)
    df["prop_delay"] = decay * df["prev_arr_delay"].fillna(0)
    df["buffer_exceeded"] = (
        (df["turnaround_minutes"].fillna(60) - df["prev_arr_delay"].fillna(0)) < 0
    ).astype(int)
    return df


def _hhmm_diff_minutes(t_from: pd.Series, t_to: pd.Series) -> pd.Series:
    """Difference in minutes between two HHMM time-of-day series (same day)."""
    def to_min(s):
        s = pd.to_numeric(s, errors="coerce")
        return (s // 100) * 60 + (s % 100)
    return (to_min(t_to) - to_min(t_from)).where(lambda x: x >= 0, other=np.nan)


# ──────────────────────────────────────────────────────────────
# Build route-level priors from prior years
# ──────────────────────────────────────────────────────────────

def build_route_priors(
    df: pd.DataFrame,
    prior_years: Iterable[int],
    year_col: str = "Year_raw",
    origin_col: str = "Origin_str",
    dest_col: str = "Dest_str",
    delay_col: str = "DepDelay",
    beta: float = 0.8,
) -> dict:
    """
    Build leak-safe priors. Every statistic is computed from flights whose Year_raw
    is in `prior_years` only — strictly before any year used for val/test.

    Returns a dict of DataFrames keyed by join level.
    """
    prior = df[df[year_col].isin(list(prior_years))].copy()
    if origin_col not in prior.columns:
        prior[origin_col] = prior["Origin"].astype(str)
    if dest_col not in prior.columns:
        prior[dest_col] = prior["Dest"].astype(str)

    # route-level delay priors
    route = prior.groupby([origin_col, dest_col])[delay_col].agg(
        route_avg_delay_prior="mean",
        route_p90_delay_prior=lambda s: np.percentile(s, 90),
        route_std_delay_prior="std",
        route_extreme_rate_prior=lambda s: float((s > 180).mean()),
    ).reset_index().rename(columns={origin_col: "Origin_str", dest_col: "Dest_str"})

    # route-level historical propagation
    prior_prop = _compute_same_day_propagation(prior, beta=beta)
    prop = prior_prop.groupby([origin_col, dest_col]).agg(
        route_prop_delay_prior=("prop_delay", "mean"),
        route_buffer_exceeded_prior=("buffer_exceeded", "mean"),
    ).reset_index().rename(columns={origin_col: "Origin_str", dest_col: "Dest_str"})

    # airport-level and hour-level fallbacks
    origin_prior = prior.groupby(origin_col)[delay_col].mean().reset_index()
    origin_prior.columns = ["Origin_str", "origin_avg_delay_prior"]
    dest_prior = prior.groupby(dest_col)[delay_col].mean().reset_index()
    dest_prior.columns = ["Dest_str", "dest_avg_delay_prior"]

    prior["dep_hour"] = (pd.to_numeric(prior["CRSDepTime_raw"], errors="coerce") // 100)
    hour_prior = prior.groupby("dep_hour")[delay_col].mean().reset_index()
    hour_prior.columns = ["dep_hour", "dep_hour_avg_delay_prior"]

    # global fallbacks
    globals_ = {
        "route_avg_delay_prior": float(prior[delay_col].mean()),
        "route_p90_delay_prior": float(np.percentile(prior[delay_col], 90)),
        "route_std_delay_prior": float(prior[delay_col].std()),
        "route_extreme_rate_prior": float((prior[delay_col] > 180).mean()),
        "route_prop_delay_prior": float(prior_prop["prop_delay"].mean()),
        "route_buffer_exceeded_prior": float(prior_prop["buffer_exceeded"].mean()),
        "origin_avg_delay_prior": float(prior[delay_col].mean()),
        "dest_avg_delay_prior": float(prior[delay_col].mean()),
        "dep_hour_avg_delay_prior": float(prior[delay_col].mean()),
    }

    return {
        "route": route, "prop": prop,
        "origin": origin_prior, "dest": dest_prior, "hour": hour_prior,
        "globals": globals_,
    }


def attach_priors(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """
    Left-merge priors onto df. Never touches the target. Fills missing with the
    prior-year global mean so unseen routes get a neutral prior, not NaN.
    """
    df = df.copy()
    if "Origin_str" not in df.columns:
        df["Origin_str"] = df["Origin"].astype(str)
    if "Dest_str" not in df.columns:
        df["Dest_str"] = df["Dest"].astype(str)
    df["dep_hour"] = (pd.to_numeric(df["CRSDepTime_raw"], errors="coerce") // 100)

    df = df.merge(priors["route"], on=["Origin_str", "Dest_str"], how="left")
    df = df.merge(priors["prop"], on=["Origin_str", "Dest_str"], how="left")
    df = df.merge(priors["origin"], on="Origin_str", how="left")
    df = df.merge(priors["dest"], on="Dest_str", how="left")
    df = df.merge(priors["hour"], on="dep_hour", how="left")

    for col, fill in priors["globals"].items():
        if col in df.columns:
            df[col] = df[col].fillna(fill)

    return df


# ──────────────────────────────────────────────────────────────
# Pair scorer — causal version
# ──────────────────────────────────────────────────────────────

def score_pairs_causal(
    flight_preds: pd.DataFrame,
    proxy_df: pd.DataFrame | None = None,
    hub: str = "DFW",
) -> pd.DataFrame:
    """
    Pair risk from causal pipeline.
    - delay_risk:       from pred_delay
    - propagation_risk: from route_prop_delay_prior (leak-safe)
    - volatility_risk:  from std of pred_delay
    - extreme_risk:     from predicted extremes: (pred_delay > 180).mean()
                        NOT from actuals — this is the leak fix.
    """
    df = flight_preds.copy()
    df["origin"] = df["origin"].astype(str)
    df["dest"] = df["dest"].astype(str)
    inb = df[df["dest"] == hub]
    outb = df[df["origin"] == hub]
    if len(inb) == 0 or len(outb) == 0:
        return pd.DataFrame()

    def agg(part, key, suffix):
        g = part.groupby(key).agg(
            **{
                f"avg_delay_{suffix}": ("pred_delay", "mean"),
                f"p90_delay_{suffix}": ("pred_delay", lambda x: float(np.percentile(x, 90))),
                f"std_delay_{suffix}": ("pred_delay", "std"),
                f"pred_extreme_rate_{suffix}": ("pred_delay", lambda x: float((x > 180).mean())),
                f"prop_prior_{suffix}": ("route_prop_delay_prior", "mean"),
                f"n_{suffix}": ("pred_delay", "count"),
            }
        ).reset_index().rename(columns={key: f"airport_{suffix}"})
        return g

    a = agg(inb, "origin", "a")
    b = agg(outb, "dest", "b")
    a["_k"] = 1; b["_k"] = 1
    pairs = a.merge(b, on="_k").drop(columns="_k")

    # normalize each to [0,1] using 95th percentile as ceiling
    def norm(col):
        ceil = max(pairs[col].quantile(0.95), 1e-6)
        return (pairs[col] / ceil).clip(0, 1)

    pairs["combined_p90"] = (pairs["p90_delay_a"] + pairs["p90_delay_b"]) / 2
    pairs["delay_risk"] = norm("combined_p90")
    pairs["avg_prop"] = (pairs["prop_prior_a"] + pairs["prop_prior_b"]) / 2
    pairs["propagation_risk"] = norm("avg_prop")
    pairs["avg_std"] = (pairs["std_delay_a"].fillna(0) + pairs["std_delay_b"].fillna(0)) / 2
    pairs["volatility_risk"] = norm("avg_std")
    pairs["avg_extreme"] = (pairs["pred_extreme_rate_a"] + pairs["pred_extreme_rate_b"]) / 2
    pairs["extreme_risk"] = norm("avg_extreme")

    pairs["ml_risk_score"] = (
        0.35 * pairs["delay_risk"]
        + 0.25 * pairs["propagation_risk"]
        + 0.25 * pairs["volatility_risk"]
        + 0.15 * pairs["extreme_risk"]
    ).round(4)

    # proxy overlay (duty/MCT/WOCL) — these are rule-based on schedules, always causal
    for c in ["duty_flag", "mct_violation", "wocl_flag"]:
        pairs[c] = 0
    pairs["wocl_multiplier"] = 1.0
    if proxy_df is not None and len(proxy_df):
        keep = [c for c in ["airport_a", "airport_b", "duty_flag", "mct_violation",
                            "wocl_flag", "wocl_multiplier"] if c in proxy_df.columns]
        if "airport_a" in keep and "airport_b" in keep:
            pg = proxy_df.groupby(["airport_a", "airport_b"])[
                [c for c in keep if c not in ("airport_a", "airport_b")]
            ].max().reset_index()
            pairs = pairs.drop(columns=["duty_flag", "mct_violation", "wocl_flag",
                                        "wocl_multiplier"], errors="ignore")
            pairs = pairs.merge(pg, on=["airport_a", "airport_b"], how="left")

    for c in ["duty_flag", "mct_violation", "wocl_flag"]:
        pairs[c] = pairs.get(c, 0)
        pairs[c] = pairs[c].fillna(0).astype(int)
    pairs["wocl_multiplier"] = pairs.get("wocl_multiplier", 1.0)
    pairs["wocl_multiplier"] = pairs["wocl_multiplier"].fillna(1.0)

    pairs["final_score"] = (
        pairs["ml_risk_score"] * pairs["wocl_multiplier"] * (1 - pairs["mct_violation"])
    ).round(4)

    pairs["flag"] = "OK"
    pairs.loc[pairs["final_score"] >= 0.6, "flag"] = "AVOID"
    pairs.loc[pairs["final_score"] >= 0.8, "flag"] = "CRITICAL"
    pairs.loc[pairs["mct_violation"] == 1, "flag"] = "MCT_VIOLATION"

    return pairs.sort_values("final_score", ascending=False).reset_index(drop=True)
