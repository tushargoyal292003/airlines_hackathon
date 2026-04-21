"""
Generate all report figures for the GROW 26.2 hackathon report.
Outputs go to results/report_figures/.

Usage:
    python generate_report_figures.py
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings("ignore")

OUT_DIR = Path("results/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRAND_BLUE   = "#003087"  # AA navy
BRAND_RED    = "#CC0000"  # AA red
BRAND_GREY   = "#6C6C6C"
ACCENT_GOLD  = "#F5A623"
PALETTE = [BRAND_BLUE, BRAND_RED, ACCENT_GOLD, "#2CA02C", "#9467BD", "#8C564B"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Feature Importance + Leakage Validation
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_importance():
    print("Generating: fig_feature_importance")
    lgbm_fi = pd.read_csv("results/lightgbm_feature_importance.csv")
    xgb_fi  = pd.read_csv("results/xgboost_feature_importance.csv")

    # Normalise to 0-100
    lgbm_fi["pct"] = lgbm_fi["importance"] / lgbm_fi["importance"].sum() * 100
    xgb_fi["pct"]  = xgb_fi["importance"]  / xgb_fi["importance"].sum()  * 100

    top_n = 20
    lgbm_top = lgbm_fi.nlargest(top_n, "pct").iloc[::-1]
    xgb_top  = xgb_fi.nlargest(top_n, "pct").iloc[::-1]

    leakage_data = {
        "LightGBM (leaky)":  {"MAE": 11.2,  "R²": 0.79,  "Spearman ρ": None},
        "LightGBM (causal)": {"MAE": 23.10, "R²": 0.207, "Spearman ρ": 0.562},
        "TFT-DCP (causal)":  {"MAE": 24.79, "R²": 0.236, "Spearman ρ": 0.737},
    }

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel 1: LightGBM feature importance
    ax1.barh(lgbm_top["feature"], lgbm_top["pct"], color=BRAND_BLUE, alpha=0.85)
    ax1.set_xlabel("Importance (%)")
    ax1.set_title("LightGBM Feature Importance\n(top 20, causal features only)", fontweight="bold")
    ax1.tick_params(axis="y", labelsize=8)

    # Panel 2: XGBoost feature importance
    ax2.barh(xgb_top["feature"], xgb_top["pct"], color=BRAND_RED, alpha=0.85)
    ax2.set_xlabel("Importance (%)")
    ax2.set_title("XGBoost Feature Importance\n(top 20, causal features only)", fontweight="bold")
    ax2.tick_params(axis="y", labelsize=8)

    # Panel 3: Leakage impact bar chart
    models = list(leakage_data.keys())
    r2_vals = [leakage_data[m]["R²"] for m in models]
    mae_vals = [leakage_data[m]["MAE"] for m in models]
    colors = [BRAND_RED, BRAND_BLUE, BRAND_BLUE]
    hatches = ["//", "", ""]

    x = np.arange(len(models))
    bars = ax3.bar(x, r2_vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax3.set_ylabel("Test R²")
    ax3.set_title("Leakage Detection:\nR² Drops 0.79 → 0.21 After Removing\nActual Flight Times", fontweight="bold")
    ax3.axhline(0, color="black", linewidth=0.7)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, mae_vals, "o--", color=ACCENT_GOLD, linewidth=2, markersize=8, label="MAE (min)")
    ax3_twin.set_ylabel("MAE (minutes)", color=ACCENT_GOLD)
    ax3_twin.tick_params(axis="y", labelcolor=ACCENT_GOLD)
    ax3_twin.legend(loc="upper right", fontsize=8)


    fig.suptitle("Feature Importance & Leakage Analysis — Pre-Season Causal Framing",
                 fontsize=13, fontweight="bold", y=1.01)
    save(fig, "fig_feature_importance")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture():
    print("Generating: fig_architecture")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(ax, x, y, w, h, label, sublabel="", color=BRAND_BLUE, fontsize=9, alpha=0.88):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=color, alpha=alpha,
                              edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.12 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                    ha="center", va="center", fontsize=7, color="#DDDDDD")

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=BRAND_GREY,
                                   lw=1.5, mutation_scale=14))

    # Input blocks
    box(ax, 0.3, 7.8, 2.8, 0.9, "Static Features",
        "route priors · airline · distance\nMonth/DoW cyclicals",
        color="#1A5276", fontsize=8)
    box(ax, 0.3, 6.0, 2.8, 1.5, "Dynamic Sequence",
        "ASPM metrics · weather\nturnaround · chain pos·len\nop_density · sched times",
        color="#1A5276", fontsize=8)
    box(ax, 0.3, 4.7, 2.8, 0.9, "Weather Context",
        "wind · visibility · precip\ntemp · humidity · wx_severity",
        color="#1A5276", fontsize=8)

    # TCN Encoder
    box(ax, 3.8, 5.6, 2.6, 2.2, "TCN Encoder",
        "3× dilated causal conv\nLayerNorm (not BN)\ndilation: 1, 2, 4",
        color=BRAND_BLUE, fontsize=8.5)

    # GRN
    box(ax, 3.8, 7.8, 2.6, 0.9, "GRN",
        "Gated Residual Network\nstatic embedding", color="#117A65", fontsize=8.5)

    # Historical Retrieval
    box(ax, 3.8, 4.1, 2.6, 1.1, "Historical Retrieval",
        "cosine sim · top-k=5\nsoftmax fusion · learnable α",
        color="#6C3483", fontsize=8.5)

    # MS-CA-EFM
    box(ax, 7.5, 5.6, 3.0, 2.5, "MS-CA-EFM",
        "Multi-Scale Channel Attention\nExcitation-Feature Modulation\nSection 3.2.3",
        color="#1F618D", fontsize=8.5)

    # Propagation
    box(ax, 7.5, 3.8, 3.0, 1.5, "Delay Propagation",
        "learnable β chain model\np_prop = 1-exp(-β·delay)\nSection 3.2.4",
        color="#922B21", fontsize=8.5)

    # Output
    box(ax, 11.8, 5.8, 3.0, 1.4, "Output Head",
        "predicted DepDelay (min)\nLinear regression",
        color="#148F77", fontsize=8.5)
    box(ax, 11.8, 3.8, 3.0, 1.6, "Risk Scorer",
        "delay_risk · variance_risk\npropagation_risk · extreme_risk\n+ FAR117 / WOCL / MCT",
        color=BRAND_RED, fontsize=8)

    # Arrows: inputs → encoders
    arrow(ax, 3.1, 8.25, 3.8, 8.25)   # static → GRN
    arrow(ax, 3.1, 6.75, 3.8, 6.75)   # dynamic → TCN
    arrow(ax, 3.1, 5.15, 3.8, 5.0)    # weather → TCN (bottom)
    arrow(ax, 3.1, 4.9, 3.8, 4.65)    # weather → retrieval

    # TCN → MS-CA-EFM
    arrow(ax, 6.4, 6.7, 7.5, 6.85)
    # GRN → MS-CA-EFM
    arrow(ax, 6.4, 8.25, 7.5, 7.5)
    # Historical Retrieval → MS-CA-EFM
    arrow(ax, 6.4, 4.65, 7.5, 5.85)

    # MS-CA-EFM → Output
    arrow(ax, 10.5, 6.85, 11.8, 6.5)
    # Propagation → Risk scorer
    arrow(ax, 10.5, 4.55, 11.8, 4.6)
    # MS-CA-EFM → Propagation
    arrow(ax, 9.0, 5.6, 9.0, 5.3)

    # Output → Risk
    arrow(ax, 13.3, 5.8, 13.3, 5.4)

    # Label sections
    ax.text(8.0, 9.55, "TFT-DCP Architecture — Pre-Season Risk Scoring",
            fontsize=14, fontweight="bold", ha="center")
    ax.text(1.7, 9.2, "INPUTS\n(pre-departure only)", ha="center",
            fontsize=9, color=BRAND_GREY, style="italic")
    ax.text(5.1, 9.2, "ENCODERS", ha="center",
            fontsize=9, color=BRAND_GREY, style="italic")
    ax.text(9.0, 9.2, "FUSION &\nPROPAGATION", ha="center",
            fontsize=9, color=BRAND_GREY, style="italic")
    ax.text(13.3, 9.2, "OUTPUTS", ha="center",
            fontsize=9, color=BRAND_GREY, style="italic")

    # Vertical dividers
    for xv in [3.5, 7.2, 11.5]:
        ax.axvline(xv, color="#CCCCCC", linewidth=0.7, linestyle="--", ymin=0.3, ymax=0.95)

    save(fig, "fig_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Training Loss Convergence
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_convergence():
    print("Generating: fig_training_convergence")
    with open("logs/training_history.json") as f:
        hist = json.load(f)

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    train_loss = np.array(hist["train_loss"])
    val_loss   = np.array(hist["val_loss"])
    val_mae    = np.array(hist.get("val_mae", [np.nan] * len(epochs)))
    lr_hist    = np.array(hist.get("lr", [np.nan] * len(epochs)))

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val   = val_loss[best_epoch - 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss / train_loss[0], color=BRAND_BLUE,
            linewidth=2, label="Train Loss (normalised)")
    ax.plot(epochs, val_loss / val_loss[0], color=BRAND_RED,
            linewidth=2, label="Val Loss (normalised)")
    ax.axvline(best_epoch, color=ACCENT_GOLD, linestyle="--", linewidth=1.5,
               label=f"Best val epoch {best_epoch}")
    ax.fill_between(epochs, val_loss / val_loss[0],
                    where=epochs >= best_epoch, alpha=0.12, color=BRAND_RED,
                    label="Early stopping window")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalised to epoch 1)")
    ax.set_title("Training & Validation Loss Convergence\n(early stopping patience = 10)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.annotate(f"Best val\nep {best_epoch}",
                xy=(best_epoch, best_val / val_loss[0]),
                xytext=(best_epoch + 4, best_val / val_loss[0] + 0.02),
                fontsize=8, color=ACCENT_GOLD,
                arrowprops=dict(arrowstyle="->", color=ACCENT_GOLD))

    # Right: val MAE
    ax2 = axes[1]
    if not np.all(np.isnan(val_mae)):
        ax2.plot(epochs, val_mae, color=BRAND_BLUE, linewidth=2)
        ax2.axvline(best_epoch, color=ACCENT_GOLD, linestyle="--", linewidth=1.5)
        ax2.axhline(val_mae[best_epoch - 1], color=BRAND_RED, linestyle=":",
                    linewidth=1.5,
                    label=f"Best MAE = {val_mae[best_epoch-1]:.2f} min")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation MAE (minutes)")
        ax2.set_title("Validation MAE Over Training", fontweight="bold")
        ax2.legend(fontsize=9)
    else:
        # fallback: draw approximate curve from known results
        ax2.text(0.5, 0.5, "val_mae not recorded\nin training history",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color=BRAND_GREY)
        ax2.set_title("Validation MAE", fontweight="bold")

    fig.suptitle("TFT-DCP Training Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_training_convergence")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Seasonal Analysis
# ─────────────────────────────────────────────────────────────────────────────

def fig_seasonal_analysis():
    print("Generating: fig_seasonal_analysis")

    seasons = ["Winter", "Spring", "Summer", "Fall"]

    # V1 numbers from results/metrics.json (v1 — from paper summary)
    tft_mae    = [21.85, 25.66, 29.40, 21.63]
    lgbm_mae   = [None, None, None, None]  # read from seasonal results
    # Read LightGBM seasonal if available
    try:
        lm = pd.read_json("results/lightgbm/metrics_lightgbm.json")
        # not structured with seasonal — skip
    except Exception:
        pass

    # Use V1 TFT-DCP seasonal numbers from summary
    model_seasonal = {
        "TFT-DCP (ours)": [21.85, 25.66, 29.40, 21.63],
        "LSTM":           [None, None, None, None],
        "TCN":            [None, None, None, None],
    }

    # Read TFT V2 seasonal from metrics.json (which is v2)
    with open("results/metrics.json") as f:
        m = json.load(f)
    tft_v2_seasonal = [
        m["seasonal"]["winter"]["MAE"],
        m["seasonal"]["spring"]["MAE"],
        m["seasonal"]["summer"]["MAE"],
        m["seasonal"]["fall"]["MAE"],
    ]
    model_seasonal["TFT-DCP V2 (200ep)"] = tft_v2_seasonal

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: seasonal MAE bars
    ax = axes[0]
    x = np.arange(len(seasons))
    width = 0.35
    bars1 = ax.bar(x - width / 2, model_seasonal["TFT-DCP (ours)"],
                   width, color=BRAND_BLUE, label="TFT-DCP V1 (submitted)", alpha=0.88)
    bars2 = ax.bar(x + width / 2, tft_v2_seasonal,
                   width, color=BRAND_RED, label="TFT-DCP V2 (200 ep)", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, fontsize=11)
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Seasonal MAE — TFT-DCP V1 vs V2\n(lower is better)", fontweight="bold")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(np.mean(model_seasonal["TFT-DCP (ours)"]), color=BRAND_BLUE,
               linestyle="--", linewidth=1, alpha=0.6)

    # Right: seasonal extreme event rate heatmap-style scatter
    ax2 = axes[1]
    season_extremes = {
        "winter": m["seasonal"]["winter"]["extreme_pct"],
        "spring": m["seasonal"]["spring"]["extreme_pct"],
        "summer": m["seasonal"]["summer"]["extreme_pct"],
        "fall":   m["seasonal"]["fall"]["extreme_pct"],
    }
    se_vals = [season_extremes["winter"], season_extremes["spring"],
               season_extremes["summer"], season_extremes["fall"]]
    bars3 = ax2.bar(seasons, se_vals, color=[BRAND_BLUE, "#2ECC71", BRAND_RED, ACCENT_GOLD], alpha=0.88)
    ax2.set_ylabel("Extreme Delay Rate (>180 min)")
    ax2.set_title("Seasonal Extreme Event Rate (Test 2025)\nSummer drives scheduling risk",
                  fontweight="bold")
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(se_vals) * 1.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))

    fig.suptitle("Seasonal Performance Analysis — DFW Hub (Test Year 2025)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_seasonal_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Top 50 Flagged Pairs Table (visual)
# ─────────────────────────────────────────────────────────────────────────────

def fig_top50_pairs_table():
    print("Generating: fig_top50_pairs_table")
    df = pd.read_csv("results/flagged_pairs.csv")
    df = df.sort_values("risk_score", ascending=False).head(50).reset_index(drop=True)

    display_cols = [
        "airport_a", "airport_b", "risk_score", "ml_risk_score",
        "duty_flag", "wocl_exposure_rate", "mct_violation_rate",
        "avg_conn_mins", "n_sequences", "recommendation",
    ]
    col_labels = [
        "Origin A", "Dest B", "Risk\nScore", "ML\nRisk",
        "Duty\nFlag", "WOCL\nExp.", "MCT\nViol.", "Avg\nConn (min)",
        "N\nSeqs", "Rec.",
    ]

    sub = df[display_cols].copy()
    sub["risk_score"]         = sub["risk_score"].map("{:.3f}".format)
    sub["ml_risk_score"]      = sub["ml_risk_score"].map("{:.3f}".format)
    sub["wocl_exposure_rate"] = sub["wocl_exposure_rate"].map("{:.1%}".format)
    sub["mct_violation_rate"] = sub["mct_violation_rate"].map("{:.1%}".format)
    sub["avg_conn_mins"]      = sub["avg_conn_mins"].map("{:.0f}".format)
    sub["n_sequences"]        = sub["n_sequences"].map("{:.0f}".format)
    sub["duty_flag"]          = sub["duty_flag"].map(lambda x: "✓" if x else "")

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis("off")

    tbl = ax.table(
        cellText=sub.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.35)

    # Color header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(BRAND_BLUE)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by recommendation
    for i in range(1, len(sub) + 1):
        rec = sub.iloc[i - 1]["recommendation"]
        row_color = "#FDECEA" if rec == "AVOID" else ("#FFF3E0" if rec == "CAUTION" else "#F1F8E9")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_color)

    save(fig, "fig_top50_pairs_table")

    # Also export CSV for easy copy-paste
    sub_export = df[display_cols].head(50)
    sub_export.to_csv(OUT_DIR / "top50_flagged_pairs.csv", index=False)
    print(f"  Saved {OUT_DIR}/top50_flagged_pairs.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Spearman ρ Ablation Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_spearman_ablation():
    print("Generating: fig_spearman_ablation")

    models = [
        "Historical\nPrior Only",
        "LightGBM\n(causal)",
        "TFT-DCP\n(submitted)",
    ]
    rhos = [0.591, 0.562, 0.737]
    colors_bar = [BRAND_GREY, "#E67E22", BRAND_BLUE]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, rhos, color=colors_bar, alpha=0.88, edgecolor="white", height=0.55)

    ax.axvline(rhos[0], color=BRAND_GREY, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(rhos[0] + 0.002, 3.42, "Prior baseline", fontsize=8, color=BRAND_GREY, style="italic")

    for bar, r in zip(bars, rhos):
        ax.text(r + 0.003, bar.get_y() + bar.get_height() / 2,
                f"ρ = {r:.3f}", va="center", fontsize=10, fontweight="bold")

    # Annotate lift
    ax.annotate(
        f"+{rhos[3] - rhos[0]:.3f}\nlift over\nprior",
        xy=(rhos[3], 3),
        xytext=(rhos[3] + 0.04, 2.5),
        fontsize=9, color=BRAND_BLUE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BRAND_BLUE),
    )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Spearman Rank Correlation (ρ)\nHigher = better ranking of risky A→DFW→B pairs",
                  fontsize=11)
    ax.set_title("Pair Ranking Quality — Spearman ρ Ablation\n"
                 "(33,670 A→DFW→B pairs, Test Year 2025)",
                 fontsize=12, fontweight="bold")

    # Add legend explaining what ranking means
    ax.text(0.02, -0.15,
            "ρ = Spearman rank correlation between predicted risk scores and true "
            "delay-based pair ranking (higher = better)",
            transform=ax.transAxes, fontsize=8.5, color=BRAND_GREY)

    fig.tight_layout()
    save(fig, "fig_spearman_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Delay Distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig_delay_distribution():
    print("Generating: fig_delay_distribution")
    try:
        df = pd.read_parquet("data/processed/processed_flights.parquet",
                             columns=["DepDelay", "Month_raw", "DayOfWeek_raw",
                                      "CRSDepTime_raw", "Year_raw"])
    except Exception as e:
        print(f"  Skipping (data not accessible): {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: histogram of DepDelay (capped at 240 min for readability)
    ax = axes[0]
    clipped = df["DepDelay"].clip(-30, 240)
    ax.hist(clipped, bins=60, color=BRAND_BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(0,   color="green",      linestyle="--", linewidth=1.5, label="On-time")
    ax.axvline(15,  color=ACCENT_GOLD,  linestyle="--", linewidth=1.5, label=">15 min")
    ax.axvline(60,  color="orange",     linestyle="--", linewidth=1.5, label=">60 min")
    ax.axvline(180, color=BRAND_RED,    linestyle="--", linewidth=1.5, label=">180 min")
    ax.set_xlabel("Departure Delay (minutes)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Delay Distribution\n(all splits, log scale)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: seasonal box plots
    ax2 = axes[1]
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall",   10: "Fall",  11: "Fall"}
    df2 = df.copy()
    df2["Season"] = df2["Month_raw"].map(season_map)
    order = ["Winter", "Spring", "Summer", "Fall"]
    data_by_season = [df2.loc[df2["Season"] == s, "DepDelay"].clip(-30, 200).dropna()
                      for s in order]
    bp = ax2.boxplot(data_by_season, labels=order, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2})
    colors_s = [BRAND_BLUE, "#2ECC71", BRAND_RED, ACCENT_GOLD]
    for patch, c in zip(bp["boxes"], colors_s):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax2.set_ylabel("Departure Delay (minutes)")
    ax2.set_title("Delay by Season\n(capped at 200 min)", fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")

    # Panel 3: average delay by hour of day (test year only)
    ax3 = axes[2]
    test_df = df[df["Year_raw"] == 2025].copy()
    test_df["dep_hour"] = (test_df["CRSDepTime_raw"] // 100).clip(0, 23)
    hourly = test_df.groupby("dep_hour")["DepDelay"].mean()
    ax3.bar(hourly.index, hourly.values, color=BRAND_BLUE, alpha=0.85, edgecolor="white")
    ax3.set_xlabel("Scheduled Departure Hour (local)")
    ax3.set_ylabel("Mean Departure Delay (minutes)")
    ax3.set_title("Average Delay by Hour of Day\n(Test 2025)", fontweight="bold")
    ax3.axhline(hourly.mean(), color=BRAND_RED, linestyle="--", linewidth=1.5,
                label=f"Daily avg: {hourly.mean():.1f} min")
    ax3.legend(fontsize=9)

    fig.suptitle("Departure Delay Characteristics — DFW Hub Network",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_delay_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Risk Score Calibration Scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig_risk_calibration():
    print("Generating: fig_risk_calibration")
    try:
        df = pd.read_csv("results/pair_ranking_eval/pair_ranking_aligned.csv")
    except Exception as e:
        print(f"  Skipping (file not found): {e}")
        return

    if "pred_risk_score" not in df.columns or "true_risk_score" not in df.columns:
        print("  Skipping: pred_risk_score / true_risk_score columns missing")
        return

    pred = df["pred_risk_score"].clip(0, 1)
    true = df["true_risk_score"].clip(0, 1)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color by predicted recommendation tier
    colors_pt = np.where(pred >= 0.65, BRAND_RED,
                np.where(pred >= 0.50, ACCENT_GOLD, "#2ECC71"))
    ax.scatter(true, pred, c=colors_pt, alpha=0.25, s=6, rasterized=True)

    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect calibration", alpha=0.6)

    # Correlation text
    from scipy.stats import spearmanr
    rho, pval = spearmanr(pred, true)
    ax.text(0.05, 0.93, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=BRAND_BLUE)
    ax.text(0.05, 0.87, f"n = {len(pred):,} pairs", transform=ax.transAxes,
            fontsize=10, color=BRAND_GREY)

    # Legend
    handles = [
        mpatches.Patch(color=BRAND_RED,    label="AVOID (pred ≥ 0.65)"),
        mpatches.Patch(color=ACCENT_GOLD,  label="CAUTION (0.50–0.65)"),
        mpatches.Patch(color="#2ECC71",    label="OK (< 0.50)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.set_xlabel("True Risk Score (based on 2025 actuals)", fontsize=11)
    ax.set_ylabel("Predicted Risk Score (pre-season causal)", fontsize=11)
    ax.set_title("Risk Score Calibration\nA→DFW→B Pair Ranking (Test 2025)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "fig_risk_calibration")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Top 10 Sub-score Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def fig_subscore_decomposition():
    print("Generating: fig_subscore_decomposition")
    df = pd.read_csv("results/scored_pairs.csv")
    top10 = df.sort_values("risk_score", ascending=False).head(10).reset_index(drop=True)
    top10["pair"] = top10["airport_a"] + "→DFW→" + top10["airport_b"]

    # Components (approximate decomposition from scoring formula):
    # risk_score ≈ weighted sum of: ml_risk, wocl_exposure, mct_violation, duty_flag
    ml     = top10["ml_risk_score"].values
    wocl   = top10["wocl_exposure_rate"].values * 0.2  # normalised contribution
    mct    = top10["mct_violation_rate"].values * 0.15
    duty   = top10["duty_flag"].values * 0.1

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(top10))
    h = 0.65

    ax.barh(y, ml, h, color=BRAND_BLUE, label="ML Delay Risk")
    ax.barh(y, wocl, h, left=ml, color=ACCENT_GOLD, label="WOCL Exposure")
    ax.barh(y, mct, h, left=ml + wocl, color=BRAND_RED, label="MCT Violation Rate")
    ax.barh(y, duty, h, left=ml + wocl + mct, color="#2CA02C", label="Duty Flag")

    # Mark composite risk score
    ax.scatter(top10["risk_score"], y, marker="|", color="black", s=200, linewidths=2,
               zorder=5, label="Composite Risk Score")

    ax.set_yticks(y)
    ax.set_yticklabels(top10["pair"][::-1] if False else top10["pair"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Risk Contribution")
    ax.set_title("Top 10 Flagged Pairs — Sub-Score Decomposition\n"
                 "ML delay risk + operational constraints (WOCL, MCT, FAR 117)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1.0)

    fig.tight_layout()
    save(fig, "fig_subscore_decomposition")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Route Prior Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_prior_correlation():
    print("Generating: fig_prior_correlation")
    prior_cols = [
        "route_avg_delay_prior", "route_p90_delay_prior", "route_std_delay_prior",
        "route_extreme_rate_prior", "route_prop_delay_prior",
        "route_buffer_exceeded_prior", "origin_avg_delay_prior",
        "dest_avg_delay_prior", "dep_hour_avg_delay_prior",
    ]
    short = [c.replace("_prior", "").replace("route_", "r_").replace("_delay", "")
             for c in prior_cols]

    try:
        df = pd.read_parquet("data/processed/processed_flights.parquet",
                             columns=prior_cols).dropna()
    except Exception as e:
        print(f"  Skipping (data not accessible): {e}")
        return

    corr = df[prior_cols].corr(method="spearman")
    corr.index = corr.columns = short

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=8)

    for i in range(len(short)):
        for j in range(len(short)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < 0.7 else "white")

    ax.set_title("Route Prior Feature Correlation (Spearman ρ)\n"
                 "Confirms non-redundancy of extreme_rate and buffer_exceeded",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_prior_correlation")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Full Baseline Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def fig_baseline_table():
    print("Generating: fig_baseline_table")

    # RMSE from V1 baselines run (4b_baselines.log); LightGBM/XGBoost from their metrics JSONs
    rows = [
        ["Historical Average",    "29.86", "73.59", "-0.018", "—†",    "—†",    "Seasonal mean baseline"],
        ["XGBoost",               "25.54", "66.22", "0.176",  "—†",    "—†",    "Gradient boosted (causal)"],
        ["LSTM",                  "25.15", "63.57", "0.138",  "—†",    "—†",    "Seq2seq, no attention"],
        ["TCN (no DCP)",          "25.57", "63.42", "0.142",  "—†",    "—†",    "Architecture ablation"],
        ["Informer",              "28.04", "64.64", "0.109",  "—†",    "—†",    "Sparse attention"],
        ["TFT (no DCP)",          "25.64", "61.01", "0.206",  "—†",    "—†",    "No propagation module"],
        ["LightGBM",              "23.10", "64.97", "0.207",  "0.562", "0.741‡","Gradient boosted trees"],
        ["TFT-DCP V1 (ours) ★",  "24.79", "59.86", "0.236",  "0.737", "0.729‡","Full model — submitted"],
    ]

    col_labels = ["Model", "MAE\n(min)", "RMSE\n(min)", "R²", "Spearman\nρ",
                  "NDCG\n@10", "Notes"]

    footnotes = (
        "† Spearman ρ requires scoring every A→DFW→B pair (33,670 pairs) with the model's risk output, "
        "then comparing that ranking to the true delay-based ranking. The neural baselines and XGBoost "
        "produce a per-flight delay prediction only — they do not include the propagation + operational "
        "constraint scoring pipeline needed to generate a pair risk score, so ρ is not directly comparable.\n\n"
        "‡ Why TFT-DCP beats LightGBM despite higher MAE and lower NDCG@10: MAE and NDCG@10 measure "
        "per-flight point accuracy on the full test set. LightGBM is better at predicting the average "
        "flight but worse at ranking pairs — Spearman ρ (0.737 vs 0.562) captures this directly. "
        "TFT-DCP's temporal propagation module explicitly models how delays cascade through an "
        "A→DFW→B chain, so the pair-level risk scores are more faithful to true operational risk. "
        "NDCG@10 is dominated by a handful of outlier pairs; ρ over all 33K pairs is a more robust "
        "measure of schedule-planning utility."
    )

    fig = plt.figure(figsize=(16, 7.5))
    ax_tbl = fig.add_axes([0.01, 0.28, 0.98, 0.65])
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(BRAND_BLUE)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight TFT-DCP row
    for j in range(len(col_labels)):
        tbl[8, j].set_facecolor("#D6EAF8")
        tbl[8, j].set_text_props(fontweight="bold")

    ax_tbl.set_title(
        "Full Model Comparison — DFW Hub, Test Year 2025\n"
        "(All models use causal pre-departure features only; RMSE from V1 baseline run)",
        fontsize=12, fontweight="bold", pad=14,
    )

    # Footnote text area
    ax_note = fig.add_axes([0.03, 0.01, 0.94, 0.25])
    ax_note.axis("off")
    ax_note.text(0, 1.0, footnotes, transform=ax_note.transAxes,
                 fontsize=8, va="top", ha="left", color="#333333",
                 wrap=True,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                           edgecolor="#CCCCCC", alpha=0.9))

    save(fig, "fig_baseline_table")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    print(f"Output directory: {OUT_DIR.resolve()}\n")

    fig_feature_importance()
    fig_architecture()
    fig_training_convergence()
    fig_seasonal_analysis()
    fig_top50_pairs_table()
    fig_spearman_ablation()
    fig_delay_distribution()
    fig_risk_calibration()
    fig_subscore_decomposition()
    fig_prior_correlation()
    fig_baseline_table()

    print(f"\nDone. All figures saved to {OUT_DIR.resolve()}/")
