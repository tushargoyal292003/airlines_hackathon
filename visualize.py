"""
Visualization for hackathon report.
Generates charts matching the paper's figures + hackathon-specific visuals.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "tft_dcp": "#2ecc71", "tft": "#3498db", "tcn": "#e74c3c",
    "lstm": "#f39c12", "informer": "#9b59b6", "ha": "#95a5a6",
    "xgboost": "#1abc9c", "lightgbm": "#e67e22",
}

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history_path: str = "logs/training_history.json"):
    """Plot training loss, MAE, RMSE, R², beta convergence."""
    with open(history_path) as f:
        h = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Train vs Val loss
    axes[0, 0].plot(h["train_loss"], label="Train", color="#3498db")
    axes[0, 0].plot(h["val_loss"], label="Validation", color="#e74c3c")
    axes[0, 0].set_title("Loss Curves", fontsize=13)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()

    # MAE
    axes[0, 1].plot(h["val_mae"], color="#2ecc71", linewidth=2)
    axes[0, 1].set_title("Validation MAE (min)", fontsize=13)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].axhline(y=min(h["val_mae"]), color="gray", linestyle="--", alpha=0.5)

    # RMSE
    axes[0, 2].plot(h["val_rmse"], color="#e67e22", linewidth=2)
    axes[0, 2].set_title("Validation RMSE (min)", fontsize=13)
    axes[0, 2].set_xlabel("Epoch")

    # R²
    axes[1, 0].plot(h["val_r2"], color="#9b59b6", linewidth=2)
    axes[1, 0].set_title("Validation R²", fontsize=13)
    axes[1, 0].set_xlabel("Epoch")

    # Beta convergence (paper: converges to 0.73-0.89)
    axes[1, 1].plot(h["beta"], color="#e74c3c", linewidth=2)
    axes[1, 1].axhspan(0.73, 0.89, alpha=0.15, color="green", label="Paper range")
    axes[1, 1].set_title("β Convergence (Decay Coefficient)", fontsize=13)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    # Learning rate
    axes[1, 2].plot(h["lr"], color="#34495e", linewidth=2)
    axes[1, 2].set_title("Learning Rate Schedule", fontsize=13)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training_curves.png")


def plot_benchmark_comparison(csv_path: str = "results/benchmark_comparison.csv"):
    """Bar chart comparing all models (Paper Table 2 / Figure 6 style)."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = df["Model"].tolist()
    colors = [COLORS.get(m.lower().replace("-", "_"), "#888") for m in models]

    # MAE
    bars = axes[0].bar(models, df["MAE"], color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("MAE (minutes) ↓", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("MAE")
    for bar, val in zip(bars, df["MAE"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # RMSE
    bars = axes[1].bar(models, df["RMSE"], color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("RMSE (minutes) ↓", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("RMSE")
    for bar, val in zip(bars, df["RMSE"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # R²
    bars = axes[2].bar(models, df["R2"], color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_title("R² ↑", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1)
    for bar, val in zip(bars, df["R2"]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved benchmark_comparison.png")


def plot_ablation_study(csv_path: str = "results/ablation_study.csv"):
    """Ablation study grouped bar chart (Paper Table 4)."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["MAE"], width, label="MAE", color="#3498db")
    ax.bar(x, df["RMSE"], width, label="RMSE", color="#e74c3c")
    ax.bar(x + width, df["R2"] * 10, width, label="R² (×10)", color="#2ecc71")

    labels = []
    for _, row in df.iterrows():
        modules = []
        if row.get("Dynamic") == "✓": modules.append("D")
        if row.get("Retrieval") == "✓": modules.append("R")
        if row.get("MS-CA-EF") == "✓": modules.append("M")
        if row.get("Chain") == "✓": modules.append("C")
        labels.append(f"Exp {int(row['Experiment'])}\n({'+'.join(modules)})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold")
    ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation_study.png")


def plot_pair_risk_heatmap(csv_path: str = "results/pair_risk_scores.csv", top_n: int = 30):
    """Risk heatmap for top airport pairs (hackathon key visual)."""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("  No pair data to plot")
        return

    # Get top N airports by frequency
    top_a = df["airport_a"].value_counts().head(top_n).index
    top_b = df["airport_b"].value_counts().head(top_n).index
    subset = df[df["airport_a"].isin(top_a) & df["airport_b"].isin(top_b)]

    pivot = subset.pivot_table(
        values="risk_score", index="airport_a", columns="airport_b",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Risk Score"},
    )
    ax.set_title("Airport Pair Risk Scores (A → DFW → B)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Destination Airport (B)")
    ax.set_ylabel("Origin Airport (A)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pair_risk_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved pair_risk_heatmap.png")


def plot_risk_decomposition(csv_path: str = "results/pair_risk_scores.csv", top_n: int = 15):
    """Stacked bar showing risk factor breakdown for top risky pairs."""
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        return

    top = df.nlargest(top_n, "risk_score")
    top["pair_label"] = top["airport_a"] + "→" + top["airport_b"]

    risk_components = ["delay_risk", "propagation_risk", "weather_risk", "turnaround_risk"]
    available = [c for c in risk_components if c in top.columns]

    if not available:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(top))
    component_colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6"]

    for i, comp in enumerate(available):
        ax.barh(top["pair_label"], top[comp], left=bottom,
                label=comp.replace("_", " ").title(),
                color=component_colors[i % len(component_colors)])
        bottom += top[comp].values

    ax.set_xlabel("Risk Score Contribution")
    ax.set_title("Risk Factor Decomposition — Top Risky Pairs", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved risk_decomposition.png")


def plot_feature_importance(csv_path: str = "results/xgboost_feature_importance.csv", top_n: int = 20):
    """Feature importance from XGBoost (for report SHAP-like analysis)."""
    try:
        df = pd.read_csv(csv_path).head(top_n)
    except FileNotFoundError:
        print("  No feature importance file found")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#2ecc71")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Features (XGBoost Baseline)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature_importance.png")


def generate_all_figures():
    """Generate all figures for the report."""
    print("\n" + "=" * 70)
    print("GENERATING REPORT FIGURES")
    print("=" * 70)

    if Path("logs/training_history.json").exists():
        plot_training_curves()

    if Path("results/benchmark_comparison.csv").exists():
        plot_benchmark_comparison()

    if Path("results/ablation_study.csv").exists():
        plot_ablation_study()

    if Path("results/pair_risk_scores.csv").exists():
        plot_pair_risk_heatmap()
        plot_risk_decomposition()

    if Path("results/xgboost_feature_importance.csv").exists():
        plot_feature_importance()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
