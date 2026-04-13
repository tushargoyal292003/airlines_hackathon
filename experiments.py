"""
Ablation Study & Benchmark Comparison Runner
Paper Section 4.3.1 (Table 2) and Section 4.3.2 (Table 4).

Runs all baselines and ablation variants, collects metrics, exports to CSV.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from config import Config
from model import TFTDCP
from model.tcn import TCNEncoder
from model.grn import GatedResidualNetwork
from model.historical_retrieval import HistoricalRetrievalModule
from model.ms_ca_efm import MSCAEFM
from model.propagation import DelayPropagationModule
from baselines import (
    HistoricalAverage, LSTMBaseline, TCNBaseline,
    InformerLite, TFTBaseline, GBMBaseline,
    compute_metrics, train_nn_baseline,
)


# ──────────────────────────────────────────────────────────────
# Ablation variants (Paper Table 4)
# ──────────────────────────────────────────────────────────────

class AblationVariant1(nn.Module):
    """Exp 1: Dynamic Embedding only (no retrieval, no MS-CA-EF, no chain)."""

    def __init__(self, num_dynamic, num_static, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.tcn = TCNEncoder(num_dynamic, [64, 128, hidden_dim], dropout=dropout)
        self.grn = GatedResidualNetwork(num_static, 64, hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dynamic, static, mask=None, **kwargs):
        _, h_global = self.tcn(dynamic, mask)
        h_static = self.grn(static)
        h = torch.cat([h_global, h_static], dim=-1)
        return {"prediction": self.head(h).squeeze(-1)}


class AblationVariant2(nn.Module):
    """Exp 2: Dynamic Embedding + Historical Retrieval (no MS-CA-EF, no chain)."""

    def __init__(self, num_dynamic, num_static, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.tcn = TCNEncoder(num_dynamic, [64, 128, hidden_dim], dropout=dropout)
        self.grn = GatedResidualNetwork(num_static, 64, hidden_dim, dropout=dropout)
        self.retrieval = HistoricalRetrievalModule(hidden_dim, db_size=10000, top_k=5)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dynamic, static, mask=None, **kwargs):
        h_dynamic, h_global = self.tcn(dynamic, mask)
        h_static = self.grn(static)

        if mask is not None:
            lengths = mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(h_dynamic.size(0), device=h_dynamic.device)
            h_current = h_dynamic[batch_idx, lengths]
        else:
            h_current = h_dynamic[:, -1, :]

        h_f = self.retrieval(h_current)
        h = torch.cat([h_f + h_static, h_global], dim=-1)
        return {"prediction": self.head(h).squeeze(-1), "h_current": h_current}


class AblationVariant3(nn.Module):
    """Exp 3: Dynamic + Retrieval + MS-CA-EFM (no chain propagation)."""

    def __init__(self, num_dynamic, num_static, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.tcn = TCNEncoder(num_dynamic, [64, 128, hidden_dim], dropout=dropout)
        self.grn = GatedResidualNetwork(num_static, 64, hidden_dim, dropout=dropout)
        self.retrieval = HistoricalRetrievalModule(hidden_dim, db_size=10000, top_k=5)
        self.ms_ca_efm = MSCAEFM(hidden_dim, reduction_ratio=4)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, dynamic, static, mask=None, **kwargs):
        h_dynamic, h_global = self.tcn(dynamic, mask)
        h_static = self.grn(static)

        if mask is not None:
            lengths = mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(h_dynamic.size(0), device=h_dynamic.device)
            h_current = h_dynamic[batch_idx, lengths]
        else:
            h_current = h_dynamic[:, -1, :]

        h_current = h_current + h_static
        h_global = h_global + h_static
        h_f = self.retrieval(h_current)
        h_fused = self.ms_ca_efm(h_current, h_f, h_global)
        return {"prediction": self.head(h_fused).squeeze(-1), "h_current": h_current}


def save_checkpoint(model, metrics, name, ckpt_dir="checkpoints/baselines"):
    """Save model checkpoint and metrics to disk."""
    import json
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{ckpt_dir}/{name}.pt")
    with open(f"{ckpt_dir}/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved checkpoint → {ckpt_dir}/{name}.pt")
    print(f"  Saved metrics    → {ckpt_dir}/{name}_metrics.json")


def run_benchmark_comparison(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_dynamic: int,
    num_static: int,
    feature_cols: list,
    config: Config,
) -> pd.DataFrame:
    """
    Run all baselines and TFT-DCP, return Table 2 results.
    """
    import json, joblib
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    ckpt_dir = "checkpoints/baselines"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON (Paper Table 2)")
    print("=" * 70)

    # 1. Historical Average
    print("\n--- Historical Average ---")
    ha = HistoricalAverage()
    ha.fit(train_df)
    ha_preds = ha.predict(test_df)
    ha_targets = test_df["DepDelay"].values
    ha_metrics = compute_metrics(ha_targets, ha_preds)
    results.append({"Model": "HA", **ha_metrics})
    print(f"  {ha_metrics}")
    joblib.dump(ha, f"{ckpt_dir}/ha.joblib")
    with open(f"{ckpt_dir}/ha_metrics.json", "w") as f:
        json.dump(ha_metrics, f, indent=2)
    print(f"  Saved checkpoint → {ckpt_dir}/ha.joblib")

    # 2. LSTM
    print("\n--- LSTM ---")
    lstm = LSTMBaseline(num_dynamic, hidden_dim=128, num_layers=2)
    lstm, lstm_metrics = train_nn_baseline(
        lstm, train_loader, test_loader, epochs=50, device=device,
    )
    results.append({"Model": "LSTM", **lstm_metrics})
    print(f"  {lstm_metrics}")
    save_checkpoint(lstm, lstm_metrics, "lstm")

    # 3. TCN
    print("\n--- TCN ---")
    tcn = TCNBaseline(num_dynamic, hidden_dim=128)
    tcn, tcn_metrics = train_nn_baseline(
        tcn, train_loader, test_loader, epochs=50, device=device,
    )
    results.append({"Model": "TCN", **tcn_metrics})
    print(f"  {tcn_metrics}")
    save_checkpoint(tcn, tcn_metrics, "tcn")

    # 4. Informer
    print("\n--- Informer ---")
    informer = InformerLite(num_dynamic, hidden_dim=128)
    informer, inf_metrics = train_nn_baseline(
        informer, train_loader, test_loader, epochs=50, device=device,
    )
    results.append({"Model": "Informer", **inf_metrics})
    print(f"  {inf_metrics}")
    save_checkpoint(informer, inf_metrics, "informer")

    # 5. TFT (without DCP)
    print("\n--- TFT (baseline) ---")
    tft = TFTBaseline(num_dynamic, num_static, hidden_dim=128)
    tft, tft_metrics = train_nn_baseline(
        tft, train_loader, test_loader, epochs=50, device=device,
    )
    results.append({"Model": "TFT", **tft_metrics})
    print(f"  {tft_metrics}")
    save_checkpoint(tft, tft_metrics, "tft_baseline")

    # 6. XGBoost
    print("\n--- XGBoost ---")
    try:
        xgb = GBMBaseline("xgboost")
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["DepDelay"].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["DepDelay"].values
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_metrics = compute_metrics(y_test, xgb_preds)
        results.append({"Model": "XGBoost", **xgb_metrics})
        print(f"  {xgb_metrics}")
        joblib.dump(xgb.model, f"{ckpt_dir}/xgboost.joblib")
        with open(f"{ckpt_dir}/xgboost_metrics.json", "w") as f:
            json.dump(xgb_metrics, f, indent=2)
        print(f"  Saved checkpoint → {ckpt_dir}/xgboost.joblib")

        # Save feature importance for report
        importance = xgb.get_feature_importance(feature_cols)
        importance.to_csv("results/xgboost_feature_importance.csv", index=False)
        print(f"  Feature importance saved")
    except ImportError:
        print("  XGBoost not installed, skipping")

    # 7. LightGBM
    print("\n--- LightGBM ---")
    try:
        lgbm = GBMBaseline("lightgbm")
        lgbm.fit(X_train, y_train)
        lgbm_preds = lgbm.predict(X_test)
        lgbm_metrics = compute_metrics(y_test, lgbm_preds)
        results.append({"Model": "LightGBM", **lgbm_metrics})
        print(f"  {lgbm_metrics}")
        joblib.dump(lgbm.model, f"{ckpt_dir}/lightgbm.joblib")
        with open(f"{ckpt_dir}/lightgbm_metrics.json", "w") as f:
            json.dump(lgbm_metrics, f, indent=2)
        print(f"  Saved checkpoint → {ckpt_dir}/lightgbm.joblib")
    except (ImportError, NameError):
        print("  LightGBM not installed, skipping")

    # TFT-DCP results loaded from checkpoint
    print("\n--- TFT-DCP (ours) ---")
    ckpt_path = Path(config.train.checkpoint_dir) / "best_model.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = TFTDCP(
            num_dynamic_features=num_dynamic,
            num_static_features=num_static,
            hidden_dim=config.model.hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    batch["dynamic"].to(device), batch["static"].to(device),
                    batch["chain_delays"].to(device), batch["turnaround_times"].to(device),
                    batch["mask"].to(device),
                )
                preds.append(out["prediction"].cpu().numpy())
                targets.append(batch["target"].numpy())

        dcp_metrics = compute_metrics(np.concatenate(targets), np.concatenate(preds))
        results.append({"Model": "TFT-DCP", **dcp_metrics})
        print(f"  {dcp_metrics}")
    else:
        print("  No checkpoint found — train TFT-DCP first")

    # Summary table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(df_results.to_string(index=False))

    df_results.to_csv("results/benchmark_comparison.csv", index=False)
    return df_results


def run_ablation_study(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_dynamic: int,
    num_static: int,
    config: Config,
) -> pd.DataFrame:
    """
    Run ablation experiments (Paper Table 4).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    print("\n" + "=" * 70)
    print("ABLATION STUDY (Paper Table 4)")
    print("=" * 70)

    # Exp 1: Dynamic Embedding only
    print("\n--- Exp 1: Dynamic Embedding only ---")
    v1 = AblationVariant1(num_dynamic, num_static)
    v1, m1 = train_nn_baseline(v1, train_loader, test_loader, epochs=50, device=device)
    results.append({
        "Experiment": 1, "Dynamic": "✓", "Retrieval": "✗",
        "MS-CA-EF": "✗", "Chain": "✗", **m1,
    })
    # Exp 2: + Historical Retrieval
    print("\n--- Exp 2: + Historical Retrieval ---")
    v2 = AblationVariant2(num_dynamic, num_static)
    v2, m2 = train_nn_baseline(v2, train_loader, test_loader, epochs=50, device=device)
    results.append({
        "Experiment": 2, "Dynamic": "✓", "Retrieval": "✓",
        "MS-CA-EF": "✗", "Chain": "✗", **m2,
    })

    # Exp 3: + MS-CA-EFM
    print("\n--- Exp 3: + MS-CA-EFM ---")
    v3 = AblationVariant3(num_dynamic, num_static)
    v3, m3 = train_nn_baseline(v3, train_loader, test_loader, epochs=50, device=device)
    results.append({
        "Experiment": 3, "Dynamic": "✓", "Retrieval": "✓",
        "MS-CA-EF": "✓", "Chain": "✗", **m3,
    })

    # Exp 4: Full TFT-DCP (loaded from checkpoint)
    ckpt_path = Path(config.train.checkpoint_dir) / "best_model.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = TFTDCP(
            num_dynamic_features=num_dynamic,
            num_static_features=num_static,
            hidden_dim=config.model.hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    batch["dynamic"].to(device), batch["static"].to(device),
                    batch["chain_delays"].to(device), batch["turnaround_times"].to(device),
                    batch["mask"].to(device),
                )
                preds.append(out["prediction"].cpu().numpy())
                targets.append(batch["target"].numpy())
        m4 = compute_metrics(np.concatenate(targets), np.concatenate(preds))
        results.append({
            "Experiment": 4, "Dynamic": "✓", "Retrieval": "✓",
            "MS-CA-EF": "✓", "Chain": "✓", **m4,
        })

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("ABLATION RESULTS")
    print("=" * 50)
    print(df_results.to_string(index=False))

    df_results.to_csv("results/ablation_study.csv", index=False)
    return df_results

