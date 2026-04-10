"""
Baseline Models for Comparison (Paper Section 4.3.1, Table 2)
Models: HA, LSTM, TCN, Informer-lite, XGBoost, LightGBM

These are needed for the 20-page hackathon report to show TFT-DCP's superiority.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ──────────────────────────────────────────────────────────────
# 1. Historical Average (HA)
# ──────────────────────────────────────────────────────────────

class HistoricalAverage:
    """Forecasts by averaging historical delays from identical time windows."""

    def __init__(self):
        self.averages = {}

    def fit(self, df: pd.DataFrame):
        """Compute averages by (origin, dest, day_of_week, hour)."""
        df = df.copy()
        df["dep_hour"] = pd.to_numeric(
            df["CRSDepTime"].astype(str).str.zfill(4).str[:2], errors="coerce"
        )
        self.averages = (
            df.groupby(["Origin", "Dest", "DayOfWeek", "dep_hour"])["DepDelay"]
            .mean()
            .to_dict()
        )
        self.global_avg = df["DepDelay"].mean()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["dep_hour"] = pd.to_numeric(
            df["CRSDepTime"].astype(str).str.zfill(4).str[:2], errors="coerce"
        )
        preds = []
        for _, row in df.iterrows():
            key = (row["Origin"], row["Dest"], row["DayOfWeek"], row["dep_hour"])
            preds.append(self.averages.get(key, self.global_avg))
        return np.array(preds)


# ──────────────────────────────────────────────────────────────
# 2. LSTM Baseline
# ──────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """Standard LSTM for sequence-based delay prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, **kwargs):
        out, _ = self.lstm(x)
        return {"prediction": self.head(out[:, -1, :]).squeeze(-1)}


# ──────────────────────────────────────────────────────────────
# 3. TCN Baseline (without DCP extensions)
# ──────────────────────────────────────────────────────────────

class TCNBaseline(nn.Module):
    """Plain TCN without propagation or retrieval modules."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        from model.tcn import TCNEncoder
        self.encoder = TCNEncoder(
            input_dim=input_dim,
            num_channels=[64, 128, hidden_dim],
            kernel_size=3,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None, **kwargs):
        _, h_global = self.encoder(x, mask)
        return {"prediction": self.head(h_global).squeeze(-1)}


# ──────────────────────────────────────────────────────────────
# 4. Informer-lite (simplified ProbSparse attention)
# ──────────────────────────────────────────────────────────────

class InformerLite(nn.Module):
    """
    Simplified Informer-style model with standard multi-head attention.
    Full ProbSparse attention is complex — this captures the Transformer baseline.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None, **kwargs):
        x = self.input_proj(x)
        if mask is not None:
            # Convert padding mask: 1=valid -> False=not masked
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return {"prediction": self.head(out[:, -1, :]).squeeze(-1)}


# ──────────────────────────────────────────────────────────────
# 5. XGBoost / LightGBM (tabular baselines)
# ──────────────────────────────────────────────────────────────

class GBMBaseline:
    """Gradient boosting baseline (XGBoost or LightGBM)."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model_type == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, n_jobs=-1, random_state=42,
            )
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
            )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)


# ──────────────────────────────────────────────────────────────
# 6. TFT Baseline (without DCP — ablation Experiment 1)
# ──────────────────────────────────────────────────────────────

class TFTBaseline(nn.Module):
    """
    Original TFT without Dynamic Chain Propagation.
    Uses TCN encoding + GRN static + basic attention fusion.
    This is effectively ablation Experiment 1 from Table 4.
    """

    def __init__(self, num_dynamic: int, num_static: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        from model.tcn import TCNEncoder
        from model.grn import GatedResidualNetwork

        self.tcn = TCNEncoder(num_dynamic, [64, 128, hidden_dim], dropout=dropout)
        self.grn = GatedResidualNetwork(num_static, 64, hidden_dim, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, dynamic, static, mask=None, **kwargs):
        _, h_global = self.tcn(dynamic, mask)
        h_static = self.grn(static)
        h = self.fusion(torch.cat([h_global, h_static], dim=-1))
        return {"prediction": self.head(h).squeeze(-1)}


# ──────────────────────────────────────────────────────────────
# Unified evaluation
# ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Paper Eq. 26-28: MAE, RMSE, R²."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def train_nn_baseline(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    """Generic training loop for PyTorch baselines."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        n = 0
        for batch in train_loader:
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            out = model(dynamic, static=static, mask=mask)
            loss = criterion(out["prediction"], target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                dynamic = batch["dynamic"].to(device)
                static = batch["static"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)
                out = model(dynamic, static=static, mask=mask)
                preds.append(out["prediction"].cpu().numpy())
                targets.append(target.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        val_loss = mean_squared_error(targets, preds)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            metrics = compute_metrics(targets, preds)
            print(f"  Epoch {epoch:3d} | MAE: {metrics['MAE']:.2f} | "
                  f"RMSE: {metrics['RMSE']:.2f} | R²: {metrics['R2']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            out = model(dynamic, static=static, mask=mask)
            preds.append(out["prediction"].cpu().numpy())
            targets.append(target.cpu().numpy())

    final_metrics = compute_metrics(np.concatenate(targets), np.concatenate(preds))
    return model, final_metrics
