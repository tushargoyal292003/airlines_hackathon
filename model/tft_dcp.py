"""
TFT-DCP: Temporal Fusion Transformer with Dynamic Chain Propagation
Full model integrating:
  - TCN Encoder (dynamic features)
  - GRN (static features)
  - Historical Retrieval Module
  - MS-CA-EFM (multi-source fusion)
  - Flight Chain Delay Propagation Module
"""
import torch
import torch.nn as nn

from .tcn import TCNEncoder
from .grn import GatedResidualNetwork
from .historical_retrieval import HistoricalRetrievalModule
from .ms_ca_efm import MSCAEFM
from .propagation import DelayPropagationModule


class TFTDCP(nn.Module):
    """
    Complete TFT-DCP architecture.
    
    Input: static features, dynamic features, chain delays, turnaround times
    Output: predicted departure delay (minutes)
    """

    def __init__(
        self,
        num_dynamic_features: int,
        num_static_features: int,
        hidden_dim: int = 128,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        grn_hidden_dim: int = 64,
        top_k_retrieval: int = 5,
        retrieval_alpha: float = 0.5,
        history_db_size: int = 50000,
        channel_reduction_ratio: int = 4,
        beta_init: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if tcn_channels is None:
            tcn_channels = [64, 128, hidden_dim]

        # --- Module 1: Dynamic Embedding (TCN) ---
        self.tcn_encoder = TCNEncoder(
            input_dim=num_dynamic_features,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        tcn_out_dim = tcn_channels[-1]

        # Project TCN output to hidden_dim if needed
        self.dynamic_proj = (
            nn.Linear(tcn_out_dim, hidden_dim)
            if tcn_out_dim != hidden_dim else nn.Identity()
        )

        # --- Module 2: Static Embedding (GRN) ---
        self.grn_static = GatedResidualNetwork(
            input_dim=num_static_features,
            hidden_dim=grn_hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # --- Module 3: Historical Retrieval ---
        self.historical_retrieval = HistoricalRetrievalModule(
            embedding_dim=hidden_dim,
            db_size=history_db_size,
            top_k=top_k_retrieval,
            alpha=retrieval_alpha,
        )

        # --- Module 4: Flight Chain Delay Propagation ---
        self.delay_propagation = DelayPropagationModule(
            beta_init=beta_init,
            hidden_dim=hidden_dim,
        )

        # --- Module 5: MS-CA-EFM ---
        self.ms_ca_efm = MSCAEFM(
            embedding_dim=hidden_dim,
            reduction_ratio=channel_reduction_ratio,
        )

        # --- Prediction Head ---
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concat with propagation
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Variable selection for static context
        self.static_context_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=grn_hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        chain_delays: torch.Tensor,
        turnaround_times: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            dynamic: (batch, seq_len, num_dynamic) — dynamic features
            static: (batch, num_static) — static features
            chain_delays: (batch, seq_len) — preceding delays
            turnaround_times: (batch, seq_len) — turnaround intervals
            mask: (batch, seq_len) — valid position mask

        Returns:
            dict with 'prediction', 'h_current', 'h_global', 'h_f', 'y_prop', 'beta'
        """
        # 1. Dynamic feature embedding via TCN (Eq. 5-6)
        h_dynamic, h_global = self.tcn_encoder(dynamic, mask)
        h_global = self.dynamic_proj(h_global)  # (B, hidden_dim)

        # Current flight embedding: last valid position in sequence
        if mask is not None:
            # Get last valid index per batch
            lengths = mask.sum(dim=1).long() - 1  # (B,)
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(h_dynamic.size(0), device=h_dynamic.device)
            h_current = h_dynamic[batch_idx, lengths]  # (B, tcn_out_dim)
        else:
            h_current = h_dynamic[:, -1, :]

        h_current = self.dynamic_proj(h_current)  # (B, hidden_dim)

        # 2. Static feature embedding via GRN (Eq. 7-10)
        h_static = self.grn_static(static)  # (B, hidden_dim)

        # Static context enrichment
        h_current = h_current + self.static_context_grn(h_static)
        h_global = h_global + h_static

        # 3. Historical retrieval (Eq. 11-15)
        h_f = self.historical_retrieval(h_current)  # (B, hidden_dim)

        # 4. MS-CA-EFM fusion (Eq. 16-22)
        h_fused = self.ms_ca_efm(h_current, h_f, h_global)  # (B, hidden_dim)

        # 5. Flight chain delay propagation (Eq. 23-24)
        y_prop, h_prop = self.delay_propagation(
            chain_delays, turnaround_times, mask
        )

        # 6. Concatenate fused embedding with propagation embedding
        h_final = torch.cat([h_fused, h_prop], dim=-1)  # (B, hidden_dim*2)

        # 7. Prediction
        prediction = self.prediction_head(h_final).squeeze(-1)  # (B,)

        return {
            "prediction": prediction,
            "h_current": h_current,
            "h_global": h_global,
            "h_f": h_f,
            "y_prop": y_prop.squeeze(-1),
            "beta": nn.functional.softplus(self.delay_propagation.beta),
        }

    @torch.no_grad()
    def update_history(self, h_current: torch.Tensor):
        """Update the historical database after predictions."""
        self.historical_retrieval.update_database(h_current)


class TFTDCPWithBaselines(nn.Module):
    """
    Wrapper that also provides baseline comparison outputs.
    Useful for the hackathon report — run all models in one forward pass.
    """

    def __init__(self, tft_dcp: TFTDCP, num_dynamic: int, hidden_dim: int):
        super().__init__()
        self.tft_dcp = tft_dcp

        # Simple LSTM baseline for comparison
        self.lstm_baseline = nn.LSTM(
            input_size=num_dynamic,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.lstm_head = nn.Linear(hidden_dim, 1)

    def forward(self, dynamic, static, chain_delays, turnaround_times, mask=None):
        # TFT-DCP prediction
        tft_out = self.tft_dcp(dynamic, static, chain_delays, turnaround_times, mask)

        # LSTM baseline prediction
        lstm_out, _ = self.lstm_baseline(dynamic)
        lstm_pred = self.lstm_head(lstm_out[:, -1, :]).squeeze(-1)

        tft_out["lstm_prediction"] = lstm_pred
        return tft_out
