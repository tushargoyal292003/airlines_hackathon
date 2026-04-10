"""
Flight Chain Delay Propagation Module
Paper Section 3.2.4, Equations 23-25.

Models "preceding delay × exponential time decay" as a learnable feature.
β is initialized to 1.0 and converges to 0.73–0.89 per the paper.
"""
import torch
import torch.nn as nn


class DelayPropagationModule(nn.Module):
    """
    Computes propagated delay feature using exponential time decay.
    
    σ_i = exp(-β·Δt_i) / Σ exp(-β·Δt_j)  (Eq. 23)
    y_prop = Σ σ_i · y_prev,i  (Eq. 24)
    
    β is learnable via backpropagation (Eq. 25):
      ∂L/∂β = (∂L/∂y_prop) × (∂y_prop/∂σ) × (∂σ/∂β)
    """

    def __init__(self, beta_init: float = 1.0, hidden_dim: int = 128):
        super().__init__()

        # Learnable decay coefficient (paper: converges to 0.73-0.89)
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # Project propagated delay into embedding space
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        chain_delays: torch.Tensor,
        turnaround_times: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            chain_delays: (batch, seq_len) — preceding flight delays (minutes)
            turnaround_times: (batch, seq_len) — Δt between consecutive flights (minutes)
            mask: (batch, seq_len) — valid positions

        Returns:
            y_prop: (batch, 1) — propagated delay value
            h_prop: (batch, hidden_dim) — propagated delay embedding
        """
        # Normalize turnaround times to hours for numerical stability
        delta_t = turnaround_times / 60.0  # convert minutes to hours

        # Eq. 23: Exponential decay weights
        # Use softplus on beta to ensure positive decay
        beta = nn.functional.softplus(self.beta)
        raw_weights = torch.exp(-beta * delta_t)  # (B, seq_len)

        # Apply mask
        if mask is not None:
            raw_weights = raw_weights * mask

        # Normalize weights (softmax-style but with exponential decay)
        weight_sum = raw_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sigma = raw_weights / weight_sum  # (B, seq_len)

        # Eq. 24: Weighted sum of predecessor delays
        y_prop = (sigma * chain_delays).sum(dim=-1, keepdim=True)  # (B, 1)

        # Project to embedding space
        h_prop = self.proj(y_prop)  # (B, hidden_dim)

        return y_prop, h_prop
