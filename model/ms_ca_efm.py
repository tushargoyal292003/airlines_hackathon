"""
Multi-Source Channel-Attention Embedding Fusion Module (MS-CA-EFM)
Paper Section 3.2.3, Equations 16-22.

Implements squeeze-and-excitation style channel attention
to fuse H_current, H_f (retrieved), and H_global embeddings.
"""
import torch
import torch.nn as nn


class MSCAEFM(nn.Module):
    """
    Fuses three embedding sources via channel attention.
    
    Step 1: H = H_current + H_f + H_global (Eq. 16)
    Step 2: Z = GAP(H) (Eq. 17)
    Step 3: S = ReLU(V1·Z), A = V2·S (Eq. 18-19)
    Step 4: A'_i = σ(A_i) — per-source weights (Eq. 20)
    Step 5: H'_i = A'_i ⊙ H_i, H_fused = Σ H'_i (Eq. 21-22)
    """

    def __init__(self, embedding_dim: int, reduction_ratio: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        reduced_dim = max(embedding_dim // reduction_ratio, 8)

        # Squeeze: Global Average Pooling is just mean
        # Excitation: two FC layers (Eq. 18-19)
        self.fc1 = nn.Linear(embedding_dim, reduced_dim)  # V1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(reduced_dim, embedding_dim * 3)  # V2 -> 3 channels
        self.sigmoid = nn.Sigmoid()

        # Residual + LayerNorm for final output (Section 3.2.3 part 3)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.relu_out = nn.ReLU()

    def forward(
        self,
        h_current: torch.Tensor,
        h_f: torch.Tensor,
        h_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_current: (batch, D) — target flight embedding
            h_f: (batch, D) — historical-fused embedding
            h_global: (batch, D) — global pooled embedding
        Returns:
            h_fused: (batch, D) — attention-weighted fusion
        """
        # Step 1: Element-wise summation (Eq. 16)
        H = h_current + h_f + h_global  # (B, D)

        # Step 2: Global average pooling (Eq. 17)
        # Already (B, D) — no spatial dims to pool over, Z = H
        Z = H

        # Step 3: Channel attention (Eq. 18-19)
        S = self.relu(self.fc1(Z))  # (B, reduced_dim)
        A = self.fc2(S)  # (B, 3*D)

        # Step 4: Split into 3 channel weights + sigmoid (Eq. 20)
        A1, A2, A3 = A.chunk(3, dim=-1)  # each (B, D)
        A1_prime = self.sigmoid(A1)
        A2_prime = self.sigmoid(A2)
        A3_prime = self.sigmoid(A3)

        # Step 5: Channel-wise weighting (Eq. 21)
        H_global_prime = A1_prime * h_global
        H_f_prime = A2_prime * h_f
        H_current_prime = A3_prime * h_current

        # Sum to get fused representation (Eq. 22)
        H_fused = H_global_prime + H_f_prime + H_current_prime

        # Residual enhancement and prediction (Section 3.2.3 part 3)
        H_out = self.layer_norm(H_fused + h_current)  # residual with current
        H_out = self.relu_out(self.fc_out(H_out))

        return H_out
