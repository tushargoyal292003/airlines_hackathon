"""
Gated Residual Network (GRN)
Paper Section 3.2.1, Equations 7-10:
  η2 = ELU(W2·a + W3·c + b2)
  η1 = W1·η2 + b1
  GLU(γ) = σ(W4·γ + b4) ⊙ (W5·γ + b5)
  h_static = LayerNorm(a + GLU(η1))
"""
import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU as defined in Eq. 9."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_gate = nn.Linear(input_dim, output_dim)
        self.fc_value = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc_gate(x)) * self.fc_value(x)


class GatedResidualNetwork(nn.Module):
    """
    GRN with optional context vector.
    Used for static feature embedding and variable selection.
    
    Input: a (primary input), c (optional context)
    Output: h_static = LayerNorm(a + GLU(η1))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Eq. 7: η2 = ELU(W2·a + W3·c + b2)
        self.fc_primary = nn.Linear(input_dim, hidden_dim)
        self.fc_context = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.elu = nn.ELU()

        # Eq. 8: η1 = W1·η2 + b1
        self.fc_eta1 = nn.Linear(hidden_dim, output_dim)

        # Eq. 9-10: GLU + LayerNorm + residual
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection (if input_dim != output_dim)
        self.skip_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim else nn.Identity()
        )

    def forward(self, a, c=None):
        """
        Args:
            a: (batch, input_dim) — primary input
            c: (batch, context_dim) — optional context vector
        Returns:
            h: (batch, output_dim) — static embedding
        """
        # Eq. 7
        eta2 = self.fc_primary(a)
        if self.fc_context is not None and c is not None:
            eta2 = eta2 + self.fc_context(c)
        eta2 = self.elu(eta2)

        # Eq. 8
        eta1 = self.fc_eta1(eta2)
        eta1 = self.dropout(eta1)

        # Eq. 9-10: GLU + residual + LayerNorm
        glu_out = self.glu(eta1)
        skip = self.skip_proj(a)
        h = self.layer_norm(skip + glu_out)

        return h
