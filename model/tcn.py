"""
Temporal Convolutional Network (TCN) Encoder
Paper Section 3.2.1 Eq. 5-6:
  H_dynamic = TCN(X_dynamic) = {h1, h2, ..., hN}
  h_global = Pooling(H_dynamic)
"""
import torch
import torch.nn as nn
from typing import List


class CausalConv1d(nn.Module):
    """Causal convolution: no future information leaks."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block with dilated causal convolutions.

    FIX: BatchNorm1d replaced with LayerNorm.
    BatchNorm1d normalises across (batch, seq_len) per channel — zero-padded
    positions contaminate the running statistics and bias all activations.
    LayerNorm normalises per (position, channel) independently, so padded
    positions only affect themselves.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = self.downsample(x)
        # LayerNorm expects (..., channels) — transpose, norm, transpose back
        out = self.conv1(x).transpose(1, 2)          # (B, seq_len, C)
        out = self.dropout(self.relu(self.ln1(out))).transpose(1, 2)  # (B, C, seq_len)
        out = self.conv2(out).transpose(1, 2)
        out = self.dropout(self.relu(self.ln2(out))).transpose(1, 2)
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Full TCN encoder for dynamic feature embedding.
    Input: (batch, seq_len, num_features)
    Output:
      - per_step: (batch, seq_len, hidden_dim) — per-flight embeddings
      - global: (batch, hidden_dim) — global pooled representation
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int] = [64, 128, 128],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, features)
            mask: (batch, seq_len) — valid position mask
        Returns:
            h_dynamic: (batch, seq_len, hidden_dim)
            h_global: (batch, hidden_dim)
        """
        # TCN expects (batch, channels, seq_len)
        out = x.transpose(1, 2)
        out = self.network(out)
        h_dynamic = out.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # Global pooling (Eq. 6) with mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            h_masked = h_dynamic * mask_expanded
            h_global = h_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h_global = h_dynamic.mean(dim=1)

        return h_dynamic, h_global
