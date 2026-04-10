"""
TFT-DCP Configuration
Hyperparameters follow the paper: lr=0.001, dropout=0.1, hidden=128, batch=256, seq_len=14
Adapted for DFW airport pair risk scoring problem.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    bts_data_dir: str = "./data/data_bts/raw/bts"
    noaa_data_dir: str = "./data/data_noaa/raw/noaa"
    aspm_data_dir: str = "./data/data_aspm"
    processed_dir: str = "./data/processed"
    hub_airport: str = "DFW"

    # Extreme delay threshold (minutes) per paper Section 3.1.2
    extreme_delay_threshold: int = 180

    # Train/val/test split
    train_months: List[int] = field(default_factory=lambda: list(range(1, 11)))  # Jan-Oct
    val_months: List[int] = field(default_factory=lambda: [11, 12])  # Nov-Dec
    # Test: extreme weather periods (paper Section 4.2.2)

    # Normalization
    normalization: str = "minmax"  # minmax as per paper


@dataclass
class ModelConfig:
    # Paper Section 4.2.2 hyperparameters
    hidden_dim: int = 128
    dropout: float = 0.1
    sequence_length: int = 14

    # TCN encoder
    tcn_num_channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    tcn_kernel_size: int = 3

    # GRN
    grn_hidden_dim: int = 64

    # Historical retrieval
    top_k_retrieval: int = 5
    retrieval_alpha: float = 0.5  # fusion coefficient (Eq. 15)
    history_db_size: int = 50000

    # MS-CA-EFM (Section 3.2.3)
    channel_reduction_ratio: int = 4  # reduction ratio r

    # Delay propagation (Section 3.2.4)
    beta_init: float = 1.0  # initial decay coefficient
    max_chain_length: int = 6

    # Feature dimensions (will be set dynamically)
    num_static_features: int = 8
    num_dynamic_features: int = 22
    num_weather_features: int = 12
    num_airport_features: int = 5


@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5

    # Multi-GPU (4x A6000)
    num_gpus: int = 4
    distributed: bool = True

    # DataLoader memory/perf controls
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Logging
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
