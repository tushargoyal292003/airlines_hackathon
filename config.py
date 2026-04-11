"""
TFT-DCP Configuration
Hyperparameters follow the paper: lr=0.001, dropout=0.1, hidden=128, batch=256, seq_len=14
Adapted for DFW airport pair risk scoring problem.

Data directory layout expected:
  data/raw/bts/     ← On_Time_*.csv files from BTS
  data/raw/noaa/    ← {AIRPORT}_{StationID}_{Year}.csv files from NOAA LCD
  data/raw/aspm/    ← {Year}-A.xls and {Year}-D.xls files from FAA ASPM
  data/processed/   ← written by preprocessor
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    # ── Data source directories ──────────────────────────────────────────────
    # Update these to match your actual layout, or pass via CLI args in main.py:
    #   python main.py --bts-dir ./data/raw/bts --noaa-dir ./data/raw/noaa \
    #                  --aspm-dir ./data/raw/aspm
    bts_data_dir: str = "./data/raw/bts"
    noaa_data_dir: str = "./data/raw/noaa"
    aspm_data_dir: str = "./data/raw/aspm"
    processed_dir: str = "./data/processed"
    results_dir:   str = "./results"

    hub_airport: str = "DFW"

    # Extreme delay threshold (minutes) per paper Section 3.1.2
    extreme_delay_threshold: int = 180

    # Train/val/test split (months)
    train_months: List[int] = field(default_factory=lambda: list(range(1, 11)))  # Jan-Oct
    val_months: List[int] = field(default_factory=lambda: [11, 12])             # Nov-Dec

    # Normalization
    normalization: str = "minmax"


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
    retrieval_alpha: float = 0.5
    history_db_size: int = 50000

    # MS-CA-EFM (Section 3.2.3)
    channel_reduction_ratio: int = 4

    # Delay propagation (Section 3.2.4)
    beta_init: float = 1.0
    max_chain_length: int = 6

    # Feature dimensions (set dynamically during training)
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

    # Multi-GPU
    num_gpus: int = 4
    distributed: bool = True

    # DataLoader
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
