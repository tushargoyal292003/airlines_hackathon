"""
PyTorch Dataset for TFT-DCP
Constructs flight chain sequences with static/dynamic feature separation.
Paper Section 3.1.4: Each chain Ck = [x1, x2, ..., xk]
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FlightChainDataset(Dataset):
    """
    Dataset that yields flight chain sequences.
    Each sample is a chain of flights operated by the same aircraft on the same day.
    
    Returns:
        dynamic_features: (seq_len, num_dynamic) - time-varying features
        static_features: (num_static,) - time-invariant features
        chain_delays: (seq_len,) - preceding flight delays for propagation module
        turnaround_times: (seq_len,) - time gaps between consecutive flights
        target: (1,) - departure delay of the target flight
        mask: (seq_len,) - valid positions mask (for variable-length chains)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        static_cols: List[str],
        dynamic_cols: List[str],
        weather_cols: List[str],
        target_col: str = "DepDelay",
        seq_len: int = 14,
        hub: str = "DFW",
    ):
        self.seq_len = seq_len
        self.target_col = target_col
        self.hub = hub

        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols
        self.weather_cols = weather_cols
        self.all_dynamic = dynamic_cols + weather_cols

        # Group by chain_id and build sequences
        self.samples = self._build_samples(df)
        print(f"  Built {len(self.samples)} samples from {df['chain_id'].nunique()} chains")

    def _build_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Build prediction samples from flight chains."""
        samples = []

        for chain_id, chain_df in df.groupby("chain_id"):
            chain_df = chain_df.sort_values("CRSDepTime")

            if len(chain_df) < 2:
                continue  # Need at least 2 flights for propagation

            # Each flight in the chain (except the first) is a prediction target
            for idx in range(1, len(chain_df)):
                target_row = chain_df.iloc[idx]

                # Historical context: preceding flights in the chain
                history = chain_df.iloc[max(0, idx - self.seq_len):idx]

                # Extract features
                dynamic = history[self.all_dynamic].values.astype(np.float32)
                static = target_row[self.static_cols].values.astype(np.float32)
                target = np.float32(target_row[self.target_col])

                # Chain delay propagation features
                chain_delays = history["prev_arr_delay"].values.astype(np.float32)
                turnarounds = history["turnaround_minutes"].values.astype(np.float32)

                # Pad/truncate to seq_len
                actual_len = len(history)
                pad_len = self.seq_len - actual_len

                if pad_len > 0:
                    dynamic = np.pad(dynamic, ((pad_len, 0), (0, 0)), mode="constant")
                    chain_delays = np.pad(chain_delays, (pad_len, 0), mode="constant")
                    turnarounds = np.pad(turnarounds, (pad_len, 0), mode="constant",
                                        constant_values=60)
                    mask = np.array([0] * pad_len + [1] * actual_len, dtype=np.float32)
                else:
                    dynamic = dynamic[-self.seq_len:]
                    chain_delays = chain_delays[-self.seq_len:]
                    turnarounds = turnarounds[-self.seq_len:]
                    mask = np.ones(self.seq_len, dtype=np.float32)

                samples.append({
                    "dynamic": dynamic,
                    "static": static,
                    "chain_delays": chain_delays,
                    "turnaround_times": turnarounds,
                    "target": target,
                    "mask": mask,
                    "chain_id": chain_id,
                    # For pair risk scoring
                    "origin": target_row.get("Origin", ""),
                    "dest": target_row.get("Dest", ""),
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "dynamic": torch.tensor(s["dynamic"], dtype=torch.float32),
            "static": torch.tensor(s["static"], dtype=torch.float32),
            "chain_delays": torch.tensor(s["chain_delays"], dtype=torch.float32),
            "turnaround_times": torch.tensor(s["turnaround_times"], dtype=torch.float32),
            "target": torch.tensor(s["target"], dtype=torch.float32),
            "mask": torch.tensor(s["mask"], dtype=torch.float32),
            # Metadata as strings — handled by custom collate_fn
            "origin": s["origin"],
            "dest": s["dest"],
            "chain_id": s["chain_id"],
        }


def flight_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate that handles mixed tensor + string fields.
    DataLoader default collate can't handle strings.
    """
    result = {}
    tensor_keys = ["dynamic", "static", "chain_delays", "turnaround_times", "target", "mask"]
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch])
    # String metadata as lists
    for key in ["origin", "dest"]:
        result[key] = [b[key] for b in batch]
    # Chain IDs as list
    result["chain_id"] = [b["chain_id"] for b in batch]
    return result


class DFWPairDataset(FlightChainDataset):
    """
    Extended dataset for DFW pair risk scoring.
    Constructs A -> DFW -> B pairs and computes pair-level risk.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, **kwargs)
        self.pairs = self._build_pairs(df)

    def _build_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build A -> DFW -> B pairs.
        Match inbound flights to DFW with subsequent outbound flights.
        """
        hub = self.hub
        inbound = df[df["Dest"] == hub].copy()
        outbound = df[df["Origin"] == hub].copy()

        # Match by tail number and date (same aircraft)
        inbound["pair_key"] = inbound["Tail_Number"].astype(str) + "_" + \
                              inbound["FlightDate"].astype(str)
        outbound["pair_key"] = outbound["Tail_Number"].astype(str) + "_" + \
                               outbound["FlightDate"].astype(str)

        pairs = inbound.merge(
            outbound,
            on="pair_key",
            suffixes=("_in", "_out"),
            how="inner",
        )

        # Filter: outbound departs after inbound arrives
        if "ArrTime_in" in pairs.columns and "CRSDepTime_out" in pairs.columns:
            pairs = pairs[
                pd.to_numeric(pairs["CRSDepTime_out"], errors="coerce") >
                pd.to_numeric(pairs["ArrTime_in"], errors="coerce")
            ]

        print(f"  Built {len(pairs):,} A->DFW->B pairs")
        return pairs


def create_dataloaders(
    df: pd.DataFrame,
    static_cols: List[str],
    dynamic_cols: List[str],
    weather_cols: List[str],
    config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with temporal split."""

    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    train_df = df[df["Month"].isin(config.data.train_months)]
    val_df = df[df["Month"].isin(config.data.val_months)]
    # Test: remaining data (extreme weather periods)
    test_months = [m for m in range(1, 13)
                   if m not in config.data.train_months and m not in config.data.val_months]
    test_df = df[df["Month"].isin(test_months)] if test_months else val_df

    common_kwargs = dict(
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
        weather_cols=weather_cols,
        seq_len=config.model.sequence_length,
        hub=config.data.hub_airport,
    )

    train_ds = FlightChainDataset(train_df, **common_kwargs)
    val_ds = FlightChainDataset(val_df, **common_kwargs)
    test_ds = FlightChainDataset(test_df, **common_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=flight_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=flight_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=flight_collate_fn,
    )

    return train_loader, val_loader, test_loader
