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

        if "chain_id" not in df.columns:
            raise ValueError("Missing required column: chain_id")

        keep_cols = list(dict.fromkeys(
            ["chain_id", "CRSDepTime", "Origin", "Dest", "Origin_str", "Dest_str",
             "Year", "Year_raw", "Month", "Month_raw", "Month_sin", "Month_cos",
             "DayOfWeek_sin", "DayOfWeek_cos",
             "prev_arr_delay", "turnaround_minutes"]
            + self.static_cols
            + self.all_dynamic
            + [self.target_col]
        ))
        available_cols = [c for c in keep_cols if c in df.columns]
        work_df = df[available_cols].copy()

        sort_cols = ["chain_id"]
        if "CRSDepTime" in work_df.columns:
            sort_cols.append("CRSDepTime")
        work_df = work_df.sort_values(sort_cols).reset_index(drop=True)

        numeric_cols = list(dict.fromkeys(
            self.static_cols
            + self.all_dynamic
            + [self.target_col, "prev_arr_delay", "turnaround_minutes"]
        ))
        for col in numeric_cols:
            if col not in work_df.columns:
                if col == "turnaround_minutes":
                    work_df[col] = 60.0
                else:
                    work_df[col] = 0.0
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

        fill_defaults = {
            self.target_col: 0.0,
            "prev_arr_delay": 0.0,
            "turnaround_minutes": 60.0,
        }
        for col in self.static_cols + self.all_dynamic:
            fill_defaults[col] = 0.0
        work_df = work_df.fillna(fill_defaults)

        self.dynamic_matrix = work_df[self.all_dynamic].to_numpy(dtype=np.float32, copy=True)
        self.static_matrix = work_df[self.static_cols].to_numpy(dtype=np.float32, copy=True)
        self.target_vector = work_df[self.target_col].to_numpy(dtype=np.float32, copy=True)
        self.prev_arr_delay = work_df["prev_arr_delay"].to_numpy(dtype=np.float32, copy=True)
        self.turnaround = work_df["turnaround_minutes"].to_numpy(dtype=np.float32, copy=True)

        n_rows = len(work_df)
        origin_src = "Origin_str" if "Origin_str" in work_df.columns else "Origin"
        dest_src   = "Dest_str"   if "Dest_str"   in work_df.columns else "Dest"
        if origin_src in work_df.columns:
            self.origin = work_df[origin_src].astype(str).to_numpy(copy=True)
        else:
            self.origin = np.full(n_rows, "", dtype=object)
        if dest_src in work_df.columns:
            self.dest = work_df[dest_src].astype(str).to_numpy(copy=True)
        else:
            self.dest = np.full(n_rows, "", dtype=object)
        self.chain_id_values = work_df["chain_id"].to_numpy(copy=True)
        year_src = "Year_raw" if "Year_raw" in work_df.columns else "Year"
        month_src = "Month_raw" if "Month_raw" in work_df.columns else "Month"
        if year_src in work_df.columns:
            self.year_values = pd.to_numeric(work_df[year_src], errors="coerce").fillna(0).astype(int).to_numpy(copy=True)
        else:
            self.year_values = np.zeros(n_rows, dtype=int)
        if month_src in work_df.columns:
            self.month_values = pd.to_numeric(work_df[month_src], errors="coerce").fillna(0).astype(int).to_numpy(copy=True)
        else:
            self.month_values = np.zeros(n_rows, dtype=int)

        self.chain_rows: List[np.ndarray] = []
        self.chain_ids: List[str] = []
        self.sample_index: List[Tuple[int, int]] = []

        for chain_id, row_idx in work_df.groupby("chain_id", sort=False).indices.items():
            rows = np.asarray(row_idx, dtype=np.int32)
            if rows.size < 2:
                continue
            chain_idx = len(self.chain_rows)
            self.chain_rows.append(rows)
            self.chain_ids.append(chain_id)
            self.sample_index.extend((chain_idx, pos) for pos in range(1, rows.size))

        print(
            f"  Built {len(self.sample_index)} samples from "
            f"{len(self.chain_rows)} chains (memory-efficient indexing)"
        )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        chain_idx, pos = self.sample_index[idx]
        rows = self.chain_rows[chain_idx]

        target_row = rows[pos]
        history_rows = rows[max(0, pos - self.seq_len):pos]

        dynamic = self.dynamic_matrix[history_rows]
        chain_delays = self.prev_arr_delay[history_rows]
        turnarounds = self.turnaround[history_rows]

        actual_len = history_rows.size
        pad_len = self.seq_len - actual_len

        if pad_len > 0:
            dynamic = np.pad(dynamic, ((pad_len, 0), (0, 0)), mode="constant")
            chain_delays = np.pad(chain_delays, (pad_len, 0), mode="constant")
            turnarounds = np.pad(
                turnarounds,
                (pad_len, 0),
                mode="constant",
                constant_values=60,
            )
            mask = np.concatenate(
                [np.zeros(pad_len, dtype=np.float32), np.ones(actual_len, dtype=np.float32)]
            )
        else:
            dynamic = dynamic[-self.seq_len:]
            chain_delays = chain_delays[-self.seq_len:]
            turnarounds = turnarounds[-self.seq_len:]
            mask = np.ones(self.seq_len, dtype=np.float32)

        static = self.static_matrix[target_row]
        target = self.target_vector[target_row]

        return {
            "dynamic": torch.from_numpy(dynamic.astype(np.float32, copy=False)),
            "static": torch.from_numpy(static.astype(np.float32, copy=False)),
            "chain_delays": torch.from_numpy(chain_delays.astype(np.float32, copy=False)),
            "turnaround_times": torch.from_numpy(turnarounds.astype(np.float32, copy=False)),
            "target": torch.tensor(target, dtype=torch.float32),
            "mask": torch.from_numpy(mask),
            "origin": self.origin[target_row],
            "dest": self.dest[target_row],
            "chain_id": self.chain_id_values[target_row],
            "year": int(self.year_values[target_row]),
            "month": int(self.month_values[target_row]),
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
    for key in ["origin", "dest"]:
        result[key] = [b[key] for b in batch]
    result["chain_id"] = [b["chain_id"] for b in batch]
    result["year"] = [b["year"] for b in batch]
    result["month"] = [b["month"] for b in batch]
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

        if "ArrTime_in" in pairs.columns and "CRSDepTime_out" in pairs.columns:
            pairs = pairs[
                pd.to_numeric(pairs["CRSDepTime_out"], errors="coerce") >
                pd.to_numeric(pairs["ArrTime_in"], errors="coerce")
            ]

        print(f"  Built {len(pairs):,} A->DFW->B pairs")
        return pairs


def _loader_kwargs(config, drop_last: bool) -> Dict:
    kwargs = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "pin_memory": config.train.pin_memory,
        "drop_last": drop_last,
        "collate_fn": flight_collate_fn,
    }
    if config.train.num_workers > 0:
        kwargs["prefetch_factor"] = config.train.prefetch_factor
        kwargs["persistent_workers"] = config.train.persistent_workers
    return kwargs


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
        train_ds,
        shuffle=True,
        **_loader_kwargs(config, drop_last=True),
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **_loader_kwargs(config, drop_last=False),
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **_loader_kwargs(config, drop_last=False),
    )

    return train_loader, val_loader, test_loader
