"""
Training script for TFT-DCP with multi-GPU support (4x A6000).
Implements paper Section 4.2.2 training protocol.
"""
import os
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from config import Config
from model import TFTDCP
from data.dataset import FlightChainDataset, flight_collate_fn


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def _dataloader_kwargs(config: Config, drop_last: bool) -> dict:
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


class Trainer:
    """TFT-DCP Trainer with early stopping and history database updates."""

    def __init__(
        self,
        model: TFTDCP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        rank: int = 0,
        world_size: int = 1,
        resume_from: str = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.is_main = rank == 0

        # Model
        self.model = model.to(self.device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        self.raw_model = self.model.module if world_size > 1 else self.model

        # Optimizer & scheduler (paper: lr=0.001)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=self.is_main
        )

        # Loss: MSE (paper Section 3.2.4 mentions MSE loss)
        self.criterion = nn.MSELoss()
        # Additional L1 loss for extreme delays
        self.l1_loss = nn.L1Loss()

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 1

        # Resume from checkpoint if provided
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.raw_model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_loss = ckpt.get("val_loss", float("inf"))
            if self.is_main:
                print(f"  Resumed from {resume_from} (epoch {ckpt['epoch']}, "
                      f"val_loss {self.best_val_loss:.4f})")

        # Logging
        self.log_dir = Path(config.train.log_dir)
        self.ckpt_dir = Path(config.train.checkpoint_dir)
        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.resource_log_path = self.log_dir / "resource_usage.jsonl"
            with open(self.resource_log_path, "a", encoding="utf-8"):
                pass

        self.history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [],
                        "val_r2": [], "beta": [], "lr": [], "resource": []}

    def _collect_resource_snapshot(self, epoch: int, elapsed: float) -> dict:
        """Collect host + GPU memory stats for debugging OOM issues."""
        snapshot = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 3),
        }

        meminfo = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.split(":", maxsplit=1)
                    parts = value.strip().split()
                    if parts:
                        meminfo[key] = int(parts[0])  # kB
        except OSError:
            meminfo = {}

        mem_total_kb = meminfo.get("MemTotal", 0)
        mem_avail_kb = meminfo.get("MemAvailable", 0)
        swap_total_kb = meminfo.get("SwapTotal", 0)
        swap_free_kb = meminfo.get("SwapFree", 0)

        snapshot["host_mem_total_gb"] = round(mem_total_kb / (1024 ** 2), 3)
        snapshot["host_mem_used_gb"] = round((mem_total_kb - mem_avail_kb) / (1024 ** 2), 3)
        snapshot["host_mem_available_gb"] = round(mem_avail_kb / (1024 ** 2), 3)
        snapshot["swap_total_gb"] = round(swap_total_kb / (1024 ** 2), 3)
        snapshot["swap_used_gb"] = round((swap_total_kb - swap_free_kb) / (1024 ** 2), 3)

        status_map = {}
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", maxsplit=1)
                    status_map[key] = value.strip()
        except OSError:
            status_map = {}

        def _status_kb(name: str) -> int:
            raw = status_map.get(name, "0 kB")
            parts = raw.split()
            return int(parts[0]) if parts else 0

        proc_rss_kb = _status_kb("VmRSS")
        proc_hwm_kb = _status_kb("VmHWM")
        snapshot["proc_rss_gb"] = round(proc_rss_kb / (1024 ** 2), 3)
        snapshot["proc_hwm_gb"] = round(proc_hwm_kb / (1024 ** 2), 3)

        gpu_stats = []
        if torch.cuda.is_available():
            for dev_idx in range(torch.cuda.device_count()):
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
                    used_bytes = total_bytes - free_bytes
                    gpu_stats.append(
                        {
                            "gpu": dev_idx,
                            "used_gb": round(used_bytes / (1024 ** 3), 3),
                            "free_gb": round(free_bytes / (1024 ** 3), 3),
                            "total_gb": round(total_bytes / (1024 ** 3), 3),
                        }
                    )
                except RuntimeError:
                    continue

            if self.device.type == "cuda" and self.device.index is not None:
                idx = self.device.index
                snapshot["local_gpu_allocated_gb"] = round(
                    torch.cuda.memory_allocated(idx) / (1024 ** 3), 3
                )
                snapshot["local_gpu_reserved_gb"] = round(
                    torch.cuda.memory_reserved(idx) / (1024 ** 3), 3
                )
                snapshot["local_gpu_peak_allocated_gb"] = round(
                    torch.cuda.max_memory_allocated(idx) / (1024 ** 3), 3
                )
                snapshot["local_gpu_peak_reserved_gb"] = round(
                    torch.cuda.max_memory_reserved(idx) / (1024 ** 3), 3
                )

        snapshot["gpus"] = gpu_stats
        return snapshot

    def _log_resource_usage(self, epoch: int, elapsed: float):
        if not self.is_main:
            return
        snapshot = self._collect_resource_snapshot(epoch, elapsed)
        self.history["resource"].append(snapshot)

        with open(self.resource_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")

        gpu_summary = ", ".join(
            f"gpu{g['gpu']} {g['used_gb']:.1f}/{g['total_gb']:.1f}GB"
            for g in snapshot["gpus"]
        )
        if not gpu_summary:
            gpu_summary = "cpu-only"

        print(
            "  Resource | "
            f"RAM used {snapshot['host_mem_used_gb']:.1f}GB "
            f"(avail {snapshot['host_mem_available_gb']:.1f}GB) | "
            f"Proc RSS {snapshot['proc_rss_gb']:.1f}GB "
            f"(HWM {snapshot['proc_hwm_gb']:.1f}GB) | "
            f"Swap used {snapshot['swap_used_gb']:.1f}GB | "
            f"{gpu_summary}"
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            dynamic = batch["dynamic"].to(self.device)
            static = batch["static"].to(self.device)
            chain_delays = batch["chain_delays"].to(self.device)
            turnarounds = batch["turnaround_times"].to(self.device)
            mask = batch["mask"].to(self.device)
            target = batch["target"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(dynamic, static, chain_delays, turnarounds, mask)

            # Combined loss: MSE + weighted L1 for extreme delays
            loss_mse = self.criterion(output["prediction"], target)

            # Extra weight on extreme delays.
            # Important: compute L1 only on extreme rows (no zero-masking dilution).
            extreme_thr = float(self.config.data.extreme_delay_threshold)
            extreme_mask = target > extreme_thr
            if extreme_mask.any():
                extreme_loss = self.l1_loss(
                    output["prediction"][extreme_mask],
                    target[extreme_mask],
                )
                loss = loss_mse + 0.3 * extreme_loss
            else:
                loss = loss_mse

            # Backward + optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Update historical database periodically
            if n_batches % 50 == 0:
                with torch.no_grad():
                    self.raw_model.update_history(output["h_current"].detach())

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate and compute metrics (MAE, RMSE, R²)."""
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            dynamic = batch["dynamic"].to(self.device)
            static = batch["static"].to(self.device)
            chain_delays = batch["chain_delays"].to(self.device)
            turnarounds = batch["turnaround_times"].to(self.device)
            mask = batch["mask"].to(self.device)
            target = batch["target"].to(self.device)

            output = self.model(dynamic, static, chain_delays, turnarounds, mask)
            all_preds.append(output["prediction"].cpu())
            all_targets.append(target.cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        # Paper Eq. 26-28
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        ss_res = np.sum((preds - targets) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / max(ss_tot, 1e-8))

        val_loss = np.mean((preds - targets) ** 2)

        return {"val_loss": val_loss, "mae": mae, "rmse": rmse, "r2": r2}

    def train(self):
        """Full training loop."""
        if self.is_main:
            print("\n" + "=" * 70)
            print("TFT-DCP TRAINING")
            print(f"  Epochs: {self.config.train.num_epochs}")
            print(f"  Batch size: {self.config.train.batch_size} x {self.world_size} GPUs")
            print(f"  Learning rate: {self.config.train.learning_rate}")
            print("=" * 70)

            # Pre-seed historical database with extreme delay cases
            print("\n  Pre-seeding historical retrieval database...")
            self.raw_model.historical_retrieval.preseed_from_extreme_cases(
                encoder=self.raw_model.tcn_encoder,
                dataloader=self.train_loader,
                device=self.device,
                max_batches=50,
            )

        for epoch in range(self.start_epoch, self.config.train.num_epochs + 1):
            if self.device.type == "cuda" and self.device.index is not None:
                torch.cuda.reset_peak_memory_stats(self.device.index)

            # Set epoch for distributed sampler
            if hasattr(self.train_loader, "sampler") and isinstance(
                self.train_loader.sampler, DistributedSampler
            ):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            self.scheduler.step(val_metrics["val_loss"])

            # Log beta value
            beta_val = nn.functional.softplus(
                self.raw_model.delay_propagation.beta
            ).item()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_r2"].append(val_metrics["r2"])
            self.history["beta"].append(beta_val)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.is_main:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:3d}/{self.config.train.num_epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.2f} | "
                    f"RMSE: {val_metrics['rmse']:.2f} | "
                    f"R²: {val_metrics['r2']:.4f} | "
                    f"β: {beta_val:.3f} | "
                    f"{elapsed:.1f}s"
                )
                self._log_resource_usage(epoch, elapsed)

                # Save checkpoint
                if epoch % self.config.train.save_every == 0:
                    self.save_checkpoint(epoch)

                # Early stopping
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.train.early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

        if self.is_main:
            self.save_history()
            print("\nTraining complete!")
            print(f"  Best MAE: {min(self.history['val_mae']):.2f}")
            print(f"  Best RMSE: {min(self.history['val_rmse']):.2f}")
            print(f"  Best R²: {max(self.history['val_r2']):.4f}")
            print(f"  Final β: {self.history['beta'][-1]:.3f}")

    def save_checkpoint(self, epoch, best=False):
        name = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        path = self.ckpt_dir / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": self.history["val_loss"][-1],
            "config": self.config,
        }, path)

    def save_history(self):
        def _coerce(o):
            import numpy as np
            if isinstance(o, (np.floating,)):  return float(o)
            if isinstance(o, (np.integer,)):   return int(o)
            if isinstance(o, np.ndarray):      return o.tolist()
            raise TypeError(f"not serializable: {type(o).__name__}")
        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=_coerce)


def train_distributed(rank, world_size, config, train_dataset, val_dataset, resume_from=None):
    """Entry point for distributed training."""
    setup_distributed(rank, world_size)

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **_dataloader_kwargs(config, drop_last=True),
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        **_dataloader_kwargs(config, drop_last=False),
    )

    # Build model
    model = TFTDCP(
        num_dynamic_features=config.model.num_dynamic_features,
        num_static_features=config.model.num_static_features,
        hidden_dim=config.model.hidden_dim,
        tcn_channels=config.model.tcn_num_channels,
        tcn_kernel_size=config.model.tcn_kernel_size,
        grn_hidden_dim=config.model.grn_hidden_dim,
        top_k_retrieval=config.model.top_k_retrieval,
        retrieval_alpha=config.model.retrieval_alpha,
        history_db_size=config.model.history_db_size,
        channel_reduction_ratio=config.model.channel_reduction_ratio,
        beta_init=config.model.beta_init,
        dropout=config.model.dropout,
    )

    trainer = Trainer(model, train_loader, val_loader, config, rank, world_size,
                      resume_from=resume_from)
    trainer.train()

    cleanup()


def train_single_gpu(config, train_dataset, val_dataset, resume_from=None):
    """Single GPU training (fallback)."""
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **_dataloader_kwargs(config, drop_last=True),
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **_dataloader_kwargs(config, drop_last=False),
    )

    model = TFTDCP(
        num_dynamic_features=config.model.num_dynamic_features,
        num_static_features=config.model.num_static_features,
        hidden_dim=config.model.hidden_dim,
        tcn_channels=config.model.tcn_num_channels,
        grn_hidden_dim=config.model.grn_hidden_dim,
        top_k_retrieval=config.model.top_k_retrieval,
        retrieval_alpha=config.model.retrieval_alpha,
        history_db_size=config.model.history_db_size,
        channel_reduction_ratio=config.model.channel_reduction_ratio,
        beta_init=config.model.beta_init,
        dropout=config.model.dropout,
    )

    trainer = Trainer(model, train_loader, val_loader, config, resume_from=resume_from)
    trainer.train()

    return trainer
