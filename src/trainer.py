"""Resilient training pipeline for Chess Supervised Learning.

Includes:
- Dual-head loss computation (Cross-Entropy policy + MSE value).
- Top-1 and Top-5 accuracy tracking.
- AMP (Automatic Mixed Precision) and gradient clipping.
- Metric logging to CSV and console.
- Early stopping & best checkpoint serialization.
- Runtime anomaly protection.
"""

import csv
import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.health_checks import RuntimeHealthMonitor, TrainingAnomalyError


class ChessTrainer:
    """Manages the full lifecycle of training a Chess Dual-Head ResNet model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 20)
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.min_lr = float(train_cfg.get("min_lr", 1e-5))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.policy_weight = float(train_cfg.get("policy_weight", 1.0))
        self.value_weight = float(train_cfg.get("value_weight", 1.0))
        self.clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))
        self.patience = int(train_cfg.get("early_stopping_patience", 5))

        # Setup AMP scaler
        use_amp = train_cfg.get("use_amp", True) and (device.type == "cuda")
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Loss functions
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.min_lr
        )

        # Health monitor
        hc_cfg = config.get("health_checks", {})
        max_loss_thresh = float(hc_cfg.get("max_loss_threshold", 50.0))
        self.health_monitor = RuntimeHealthMonitor(max_loss_threshold=max_loss_thresh)

        # Paths
        out_cfg = config.get("output", {})
        self.checkpoint_dir = out_cfg.get("checkpoint_dir", "checkpoints")
        self.log_dir = out_cfg.get("log_dir", "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "training_history.csv")
        self._init_csv()

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _init_csv(self):
        """Initializes the CSV metric history file."""
        headers = [
            "epoch",
            "train_loss",
            "train_p_loss",
            "train_v_loss",
            "train_top1",
            "train_top5",
            "train_v_mae",
            "val_loss",
            "val_p_loss",
            "val_v_loss",
            "val_top1",
            "val_top5",
            "val_v_mae",
            "lr",
            "elapsed_s",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    @staticmethod
    def _compute_accuracies(
        policy_logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float]:
        """Calculates Top-1 and Top-5 accuracy percentages."""
        with torch.no_grad():
            batch_size = targets.size(0)
            _, top5_pred = policy_logits.topk(5, dim=1, largest=True, sorted=True)
            top1_correct = (top5_pred[:, 0] == targets).sum().item()
            top5_correct = (top5_pred == targets.unsqueeze(1)).any(dim=1).sum().item()
            return (top1_correct / batch_size) * 100.0, (top5_correct / batch_size) * 100.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes a single training epoch."""
        self.model.train()
        total_p_loss, total_v_loss, total_loss = 0.0, 0.0, 0.0
        total_top1, total_top5, total_v_mae = 0.0, 0.0, 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Train]",
            leave=False,
        )

        for step, (boards, target_moves, target_values) in enumerate(pbar):
            boards = boards.to(self.device, non_blocking=True)
            target_moves = target_moves.to(self.device, non_blocking=True)
            target_values = target_values.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                policy_logits, value_pred = self.model(boards)
                loss_p = self.criterion_policy(policy_logits, target_moves)
                loss_v = self.criterion_value(value_pred, target_values)
                loss = (self.policy_weight * loss_p) + (self.value_weight * loss_v)

            # Backward with AMP scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            ).item()

            # Health check on current step
            self.health_monitor.check_step(
                step=epoch * num_batches + step,
                loss_policy=loss_p.item(),
                loss_value=loss_v.item(),
                grad_norm=grad_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            top1, top5 = self._compute_accuracies(policy_logits, target_moves)
            v_mae = torch.abs(value_pred - target_values).mean().item()

            total_loss += loss.item()
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            total_top1 += top1
            total_top5 += top5
            total_v_mae += v_mae

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "top1": f"{top1:.1f}%",
                    "v_mae": f"{v_mae:.3f}",
                    "gnorm": f"{grad_norm:.2f}",
                }
            )

        return {
            "loss": total_loss / num_batches,
            "p_loss": total_p_loss / num_batches,
            "v_loss": total_v_loss / num_batches,
            "top1": total_top1 / num_batches,
            "top5": total_top5 / num_batches,
            "v_mae": total_v_mae / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluates model performance on validation data."""
        self.model.eval()
        total_p_loss, total_v_loss, total_loss = 0.0, 0.0, 0.0
        total_top1, total_top5, total_v_mae = 0.0, 0.0, 0.0
        num_batches = len(self.val_loader)

        if num_batches == 0:
            return {"loss": 0.0, "p_loss": 0.0, "v_loss": 0.0, "top1": 0.0, "top5": 0.0, "v_mae": 0.0}

        for boards, target_moves, target_values in self.val_loader:
            boards = boards.to(self.device, non_blocking=True)
            target_moves = target_moves.to(self.device, non_blocking=True)
            target_values = target_values.to(self.device, non_blocking=True)

            policy_logits, value_pred = self.model(boards)
            loss_p = self.criterion_policy(policy_logits, target_moves)
            loss_v = self.criterion_value(value_pred, target_values)
            loss = (self.policy_weight * loss_p) + (self.value_weight * loss_v)

            top1, top5 = self._compute_accuracies(policy_logits, target_moves)
            v_mae = torch.abs(value_pred - target_values).mean().item()

            total_loss += loss.item()
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            total_top1 += top1
            total_top5 += top5
            total_v_mae += v_mae

        return {
            "loss": total_loss / num_batches,
            "p_loss": total_p_loss / num_batches,
            "v_loss": total_v_loss / num_batches,
            "top1": total_top1 / num_batches,
            "top5": total_top5 / num_batches,
            "v_mae": total_v_mae / num_batches,
        }

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Serializes current model and optimizer state."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(state, path)

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print(f"\n[Training Started] Device: {self.device} | Epochs: {self.epochs} | Mixed-Precision: {self.use_amp}")
        print("-" * 90)
        print(
            f"{'Epoch':<6} | {'Tr Loss':<8} {'Tr P-Loss':<9} {'Tr Top1':<8} {'Tr Top5':<8} | "
            f"{'Val Loss':<8} {'Val Top1':<8} {'Val MAE':<7} | {'LR':<9} | {'Status'}"
        )
        print("-" * 90)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            try:
                tr_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
            except TrainingAnomalyError as e:
                # Emergency safe checkpoint before dying
                emergency_path = os.path.join(self.checkpoint_dir, f"diverged_checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(emergency_path, epoch)
                print(f"\n[EMERGENCY STOP] {e}")
                print(f"[EMERGENCY SAVED] Diverged state preserved at {emergency_path}")
                raise e

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Checkpoint best model
            is_best = val_metrics["loss"] < self.best_val_loss
            status = " "
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                best_path = os.path.join(
                    self.checkpoint_dir,
                    self.config.get("output", {}).get("best_model_name", "best_model.pt"),
                )
                self.save_checkpoint(best_path, epoch, is_best=True)
                status = "[BEST]"
            else:
                self.epochs_without_improvement += 1

            # Save latest checkpoint
            latest_path = os.path.join(
                self.checkpoint_dir,
                self.config.get("output", {}).get("latest_model_name", "latest_model.pt"),
            )
            self.save_checkpoint(latest_path, epoch)

            elapsed = time.time() - epoch_start
            print(
                f"{epoch:<6d} | {tr_metrics['loss']:<8.4f} {tr_metrics['p_loss']:<9.4f} "
                f"{tr_metrics['top1']:<7.1f}% {tr_metrics['top5']:<7.1f}% | "
                f"{val_metrics['loss']:<8.4f} {val_metrics['top1']:<7.1f}% "
                f"{val_metrics['v_mae']:<7.3f} | {current_lr:<9.6f} | {status} ({elapsed:.1f}s)"
            )

            # Log to CSV
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        tr_metrics["loss"],
                        tr_metrics["p_loss"],
                        tr_metrics["v_loss"],
                        tr_metrics["top1"],
                        tr_metrics["top5"],
                        tr_metrics["v_mae"],
                        val_metrics["loss"],
                        val_metrics["p_loss"],
                        val_metrics["v_loss"],
                        val_metrics["top1"],
                        val_metrics["top5"],
                        val_metrics["v_mae"],
                        current_lr,
                        elapsed,
                    ]
                )

            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                print(
                    f"\n[Early Stopping] No validation loss improvement for {self.patience} consecutive epochs. Stopping."
                )
                break

        total_time = time.time() - start_time
        print("-" * 90)
        print(f"[Training Complete] Total elapsed: {total_time / 60:.2f} mins | Best Val Loss: {self.best_val_loss:.4f}")
        return {"best_val_loss": self.best_val_loss, "epochs_completed": epoch}
