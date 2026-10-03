"""Sanity checks and runtime training health monitoring for Chess Supervised Learning.

Validates:
1. Environment & CUDA capabilities.
2. Model forward & backward pass integrity.
3. Overfitting capacity test (proves network capacity & gradient flow).
4. Runtime anomaly detection (NaN/Inf losses, gradient explosion, divergence).
"""

import sys
from typing import Dict, Any, Tuple
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.board_encoder import board_to_tensor
from src.move_encoder import encode_move, get_legal_moves_mask, ACTION_SIZE
from src.model import ChessResNet


class TrainingAnomalyError(RuntimeError):
    """Raised when training diverges, produces NaNs, or encounters unrecoverable errors."""
    pass


def run_environment_check() -> Dict[str, Any]:
    """Inspects and reports system and PyTorch execution environment."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    print("=" * 60)
    print(" [HEALTH CHECK 1/4] Environment Diagnostics")
    print(f" - Python: {info['python_version']}")
    print(f" - PyTorch: {info['torch_version']}")
    print(f" - Accelerator: {info['device_name']} (CUDA Available: {info['cuda_available']})")
    print("=" * 60)
    return info


def run_model_forward_backward_check(model: nn.Module, device: torch.device) -> bool:
    """Verifies forward pass shapes, backward gradient computation, and NaN resistance."""
    print(" [HEALTH CHECK 2/4] Testing Model Forward & Backward Pass...")
    model.train()
    model.to(device)

    dummy_input = torch.randn(4, 18, 8, 8, device=device)
    dummy_policy_target = torch.randint(0, ACTION_SIZE, (4,), device=device)
    dummy_value_target = torch.FloatTensor([[-0.5], [0.0], [0.5], [1.0]]).to(device)

    # 1. Forward pass
    policy_logits, value_pred = model(dummy_input)

    assert policy_logits.shape == (4, ACTION_SIZE), (
        f"Expected policy logits shape (4, {ACTION_SIZE}), got {policy_logits.shape}"
    )
    assert value_pred.shape == (4, 1), (
        f"Expected value prediction shape (4, 1), got {value_pred.shape}"
    )

    # 2. Loss computation
    loss_p = nn.CrossEntropyLoss()(policy_logits, dummy_policy_target)
    loss_v = nn.MSELoss()(value_pred, dummy_value_target)
    total_loss = loss_p + loss_v

    assert not torch.isnan(total_loss), "Forward loss produced NaN!"
    assert not torch.isinf(total_loss), "Forward loss produced Inf!"

    # 3. Backward pass
    model.zero_grad()
    total_loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN!"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf!"

    print(" -> Forward & Backward passes validated successfully. Gradients healthy.")
    return True


def run_legal_masking_check(model: nn.Module, device: torch.device) -> bool:
    """Verifies that illegal move masking guarantees 100% legal moves."""
    print(" [HEALTH CHECK 3/4] Testing Legal Move Masking Integrity...")
    model.eval()
    board = chess.Board()
    tensor = board_to_tensor(board, canonical=True)
    input_t = torch.from_numpy(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(input_t)
        logits = policy_logits[0].cpu().numpy()

    mask = get_legal_moves_mask(board, canonical=True)
    legal_indices = np.where(mask)[0]

    # Mask illegal moves
    masked_logits = logits.copy()
    masked_logits[~mask] = -1e9
    best_move_idx = int(np.argmax(masked_logits))

    assert best_move_idx in legal_indices, (
        f"Masking failed: Selected action {best_move_idx} is not in legal indices!"
    )
    print(f" -> Legal move mask confirmed. Tested {len(legal_indices)} legal opening moves.")
    return True


def run_overfit_capacity_check(model: nn.Module, device: torch.device, iterations: int = 40) -> bool:
    """Verifies model learning capacity by testing overfitting on a tiny batch.

    If a model cannot overfit on 4 samples in 40 iterations, there is an architecture
    or gradient flow defect.
    """
    print(" [HEALTH CHECK 4/4] Testing Learning Capacity (Micro-batch Overfitting)...")
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dummy_input = torch.randn(4, 18, 8, 8, device=device)
    target_moves = torch.tensor([10, 250, 1000, 4200], device=device)
    target_values = torch.tensor([[1.0], [-1.0], [0.5], [0.0]], device=device)

    initial_loss = None
    final_loss = None

    for step in range(iterations):
        optimizer.zero_grad()
        p_logits, v_pred = model(dummy_input)
        lp = nn.CrossEntropyLoss()(p_logits, target_moves)
        lv = nn.MSELoss()(v_pred, target_values)
        loss = lp + lv

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, (
        f"Model failed to overfit: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}"
    )
    print(
        f" -> Overfitting test passed: Loss reduced from {initial_loss:.4f} to {final_loss:.4f} in {iterations} steps."
    )
    return True


class RuntimeHealthMonitor:
    """Monitors live training health to abort immediately upon divergence or anomalies."""

    def __init__(self, max_loss_threshold: float = 50.0):
        self.max_loss_threshold = max_loss_threshold
        self.consecutive_nan_count = 0

    def check_step(self, step: int, loss_policy: float, loss_value: float, grad_norm: float):
        """Checks metric values of a single training step."""
        total_loss = loss_policy + loss_value

        if np.isnan(total_loss) or np.isinf(total_loss):
            self.consecutive_nan_count += 1
            if self.consecutive_nan_count >= 1:
                raise TrainingAnomalyError(
                    f"[CRITICAL ERROR] Training diverged at step {step}! Loss is NaN/Inf "
                    f"(Policy Loss: {loss_policy}, Value Loss: {loss_value}). Stopping training."
                )
        else:
            self.consecutive_nan_count = 0

        if total_loss > self.max_loss_threshold:
            raise TrainingAnomalyError(
                f"[CRITICAL ERROR] Total loss {total_loss:.4f} exceeded threshold "
                f"{self.max_loss_threshold} at step {step}. Training is diverging."
            )

        if np.isnan(grad_norm) or np.isinf(grad_norm):
            raise TrainingAnomalyError(
                f"[CRITICAL ERROR] Gradient norm is NaN/Inf at step {step}."
            )
