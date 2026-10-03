"""Deep Dual-Head Residual CNN for Chess.

Predicts both:
1. Move Policy: Probability distribution over 16,384 candidate moves.
2. Position Value: Expected game outcome in [-1.0, 1.0] from current player's perspective.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard 2D Residual Block with Batch Normalization and ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class ChessResNet(nn.Module):
    """Dual-Head Chess Residual Network (Policy + Value)."""

    def __init__(
        self,
        in_channels: int = 18,
        num_filters: int = 128,
        num_blocks: int = 6,
        num_actions: int = 16384,
        policy_head_filters: int = 32,
        value_head_filters: int = 4,
        value_hidden_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        # 1. Initial representation extraction
        self.initial_conv = nn.Conv2d(
            in_channels, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # 2. ResNet Backbone
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

        # 3. Policy Head: Predicts move probabilities
        self.policy_conv = nn.Conv2d(
            num_filters, policy_head_filters, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(policy_head_filters)
        self.policy_fc = nn.Linear(policy_head_filters * 8 * 8, num_actions)

        # 4. Value Head: Predicts expected game outcome [-1, 1]
        self.value_conv = nn.Conv2d(
            num_filters, value_head_filters, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(value_head_filters)
        self.value_fc1 = nn.Linear(value_head_filters * 8 * 8, value_hidden_dim)
        self.value_fc2 = nn.Linear(value_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, in_channels, 8, 8)

        Returns:
            Tuple of:
            - policy_logits: Tensor of shape (B, num_actions)
            - value: Tensor of shape (B, 1) in range [-1.0, 1.0]
        """
        # Feature extraction
        out = F.relu(self.initial_bn(self.initial_conv(x)), inplace=True)
        for block in self.blocks:
            out = block(out)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)), inplace=True)
        p = torch.flatten(p, start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)), inplace=True)
        v = torch.flatten(v, start_dim=1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def build_model(config: Dict[str, Any]) -> ChessResNet:
    """Factory function to build ChessResNet from configuration dictionary."""
    cfg = config.get("model", {})
    return ChessResNet(
        in_channels=cfg.get("in_channels", 18),
        num_filters=cfg.get("num_filters", 128),
        num_blocks=cfg.get("num_blocks", 6),
        num_actions=cfg.get("num_actions", 16384),
        policy_head_filters=cfg.get("policy_head_filters", 32),
        value_head_filters=cfg.get("value_head_filters", 4),
        value_hidden_dim=cfg.get("value_hidden_dim", 256),
    )


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
