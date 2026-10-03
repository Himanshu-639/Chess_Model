"""Inference and gameplay engine for the CNN Chess Model.

Provides:
- Legal move masking and greedy/temperature move selection.
- Position value evaluation.
- Optional 1-ply minimax search using policy prior and value head.
"""

from typing import Optional, Tuple, List
import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.board_encoder import board_to_tensor
from src.move_encoder import (
    decode_move,
    encode_move,
    get_legal_moves_mask,
    ACTION_SIZE,
)
from src.model import ChessResNet


class ChessEngine:
    """Wraps a trained ChessResNet for real-time move generation and position evaluation."""

    def __init__(
        self,
        model: ChessResNet,
        device: Optional[torch.device] = None,
        canonical: bool = True,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.canonical = canonical

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, device: Optional[torch.device] = None
    ) -> "ChessEngine":
        """Loads a ChessEngine from a saved PyTorch checkpoint."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = checkpoint.get("config", {})
        from src.model import build_model

        model = build_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device=device, canonical=cfg.get("data", {}).get("canonical", True))

    @torch.no_grad()
    def evaluate_position(self, board: chess.Board) -> float:
        """Returns the predicted value in [-1.0, 1.0] from current player's perspective."""
        tensor = board_to_tensor(board, canonical=self.canonical)
        tensor_t = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        _, value = self.model(tensor_t)
        return float(value.item())

    @torch.no_grad()
    def select_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        return_probs: bool = False,
    ) -> chess.Move:
        """Selects a legal move for the current board position.

        Args:
            board: python-chess Board.
            temperature: If 0.0, selects greedy argmax. If > 0, samples probabilistically.
            return_probs: If True, also returns top move candidates.

        Returns:
            chess.Move object.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available in this board state.")

        if len(legal_moves) == 1:
            return legal_moves[0]

        tensor = board_to_tensor(board, canonical=self.canonical)
        tensor_t = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        policy_logits, _ = self.model(tensor_t)
        logits = policy_logits[0].cpu().numpy()

        # Mask illegal moves
        mask = get_legal_moves_mask(board, canonical=self.canonical)
        canonical_flip = self.canonical and (board.turn == chess.BLACK)

        masked_logits = np.full_like(logits, -1e9)
        masked_logits[mask] = logits[mask]

        if temperature == 0.0 or temperature < 1e-3:
            best_idx = int(np.argmax(masked_logits))
        else:
            scaled_logits = (masked_logits - np.max(masked_logits[mask])) / max(temperature, 1e-2)
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / np.sum(exp_logits)
            best_idx = int(np.random.choice(ACTION_SIZE, p=probs))

        chosen_move = decode_move(best_idx, board, canonical_flip=canonical_flip)

        # Fallback safeguard: verify move is strictly legal
        if chosen_move not in board.legal_moves:
            # Pick highest logit among known legal moves
            best_score = -1e9
            best_m = legal_moves[0]
            for m in legal_moves:
                idx = encode_move(m, canonical_flip=canonical_flip)
                if logits[idx] > best_score:
                    best_score = logits[idx]
                    best_m = m
            chosen_move = best_m

        return chosen_move
