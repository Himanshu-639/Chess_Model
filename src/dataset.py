"""Dataset utilities and PGN processing for Chess Supervised Learning.

Handles:
1. Converting PGN games into (board_tensor, move_target, value_target) triples.
2. High-performance NPZ sharded dataset for fast multi-epoch GPU training.
3. Synthetic PGN generation for local testing and pipeline verification.
"""

import glob
import os
import random
from typing import Dict, Generator, List, Optional, Tuple
import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

from src.board_encoder import board_to_tensor
from src.move_encoder import encode_move


def parse_game_result(result_str: str) -> Optional[float]:
    """Parses PGN Result tag into outcome value for White.

    1-0 -> 1.0 (White won)
    0-1 -> -1.0 (Black won)
    1/2-1/2 -> 0.0 (Draw)
    """
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return -1.0
    elif result_str == "1/2-1/2":
        return 0.0
    return None


def extract_samples_from_game(
    game: chess.pgn.Game, canonical: bool = True
) -> List[Tuple[np.ndarray, int, float]]:
    """Extracts (board_tensor, move_index, value) tuples from a chess.pgn.Game.

    Args:
        game: A python-chess Game object.
        canonical: If True, values and boards are canonicalized to player perspective.

    Returns:
        List of (tensor, move_idx, value) tuples.
    """
    result_str = game.headers.get("Result", "*")
    white_outcome = parse_game_result(result_str)
    if white_outcome is None:
        return []

    samples = []
    board = game.board()

    for move in game.mainline_moves():
        # Current player perspective value
        if canonical:
            val = white_outcome if board.turn == chess.WHITE else -white_outcome
            canonical_flip = board.turn == chess.BLACK
        else:
            val = white_outcome
            canonical_flip = False

        board_tensor = board_to_tensor(board, canonical=canonical)
        move_idx = encode_move(move, canonical_flip=canonical_flip)
        samples.append((board_tensor, move_idx, val))

        board.push(move)

    return samples


class InMemoryChessDataset(Dataset):
    """Simple in-memory dataset for verified test batches and small experiments."""

    def __init__(
        self,
        boards: np.ndarray,
        moves: np.ndarray,
        values: np.ndarray,
    ):
        self.boards = torch.from_numpy(boards).float()
        self.moves = torch.from_numpy(moves).long()
        self.values = torch.from_numpy(values).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.moves[idx], self.values[idx]


class ShardedNpzChessDataset(Dataset):
    """Memory-efficient sharded NPZ dataset for training on millions of positions.

    Loads shards on-demand or pre-indexes shard boundaries.
    """

    def __init__(self, data_dir: str):
        self.shard_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.shard_files:
            raise FileNotFoundError(f"No .npz shard files found in {data_dir}")

        self.shard_lengths = []
        for sf in self.shard_files:
            with np.load(sf) as data:
                self.shard_lengths.append(len(data["moves"]))

        self.cumulative_lengths = np.cumsum(self.shard_lengths)
        self.total_samples = int(self.cumulative_lengths[-1])

        self.current_shard_idx = -1
        self.current_data = None

    def __len__(self) -> int:
        return self.total_samples

    def _load_shard(self, shard_idx: int):
        if self.current_shard_idx != shard_idx:
            data = np.load(self.shard_files[shard_idx])
            self.current_data = {
                "boards": data["boards"],
                "moves": data["moves"],
                "values": data["values"],
            }
            self.current_shard_idx = shard_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        local_idx = idx if shard_idx == 0 else idx - int(self.cumulative_lengths[shard_idx - 1])

        self._load_shard(shard_idx)
        board = torch.from_numpy(self.current_data["boards"][local_idx]).float()
        move = torch.tensor(self.current_data["moves"][local_idx], dtype=torch.long)
        value = torch.tensor([self.current_data["values"][local_idx]], dtype=torch.float32)

        return board, move, value


def generate_synthetic_pgn(
    output_path: str, num_games: int = 50, max_moves_per_game: int = 60
) -> str:
    """Generates synthetic chess games and saves to PGN format.

    Enables immediate out-of-the-box verification without downloading datasets.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for game_id in range(num_games):
            game = chess.pgn.Game()
            game.headers["Event"] = f"Synthetic Verification Match {game_id + 1}"
            game.headers["White"] = "SyntheticBot_Alpha"
            game.headers["Black"] = "SyntheticBot_Beta"

            board = game.board()
            node = game

            # Play moves using random legal moves with piece capture preference
            for _ in range(max_moves_per_game):
                if board.is_game_over():
                    break
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                # Bias towards captures to make realistic chess moves
                captures = [m for m in legal_moves if board.is_capture(m)]
                if captures and random.random() < 0.6:
                    chosen_move = random.choice(captures)
                else:
                    chosen_move = random.choice(legal_moves)

                board.push(chosen_move)
                node = node.add_variation(chosen_move)

            # Assign valid game outcome
            if board.is_checkmate():
                result = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                result = "1/2-1/2"
            else:
                # Assign outcome based on piece count heuristic
                white_material = sum(
                    len(board.pieces(pt, chess.WHITE)) * weight
                    for pt, weight in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)]
                )
                black_material = sum(
                    len(board.pieces(pt, chess.BLACK)) * weight
                    for pt, weight in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)]
                )
                if white_material > black_material + 2:
                    result = "1-0"
                elif black_material > white_material + 2:
                    result = "0-1"
                else:
                    result = "1/2-1/2"

            game.headers["Result"] = result
            f.write(str(game) + "\n\n")

    return output_path
