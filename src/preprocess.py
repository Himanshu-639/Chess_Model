"""Preprocessing pipeline: Converts raw PGN games into fast NPZ training shards."""

import argparse
import os
import random
from typing import List, Tuple
import chess.pgn
import numpy as np
from tqdm import tqdm

from src.dataset import extract_samples_from_game


def save_shard(
    boards: List[np.ndarray],
    moves: List[int],
    values: List[float],
    output_path: str,
):
    """Saves accumulated samples into a compressed NPZ file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        boards=np.array(boards, dtype=np.float32),
        moves=np.array(moves, dtype=np.int32),
        values=np.array(values, dtype=np.float32),
    )


def preprocess_pgn_file(
    pgn_path: str,
    output_dir: str,
    max_positions_per_shard: int = 50000,
    val_split: float = 0.1,
    max_games: int = -1,
    canonical: bool = True,
) -> Tuple[int, int]:
    """Parses a PGN file and writes train/val NPZ shards.

    Args:
        pgn_path: Path to the input .pgn file.
        output_dir: Base directory to store output shards.
        max_positions_per_shard: Maximum number of positions per .npz file.
        val_split: Fraction of samples to allocate to validation.
        max_games: Maximum number of games to process (-1 for all).
        canonical: Whether to canonicalize board orientations.

    Returns:
        Tuple of (train_sample_count, val_sample_count).
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_boards, train_moves, train_values = [], [], []
    val_boards, val_moves, val_values = [], [], []

    train_shard_idx = 0
    val_shard_idx = 0
    total_train_samples = 0
    total_val_samples = 0
    games_processed = 0

    print(f"[Preprocessing] Reading {pgn_path} ...")
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            if max_games > 0 and games_processed >= max_games:
                break
            try:
                game = chess.pgn.read_game(f)
            except Exception as e:
                print(f"[Preprocessing Warning] Error reading game: {e}")
                continue

            if game is None:
                break

            samples = extract_samples_from_game(game, canonical=canonical)
            if not samples:
                continue

            games_processed += 1
            # Allocate to val or train
            is_val = random.random() < val_split

            for b_tensor, m_idx, v_val in samples:
                if is_val:
                    val_boards.append(b_tensor)
                    val_moves.append(m_idx)
                    val_values.append(v_val)
                    total_val_samples += 1
                    if len(val_moves) >= max_positions_per_shard:
                        save_shard(
                            val_boards,
                            val_moves,
                            val_values,
                            os.path.join(val_dir, f"shard_{val_shard_idx:05d}.npz"),
                        )
                        val_shard_idx += 1
                        val_boards, val_moves, val_values = [], [], []
                else:
                    train_boards.append(b_tensor)
                    train_moves.append(m_idx)
                    train_values.append(v_val)
                    total_train_samples += 1
                    if len(train_moves) >= max_positions_per_shard:
                        save_shard(
                            train_boards,
                            train_moves,
                            train_values,
                            os.path.join(train_dir, f"shard_{train_shard_idx:05d}.npz"),
                        )
                        train_shard_idx += 1
                        train_boards, train_moves, train_values = [], [], []

    # Flush remaining samples
    if train_moves:
        save_shard(
            train_boards,
            train_moves,
            train_values,
            os.path.join(train_dir, f"shard_{train_shard_idx:05d}.npz"),
        )
    if val_moves:
        save_shard(
            val_boards,
            val_moves,
            val_values,
            os.path.join(val_dir, f"shard_{val_shard_idx:05d}.npz"),
        )

    print(
        f"[Preprocessing Complete] Games: {games_processed} | Train Samples: {total_train_samples} | Val Samples: {total_val_samples}"
    )
    return total_train_samples, total_val_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PGN games to NPZ shards.")
    parser.add_argument("--pgn", type=str, required=True, help="Input PGN file")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=50000, help="Positions per shard")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--max-games", type=int, default=-1, help="Limit games (-1 for all)")
    args = parser.parse_args()

    preprocess_pgn_file(
        args.pgn,
        args.out,
        max_positions_per_shard=args.shard_size,
        val_split=args.val_split,
        max_games=args.max_games,
    )
