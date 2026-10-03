"""Unified runner script for the Supervised CNN Chess Pipeline.

Usage Modes:
1. Verification / Self-Test (local instant validation):
   python run_pipeline.py --mode verify

2. PGN Preprocessing (prepare large PGN for fast training):
   python run_pipeline.py --mode preprocess --pgn lichess_elite.pgn --out data/processed

3. Full Training (train model on full dataset with all checks & logging):
   python run_pipeline.py --mode train --config config.yaml

4. Evaluation (benchmark trained model against baseline bots):
   python run_pipeline.py --mode evaluate --checkpoint checkpoints/best_model.pt --games 20

5. Interactive Play (watch or play against the model):
   python run_pipeline.py --mode play --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import sys
import time
import yaml
import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.board_encoder import board_to_tensor
from src.move_encoder import encode_move, decode_move, get_legal_moves_mask, ACTION_SIZE
from src.model import build_model, count_parameters, ChessResNet
from src.dataset import (
    generate_synthetic_pgn,
    ShardedNpzChessDataset,
    InMemoryChessDataset,
    ShardedBatchSampler,
)
from src.preprocess import preprocess_pgn_file
from src.health_checks import (
    run_environment_check,
    run_model_forward_backward_check,
    run_legal_masking_check,
    run_overfit_capacity_check,
    TrainingAnomalyError,
)
from src.trainer import ChessTrainer
from src.engine import ChessEngine
from src.evaluator import evaluate_tournament, play_game, random_player


def load_yaml_config(config_path: str) -> dict:
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"[Warning] Config {config_path} not found. Using default parameters.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mode_verify(args):
    """Executes an end-to-end self-test of all components on a generated micro-dataset."""
    print("=" * 70)
    print(" [STAGE 1] SYSTEM ENVIRONMENT & GPU DETECTION")
    print("=" * 70)
    env_info = run_environment_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print(" [STAGE 2] ARCHITECTURE & INTEGRITY SANITY CHECKS")
    print("=" * 70)
    # Instantiate small verification model (3 blocks, 64 filters for rapid test)
    test_config = {
        "model": {
            "in_channels": 18,
            "num_filters": 64,
            "num_blocks": 3,
            "num_actions": ACTION_SIZE,
            "policy_head_filters": 16,
            "value_head_filters": 2,
            "value_hidden_dim": 128,
        },
        "training": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "clip_grad_norm": 1.0,
            "early_stopping_patience": 3,
        },
        "health_checks": {
            "max_loss_threshold": 50.0,
        },
        "output": {
            "checkpoint_dir": "checkpoints/test",
            "log_dir": "logs/test",
            "best_model_name": "test_best.pt",
            "latest_model_name": "test_latest.pt",
        },
    }

    model = build_model(test_config).to(device)
    print(f" -> Verification Model Instantiated with {count_parameters(model):,} parameters.")

    run_model_forward_backward_check(model, device)
    run_legal_masking_check(model, device)
    run_overfit_capacity_check(model, device, iterations=30)

    print("\n" + "=" * 70)
    print(" [STAGE 3] SYNTHETIC DATASET GENERATION & PREPROCESSING")
    print("=" * 70)
    test_pgn_path = "data/test_games.pgn"
    test_proc_dir = "data/test_processed"
    print(f" -> Generating synthetic chess games at {test_pgn_path} ...")
    generate_synthetic_pgn(test_pgn_path, num_games=15, max_moves_per_game=50)

    print(f" -> Preprocessing synthetic PGN into NPZ shards at {test_proc_dir} ...")
    tr_samples, val_samples = preprocess_pgn_file(
        test_pgn_path,
        test_proc_dir,
        max_positions_per_shard=2000,
        val_split=0.2,
    )
    assert tr_samples > 0, "No training samples generated!"

    train_ds = ShardedNpzChessDataset(os.path.join(test_proc_dir, "train"))
    val_ds = ShardedNpzChessDataset(os.path.join(test_proc_dir, "val"))

    train_sampler = ShardedBatchSampler(train_ds.cumulative_lengths, batch_size=32, shuffle=True)
    val_sampler = ShardedBatchSampler(val_ds.cumulative_lengths, batch_size=32, shuffle=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
    print(f" -> DataLoaders active: Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    print("\n" + "=" * 70)
    print(" [STAGE 4] VERIFYING RESILIENT TRAINING & METRIC LOGGING")
    print("=" * 70)
    trainer = ChessTrainer(model, train_loader, val_loader, test_config, device)
    train_result = trainer.train()

    best_ckpt = os.path.join(test_config["output"]["checkpoint_dir"], "test_best.pt")
    assert os.path.exists(best_ckpt), f"Checkpoint was not created at {best_ckpt}"
    print(f" -> Training test passed. Checkpoint verified at: {best_ckpt}")

    print("\n" + "=" * 70)
    print(" [STAGE 5] EVALUATION BENCHMARK & ENGINE INFERENCE")
    print("=" * 70)
    engine = ChessEngine.load_from_checkpoint(best_ckpt, device=device)
    print(" -> ChessEngine successfully restored from checkpoint.")

    # Play a match against Random bot
    print(" -> Running 4-game benchmark tournament against Random bot...")
    tourney_stats = evaluate_tournament(engine, opponent_type="random", num_games=4)
    print(f" -> Tournament Results: {tourney_stats}")

    # Play demo game in console
    print("\n -> Demonstration Self-Play Mini Game:")
    board = chess.Board()
    for move_num in range(1, 7):
        if board.is_game_over():
            break
        m_w = engine.select_move(board, temperature=0.1)
        board.push(m_w)
        if board.is_game_over():
            break
        m_b = engine.select_move(board, temperature=0.1)
        board.push(m_b)
        print(f"    Move {move_num}: White: {m_w.uci()} | Black: {m_b.uci()}")

    print("\n" + "=" * 70)
    print(" [VERIFICATION COMPLETE] ALL PIPELINE COMPONENTS VERIFIED SUCCESSFULLY!")
    print("=" * 70)
    print(" Summary:")
    print("  1. Board & Move Encoders: PASS (Canonical, 100% legal mask)")
    print("  2. ResNet Dual-Head CNN:  PASS (Forward, Backward, Gradient health)")
    print("  3. Learning Capacity:     PASS (Loss reduction confirmed)")
    print("  4. Sharded Data Pipeline: PASS (PGN -> NPZ -> DataLoader)")
    print("  5. Trainer & Safeguards:  PASS (AMP, Anomaly checks, CSV logging)")
    print("  6. Engine Inference:      PASS (Autonomous legal chess gameplay)")
    print("=" * 70)


def mode_preprocess(args):
    """Preprocesses a raw PGN file into NPZ shards."""
    if not args.pgn or not os.path.exists(args.pgn):
        print(f"[Error] PGN file not found: {args.pgn}")
        sys.exit(1)

    out_dir = args.out if args.out else "data/processed"
    preprocess_pgn_file(
        args.pgn,
        out_dir,
        max_positions_per_shard=args.shard_size,
        val_split=args.val_split,
        max_games=args.max_games,
    )


def mode_train(args):
    """Executes full model training on preprocessed dataset."""
    config_path = args.config if args.config else "config.yaml"
    config = load_yaml_config(config_path)

    # Command line overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    train_data_dir = args.train_data or config.get("data", {}).get("train_data_dir", "data/processed/train")
    val_data_dir = args.val_data or config.get("data", {}).get("val_data_dir", "data/processed/val")

    if not os.path.exists(train_data_dir):
        print(f"[Error] Training data directory not found: {train_data_dir}")
        print("Please run preprocessing first using: python run_pipeline.py --mode preprocess --pgn <your_pgn>")
        sys.exit(1)

    # Pre-flight health checks
    run_environment_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    print(f"[Model Architecture] Trainable parameters: {count_parameters(model):,}")

    if config.get("health_checks", {}).get("gradient_flow_test", True):
        run_model_forward_backward_check(model, device)
        run_legal_masking_check(model, device)

    # Datasets and Loaders
    print(f"[Data Loading] Loading shards from {train_data_dir} ...")
    train_ds = ShardedNpzChessDataset(train_data_dir)
    val_ds = ShardedNpzChessDataset(val_data_dir) if os.path.exists(val_data_dir) else None

    batch_size = config.get("training", {}).get("batch_size", 256)
    train_sampler = ShardedBatchSampler(train_ds.cumulative_lengths, batch_size=batch_size, shuffle=True)
    val_sampler = (
        ShardedBatchSampler(val_ds.cumulative_lengths, batch_size=batch_size, shuffle=False)
        if val_ds
        else None
    )

    num_workers = config.get("training", {}).get("num_workers", 4)
    # On Windows or CPU, set num_workers=0 to eliminate multi-process RAM duplication
    if device.type == "cpu" or sys.platform.startswith("win32"):
        num_workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if val_ds
        else DataLoader([])
    )

    trainer = ChessTrainer(model, train_loader, val_loader, config, device)

    # Optional Resume
    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading weights from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    try:
        trainer.train()
    except TrainingAnomalyError as e:
        print(f"\n[FATAL] Training halted: {e}")
        sys.exit(2)


def mode_evaluate(args):
    """Benchmarks a trained model against baseline bots."""
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Loading model from {args.checkpoint} on {device} ...")
    engine = ChessEngine.load_from_checkpoint(args.checkpoint, device=device)

    num_games = args.games if args.games else 20
    opp_type = args.opponent if args.opponent else "random"

    print(f"[Evaluation] Playing {num_games} matches against '{opp_type}' player...")
    stats = evaluate_tournament(engine, opponent_type=opp_type, num_games=num_games)
    print("-" * 50)
    print(f" Opponent:     {stats['opponent']}")
    print(f" Total Games:  {stats['games']}")
    print(f" Wins:         {stats['wins']}")
    print(f" Draws:        {stats['draws']}")
    print(f" Losses:       {stats['losses']}")
    print(f" Score %:      {stats['score_pct']:.1f}%")
    print(f" Avg Plies:    {stats['avg_plies']:.1f}")
    print("-" * 50)


def mode_play(args):
    """Plays an interactive or demo match in terminal."""
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = ChessEngine.load_from_checkpoint(args.checkpoint, device=device)
    board = chess.Board()

    print("\n=== Chess Model Terminal Game ===")
    print("Commands: Enter move in UCI format (e.g., 'e2e4'), or 'quit'.")

    while not board.is_game_over():
        print(board)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

        if args.interactive and board.turn == chess.WHITE:
            user_input = input("Your move: ").strip()
            if user_input.lower() == "quit":
                break
            try:
                move = chess.Move.from_uci(user_input)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    continue
            except Exception:
                print("Invalid UCI format. Try again.")
                continue
        else:
            comp_move = engine.select_move(board, temperature=0.0)
            eval_score = engine.evaluate_position(board)
            print(f"Engine played: {comp_move.uci()} (Eval: {eval_score:+.3f})")
            board.push(comp_move)

    print("\nGame Over! Result:", board.result())


def main():
    parser = argparse.ArgumentParser(description="Supervised CNN Chess Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="verify",
        choices=["verify", "preprocess", "train", "evaluate", "play"],
        help="Pipeline mode to execute",
    )
    # Preprocess flags
    parser.add_argument("--pgn", type=str, help="Path to input PGN file")
    parser.add_argument("--out", type=str, help="Output directory for processed shards")
    parser.add_argument("--shard-size", type=int, default=50000, help="Samples per shard")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--max-games", type=int, default=-1, help="Max games to preprocess")

    # Training flags
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML config file")
    parser.add_argument("--train-data", type=str, help="Path to training NPZ directory")
    parser.add_argument("--val-data", type=str, help="Path to validation NPZ directory")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume")

    # Evaluation / Play flags
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--games", type=int, default=20, help="Number of evaluation games")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy"])
    parser.add_argument("--interactive", action="store_true", help="Play interactively as White")

    args = parser.parse_args()

    if args.mode == "verify":
        mode_verify(args)
    elif args.mode == "preprocess":
        mode_preprocess(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "evaluate":
        mode_evaluate(args)
    elif args.mode == "play":
        mode_play(args)


if __name__ == "__main__":
    main()
