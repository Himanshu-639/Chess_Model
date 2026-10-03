"""Unit and integration test suite for the Chess CNN pipeline."""

import os
import shutil
import unittest
import chess
import numpy as np
import torch

from src.board_encoder import board_to_tensor
from src.move_encoder import encode_move, decode_move, get_legal_moves_mask, ACTION_SIZE
from src.model import ChessResNet, count_parameters
from src.dataset import generate_synthetic_pgn
from src.preprocess import preprocess_pgn_file
from src.engine import ChessEngine


class TestChessPipeline(unittest.TestCase):

    def test_board_encoder_shape_and_channels(self):
        board = chess.Board()
        tensor_w = board_to_tensor(board, canonical=True)
        self.assertEqual(tensor_w.shape, (18, 8, 8))
        self.assertEqual(tensor_w.dtype, np.float32)

        # White pawns on rank 2 (index 1)
        self.assertEqual(np.sum(tensor_w[0, 1, :]), 8.0)
        # Black pawns on rank 7 (index 6)
        self.assertEqual(np.sum(tensor_w[6, 6, :]), 8.0)

    def test_canonical_perspective_flip(self):
        # When Black moves in canonical mode, Black's pieces should appear in planes 0-5
        board = chess.Board()
        board.push_san("e4")  # Now it's Black's turn
        tensor_b = board_to_tensor(board, canonical=True)
        self.assertEqual(tensor_b.shape, (18, 8, 8))
        # Black pawns (now active player) should be on rank index 1 in mirrored coordinates
        self.assertEqual(np.sum(tensor_b[0, 1, :]), 8.0)

    def test_move_encoder_roundtrip(self):
        board = chess.Board()
        for move in board.legal_moves:
            idx = encode_move(move, canonical_flip=False)
            self.assertTrue(0 <= idx < ACTION_SIZE)
            reconstructed = decode_move(idx, board, canonical_flip=False)
            self.assertEqual(move, reconstructed)

    def test_promotion_encoding(self):
        board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        for move in board.legal_moves:
            idx = encode_move(move, canonical_flip=False)
            reconstructed = decode_move(idx, board, canonical_flip=False)
            self.assertEqual(move, reconstructed)

    def test_model_forward_backward(self):
        model = ChessResNet(
            in_channels=18,
            num_filters=32,
            num_blocks=2,
            num_actions=ACTION_SIZE,
            policy_head_filters=8,
            value_head_filters=2,
            value_hidden_dim=64,
        )
        dummy_x = torch.randn(2, 18, 8, 8)
        p_logits, v_pred = model(dummy_x)

        self.assertEqual(p_logits.shape, (2, ACTION_SIZE))
        self.assertEqual(v_pred.shape, (2, 1))
        self.assertTrue((v_pred >= -1.0).all() and (v_pred <= 1.0).all())

        loss = p_logits.sum() + v_pred.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_legal_mask_and_engine(self):
        model = ChessResNet(
            in_channels=18,
            num_filters=32,
            num_blocks=2,
            num_actions=ACTION_SIZE,
            policy_head_filters=8,
            value_head_filters=2,
            value_hidden_dim=64,
        )
        engine = ChessEngine(model, device=torch.device("cpu"), canonical=True)
        board = chess.Board()

        # Engine must pick legal moves for 10 consecutive turns
        for _ in range(10):
            if board.is_game_over():
                break
            move = engine.select_move(board, temperature=0.0)
            self.assertIn(move, board.legal_moves)
            board.push(move)

    def test_sharded_batch_sampler(self):
        from src.dataset import ShardedBatchSampler
        cumulative_lengths = np.array([100, 250, 400])
        sampler = ShardedBatchSampler(cumulative_lengths, batch_size=32, shuffle=True)
        batches = list(sampler)
        self.assertTrue(len(batches) > 0)
        # All indices must be within [0, 400)
        all_indices = [idx for b in batches for idx in b]
        self.assertEqual(len(all_indices), 400)
        self.assertEqual(set(all_indices), set(range(400)))


if __name__ == "__main__":
    unittest.main()
