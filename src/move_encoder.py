"""Move encoder and decoder for Chess Supervised Learning.

Maps chess moves to discrete integer indices and provides legal move masking.
Supports 16,384 discrete actions (64x64 standard moves + underpromotions).
"""

from typing import List, Optional
import chess
import numpy as np

ACTION_SIZE = 16384  # 4 * 64 * 64


def encode_move(move: chess.Move, canonical_flip: bool = False) -> int:
    """Encodes a chess.Move to an action index in [0, ACTION_SIZE).

    Args:
        move: python-chess Move.
        canonical_flip: If True, mirror the move squares across ranks
          (sq ^ 56), matching canonical board perspective.

    Returns:
        Integer action index in [0, 16383].
    """
    f = chess.square_mirror(move.from_square) if canonical_flip else move.from_square
    t = chess.square_mirror(move.to_square) if canonical_flip else move.to_square

    base = f * 64 + t
    promo = move.promotion

    if promo is None or promo == chess.QUEEN:
        return base
    elif promo == chess.KNIGHT:
        return 4096 + base
    elif promo == chess.BISHOP:
        return 2 * 4096 + base
    elif promo == chess.ROOK:
        return 3 * 4096 + base
    return base


def decode_move(idx: int, board: chess.Board, canonical_flip: bool = False) -> chess.Move:
    """Decodes an action index back into a chess.Move.

    Args:
        idx: Integer action index in [0, 16383].
        board: The chess.Board in current state.
        canonical_flip: If True, reverses canonical mirroring back to actual board squares.

    Returns:
        chess.Move object.
    """
    plane = idx // 4096
    base = idx % 4096
    f = base // 64
    t = base % 64

    if canonical_flip:
        actual_f = chess.square_mirror(f)
        actual_t = chess.square_mirror(t)
    else:
        actual_f = f
        actual_t = t

    # Check if this move requires pawn promotion
    piece = board.piece_at(actual_f)
    promo = None
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(actual_t)
        if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
            if plane == 0:
                promo = chess.QUEEN
            elif plane == 1:
                promo = chess.KNIGHT
            elif plane == 2:
                promo = chess.BISHOP
            elif plane == 3:
                promo = chess.ROOK

    return chess.Move(actual_f, actual_t, promotion=promo)


def get_legal_moves_mask(board: chess.Board, canonical: bool = True) -> np.ndarray:
    """Generates a boolean mask of legal moves for the given board.

    Args:
        board: python-chess Board.
        canonical: Whether canonical perspective is active.

    Returns:
        np.ndarray of shape (ACTION_SIZE,) with dtype bool (True for legal moves).
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    canonical_flip = canonical and (board.turn == chess.BLACK)

    for move in board.legal_moves:
        action_idx = encode_move(move, canonical_flip=canonical_flip)
        mask[action_idx] = True

    return mask


def get_legal_move_indices(board: chess.Board, canonical: bool = True) -> List[int]:
    """Returns a list of integer indices for all currently legal moves."""
    canonical_flip = canonical and (board.turn == chess.BLACK)
    return [encode_move(m, canonical_flip=canonical_flip) for m in board.legal_moves]
