"""Board representation encoder for Chess Supervised Learning.

Converts a python-chess Board state into an 18 x 8 x 8 tensor.
Supports canonical perspective (rotating board so current player is always White),
which makes learning 2x more sample efficient.
"""

from typing import Tuple
import chess
import numpy as np

NUM_PLANES = 18

# Piece order for planes: Pawns, Knights, Bishops, Rooks, Queens, King
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_tensor(board: chess.Board, canonical: bool = True) -> np.ndarray:
    """Converts a chess.Board to an (18, 8, 8) float32 numpy array.

    Planes description:
    0-5: Current player's pieces (P, N, B, R, Q, K)
    6-11: Opponent's pieces (P, N, B, R, Q, K)
    12: Current player Kingside castling right (all 1s if available, else 0)
    13: Current player Queenside castling right
    14: Opponent Kingside castling right
    15: Opponent Queenside castling right
    16: En passant target square (1 at square, else 0)
    17: Halfmove clock normalized by 100 (for 50-move rule awareness)

    Args:
        board: python-chess Board object.
        canonical: If True, mirror the board when it's Black's turn so that
          the active player is always White.

    Returns:
        np.ndarray of shape (18, 8, 8) with dtype float32.
    """
    is_black_turn = board.turn == chess.BLACK
    flip = canonical and is_black_turn

    b = board.mirror() if flip else board
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # 1. Piece positions
    for p_idx, piece_type in enumerate(PIECE_TYPES):
        # Current player pieces (always WHITE in perspective of b)
        for sq in b.pieces(piece_type, chess.WHITE):
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            tensor[p_idx, r, f] = 1.0

        # Opponent pieces (always BLACK in perspective of b)
        for sq in b.pieces(piece_type, chess.BLACK):
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            tensor[6 + p_idx, r, f] = 1.0

    # 2. Castling rights
    if b.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    # 3. En passant square
    if b.ep_square is not None:
        ep_r = chess.square_rank(b.ep_square)
        ep_f = chess.square_file(b.ep_square)
        tensor[16, ep_r, ep_f] = 1.0

    # 4. Halfmove clock (rule of 50 moves)
    tensor[17, :, :] = min(b.halfmove_clock, 100) / 100.0

    return tensor
