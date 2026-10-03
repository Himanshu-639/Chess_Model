"""Evaluation and benchmarking module for the CNN Chess model.

Runs matches against baseline bots:
1. Uniform Random player.
2. Material Greedy player (evaluates captures and basic piece values).
"""

import random
from typing import Callable, Dict, Any, Tuple
import chess

from src.engine import ChessEngine


def random_player(board: chess.Board) -> chess.Move:
    """Selects a legal move uniformly at random."""
    return random.choice(list(board.legal_moves))


def greedy_material_player(board: chess.Board) -> chess.Move:
    """Selects move maximizing immediate material advantage."""
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    legal_moves = list(board.legal_moves)
    best_move = legal_moves[0]
    best_eval = -999999

    for move in legal_moves:
        board.push(move)
        # Material evaluation from current perspective
        my_mat = sum(
            len(board.pieces(pt, board.turn == False)) * val
            for pt, val in piece_values.items()
        )
        opp_mat = sum(
            len(board.pieces(pt, board.turn == True)) * val
            for pt, val in piece_values.items()
        )
        eval_score = my_mat - opp_mat
        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move
        board.pop()

    return best_move


def play_game(
    white_player: Callable[[chess.Board], chess.Move],
    black_player: Callable[[chess.Board], chess.Move],
    max_moves: int = 150,
) -> Tuple[str, int]:
    """Plays a single game between two bot functions.

    Returns:
        Tuple of (result_string: '1-0', '0-1', '1/2-1/2', move_count)
    """
    board = chess.Board()
    ply = 0

    while not board.is_game_over() and ply < max_moves:
        if board.turn == chess.WHITE:
            move = white_player(board)
        else:
            move = black_player(board)

        if move not in board.legal_moves:
            # Forfeit on illegal move
            return ("0-1" if board.turn == chess.WHITE else "1-0"), ply

        board.push(move)
        ply += 1

    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), ply
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw() or ply >= max_moves:
        return "1/2-1/2", ply

    return "1/2-1/2", ply


def evaluate_tournament(
    engine: ChessEngine,
    opponent_type: str = "random",
    num_games: int = 10,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Evaluates the neural engine across multiple games against a specified opponent."""
    if opponent_type == "random":
        opp_fn = random_player
    elif opponent_type == "greedy":
        opp_fn = greedy_material_player
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    wins = 0
    draws = 0
    losses = 0
    total_plies = 0

    engine_fn = lambda b: engine.select_move(b, temperature=temperature)

    for game_idx in range(num_games):
        # Alternate colors: even games engine is White, odd games engine is Black
        engine_is_white = (game_idx % 2 == 0)
        if engine_is_white:
            result, plies = play_game(white_player=engine_fn, black_player=opp_fn)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            result, plies = play_game(white_player=opp_fn, black_player=engine_fn)
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1

        total_plies += plies

    win_rate = (wins + 0.5 * draws) / num_games * 100.0
    return {
        "opponent": opponent_type,
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": win_rate,
        "avg_plies": total_plies / num_games,
    }
