import chess

from src.sft.formatting_sft import add_messages_with_reasoning_trace, position_with_eval_to_distill_analysis
from src.utils.chess_utils import get_first_legal_move, get_legal_moves_uci, render_board_utf


def test_sft_eval_bridge_builds_messages_and_probs():
    board = chess.Board()
    legal_moves_uci = get_legal_moves_uci(board)
    first_legal = get_first_legal_move(board) or ""

    # Minimal eval-style payload as produced by `data_processing_with_eval`.
    example = {
        "fen": board.fen(),
        "legal_moves_uci": legal_moves_uci,
        "target_move_uci": "e2e4",
        "first_legal_move": first_legal,
        "board_utf": render_board_utf(board),
        "side_to_move": "White",
        "source": "game",
        "best_move_uci": "e2e4",
        "best_move_san": "e4",
        "best_score_cp": 20,
        "move_evaluations": [
            {"uci": "e2e4", "san": "e4", "centipawn": 20, "mate_in": None},
            {"uci": "d2d4", "san": "d4", "centipawn": 10, "mate_in": None},
        ],
    }

    analysis = position_with_eval_to_distill_analysis(example, min_probability=0.001, stockfish_temperature=100.0)
    assert analysis.best_move_uci == "e2e4"
    assert abs(sum(analysis.move_probs.values()) - 1.0) < 1e-6
    assert analysis.move_probs.get("e2e4", 0.0) > analysis.move_probs.get("a2a3", 0.0)

    messages = add_messages_with_reasoning_trace(
        example,
        reasoning_trace_generator=None,
        include_board=False,
        always_choose_best_move=True,
        min_probability=0.001,
        stockfish_temperature=100.0,
    )["messages"]
    assert len(messages) == 2
    assert "<uci_move>" in messages[1]["content"]
    assert "</uci_move>" in messages[1]["content"]
    assert "e2e4" in messages[1]["content"]

