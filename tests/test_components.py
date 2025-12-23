#!/usr/bin/env python3
"""
Unit tests for the lightweight helpers used by training/eval scripts.

Run:
  python -m unittest tests.test_components -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import chess

# Ensure repo root is importable as a package (so `import src...` works).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.chess_utils import (
    extract_uci_from_response,
    get_first_legal_move,
    get_legal_moves_uci,
    parse_movetext,
    position_from_board,
    render_board_utf,
    validate_uci_move,
)
from src.utils.formatting import position_to_messages


class TestChessUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.starting_board = chess.Board()

    def test_render_board_utf_starting(self) -> None:
        rendered = render_board_utf(self.starting_board)
        self.assertIn("K", rendered)
        self.assertIn("k", rendered)
        self.assertIn("P", rendered)
        self.assertIn("p", rendered)
        self.assertIn("a", rendered)
        self.assertIn("8", rendered)

    def test_parse_movetext_basic(self) -> None:
        moves = parse_movetext("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6")
        self.assertEqual(moves, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])

    def test_parse_movetext_with_result(self) -> None:
        moves = parse_movetext("1. e4 e5 2. Nf3 1-0")
        self.assertEqual(moves, ["e4", "e5", "Nf3"])

    def test_parse_movetext_with_clock(self) -> None:
        moves = parse_movetext("1. e4 { [%clk 0:05:00] } e5 { [%clk 0:05:00] } 2. Nf3")
        self.assertEqual(moves, ["e4", "e5", "Nf3"])

    def test_get_legal_moves_uci(self) -> None:
        moves = get_legal_moves_uci(self.starting_board).split()
        self.assertEqual(len(moves), 20)
        self.assertIn("e2e4", moves)
        self.assertIn("g1f3", moves)

    def test_get_first_legal_move(self) -> None:
        first_move = get_first_legal_move(self.starting_board)
        self.assertIsNotNone(first_move)
        self.assertRegex(first_move, r"^[a-h][1-8][a-h][1-8][qrbn]?$")

    def test_validate_uci_move_valid(self) -> None:
        self.assertTrue(validate_uci_move(self.starting_board, "e2e4"))
        self.assertTrue(validate_uci_move(self.starting_board, "g1f3"))

    def test_validate_uci_move_invalid(self) -> None:
        self.assertFalse(validate_uci_move(self.starting_board, "e2e5"))
        self.assertFalse(validate_uci_move(self.starting_board, "e1e2"))
        self.assertFalse(validate_uci_move(self.starting_board, "xyz"))

    def test_extract_uci_from_response(self) -> None:
        response = "<think>...</think>\n<uci_move>e2e4</uci_move>"
        self.assertEqual(extract_uci_from_response(response), "e2e4")

    def test_extract_uci_from_response_promotion(self) -> None:
        response = "<think>...</think>\n<uci_move>e7e8q</uci_move>"
        self.assertEqual(extract_uci_from_response(response), "e7e8q")

    def test_extract_uci_from_response_missing(self) -> None:
        self.assertIsNone(extract_uci_from_response("I think the best move is e4."))

    def test_extract_uci_from_response_bare_uci_fallback(self) -> None:
        response = "Main line: e2e4 e7e5. The pick is e2e4."
        self.assertEqual(extract_uci_from_response(response), "e2e4")

    def test_position_from_board(self) -> None:
        board = self.starting_board.copy()
        board.push_san("e4")

        pos = position_from_board(
            board=board,
            target_move_uci="e7e5",
            white_elo=1500,
            black_elo=1600,
            move_number=1,
            source="test",
        )
        self.assertEqual(pos.target_move_uci, "e7e5")
        self.assertEqual(pos.side_to_move, "Black")
        self.assertEqual(pos.white_elo, 1500)
        self.assertEqual(pos.source, "test")
        self.assertIn("e7e5", pos.legal_moves_uci)


class TestFormatting(unittest.TestCase):
    def test_position_to_messages(self) -> None:
        sample_position = {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "legal_moves_uci": "a7a6 a7a5 b7b6 b7b5 c7c6 c7c5 d7d6 d7d5 e7e6 e7e5",
            "target_move_uci": "e7e5",
            "first_legal_move": "a7a6",
            "board_utf": render_board_utf(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
            "side_to_move": "Black",
            "white_elo": 1800,
            "black_elo": 1750,
            "move_number": 1,
            "source": "game",
        }

        result = position_to_messages(sample_position)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][1]["role"], "assistant")

        user_content = result["messages"][0]["content"]
        assistant_content = result["messages"][1]["content"]
        self.assertIn(sample_position["fen"], user_content)
        self.assertIn(sample_position["legal_moves_uci"], user_content)
        self.assertIn("<uci_move>e7e5</uci_move>", assistant_content)


class TestDataProcessing(unittest.TestCase):
    def test_elo_weighting(self) -> None:
        from src.utils.data_processing import get_elo_weight

        elo_weights = {
            1200: 0.05,
            1400: 0.10,
            1600: 0.15,
            1800: 0.25,
            2000: 0.30,
            2200: 0.20,
        }

        self.assertEqual(get_elo_weight(1200, 1200, elo_weights), 0.05)
        self.assertEqual(get_elo_weight(2000, 2000, elo_weights), 0.30)
        self.assertEqual(get_elo_weight(2200, 2200, elo_weights), 0.20)

    def test_game_to_positions(self) -> None:
        from src.utils.data_processing import game_to_positions

        movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
        positions = list(
            game_to_positions(
                movetext=movetext,
                white_elo=1800,
                black_elo=1750,
                skip_first_n=2,
                skip_last_n=1,
                sample_rate=1.0,
            )
        )
        self.assertGreater(len(positions), 0)
        for pos in positions:
            self.assertTrue(pos.fen)
            self.assertTrue(pos.target_move_uci)
            self.assertTrue(pos.legal_moves_uci)


class TestLossWeighting(unittest.TestCase):
    def test_loss_weight_linear(self) -> None:
        from src.utils.data_processing import compute_loss_weight_linear

        w_min = compute_loss_weight_linear(1200, elo_min=1200, elo_max=2400)
        self.assertAlmostEqual(w_min, 0.5, places=2)
        w_max = compute_loss_weight_linear(2400, elo_min=1200, elo_max=2400)
        self.assertAlmostEqual(w_max, 2.0, places=2)
        w_mid = compute_loss_weight_linear(1800, elo_min=1200, elo_max=2400)
        self.assertAlmostEqual(w_mid, 1.25, places=2)

    def test_loss_weight_gaussian(self) -> None:
        from src.utils.data_processing import compute_loss_weight_gaussian

        w_target = compute_loss_weight_gaussian(1900, target_elo=1900)
        self.assertAlmostEqual(w_target, 2.0, places=2)
        w_far = compute_loss_weight_gaussian(1200, target_elo=1900, sigma=400)
        self.assertLess(w_far, 1.0)
        w_above = compute_loss_weight_gaussian(2200, target_elo=1900, sigma=400)
        w_below = compute_loss_weight_gaussian(1600, target_elo=1900, sigma=400)
        self.assertAlmostEqual(w_above, w_below, places=2)

    def test_loss_weight_bounds(self) -> None:
        from src.utils.data_processing import compute_loss_weight_gaussian, compute_loss_weight_linear

        for elo in [0, 500, 1000, 1500, 2000, 2500, 3000, 4000]:
            w_lin = compute_loss_weight_linear(elo)
            w_gauss = compute_loss_weight_gaussian(elo)
            self.assertGreaterEqual(w_lin, 0.5)
            self.assertLessEqual(w_lin, 2.0)
            self.assertGreaterEqual(w_gauss, 0.5)
            self.assertLessEqual(w_gauss, 2.0)

    def test_compute_loss_weight_with_config(self) -> None:
        from src.utils.data_processing import compute_loss_weight

        config = {
            "loss_weighting": {
                "enabled": True,
                "function": "gaussian",
                "target_elo": 1900,
                "gaussian_sigma": 400,
                "min_weight": 0.5,
                "max_weight": 2.0,
            }
        }
        w = compute_loss_weight(1800, 2000, config)
        self.assertAlmostEqual(w, 2.0, places=2)

        w_disabled = compute_loss_weight(1800, 2000, {"loss_weighting": {"enabled": False}})
        self.assertEqual(w_disabled, 1.0)

    def test_positions_have_loss_weight(self) -> None:
        from src.utils.data_processing import game_to_positions

        movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
        positions = list(
            game_to_positions(
                movetext=movetext,
                white_elo=1900,
                black_elo=1900,
                skip_first_n=2,
                skip_last_n=1,
                sample_rate=1.0,
            )
        )
        for pos in positions:
            self.assertIsInstance(pos.loss_weight, float)
            self.assertGreaterEqual(pos.loss_weight, 0.5)
            self.assertLessEqual(pos.loss_weight, 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
