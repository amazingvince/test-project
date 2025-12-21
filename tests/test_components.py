#!/usr/bin/env python3
"""
Unit tests for Chess LLM SFT training components.

Run with: python -m pytest tests/ -v
Or: python tests/test_components.py
"""

import unittest
import chess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chess_utils import (
    render_board_utf,
    parse_movetext,
    get_legal_moves_uci,
    get_first_legal_move,
    validate_uci_move,
    extract_uci_from_response,
    position_from_board,
)

from src.formatting import (
    position_to_messages,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_RESPONSE_TEMPLATE,
)


class TestChessUtils(unittest.TestCase):
    """Test chess utility functions."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.starting_board = chess.Board()
        self.sicilian_board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
    
    def test_render_board_utf_starting(self):
        """Test board rendering for starting position."""
        rendered = render_board_utf(self.starting_board)
        
        # Check that it contains expected elements
        self.assertIn("♔", rendered)  # White king
        self.assertIn("♚", rendered)  # Black king
        self.assertIn("♙", rendered)  # White pawn
        self.assertIn("♟", rendered)  # Black pawn
        self.assertIn("a", rendered)   # File labels
        self.assertIn("8", rendered)   # Rank labels
    
    def test_parse_movetext_basic(self):
        """Test parsing basic movetext."""
        movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
        moves = parse_movetext(movetext)
        
        self.assertEqual(moves, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    
    def test_parse_movetext_with_result(self):
        """Test parsing movetext with game result."""
        movetext = "1. e4 e5 2. Nf3 1-0"
        moves = parse_movetext(movetext)
        
        self.assertEqual(moves, ["e4", "e5", "Nf3"])
    
    def test_parse_movetext_with_clock(self):
        """Test parsing movetext with clock annotations."""
        movetext = "1. e4 { [%clk 0:05:00] } e5 { [%clk 0:05:00] } 2. Nf3"
        moves = parse_movetext(movetext)
        
        self.assertEqual(moves, ["e4", "e5", "Nf3"])
    
    def test_get_legal_moves_uci(self):
        """Test getting legal moves in UCI format."""
        legal_moves = get_legal_moves_uci(self.starting_board)
        moves = legal_moves.split()
        
        # Starting position has 20 legal moves
        self.assertEqual(len(moves), 20)
        self.assertIn("e2e4", moves)
        self.assertIn("g1f3", moves)
    
    def test_get_first_legal_move(self):
        """Test getting first legal move."""
        first_move = get_first_legal_move(self.starting_board)
        
        self.assertIsNotNone(first_move)
        # Should be a valid UCI move
        self.assertRegex(first_move, r'^[a-h][1-8][a-h][1-8][qrbn]?$')
    
    def test_validate_uci_move_valid(self):
        """Test validating a legal move."""
        self.assertTrue(validate_uci_move(self.starting_board, "e2e4"))
        self.assertTrue(validate_uci_move(self.starting_board, "g1f3"))
    
    def test_validate_uci_move_invalid(self):
        """Test validating an illegal move."""
        self.assertFalse(validate_uci_move(self.starting_board, "e2e5"))  # Too far
        self.assertFalse(validate_uci_move(self.starting_board, "e1e2"))  # King blocked
        self.assertFalse(validate_uci_move(self.starting_board, "xyz"))   # Invalid format
    
    def test_extract_uci_from_response(self):
        """Test extracting UCI move from response."""
        response = "<think>Best move for development.</think>\n<uci_move>e2e4</uci_move>"
        move = extract_uci_from_response(response)
        
        self.assertEqual(move, "e2e4")
    
    def test_extract_uci_from_response_promotion(self):
        """Test extracting UCI move with promotion."""
        response = "<think>Promote to queen.</think>\n<uci_move>e7e8q</uci_move>"
        move = extract_uci_from_response(response)
        
        self.assertEqual(move, "e7e8q")
    
    def test_extract_uci_from_response_missing(self):
        """Test extracting UCI move when not present."""
        response = "I think the best move is e4."
        move = extract_uci_from_response(response)
        
        self.assertIsNone(move)
    
    def test_position_from_board(self):
        """Test creating ChessPosition from board."""
        board = self.starting_board.copy()
        board.push_san("e4")
        
        pos = position_from_board(
            board=board,
            target_move_uci="e7e5",
            white_elo=1500,
            black_elo=1600,
            move_number=1,
            source="test"
        )
        
        self.assertEqual(pos.target_move_uci, "e7e5")
        self.assertEqual(pos.side_to_move, "Black")
        self.assertEqual(pos.white_elo, 1500)
        self.assertEqual(pos.source, "test")
        self.assertIn("e7e5", pos.legal_moves_uci)


class TestFormatting(unittest.TestCase):
    """Test formatting functions."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.sample_position = {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "legal_moves_uci": "a7a6 a7a5 b7b6 b7b5 c7c6 c7c5 d7d6 d7d5 e7e6 e7e5",
            "target_move_uci": "e7e5",
            "first_legal_move": "a7a6",
            "board_utf": "   a  b  c  d  e  f  g  h  \n   +------------------------+\n8 | ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ | 8\n...",
            "side_to_move": "Black",
            "white_elo": 1800,
            "black_elo": 1750,
            "move_number": 1,
            "source": "game"
        }
    
    def test_position_to_messages(self):
        """Test converting position to messages format."""
        result = position_to_messages(self.sample_position)
        
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 2)
        
        user_msg = result["messages"][0]
        assistant_msg = result["messages"][1]
        
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(assistant_msg["role"], "assistant")
        
        # Check user message contains expected content
        self.assertIn(self.sample_position["fen"], user_msg["content"])
        self.assertIn(self.sample_position["legal_moves_uci"], user_msg["content"])
        
        # Check assistant message contains the move
        self.assertIn("<uci_move>e7e5</uci_move>", assistant_msg["content"])


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""
    
    def test_elo_weighting(self):
        """Test ELO-based weighting."""
        from src.data_processing import get_elo_weight
        
        elo_weights = {
            1200: 0.05, 1400: 0.10, 1600: 0.15,
            1800: 0.25, 2000: 0.30, 2200: 0.20
        }
        
        # Low ELO should have low weight
        self.assertEqual(get_elo_weight(1200, 1200, elo_weights), 0.05)
        
        # Target ELO should have high weight
        self.assertEqual(get_elo_weight(2000, 2000, elo_weights), 0.30)
        
        # High ELO should have moderate weight
        self.assertEqual(get_elo_weight(2200, 2200, elo_weights), 0.20)
    
    def test_game_to_positions(self):
        """Test converting game to positions."""
        from src.data_processing import game_to_positions
        
        # Simple game
        movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
        
        positions = list(game_to_positions(
            movetext=movetext,
            white_elo=1800,
            black_elo=1750,
            skip_first_n=2,
            skip_last_n=1,
            sample_rate=1.0  # Take all positions
        ))
        
        # Should get some positions (depends on sampling)
        self.assertGreater(len(positions), 0)
        
        # Check position structure
        for pos in positions:
            self.assertIsNotNone(pos.fen)
            self.assertIsNotNone(pos.target_move_uci)
            self.assertIsNotNone(pos.legal_moves_uci)


class TestLossWeighting(unittest.TestCase):
    """Test loss weight computation functions."""
    
    def test_loss_weight_linear(self):
        """Test linear loss weighting."""
        from src.data_processing import compute_loss_weight_linear
        
        # Min ELO should give min weight
        w_min = compute_loss_weight_linear(1200, elo_min=1200, elo_max=2400)
        self.assertAlmostEqual(w_min, 0.5, places=2)
        
        # Max ELO should give max weight
        w_max = compute_loss_weight_linear(2400, elo_min=1200, elo_max=2400)
        self.assertAlmostEqual(w_max, 2.0, places=2)
        
        # Mid ELO should give mid weight
        w_mid = compute_loss_weight_linear(1800, elo_min=1200, elo_max=2400)
        self.assertAlmostEqual(w_mid, 1.25, places=2)
    
    def test_loss_weight_gaussian(self):
        """Test Gaussian loss weighting."""
        from src.data_processing import compute_loss_weight_gaussian
        
        # Target ELO should give max weight
        w_target = compute_loss_weight_gaussian(1900, target_elo=1900)
        self.assertAlmostEqual(w_target, 2.0, places=2)
        
        # Far from target should give lower weight
        w_far = compute_loss_weight_gaussian(1200, target_elo=1900, sigma=400)
        self.assertLess(w_far, 1.0)
        
        # Symmetric around target
        w_above = compute_loss_weight_gaussian(2200, target_elo=1900, sigma=400)
        w_below = compute_loss_weight_gaussian(1600, target_elo=1900, sigma=400)
        self.assertAlmostEqual(w_above, w_below, places=2)
    
    def test_loss_weight_bounds(self):
        """Test that loss weights stay within bounds."""
        from src.data_processing import compute_loss_weight_linear, compute_loss_weight_gaussian
        
        # Test extreme values
        for elo in [0, 500, 1000, 1500, 2000, 2500, 3000, 4000]:
            w_lin = compute_loss_weight_linear(elo)
            w_gauss = compute_loss_weight_gaussian(elo)
            
            self.assertGreaterEqual(w_lin, 0.5)
            self.assertLessEqual(w_lin, 2.0)
            self.assertGreaterEqual(w_gauss, 0.5)
            self.assertLessEqual(w_gauss, 2.0)
    
    def test_compute_loss_weight_with_config(self):
        """Test full compute_loss_weight with config."""
        from src.data_processing import compute_loss_weight
        
        config = {
            'loss_weighting': {
                'enabled': True,
                'function': 'gaussian',
                'target_elo': 1900,
                'gaussian_sigma': 400,
                'min_weight': 0.5,
                'max_weight': 2.0
            }
        }
        
        # Average ELO = 1900 should give max weight
        w = compute_loss_weight(1800, 2000, config)
        self.assertAlmostEqual(w, 2.0, places=2)
        
        # Disabled should return 1.0
        config_disabled = {'loss_weighting': {'enabled': False}}
        w_disabled = compute_loss_weight(1800, 2000, config_disabled)
        self.assertEqual(w_disabled, 1.0)
    
    def test_positions_have_loss_weight(self):
        """Test that generated positions include loss_weight."""
        from src.data_processing import game_to_positions
        
        movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
        
        positions = list(game_to_positions(
            movetext=movetext,
            white_elo=1900,
            black_elo=1900,
            skip_first_n=2,
            skip_last_n=1,
            sample_rate=1.0
        ))
        
        for pos in positions:
            self.assertTrue(hasattr(pos, 'loss_weight'))
            self.assertIsInstance(pos.loss_weight, float)
            self.assertGreaterEqual(pos.loss_weight, 0.5)
            self.assertLessEqual(pos.loss_weight, 2.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
