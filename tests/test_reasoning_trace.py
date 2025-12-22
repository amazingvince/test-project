#!/usr/bin/env python3
"""
Unit tests for reasoning trace generation.
"""

import random
import unittest
from typing import List, Tuple

import chess

from src.distill.reasoning_trace import ReasoningTraceGenerator, TablebaseInfo
from src.distill.stockfish_teacher import MoveAnalysis, PositionAnalysis


def make_analysis(
    board: chess.Board,
    moves: List[Tuple[str, int, float]],
) -> PositionAnalysis:
    best_cp = max(cp for _, cp, _ in moves)
    move_analyses = []
    for uci, cp, win_prob in moves:
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        cp_loss = best_cp - cp
        category = "best move" if cp_loss == 0 else "good"
        move_analyses.append(
            MoveAnalysis(
                uci=uci,
                san=san,
                centipawn=cp,
                cp_loss=cp_loss,
                category=category,
                mate_in=None,
                win_probability=win_prob,
            )
        )
    move_analyses.sort(key=lambda m: m.centipawn, reverse=True)
    best_move_uci = move_analyses[0].uci
    best_move_san = move_analyses[0].san
    move_probs = {uci: prob for uci, _, prob in moves}
    return PositionAnalysis(
        fen=board.fen(),
        move_analyses=move_analyses,
        best_move_uci=best_move_uci,
        best_move_san=best_move_san,
        best_score_cp=best_cp,
        best_pv=[best_move_san],
        move_probs=move_probs,
        top_k_moves=[m.uci for m in move_analyses],
    )


class TestReasoningTraceThreatScan(unittest.TestCase):
    def test_threat_scan_includes_checks(self):
        board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("e2e7", 100, 0.60),
                ("e2e4", 0, 0.50),
            ],
        )
        cfg = {
            "style": "thorough",  # Explicit style to ensure threat_scan flow
            "include_threat_scan": True,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(0))
        self.assertIn("Checks to consider", text)
        self.assertIn("e2e7", text)

    def test_threat_scan_hanging_piece(self):
        board = chess.Board("4k3/8/8/8/8/8/1b6/R3K3 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("a1a2", 0, 0.50),
                ("a1a3", -10, 0.48),
            ],
        )
        cfg = {
            "style": "thorough",  # Explicit style to ensure threat_scan flow
            "include_threat_scan": True,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(1))
        self.assertIn("Hanging pieces", text)
        self.assertIn("a1", text)


class TestReasoningTraceConsistency(unittest.TestCase):
    def test_unclear_assessment_language(self):
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
                ("d2d4", 5, 0.49),
            ],
        )
        cfg = {
            "style": "thorough",  # Explicit style to ensure assessment flow
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": True,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
            "unclear_win_margin": 0.04,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(2))
        lowered = text.lower()
        self.assertTrue(
            "balanced" in lowered
            or "equal" in lowered
            or "unclear" in lowered
        )

    def test_tablebase_draw_assessment(self):
        board = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
        generator = ReasoningTraceGenerator({"include_threat_scan": False})
        line = generator._assessment_line(
            board,
            best_analysis=None,
            tablebase=TablebaseInfo(result="drawing", dtz=0, piece_count=2),
            style="concise",
            rng=random.Random(0),
            cfg={"unclear_win_margin": 0.04},
        )
        self.assertIn("draw", line.lower())


class TestReasoningTraceMotifs(unittest.TestCase):
    def test_candidate_fork_motif(self):
        board = chess.Board("3qk2r/8/8/6N1/8/8/8/4K3 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("g5f7", 50, 0.60),
                ("g5e6", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(3))
        self.assertIn("fork", text.lower())

    def test_candidate_weak_square_motif(self):
        board = chess.Board("4k3/8/8/8/2N5/8/8/4K3 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("c4e5", 10, 0.55),
                ("c4d6", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(4))
        self.assertIn("weak square", text.lower())

    def test_positional_cues_bishop_pair(self):
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
                ("d2d4", 5, 0.49),
            ],
        )
        cfg = {
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": True,
            "include_positional_cues": True,
            "max_positional_cues": 4,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(5))
        self.assertIn("bishop pair", text.lower())

    def test_positional_cues_isolated_pawn(self):
        board = chess.Board("8/8/8/8/3P4/8/4K3/7k w - - 0 1")
        analysis = make_analysis(board, [("e2e3", 0, 0.50)])
        cfg = {
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": True,
            "include_positional_cues": True,
            "max_positional_cues": 4,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(8))
        self.assertIn("isolated pawn", text.lower())

    def test_positional_cues_doubled_pawns(self):
        board = chess.Board("8/8/8/8/8/2P5/2P1K3/7k w - - 0 1")
        analysis = make_analysis(board, [("e2e3", 0, 0.50)])
        cfg = {
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": True,
            "include_positional_cues": True,
            "max_positional_cues": 4,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(9))
        self.assertIn("doubled pawns", text.lower())

    def test_positional_cues_half_open_file(self):
        board = chess.Board("8/3p4/8/8/8/8/4K3/7k w - - 0 1")
        analysis = make_analysis(board, [("e2e3", 0, 0.50)])
        cfg = {
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": True,
            "include_positional_cues": True,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(10))
        self.assertIn("half-open", text.lower())

    def test_positional_cues_backward_pawn(self):
        board = chess.Board("3r4/8/8/8/8/2P1P3/3P4/4K2k w - - 0 1")
        analysis = make_analysis(board, [("e1e2", 0, 0.50)])
        cfg = {
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": True,
            "include_positional_cues": True,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(11))
        self.assertIn("backward pawn", text.lower())

    def test_candidate_skewer_motif(self):
        board = chess.Board("4r1k1/4q3/8/8/8/8/4R3/6K1 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("e2e1", 50, 0.60),
                ("g1f1", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(7))
        self.assertIn("skewer", text.lower())

    def test_candidate_double_check_motif(self):
        board = chess.Board("4k3/8/8/8/8/8/4B3/4R1K1 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("e2b5", 50, 0.60),
                ("e2f3", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(12))
        self.assertIn("double check", text.lower())

    def test_candidate_deflection_motif(self):
        board = chess.Board("3r3k/5n2/8/6N1/8/8/8/3RK3 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("g5f7", 50, 0.60),
                ("g5e6", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(13))
        self.assertIn("deflection", text.lower())

    def test_candidate_xray_motif(self):
        board = chess.Board("4q2k/4b3/8/8/8/8/8/R5K1 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("a1e1", 50, 0.60),
                ("g1f1", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(14))
        self.assertIn("x-ray", text.lower())

    def test_candidate_sacrifice_motif(self):
        board = chess.Board("4k3/5p2/8/8/2B5/8/8/4K3 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("c4f7", 50, 0.60),
                ("c4b5", 0, 0.50),
            ],
        )
        cfg = {
            "style": "thorough",  # Explicit style to ensure motif flow
            "include_motifs": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(15))
        self.assertIn("sacrifice", text.lower())

    def test_candidate_quiet_move_motif(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
        analysis = make_analysis(
            board,
            [
                ("g1f3", 10, 0.55),
                ("g1e2", 0, 0.50),
            ],
        )
        cfg = {
            "include_motifs": True,
            "include_quiet_move_motif": True,
            "max_motifs_per_candidate": 1,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(16))
        self.assertIn("quiet move", text.lower())


class TestReasoningTraceTraps(unittest.TestCase):
    def test_trap_detection_from_shallow(self):
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("d2d4", -80, 0.40),
                ("e2e4", 0, 0.50),
            ],
        )
        analysis.shallow_move_cps = {
            "e2e4": 20,
            "d2d4": 30,
        }
        analysis.shallow_move_win_probs = {
            "e2e4": 0.55,
            "d2d4": 0.54,
        }
        cfg = {
            "style": "thorough",  # Explicit style to ensure trap detection flow
            "include_trap_detection": True,
            "trap_min_cp_swing": 60,
            "trap_shallow_good_cp": 10,
            "trap_shallow_bad_cp": 60,
            "trap_min_win_prob_swing": 0.10,
            "trap_refutation_max_len": 2,
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "candidate_order": "fixed",
            "min_candidates": 2,
            "max_candidates": 2,
        }
        for ma in analysis.move_analyses:
            if ma.uci == "d2d4":
                ma.pv_uci = ["d2d4", "d7d5", "c2c4"]
                break
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(6))
        self.assertIn("trap", text.lower())
        self.assertIn("line:", text.lower())

    def test_trap_confirmation(self):
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("d2d4", -80, 0.40),
                ("e2e4", 0, 0.50),
            ],
        )
        analysis.shallow_move_cps = {
            "d2d4": 30,
            "e2e4": 20,
        }
        analysis.confirm_move_cps = {
            "d2d4": -70,
        }
        cfg = {
            "trap_min_cp_swing": 60,
            "trap_shallow_good_cp": 10,
            "trap_shallow_bad_cp": 60,
            "trap_confirm_cp_tolerance": 30,
        }
        generator = ReasoningTraceGenerator(cfg)
        candidate = next(ma for ma in analysis.move_analyses if ma.uci == "d2d4")
        trap = generator._detect_trap_for_candidate(candidate, analysis, cfg)
        self.assertIsNotNone(trap)
        self.assertEqual(trap[2], "confirmed")


class TestNewStyles(unittest.TestCase):
    """Tests for the new style generators: quick, problem_focused, intuition, comparison_focused."""

    def test_quick_style_length(self):
        """Quick style should produce 2-3 sentences (under 100 words)."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
                ("d2d4", 5, 0.49),
                ("g1f3", 0, 0.48),
            ],
        )
        cfg = {"style": "quick"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(42))

        # Should be short (under 100 words)
        word_count = len(text.split())
        self.assertLess(word_count, 100, f"Quick style too long: {word_count} words")

        # Should contain the best move
        self.assertIn("e2e4", text)

    def test_quick_style_structure(self):
        """Quick style should have position, candidates, and conclusion."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
                ("d2d4", 5, 0.49),
            ],
        )
        cfg = {"style": "quick"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(43))

        # Should mention White to move
        self.assertTrue("White" in text or "white" in text.lower())

        # Should contain best move in conclusion
        self.assertIn("e2e4", text)

    def test_problem_focused_has_attempts(self):
        """Problem-focused style should show failed attempts."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.55),
                ("d2d4", 0, 0.48),
                ("c2c4", -10, 0.45),
            ],
        )
        cfg = {"style": "problem_focused"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(44))

        # Should have problem identification
        lowered = text.lower()
        self.assertTrue(
            "challenge" in lowered
            or "problem" in lowered
            or "solving" in lowered
            or "question" in lowered
            or "issue" in lowered
            or "task" in lowered
        )

        # Should mention best move
        self.assertIn("e2e4", text)

    def test_problem_focused_has_solution(self):
        """Problem-focused style should show the solution."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.55),
                ("d2d4", 0, 0.48),
            ],
        )
        cfg = {"style": "problem_focused"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(45))

        lowered = text.lower()
        # Should have solution language
        self.assertTrue(
            "solution" in lowered
            or "works" in lowered
            or "answer" in lowered
            or "solves" in lowered
        )

    def test_intuition_has_verification(self):
        """Intuition style should have verification section."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
                ("d2d4", 5, 0.49),
            ],
        )
        cfg = {"style": "intuition", "include_pv": True}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(46))

        lowered = text.lower()
        # Should have verification language
        self.assertTrue(
            "verify" in lowered
            or "check" in lowered
            or "confirm" in lowered
            or "testing" in lowered
            or "looking" in lowered
        )

    def test_intuition_has_impression(self):
        """Intuition style should have first impression."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
            ],
        )
        cfg = {"style": "intuition"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(47))

        lowered = text.lower()
        # Should have impression/intuition language
        self.assertTrue(
            "impression" in lowered
            or "instinct" in lowered
            or "gut" in lowered
            or "feeling" in lowered
        )

    def test_comparison_has_candidates(self):
        """Comparison style should compare multiple moves."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.55),
                ("d2d4", 5, 0.52),
                ("g1f3", 0, 0.48),
            ],
        )
        cfg = {"style": "comparison_focused"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(48))

        # Should mention multiple candidates
        self.assertIn("e2e4", text)
        self.assertIn("d2d4", text)

    def test_comparison_has_pros_cons(self):
        """Comparison style should have pros/cons structure."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.55),
                ("d2d4", 5, 0.52),
            ],
        )
        cfg = {"style": "comparison_focused"}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(49))

        # Should have comparison structure with /
        self.assertIn("/", text)

    def test_comparison_fallback_to_quick(self):
        """Comparison style with < 2 candidates should fall back to quick."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.55),
            ],
        )
        cfg = {"style": "comparison_focused", "min_candidates": 1, "max_candidates": 1}
        generator = ReasoningTraceGenerator(cfg)
        text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(50))

        # Should still work and be short (fallback to quick)
        word_count = len(text.split())
        self.assertLess(word_count, 100)
        self.assertIn("e2e4", text)

    def test_style_weights_selection(self):
        """Style weights should affect selection distribution."""
        cfg = {
            "style_weights": {
                "quick": 1.0,
                "thorough": 0.0,
                "concise": 0.0,
                "tactical": 0.0,
            }
        }
        generator = ReasoningTraceGenerator(cfg)
        rng = random.Random(51)

        # With weight 1.0 for quick, should always select quick
        style = generator._select_style(cfg, None, rng)
        self.assertEqual(style, "quick")

    def test_style_weights_new_styles(self):
        """New styles should be selectable via weights."""
        for new_style in ["quick", "problem_focused", "intuition", "comparison_focused"]:
            cfg = {
                "style_weights": {
                    new_style: 1.0,
                }
            }
            generator = ReasoningTraceGenerator(cfg)
            rng = random.Random(52)

            style = generator._select_style(cfg, None, rng)
            self.assertEqual(style, new_style, f"Failed for style: {new_style}")

    def test_conclusion_uses_expanded_phrases(self):
        """Conclusion should use the expanded phrase library."""
        board = chess.Board()
        analysis = make_analysis(
            board,
            [
                ("e2e4", 10, 0.51),
                ("d2d4", 5, 0.49),
            ],
        )
        # Use thorough style to get the standard conclusion flow
        cfg = {
            "style": "thorough",
            "include_threat_scan": False,
            "include_orientation": False,
            "include_assessment": False,
            "include_opening": False,
            "include_tablebase": False,
            "include_reconsideration": False,
            "include_dead_end": False,
            "include_comparison": False,
            "include_pv": False,
            "min_candidates": 1,
            "max_candidates": 1,
        }
        generator = ReasoningTraceGenerator(cfg)

        # Generate multiple times and check for variety
        conclusions = set()
        for seed in range(10):
            text = generator.generate({"fen": board.fen()}, analysis, rng=random.Random(seed + 100))
            # Extract last sentence/conclusion
            conclusions.add(text.strip())

        # Should have variety in conclusions (not all identical)
        self.assertGreater(len(conclusions), 1, "Conclusions should vary across generations")


if __name__ == "__main__":
    unittest.main(verbosity=2)
