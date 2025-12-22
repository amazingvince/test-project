#!/usr/bin/env python3
"""
Unit tests for distillation loss utilities.

Run:
  python -m unittest tests.test_distillation_loss -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from exc

# Ensure repo root is importable as a package (so `import src...` works).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distill.distillation_loss import (
    ChessDistillationLoss,
    compute_distillation_metrics,
    create_soft_target_tensor,
)


class TestSoftTargetTensor(unittest.TestCase):
    def test_create_soft_target_tensor_normalizes(self) -> None:
        vocab_size = 10
        move_to_token_id = {"e2e4": 5, "d2d4": 7}
        move_probs = {"e2e4": 0.7, "d2d4": 0.3, "a1a9": 0.9}  # invalid move ignored

        probs = create_soft_target_tensor(
            move_probs=move_probs,
            vocab_size=vocab_size,
            move_to_token_id=move_to_token_id,
            device=torch.device("cpu"),
        )

        self.assertEqual(probs.shape, (vocab_size,))
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=6)
        self.assertAlmostEqual(float(probs[5].item()), 0.7, places=6)
        self.assertAlmostEqual(float(probs[7].item()), 0.3, places=6)

    def test_create_soft_target_tensor_fallback_uniform(self) -> None:
        vocab_size = 8
        probs = create_soft_target_tensor(
            move_probs={"e2e4": 1.0},
            vocab_size=vocab_size,
            move_to_token_id={},  # nothing maps
            device=torch.device("cpu"),
        )
        self.assertEqual(probs.shape, (vocab_size,))
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=6)
        self.assertTrue(torch.allclose(probs, torch.full((vocab_size,), 1.0 / vocab_size)))


class TestChessDistillationLoss(unittest.TestCase):
    def test_forward_kl_alpha_one_ignores_hard(self) -> None:
        torch.manual_seed(0)
        batch, vocab = 4, 7
        student_logits = torch.randn(batch, vocab)
        teacher_probs = torch.softmax(torch.randn(batch, vocab), dim=-1)
        hard_targets = torch.tensor([1, 2, 3, 4])

        loss_fn = ChessDistillationLoss(alpha=1.0, temperature=1.0, use_liger=False, loss_type="kl")
        total, info = loss_fn(student_logits, teacher_probs, hard_targets=hard_targets)

        self.assertTrue(torch.is_tensor(total))
        self.assertEqual(total.ndim, 0)
        self.assertIn("soft_loss", info)
        self.assertIn("hard_loss", info)
        self.assertAlmostEqual(info["total_loss"], info["soft_loss"], places=6)
        self.assertAlmostEqual(info["hard_loss"], 0.0, places=6)

    def test_forward_combines_soft_and_hard(self) -> None:
        torch.manual_seed(1)
        batch, vocab = 3, 5
        student_logits = torch.randn(batch, vocab)
        teacher_probs = torch.softmax(torch.randn(batch, vocab), dim=-1)
        hard_targets = torch.tensor([0, 1, 4])

        soft_only = ChessDistillationLoss(alpha=1.0, temperature=1.0, use_liger=False, loss_type="kl")
        hard_only = ChessDistillationLoss(alpha=0.0, temperature=1.0, use_liger=False, loss_type="kl")
        mixed = ChessDistillationLoss(alpha=0.5, temperature=1.0, use_liger=False, loss_type="kl")

        soft_val, _ = soft_only(student_logits, teacher_probs, hard_targets=hard_targets)
        hard_val, _ = hard_only(student_logits, teacher_probs, hard_targets=hard_targets)
        mixed_val, _ = mixed(student_logits, teacher_probs, hard_targets=hard_targets)

        expected = 0.5 * soft_val + 0.5 * hard_val
        self.assertTrue(torch.allclose(mixed_val, expected, atol=1e-6, rtol=1e-6))


class TestDistillationMetrics(unittest.TestCase):
    def test_metrics_include_expected_keys(self) -> None:
        student_logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 10.0, 0.0],
            ]
        )
        teacher_probs = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        hard_targets = torch.tensor([3, 2])

        metrics = compute_distillation_metrics(student_logits, teacher_probs, hard_targets=hard_targets)
        self.assertIn("top1_agreement", metrics)
        self.assertIn("student_mass_on_teacher_top3", metrics)
        self.assertIn("kl_divergence", metrics)
        self.assertIn("hard_accuracy", metrics)
        self.assertIn("prob_on_correct", metrics)
        self.assertAlmostEqual(metrics["top1_agreement"], 1.0, places=6)
        self.assertAlmostEqual(metrics["hard_accuracy"], 1.0, places=6)
        self.assertGreaterEqual(metrics["student_mass_on_teacher_top3"], 0.9)
        self.assertLess(metrics["kl_divergence"], 0.1)
