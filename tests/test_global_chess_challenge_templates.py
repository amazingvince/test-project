#!/usr/bin/env python3
"""
Compatibility checks for AIcrowd Global Chess Challenge 2025 assets.

These tests are intentionally lightweight (no model inference). They ensure our
submission prompt templates only reference variables that the starter kit
provides, and that they include the required output tags.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class TestGlobalChessChallengePromptTemplates(unittest.TestCase):
    def test_competition_templates_use_supported_variables(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        templates_dir = repo_root / "competition" / "global_chess_challenge_2025"
        self.assertTrue(templates_dir.exists(), f"Missing templates dir: {templates_dir}")

        allowed_vars = {
            "board_utf",
            "board_ascii",
            "FEN",
            "side_to_move",
            "last_move",
            "legal_moves_uci",
            "legal_moves_san",
            "legal_moves_uci_list",
            "legal_moves_san_list",
            "first_legal_move",
            "move_history_uci",
            "move_history_san",
            "move_history_uci_list",
            "move_history_san_list",
        }

        var_re = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
        template_paths = sorted(p for p in templates_dir.glob("*.jinja") if p.is_file())
        self.assertTrue(template_paths, f"No .jinja templates found in {templates_dir}")

        for path in template_paths:
            text = path.read_text(encoding="utf-8")
            used_vars = set(var_re.findall(text))
            unknown = used_vars - allowed_vars
            self.assertFalse(
                unknown,
                f"Template {path} references unsupported variables: {sorted(unknown)}",
            )
            self.assertIn("<uci_move>", text, f"Template {path} must mention <uci_move> tags")
            self.assertIn("<think>", text, f"Template {path} must mention <think> tags")

