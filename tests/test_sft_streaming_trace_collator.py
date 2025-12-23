import chess
import pytest

from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.distill.stockfish_teacher import MoveAnalysis, PositionAnalysis
from src.sft import streaming_trace_collator

StreamingSFTStockfishTraceCollator = streaming_trace_collator.StreamingSFTStockfishTraceCollator


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = None

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(self, text, *, truncation=False, return_tensors=None, **kwargs):
        # Simple character-level tokenizer (deterministic, no external deps).
        ids = [2 + (ord(c) % 200) for c in str(text)]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeTeacher:
    def __init__(self, analysis: PositionAnalysis):
        self._analysis = analysis

    def analyze_batch(self, fens, analysis_overrides=None):
        return [self._analysis for _ in fens]


def make_analysis(board: chess.Board) -> PositionAnalysis:
    moves = [
        ("e2e4", 20, 0.55),
        ("d2d4", 10, 0.52),
        ("g1f3", 5, 0.51),
    ]
    best_cp = max(cp for _, cp, _ in moves)
    move_analyses = []
    for uci, cp, win_prob in moves:
        san = board.san(chess.Move.from_uci(uci))
        move_analyses.append(
            MoveAnalysis(
                uci=uci,
                san=san,
                centipawn=cp,
                cp_loss=best_cp - cp,
                category="best move" if cp == best_cp else "good",
                mate_in=None,
                win_probability=win_prob,
            )
        )
    move_analyses.sort(key=lambda m: m.centipawn, reverse=True)
    return PositionAnalysis(
        fen=board.fen(),
        move_analyses=move_analyses,
        best_move_uci=move_analyses[0].uci,
        best_move_san=move_analyses[0].san,
        best_score_cp=best_cp,
        best_pv=[move_analyses[0].san],
        move_probs={m.uci: m.win_probability for m in move_analyses},
        top_k_moves=[m.uci for m in move_analyses],
    )


def test_streaming_sft_stockfish_trace_collator_builds_batch():
    if streaming_trace_collator.torch is None:
        pytest.skip("torch is not installed in this environment")
    board = chess.Board()
    analysis = make_analysis(board)
    teacher = FakeTeacher(analysis)
    tokenizer = TinyTokenizer()

    trace_cfg = {
        "enabled": True,
        "style": "quick",
        "include_opening": False,
        "include_tablebase": False,
        "include_pv": False,
        "include_assessment": False,
        "include_orientation": False,
        "include_threat_scan": False,
        "include_comparison": False,
        "include_reconsideration": False,
        "include_dead_end": False,
        "min_candidates": 1,
        "max_candidates": 1,
    }
    trace_gen = ReasoningTraceGenerator(trace_cfg, tokenizer=None)

    collator = StreamingSFTStockfishTraceCollator(
        tokenizer=tokenizer,
        teacher=teacher,  # type: ignore[arg-type]
        config={"loss_weighting": {"enabled": False}},
        reasoning_trace_generator=trace_gen,
        max_length=512,
        pad_to_multiple_of=8,
        include_board=False,
        force_best_move=True,
        seed=123,
    )

    batch = collator(
        [
            {"fen": board.fen(), "target_move_uci": "g1f3", "source": "game", "white_elo": 1500, "black_elo": 1500},
            {"fen": board.fen(), "target_move_uci": "d2d4", "source": "puzzle", "white_elo": 1500, "black_elo": 1500},
        ]
    )

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels", "loss_weights"}
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["loss_weights"].shape[0] == 2

    # Prompt masking: there should be at least some ignored tokens.
    assert (batch["labels"] == -100).any().item() is True
