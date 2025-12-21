"""
Reasoning trace generator for chess distillation.

Creates human-style analysis text using Stockfish move analysis, opening
lookups, and optional tablebase probes. Output stays in UCI notation by
default to align with move tokens.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess

try:
    from .chess_utils import parse_movetext
except ImportError:
    from chess_utils import parse_movetext


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

WEAK_SQUARE_CANDIDATES = [
    chess.D4, chess.E4, chess.D5, chess.E5,
    chess.C4, chess.F4, chess.C5, chess.F5,
]


@dataclass
class OpeningInfo:
    eco: str
    name: str
    main_line: str
    move_count: int


@dataclass
class TablebaseInfo:
    result: str
    dtz: Optional[int]
    piece_count: int


class OpeningBook:
    def __init__(self, paths: Iterable[Path]):
        self.paths = [Path(p) for p in paths if p]
        self._loaded = False
        self._fen_to_opening: Dict[str, OpeningInfo] = {}
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        self._ensure_loaded()
        return bool(self._fen_to_opening)

    def lookup(self, board: chess.Board) -> Optional[OpeningInfo]:
        self._ensure_loaded()
        if not self._fen_to_opening:
            return None
        key = self._normalize_fen(board)
        return self._fen_to_opening.get(key)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        existing = [p for p in self.paths if p.exists()]
        if not existing:
            self._load_error = "no opening files found"
            return
        for path in existing:
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline().strip().split("\t")
                col = {name: idx for idx, name in enumerate(header)}
                for line in handle:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    eco = self._get_col(parts, col, "eco") or ""
                    name = self._get_col(parts, col, "name") or ""
                    uci_moves = self._extract_moves(parts, col)
                    if not uci_moves:
                        continue
                    main_line = " ".join(uci_moves)
                    self._index_line(eco, name, uci_moves, main_line)
        except Exception as exc:
            self._load_error = f"failed to load {path}: {exc}"

    def _extract_moves(self, parts: List[str], col: Dict[str, int]) -> List[str]:
        uci_moves = self._get_col(parts, col, "uci")
        if uci_moves:
            return [m for m in uci_moves.split() if m]
        pgn_moves = self._get_col(parts, col, "pgn")
        if not pgn_moves:
            pgn_moves = self._get_col(parts, col, "moves")
        if not pgn_moves:
            return []
        san_moves = parse_movetext(pgn_moves)
        board = chess.Board()
        uci_seq = []
        for san in san_moves:
            try:
                move = board.parse_san(san)
            except Exception:
                break
            uci_seq.append(move.uci())
            board.push(move)
        return uci_seq

    @staticmethod
    def _get_col(parts: List[str], col: Dict[str, int], key: str) -> Optional[str]:
        idx = col.get(key)
        if idx is None or idx >= len(parts):
            return None
        return parts[idx]

    def _index_line(
        self,
        eco: str,
        name: str,
        uci_moves: List[str],
        main_line: str,
    ) -> None:
        board = chess.Board()
        for idx, move_uci in enumerate(uci_moves):
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            board.push(move)
            key = self._normalize_fen(board)
            existing = self._fen_to_opening.get(key)
            if existing is None or idx + 1 >= existing.move_count:
                self._fen_to_opening[key] = OpeningInfo(
                    eco=eco,
                    name=name,
                    main_line=main_line,
                    move_count=idx + 1,
                )

    @staticmethod
    def _normalize_fen(board: chess.Board) -> str:
        fen = board.fen()
        parts = fen.split(" ")
        if len(parts) < 4:
            return fen
        return " ".join(parts[:4])


class TablebaseProbe:
    def __init__(self, paths: Iterable[Path]):
        self.paths = [Path(p) for p in paths if p]
        self._tb = None
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def probe(self, board: chess.Board) -> Optional[TablebaseInfo]:
        if len(board.piece_map()) > 7:
            return None
        if not self._ensure_loaded():
            return None
        try:
            wdl = self._tb.probe_wdl(board)
            dtz = self._tb.probe_dtz(board)
        except Exception:
            return None
        result = "drawing"
        if wdl > 0:
            result = "winning"
        elif wdl < 0:
            result = "losing"
        return TablebaseInfo(result=result, dtz=dtz, piece_count=len(board.piece_map()))

    def _ensure_loaded(self) -> bool:
        if self._tb is not None:
            return True
        if not self.paths:
            return False
        try:
            import chess.syzygy
        except Exception as exc:
            self._load_error = f"syzygy unavailable: {exc}"
            return False
        existing = [p for p in self.paths if p.exists()]
        if not existing:
            self._load_error = "no tablebase path found"
            return False
        try:
            self._tb = chess.syzygy.open_tablebase(existing[0])
            for path in existing[1:]:
                self._tb.add_directory(path)
            return True
        except Exception as exc:
            self._load_error = f"failed to load tablebase: {exc}"
            return False


class ReasoningTraceGenerator:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or {}
        self.tokenizer = tokenizer
        self.seed = self.config.get("seed")
        self._rng = random.Random(self.seed)

        opening_paths = self.config.get("opening_paths")
        if isinstance(opening_paths, (str, Path)):
            opening_paths = [opening_paths]
        if not opening_paths:
            opening_paths = self._default_opening_paths()
        self._opening = OpeningBook(opening_paths)

        tablebase_paths = self.config.get("tablebase_paths") or []
        if isinstance(tablebase_paths, (str, Path)):
            tablebase_paths = [tablebase_paths]
        self._tablebase = TablebaseProbe(tablebase_paths)

    def status(self) -> Dict[str, Any]:
        return {
            "opening_available": self._opening.is_available(),
            "tablebase_available": self._tablebase.is_available(),
        }

    def generate(
        self,
        position: Dict[str, Any],
        analysis: Any,
        rng: Optional[random.Random] = None,
    ) -> str:
        source = position.get("source")
        cfg = self._resolve_config(source)
        rng = rng or self._rng

        fen = position.get("fen")
        if not fen:
            return "No position provided."
        board = chess.Board(fen)

        best_move_uci = analysis.best_move_uci or ""
        best_analysis = self._find_move(analysis, best_move_uci)
        if best_analysis is None and analysis.move_analyses:
            best_analysis = analysis.move_analyses[0]
            best_move_uci = best_analysis.uci

        candidates = self._select_candidates(analysis, cfg, rng)
        if not candidates and best_analysis is not None:
            candidates = [best_analysis]

        opening_info = None
        if cfg.get("include_opening", True):
            opening_info = self._opening.lookup(board)

        tablebase_info = None
        if cfg.get("include_tablebase", True):
            tablebase_info = self._tablebase.probe(board)

        style = self._select_style(cfg, source, rng)
        notation = cfg.get("move_notation", "uci")

        orientation = self._orientation_line(
            board,
            opening_info,
            tablebase_info,
            source,
            style,
            rng,
        ) if cfg.get("include_orientation", True) else ""

        assessment = self._assessment_line(
            board,
            best_analysis,
            tablebase_info,
            style,
            rng,
            cfg,
        ) if cfg.get("include_assessment", True) else ""

        threat_scan = self._threat_scan_line(
            board,
            style,
            rng,
            notation,
            cfg,
        ) if cfg.get("include_threat_scan", True) else ""

        plan_line = ""
        if cfg.get("include_plan", False):
            if rng.random() < float(cfg.get("plan_prob", 0.5)):
                plan_line = self._plan_line(
                    board,
                    best_analysis,
                    style,
                    rng,
                    cfg,
                )

        opponent_line = ""
        if cfg.get("include_opponent_perspective", False):
            if rng.random() < float(cfg.get("opponent_perspective_prob", 0.4)):
                opponent_line = self._opponent_perspective_line(
                    board,
                    style,
                    rng,
                    notation,
                    cfg,
                )

        candidates_line = self._candidate_line(candidates, style, rng, notation)

        exploration_paragraphs = self._exploration_paragraphs(
            board,
            candidates,
            best_analysis,
            analysis,
            style,
            rng,
            cfg,
            notation,
        )

        reconsideration = ""
        if cfg.get("include_reconsideration", True):
            if rng.random() < cfg.get("reconsideration_prob", 0.4):
                reconsideration = self._reconsideration_line(rng)

        dead_end = ""
        if cfg.get("include_dead_end", True):
            if rng.random() < cfg.get("dead_end_prob", 0.4):
                dead_end = self._dead_end_line(candidates, best_analysis, rng, notation)

        comparison = ""
        if cfg.get("include_comparison", True):
            if rng.random() < cfg.get("comparison_prob", 0.5):
                comparison = self._comparison_line(candidates, best_analysis, rng, notation)

        conclusion = self._conclusion_line(
            board,
            best_analysis,
            analysis,
            style,
            rng,
            cfg,
            notation,
        )

        sections = [
            ("orientation", orientation),
            ("assessment", assessment),
            ("threat_scan", threat_scan),
            ("plan", plan_line),
            ("opponent", opponent_line),
            ("candidates", candidates_line),
            ("exploration", "\n\n".join(exploration_paragraphs) if exploration_paragraphs else ""),
            ("reconsideration", reconsideration),
            ("dead_end", dead_end),
            ("comparison", comparison),
            ("conclusion", conclusion),
        ]

        text = self._apply_budget(sections, exploration_paragraphs, cfg)
        return self._apply_guardrails(text, tablebase_info, cfg)

    def _resolve_config(self, source: Optional[str]) -> Dict[str, Any]:
        cfg = dict(self.config)
        source_overrides = self.config.get("source_overrides", {})
        if source and source in source_overrides:
            cfg.update(source_overrides[source] or {})
        return cfg

    def _select_candidates(
        self,
        analysis: Any,
        cfg: Dict[str, Any],
        rng: random.Random,
    ) -> List[Any]:
        if not analysis.move_analyses:
            return []
        min_c = max(1, int(cfg.get("min_candidates", 3)))
        max_c = max(min_c, int(cfg.get("max_candidates", 5)))
        max_c = min(max_c, len(analysis.move_analyses))
        count = rng.randint(min_c, max_c) if max_c > min_c else max_c
        return analysis.move_analyses[:count]

    def _orientation_line(
        self,
        board: chess.Board,
        opening: Optional[OpeningInfo],
        tablebase: Optional[TablebaseInfo],
        source: Optional[str],
        style: str,
        rng: random.Random,
    ) -> str:
        side = "White" if board.turn == chess.WHITE else "Black"
        phase = self._position_phase(board)
        opener = ""
        if opening:
            opener = f"Opening: {opening.eco} {opening.name}."
        elif phase == "endgame":
            opener = "This is an endgame position."
        elif phase == "opening":
            opener = "Opening phase of the game."

        source_hint = ""
        if source == "puzzle":
            source_hint = rng.choice([
                "This looks like a tactical puzzle.",
                "Puzzle position with a likely forcing line.",
                "Tactical spot where precision matters.",
            ])
        elif source == "game":
            source_hint = rng.choice([
                "Game position with practical choices.",
                "From a game, so steady development matters.",
                "Practical game position.",
            ])

        lines = []
        if opener:
            lines.append(opener)
        if source_hint:
            lines.append(source_hint)
        lines.append(f"{side} to move.")
        return " ".join(lines)

    def _assessment_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        tablebase: Optional[TablebaseInfo],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if tablebase:
            side = "White" if board.turn == chess.WHITE else "Black"
            if tablebase.result == "drawing":
                return rng.choice([
                    "With perfect play, this is a draw.",
                    "Tablebase says the position is drawn.",
                ])
            if tablebase.result == "winning":
                return rng.choice([
                    f"With perfect play, {side} is winning.",
                    f"Tablebase indicates a win for {side}.",
                ])
            if tablebase.result == "losing":
                return rng.choice([
                    f"With perfect play, {side} is losing.",
                    f"Tablebase indicates {side} is losing.",
                ])
            return "Tablebase result is unclear."
        if best_analysis is None:
            return "No clear evaluation, so focus on solid moves."
        win_prob = getattr(best_analysis, "win_probability", 0.5)
        unclear_margin = float(cfg.get("unclear_win_margin", 0.04))
        if abs(win_prob - 0.5) <= unclear_margin:
            base = rng.choice([
                "The position looks balanced and unclear.",
                "This feels roughly equal without a clear edge.",
                "Neither side has much of an edge here.",
            ])
            return self._append_positional_cues(base, board, cfg)
        if win_prob >= 0.60:
            base = rng.choice([
                "Side to move has the initiative and can press.",
                "There is a clear edge for the side to move.",
            ])
            return self._append_positional_cues(base, board, cfg)
        if win_prob <= 0.40:
            base = rng.choice([
                "The side to move is under pressure here.",
                "It looks a bit worse, so accuracy matters.",
            ])
            return self._append_positional_cues(base, board, cfg)
        base = rng.choice([
            "The position looks roughly balanced.",
            "This feels quite equal with chances for both sides.",
        ])
        return self._append_positional_cues(base, board, cfg)

    def _candidate_line(
        self,
        candidates: List[Any],
        style: str,
        rng: random.Random,
        notation: str,
    ) -> str:
        if not candidates:
            return "Candidates are unclear in this position."
        moves = [self._move_label(cand, notation) for cand in candidates]
        prefix = rng.choice([
            "Moves to consider:",
            "Candidates:",
            "The main candidates are:",
        ])
        return f"{prefix} {', '.join(moves)}."

    def _exploration_paragraphs(
        self,
        board: chess.Board,
        candidates: List[Any],
        best_analysis: Optional[Any],
        analysis: Any,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> List[str]:
        if not candidates:
            return []
        ordered = candidates.copy()
        if cfg.get("candidate_order", "shuffled") == "shuffled":
            rng.shuffle(ordered)
        best_win = getattr(best_analysis, "win_probability", 0.5) if best_analysis else 0.5
        paragraphs = []
        for cand in ordered:
            paragraphs.append(
                self._analyze_candidate(
                    board, cand, best_win, analysis, style, rng, notation, cfg
                )
            )
        return paragraphs

    def _analyze_candidate(
        self,
        board: chess.Board,
        cand: Any,
        best_win: float,
        analysis: Any,
        style: str,
        rng: random.Random,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        move_label = self._move_label(cand, notation)
        try:
            move = chess.Move.from_uci(cand.uci)
        except Exception:
            move = None
        features = self._move_features(board, move) if move else []
        idea = self._idea_phrase(features, rng, style)
        unclear_margin = float(cfg.get("unclear_win_margin", 0.04))
        quality = self._quality_phrase(cand, best_win, rng, style, unclear_margin)
        motif_line = ""
        if cfg.get("include_motifs", False) and move is not None:
            motifs = self._detect_candidate_motifs(board, move, cand, cfg)
            motif_line = self._motif_phrase(motifs, rng)
        trap_line = ""
        if cfg.get("include_trap_detection", False):
            trap = self._detect_trap_for_candidate(cand, analysis, cfg)
            trap_line = self._trap_phrase(trap, rng, board, cand, notation, cfg)
        intro = rng.choice([
            f"Looking at {move_label},",
            f"For {move_label},",
            f"Considering {move_label},",
        ])
        details = " ".join(part for part in [motif_line, trap_line] if part)
        if details:
            return f"{intro} {idea} {quality} {details}"
        return f"{intro} {idea} {quality}"

    def _reconsideration_line(self, rng: random.Random) -> str:
        return rng.choice([
            "Actually, I should double-check the tactical details.",
            "On second thought, there may be a hidden resource here.",
            "Wait, there could be a subtle defense I missed.",
        ])

    def _dead_end_line(
        self,
        candidates: List[Any],
        best_analysis: Optional[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        if len(candidates) < 2:
            return ""
        worst = min(
            candidates,
            key=lambda c: getattr(c, "win_probability", 0.5),
        )
        move_label = self._move_label(worst, notation)
        reason = rng.choice([
            "it allows a strong reply",
            "it leaves the king exposed",
            "it drops material after a simple tactic",
            "it concedes too much activity",
        ])
        return f"I looked at {move_label}, but it does not work because {reason}."

    def _comparison_line(
        self,
        candidates: List[Any],
        best_analysis: Optional[Any],
        rng: random.Random,
        notation: str,
    ) -> str:
        if len(candidates) < 2:
            return ""
        ordered = sorted(
            candidates,
            key=lambda c: getattr(c, "win_probability", 0.5),
            reverse=True,
        )
        first = ordered[0]
        second = ordered[1]
        first_label = self._move_label(first, notation)
        second_label = self._move_label(second, notation)
        return rng.choice([
            f"Between {first_label} and {second_label}, the first looks more direct.",
            f"Comparing {first_label} vs {second_label}, the first seems more forcing.",
            f"{first_label} looks a bit cleaner than {second_label}.",
        ])

    def _conclusion_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        analysis: Any,
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
        notation: str,
    ) -> str:
        if best_analysis is None:
            return "I will choose a solid move and keep the position safe."
        best_move_label = self._move_label(best_analysis, notation)
        reason = self._idea_phrase(
            self._move_features(board, chess.Move.from_uci(best_analysis.uci)),
            rng,
            style,
        )
        intro = rng.choice([
            "After considering everything,",
            "Taking this all into account,",
            "Given these factors,",
        ])
        conclusion = f"{intro} {best_move_label} is the move."
        pv_line = ""
        pruned = False
        if reason:
            reason = reason[:1].upper() + reason[1:]
        if cfg.get("include_pv", True):
            max_len = cfg.get("max_pv_length", 8)
            if notation == "san" and analysis.best_pv:
                pv_moves, pruned = self._prune_pv_san(board, analysis.best_pv, cfg, max_len)
                if pv_moves:
                    pv_line = f"Main line: {' '.join(pv_moves)}"
            else:
                pv_uci, pruned = self._pv_line_uci(board, analysis, max_len, cfg)
                if pv_uci:
                    pv_line = f"Main line: {pv_uci}"
            if pv_line and pruned and cfg.get("pv_quiet_summary", True):
                quiet_summary = rng.choice([
                    "The line quiets down after that.",
                    "After that, the position stays quiet.",
                    "The rest of the line is quieter.",
                    "The position settles after that.",
                ])
                pv_line = f"{pv_line} {quiet_summary}"
        if pv_line:
            return f"{conclusion} {reason}\n\n{pv_line}"
        return f"{conclusion} {reason}"

    def _move_label(self, cand: Any, notation: str) -> str:
        if notation == "san":
            return getattr(cand, "san", getattr(cand, "uci", ""))
        return getattr(cand, "uci", getattr(cand, "san", ""))

    def _move_features(self, board: chess.Board, move: Optional[chess.Move]) -> List[str]:
        if move is None:
            return []
        features = []
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                features.append(f"captures the {PIECE_NAMES.get(captured.piece_type, 'piece')}")
            else:
                features.append("captures material")
        if board.gives_check(move):
            features.append("gives check")
        if board.is_castling(move):
            features.append("castles to safety")
        if move.promotion:
            features.append("promotes a pawn")
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if chess.square_rank(move.from_square) in (0, 7):
                features.append("develops a piece")
        if piece and piece.piece_type == chess.PAWN:
            if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                features.append("claims central space")
        return features

    def _idea_phrase(self, features: List[str], rng: random.Random, style: str) -> str:
        if not features:
            if style == "tactical":
                return rng.choice([
                    "it keeps the initiative and creates immediate threats.",
                    "it forces a reply and keeps pressure on the opponent.",
                    "it keeps the attack moving.",
                ])
            if style == "concise":
                return rng.choice([
                    "it keeps things simple.",
                    "it is a safe, flexible choice.",
                    "it is a solid move.",
                ])
            return rng.choice([
                "it improves coordination and keeps options open.",
                "it keeps the position flexible.",
                "it develops sensibly without committing too much.",
            ])
        if len(features) == 1:
            base = f"it {features[0]}."
            if style == "tactical":
                tail = rng.choice([
                    "This keeps the initiative.",
                    "This is forcing.",
                    "This keeps pressure on.",
                ])
                return f"{base} {tail}"
            return base
        if len(features) == 2:
            base = f"it {features[0]} and {features[1]}."
        else:
            base = f"it {features[0]}, {features[1]}, and {features[2]}."
        if style == "tactical":
            tail = rng.choice([
                "This keeps the initiative.",
                "This is the forcing option.",
                "This keeps momentum.",
            ])
            return f"{base} {tail}"
        if style == "concise":
            return base
        return base

    def _quality_phrase(
        self,
        cand: Any,
        best_win: float,
        rng: random.Random,
        style: str,
        unclear_margin: float,
    ) -> str:
        mate_in = getattr(cand, "mate_in", None)
        if mate_in is not None:
            if mate_in > 0:
                return f"This is a forced mate in {mate_in}."
            return f"This allows mate in {abs(mate_in)}."
        win_prob = getattr(cand, "win_probability", 0.5)
        delta = best_win - win_prob
        if abs(win_prob - 0.5) <= unclear_margin:
            if delta <= 0.02:
                return rng.choice([
                    "It looks like one of several reasonable options.",
                    "It is a fine choice in an equal position.",
                    "This is a sensible choice in a balanced position.",
                ])
            if delta <= 0.06:
                return rng.choice([
                    "It is a reasonable alternative in a balanced position.",
                    "It seems playable with equality.",
                    "It keeps the game about equal.",
                ])
        if delta <= 0.02:
            if style == "tactical":
                return rng.choice([
                    "This looks most forcing.",
                    "This keeps the initiative best.",
                    "This feels like the sharpest line.",
                ])
            return rng.choice([
                "It looks strongest.",
                "This seems best.",
                "It feels like the top choice.",
            ])
        if delta <= 0.06:
            if style == "concise":
                return rng.choice([
                    "It looks fine.",
                    "This is playable.",
                    "It seems solid.",
                ])
            return rng.choice([
                "It looks very solid.",
                "This is a strong alternative.",
                "It seems quite playable.",
            ])
        if delta <= 0.12:
            if style == "tactical":
                return rng.choice([
                    "It is playable, but less forcing.",
                    "It gives chances but lacks bite.",
                    "It keeps play going, but is not the sharpest.",
                ])
            return rng.choice([
                "It is a reasonable try.",
                "It seems playable but not perfect.",
                "It is usable but less forcing.",
            ])
        return rng.choice([
            "It looks risky in comparison.",
            "It seems inferior to the best move.",
            "It is probably not the most accurate.",
        ])

    def _append_positional_cues(
        self,
        base: str,
        board: chess.Board,
        cfg: Dict[str, Any],
    ) -> str:
        if not cfg.get("include_positional_cues", False):
            return base
        cues = self._positional_cues(board, cfg)
        if not cues:
            return base
        return f"{base} Positional notes: {', '.join(cues)}."

    def _positional_cues(
        self,
        board: chess.Board,
        cfg: Dict[str, Any],
    ) -> List[str]:
        cues: List[str] = []
        max_cues = int(cfg.get("max_positional_cues", 2))
        color = board.turn

        bishops = board.pieces(chess.BISHOP, color)
        if len(bishops) >= 2:
            cues.append("bishop pair")

        open_files = self._open_files(board)
        rooks = board.pieces(chess.ROOK, color)
        for square in rooks:
            file_idx = chess.square_file(square)
            if file_idx in open_files:
                file_letter = "abcdefgh"[file_idx]
                cues.append(f"rook on open {file_letter}-file")
                break

        weak_squares = self._find_weak_squares(board, color)
        if weak_squares:
            cues.append(f"weak square on {chess.square_name(weak_squares[0])}")

        passed_pawn = self._find_passed_pawn(board, color)
        if passed_pawn is not None:
            cues.append(f"passed pawn on {chess.square_name(passed_pawn)}")

        outpost = self._find_outpost(board, color)
        if outpost is not None:
            cues.append(f"outpost on {chess.square_name(outpost)}")

        isolated = self._find_isolated_pawn(board, color)
        if isolated is not None:
            cues.append(f"isolated pawn on {chess.square_name(isolated)}")

        doubled_file = self._find_doubled_pawn_file(board, color)
        if doubled_file is not None:
            file_letter = "abcdefgh"[doubled_file]
            cues.append(f"doubled pawns on {file_letter}-file")

        backward = self._find_backward_pawn(board, color)
        if backward is not None:
            cues.append(f"backward pawn on {chess.square_name(backward)}")

        half_open = self._half_open_files(board, color)
        if half_open:
            file_letter = "abcdefgh"[half_open[0]]
            cues.append(f"half-open {file_letter}-file")

        majority = self._pawn_majority(board, color)
        if majority:
            cues.append(majority)

        seventh = self._rook_on_seventh(board, color)
        if seventh:
            cues.append(seventh)

        if self._is_exposed_king(board, color):
            cues.append("exposed king")

        return cues[:max_cues]

    def _position_phase(self, board: chess.Board) -> str:
        piece_count = len(board.piece_map())
        if piece_count <= 10:
            return "endgame"
        if board.fullmove_number <= 10:
            return "opening"
        return "middlegame"

    def _find_move(self, analysis: Any, uci: str) -> Optional[Any]:
        if not uci:
            return None
        for move in analysis.move_analyses:
            if move.uci == uci:
                return move
        return None

    def _pv_line_uci(
        self,
        board: chess.Board,
        analysis: Any,
        max_len: int,
        cfg: Dict[str, Any],
    ) -> Tuple[str, bool]:
        if not analysis.best_pv:
            return "", False
        temp = board.copy()
        uci_moves = []
        for san in analysis.best_pv[:max_len]:
            try:
                move = temp.parse_san(san)
            except Exception:
                break
            uci_moves.append(move.uci())
            temp.push(move)
        pruned, was_pruned = self._prune_pv_moves(board, uci_moves, cfg)
        return " ".join(pruned), was_pruned

    def _prune_pv_san(
        self,
        board: chess.Board,
        pv_san: List[str],
        cfg: Dict[str, Any],
        max_len: int,
    ) -> Tuple[List[str], bool]:
        if max_len > 0:
            pv_san = pv_san[:max_len]
        if not cfg.get("pv_prune_quiet", True):
            return pv_san, False
        max_quiet = int(cfg.get("pv_prune_quiet_plies", 2))
        min_moves = int(cfg.get("pv_prune_min_moves", 2))
        temp = board.copy()
        quiet = 0
        filtered: List[str] = []
        for san in pv_san:
            try:
                move = temp.parse_san(san)
            except Exception:
                break
            forcing = self._is_forcing_move(temp, move)
            filtered.append(san)
            temp.push(move)
            if forcing:
                quiet = 0
            else:
                quiet += 1
            if quiet >= max_quiet and len(filtered) >= min_moves:
                return filtered, True
        return filtered, False

    def _prune_pv_moves(
        self,
        board: chess.Board,
        pv_uci: List[str],
        cfg: Dict[str, Any],
    ) -> Tuple[List[str], bool]:
        if not cfg.get("pv_prune_quiet", True):
            return pv_uci, False
        max_quiet = int(cfg.get("pv_prune_quiet_plies", 2))
        min_moves = int(cfg.get("pv_prune_min_moves", 2))
        temp = board.copy()
        quiet = 0
        filtered: List[str] = []
        for uci in pv_uci:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in temp.legal_moves:
                break
            forcing = self._is_forcing_move(temp, move)
            filtered.append(uci)
            temp.push(move)
            if forcing:
                quiet = 0
            else:
                quiet += 1
            if quiet >= max_quiet and len(filtered) >= min_moves:
                return filtered, True
        return filtered, False

    @staticmethod
    def _is_forcing_move(board: chess.Board, move: chess.Move) -> bool:
        return board.is_capture(move) or board.gives_check(move) or bool(move.promotion)

    def _select_style(self, cfg: Dict[str, Any], source: Optional[str], rng: random.Random) -> str:
        if cfg.get("style"):
            return cfg["style"]
        weights = cfg.get("style_weights")
        if weights:
            return self._weighted_choice(weights, rng)
        if source == "puzzle":
            return "tactical"
        return "thorough"

    def _weighted_choice(self, weights: Dict[str, float], rng: random.Random) -> str:
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0:
            return next(iter(weights.keys()))
        pick = rng.random() * total
        running = 0.0
        for key, weight in weights.items():
            running += max(weight, 0.0)
            if pick <= running:
                return key
        return next(iter(weights.keys()))

    def _threat_scan_line(
        self,
        board: chess.Board,
        style: str,
        rng: random.Random,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        max_checks = int(cfg.get("threat_max_checks", 2))
        max_hanging = int(cfg.get("threat_max_hanging", 2))
        include_empty = bool(cfg.get("threat_scan_include_empty", False))
        side = "White" if board.turn == chess.WHITE else "Black"
        lines = []

        if board.is_check():
            lines.append(f"{side} is in check and must respond.")

        checks = [move for move in board.legal_moves if board.gives_check(move)]
        if checks:
            moves_text = self._format_moves(board, checks, notation, max_checks)
            extra = " and others" if len(checks) > max_checks else ""
            lines.append(f"Checks to consider: {moves_text}{extra}.")

        opponent_board = board.copy()
        opponent_board.turn = not board.turn
        opp_checks = [move for move in opponent_board.legal_moves if opponent_board.gives_check(move)]
        if opp_checks:
            moves_text = self._format_moves(opponent_board, opp_checks, notation, max_checks)
            lines.append(f"Opponent has checking ideas like {moves_text}.")

        hanging = self._find_hanging_pieces(board, max_hanging)
        if hanging:
            pieces = []
            for piece, square in hanging:
                color = "White" if piece.color == chess.WHITE else "Black"
                square_name = chess.square_name(square)
                piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
                pieces.append(f"{color} {piece_name} on {square_name}")
            lines.append(f"Hanging pieces: {', '.join(pieces)}.")

        if not lines:
            if include_empty:
                return rng.choice([
                    "No immediate tactics stand out.",
                    "No forcing tactics jump out.",
                ])
            return ""

        prefix = rng.choice([
            "Quick threat scan:",
            "Immediate tactics:",
            "Tactical scan:",
        ])
        return f"{prefix} " + " ".join(lines)

    def _plan_line(
        self,
        board: chess.Board,
        best_analysis: Optional[Any],
        style: str,
        rng: random.Random,
        cfg: Dict[str, Any],
    ) -> str:
        if board.is_check():
            return rng.choice([
                "First priority is to get out of check cleanly.",
                "The immediate task is handling the check safely.",
            ])

        color = board.turn
        plan_options: List[str] = []

        king_sq = board.king(color)
        if king_sq in (chess.E1, chess.E8):
            if board.has_kingside_castling_rights(color):
                plan_options.append(rng.choice([
                    "Plan: castle kingside and consolidate.",
                    "I want to get the king safe with O-O.",
                ]))
            if board.has_queenside_castling_rights(color):
                plan_options.append(rng.choice([
                    "Plan: castle long and start activity on the kingside.",
                    "Queenside castling is an option to keep the initiative.",
                ]))

        passed_pawn = self._find_passed_pawn(board, color)
        if passed_pawn is not None:
            square_name = chess.square_name(passed_pawn)
            plan_options.append(rng.choice([
                f"The passed pawn on {square_name} is a long-term asset to push.",
                f"I should look to advance the passed pawn on {square_name}.",
            ]))

        open_files = self._open_files(board)
        rooks = board.pieces(chess.ROOK, color)
        for square in rooks:
            file_idx = chess.square_file(square)
            if file_idx in open_files:
                file_letter = "abcdefgh"[file_idx]
                plan_options.append(rng.choice([
                    f"Occupy the open {file_letter}-file with a rook.",
                    f"Rooks belong on the open {file_letter}-file.",
                ]))
                break

        if self._is_exposed_king(board, not color):
            plan_options.append(rng.choice([
                "The enemy king looks exposed, so an attack is natural.",
                "There are chances to go after the king with active pieces.",
            ]))

        if not plan_options:
            plan_options.append(rng.choice([
                "Plan: improve piece activity and keep options open.",
                "I want to coordinate pieces and avoid creating weaknesses.",
                "The plan is to finish development and centralize.",
            ]))

        return rng.choice(plan_options)

    def _opponent_perspective_line(
        self,
        board: chess.Board,
        style: str,
        rng: random.Random,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        opponent = not board.turn
        opp_board = board.copy()
        opp_board.turn = opponent
        checks = [move for move in opp_board.legal_moves if opp_board.gives_check(move)]
        if checks:
            moves_text = self._format_moves(opp_board, checks, notation, int(cfg.get("threat_max_checks", 2)))
            return rng.choice([
                f"From the opponent's view, checks like {moves_text} are the main forcing ideas.",
                f"Opponent may look for checks such as {moves_text}.",
            ])

        hanging = self._find_hanging_pieces_for_color(board, board.turn, 1)
        if hanging:
            piece, square = hanging[0]
            piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
            square_name = chess.square_name(square)
            return rng.choice([
                f"Opponent will try to pick off the {piece_name} on {square_name}.",
                f"The {piece_name} on {square_name} could be a target for the opponent.",
            ])

        return rng.choice([
            "From the opponent's side, the plan is likely to challenge the center.",
            "Opponent probably wants to simplify and relieve the pressure.",
            "Expect the opponent to activate pieces and look for counterplay.",
        ])

    def _format_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        notation: str,
        max_items: int,
    ) -> str:
        shown = moves[:max_items]
        labels = [self._move_notation(board, move, notation) for move in shown]
        return ", ".join(labels)

    def _move_notation(self, board: chess.Board, move: chess.Move, notation: str) -> str:
        if notation == "san":
            try:
                return board.san(move)
            except Exception:
                return move.uci()
        return move.uci()

    def _find_hanging_pieces(
        self,
        board: chess.Board,
        max_items: int,
    ) -> List[Tuple[chess.Piece, chess.Square]]:
        candidates: List[Tuple[int, int, chess.Square, chess.Piece]] = []
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
            attackers = board.attackers(not piece.color, square)
            if not attackers:
                continue
            defenders = board.attackers(piece.color, square)
            if defenders:
                continue
            value = PIECE_VALUES.get(piece.piece_type, 0)
            turn_bias = 1 if piece.color == board.turn else 0
            candidates.append((value, turn_bias, square, piece))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [(piece, square) for _, _, square, piece in candidates[:max_items]]

    def _find_hanging_pieces_for_color(
        self,
        board: chess.Board,
        color: bool,
        max_items: int,
    ) -> List[Tuple[chess.Piece, chess.Square]]:
        candidates: List[Tuple[int, chess.Square, chess.Piece]] = []
        for square, piece in board.piece_map().items():
            if piece.color != color or piece.piece_type == chess.KING:
                continue
            attackers = board.attackers(not color, square)
            if not attackers:
                continue
            defenders = board.attackers(color, square)
            if defenders:
                continue
            value = PIECE_VALUES.get(piece.piece_type, 0)
            candidates.append((value, square, piece))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [(piece, square) for _, square, piece in candidates[:max_items]]

    def _detect_candidate_motifs(
        self,
        board: chess.Board,
        move: chess.Move,
        cand: Any,
        cfg: Dict[str, Any],
    ) -> List[str]:
        motifs: List[str] = []
        max_motifs = int(cfg.get("max_motifs_per_candidate", 1))
        attacker_color = board.color_at(move.from_square)
        temp = board.copy()
        temp.push(move)

        double_check = self._detect_double_check(temp, attacker_color)
        if double_check:
            motifs.append(double_check)
        fork = self._detect_fork(temp, move, attacker_color)
        if fork:
            motifs.append(fork)
        pin = self._detect_pin(temp)
        if pin:
            motifs.append(pin)
        skewer = self._detect_skewer(temp, attacker_color)
        if skewer:
            motifs.append(skewer)
        discovered = self._detect_discovered_attack(board, move, attacker_color)
        if discovered:
            motifs.append(discovered)
        else:
            clearance = self._detect_clearance(board, move, attacker_color)
            if clearance:
                motifs.append(clearance)
        deflection = self._detect_deflection(board, move, attacker_color)
        if deflection:
            motifs.append(deflection)
        xray = self._detect_xray(temp, attacker_color)
        if xray:
            motifs.append(xray)
        sacrifice = self._detect_sacrifice(board, move, attacker_color)
        if sacrifice:
            motifs.append(sacrifice)
        attraction = self._detect_attraction(board, move, attacker_color)
        if attraction:
            motifs.append(attraction)
        trapped = self._detect_trapped_piece(temp, attacker_color)
        if trapped:
            motifs.append(trapped)
        overload = self._detect_overload(temp, attacker_color)
        if overload:
            motifs.append(overload)
        back_rank = self._detect_back_rank_weakness(temp, attacker_color)
        if back_rank:
            motifs.append(back_rank)
        stalemate = self._detect_stalemate_trick(temp, attacker_color)
        if stalemate:
            motifs.append(stalemate)
        perpetual = self._detect_perpetual_idea(board, move, cand, attacker_color, cfg)
        if perpetual:
            motifs.append(perpetual)
        zwischenzug = self._detect_zwischenzug(board, move, attacker_color)
        if zwischenzug:
            motifs.append(zwischenzug)
        weak = self._detect_weak_square_move(board, move, attacker_color)
        if weak:
            motifs.append(weak)
        quiet_move = self._detect_quiet_move(board, move, attacker_color, cfg)
        if quiet_move:
            motifs.append(quiet_move)

        return motifs[:max_motifs]

    def _detect_fork(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        piece = board.piece_at(move.to_square)
        if piece is None or attacker_color is None:
            return None
        enemy_color = not piece.color
        attacked = []
        for square in board.attacks(move.to_square):
            target = board.piece_at(square)
            if not target or target.color != enemy_color:
                continue
            if target.piece_type == chess.KING:
                continue
            value = PIECE_VALUES.get(target.piece_type, 0)
            if value >= 3:
                attacked.append((value, PIECE_NAMES.get(target.piece_type, "piece")))
        if len(attacked) < 2:
            return None
        attacked.sort(key=lambda x: x[0], reverse=True)
        first = attacked[0][1]
        second = attacked[1][1]
        if first == second:
            return f"fork on two {self._plural_piece(first)}"
        return f"fork on the {first} and {second}"

    def _detect_pin(self, board: chess.Board) -> Optional[str]:
        opponent_color = board.turn
        pinned: List[Tuple[int, chess.Piece, chess.Square]] = []
        for square, piece in board.piece_map().items():
            if piece.color != opponent_color:
                continue
            if board.is_pinned(opponent_color, square):
                value = PIECE_VALUES.get(piece.piece_type, 0)
                pinned.append((value, piece, square))
        if not pinned:
            return None
        pinned.sort(key=lambda x: x[0], reverse=True)
        _, piece, square = pinned[0]
        name = PIECE_NAMES.get(piece.piece_type, "piece")
        return f"pin on the {name} at {chess.square_name(square)}"

    def _detect_double_check(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        king_sq = board.king(not attacker_color)
        if king_sq is None:
            return None
        if not board.is_check():
            return None
        attackers = board.attackers(attacker_color, king_sq)
        if len(attackers) >= 2:
            return "double check"
        return None

    def _detect_skewer(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        enemy_color = not attacker_color
        sliders = (
            list(board.pieces(chess.ROOK, attacker_color)) +
            list(board.pieces(chess.BISHOP, attacker_color)) +
            list(board.pieces(chess.QUEEN, attacker_color))
        )
        for slider_sq in sliders:
            for direction in self._slider_directions(board.piece_at(slider_sq)):
                current = slider_sq
                first_enemy = None
                while True:
                    current = self._step_square(current, direction)
                    if current is None:
                        break
                    piece = board.piece_at(current)
                    if piece is None:
                        continue
                    if piece.color == attacker_color:
                        break
                    if first_enemy is None:
                        first_enemy = piece
                        continue
                    second_enemy = piece
                    if first_enemy.piece_type == chess.KING:
                        return "skewer on the king"
                    if PIECE_VALUES.get(first_enemy.piece_type, 0) >= PIECE_VALUES.get(second_enemy.piece_type, 0):
                        first_name = PIECE_NAMES.get(first_enemy.piece_type, "piece")
                        second_name = PIECE_NAMES.get(second_enemy.piece_type, "piece")
                        return f"skewer on the {first_name} to the {second_name}"
                    break
        return None

    def _detect_clearance(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        temp = board.copy()
        temp.push(move)
        king_sq = temp.king(not attacker_color)
        if king_sq is None or not temp.is_check():
            return None
        attackers = temp.attackers(attacker_color, king_sq)
        if move.to_square in attackers:
            return None
        return "clearance for a discovered check"

    def _detect_deflection(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.is_capture(move):
            return None
        captured_square, captured_piece = self._captured_piece(board, move)
        if captured_piece is None or captured_piece.color == attacker_color:
            return None
        defender_color = captured_piece.color
        temp = board.copy()
        temp.push(move)
        targets: List[Tuple[int, str, chess.Square]] = []
        for target_square in board.attacks(captured_square):
            target_piece = board.piece_at(target_square)
            if not target_piece or target_piece.color != defender_color:
                continue
            if target_piece.piece_type == chess.KING:
                continue
            if temp.is_attacked_by(attacker_color, target_square) and not temp.is_attacked_by(defender_color, target_square):
                value = PIECE_VALUES.get(target_piece.piece_type, 0)
                targets.append((value, PIECE_NAMES.get(target_piece.piece_type, "piece"), target_square))
        if not targets:
            return None
        targets.sort(key=lambda x: x[0], reverse=True)
        name = targets[0][1]
        return f"deflection of the {name}"

    def _detect_xray(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        sliders = (
            list(board.pieces(chess.ROOK, attacker_color)) +
            list(board.pieces(chess.BISHOP, attacker_color)) +
            list(board.pieces(chess.QUEEN, attacker_color))
        )
        for slider_sq in sliders:
            piece = board.piece_at(slider_sq)
            for direction in self._slider_directions(piece):
                current = slider_sq
                blocker = None
                while True:
                    current = self._step_square(current, direction)
                    if current is None:
                        break
                    target = board.piece_at(current)
                    if target is None:
                        continue
                    if blocker is None:
                        blocker = target
                        continue
                    if target.color != attacker_color and target.piece_type in (chess.KING, chess.QUEEN):
                        name = PIECE_NAMES.get(target.piece_type, "piece")
                        return f"x-ray on the {name}"
                    break
        return None

    def _detect_attraction(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.gives_check(move):
            return None
        temp = board.copy()
        temp.push(move)
        moved_piece = temp.piece_at(move.to_square)
        if moved_piece is None:
            return None
        if temp.is_attacked_by(not attacker_color, move.to_square) and not temp.is_attacked_by(attacker_color, move.to_square):
            return "decoy on the king"
        return None

    def _detect_trapped_piece(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        enemy_color = not attacker_color
        candidates: List[Tuple[int, str, chess.Square]] = []
        for square, piece in board.piece_map().items():
            if piece.color != enemy_color:
                continue
            if piece.piece_type in (chess.KING, chess.PAWN):
                continue
            if not board.is_attacked_by(attacker_color, square):
                continue
            legal_moves = [m for m in board.legal_moves if m.from_square == square]
            if not legal_moves:
                continue
            safe = False
            for mv in legal_moves:
                temp = board.copy()
                temp.push(mv)
                if not temp.is_attacked_by(attacker_color, mv.to_square):
                    safe = True
                    break
            if not safe:
                value = PIECE_VALUES.get(piece.piece_type, 0)
                name = PIECE_NAMES.get(piece.piece_type, "piece")
                candidates.append((value, name, square))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        name = candidates[0][1]
        return f"trapped {name}"

    @staticmethod
    def _captured_piece(
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[chess.Square, Optional[chess.Piece]]:
        if board.is_en_passant(move):
            offset = -8 if board.turn == chess.WHITE else 8
            capture_square = move.to_square + offset
            return capture_square, board.piece_at(capture_square)
        return move.to_square, board.piece_at(move.to_square)

    def _detect_discovered_attack(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        for slider_sq in (
            list(board.pieces(chess.ROOK, attacker_color)) +
            list(board.pieces(chess.BISHOP, attacker_color)) +
            list(board.pieces(chess.QUEEN, attacker_color))
        ):
            directions = self._slider_directions(board.piece_at(slider_sq))
            for direction in directions:
                current = slider_sq
                blocked = False
                found_mover = False
                while True:
                    current = self._step_square(current, direction)
                    if current is None:
                        break
                    if current == move.from_square:
                        found_mover = True
                        continue
                    piece = board.piece_at(current)
                    if piece is None:
                        continue
                    if not found_mover:
                        blocked = True
                        break
                    if piece.color != attacker_color:
                        return "discovered attack"
                    break
                if found_mover and not blocked:
                    continue
        return None

    def _detect_overload(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        defender_color = not attacker_color
        overloaded: List[Tuple[int, chess.Piece, chess.Square]] = []
        for square, piece in board.piece_map().items():
            if piece.color != defender_color:
                continue
            defended_targets = 0
            for target_square in board.attacks(square):
                target_piece = board.piece_at(target_square)
                if not target_piece or target_piece.color != defender_color:
                    continue
                if board.attackers(attacker_color, target_square):
                    defended_targets += 1
            if defended_targets >= 2:
                value = PIECE_VALUES.get(piece.piece_type, 0)
                overloaded.append((value, piece, square))
        if not overloaded:
            return None
        overloaded.sort(key=lambda x: x[0], reverse=True)
        _, piece, square = overloaded[0]
        name = PIECE_NAMES.get(piece.piece_type, "piece")
        return f"overloaded defender on the {name} at {chess.square_name(square)}"

    def _detect_back_rank_weakness(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        defender_color = not attacker_color
        king_sq = board.king(defender_color)
        if king_sq is None:
            return None
        back_rank = 0 if defender_color == chess.WHITE else 7
        if chess.square_rank(king_sq) != back_rank:
            return None
        escape = False
        for square in board.attacks(king_sq):
            if board.piece_at(square) is not None:
                continue
            if board.is_attacked_by(attacker_color, square):
                continue
            escape = True
            break
        if not escape:
            return "back rank weakness"
        return None

    def _detect_stalemate_trick(
        self,
        board: chess.Board,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if board.is_check():
            return None
        if any(board.legal_moves):
            return None
        return "stalemate trick"

    def _detect_perpetual_idea(
        self,
        board: chess.Board,
        move: chess.Move,
        cand: Any,
        attacker_color: Optional[bool],
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not board.gives_check(move):
            return None
        pv_uci = list(getattr(cand, "pv_uci", None) or [])
        if not pv_uci:
            return None
        max_plies = int(cfg.get("perpetual_max_plies", 6))
        temp = board.copy()
        temp.push(move)
        checks = 1
        for uci in pv_uci[1:max_plies + 1]:
            try:
                next_move = chess.Move.from_uci(uci)
            except Exception:
                break
            if next_move not in temp.legal_moves:
                break
            if temp.turn == attacker_color and temp.gives_check(next_move):
                checks += 1
            temp.push(next_move)
        if checks >= 2:
            return "perpetual check idea"
        return None

    def _detect_sacrifice(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not (board.is_capture(move) or board.gives_check(move)):
            return None
        temp = board.copy()
        temp.push(move)
        piece = temp.piece_at(move.to_square)
        if piece is None or piece.color != attacker_color:
            return None
        if not temp.is_attacked_by(not attacker_color, move.to_square):
            return None
        if temp.is_attacked_by(attacker_color, move.to_square):
            return None
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if value < 3:
            return None
        return "sacrifice"

    def _detect_zwischenzug(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not (board.is_capture(move) or board.gives_check(move)):
            return None
        hanging = self._find_hanging_pieces_for_color(board, attacker_color, 1)
        if not hanging:
            return None
        return "zwischenzug"

    def _detect_quiet_move(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        if not cfg.get("include_quiet_move_motif", True):
            return None
        if board.is_capture(move) or board.gives_check(move) or move.promotion:
            return None
        if board.is_castling(move):
            return None
        return "quiet move"

    @staticmethod
    def _slider_directions(piece: Optional[chess.Piece]) -> List[Tuple[int, int]]:
        if piece is None:
            return []
        if piece.piece_type == chess.ROOK:
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if piece.piece_type == chess.BISHOP:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if piece.piece_type == chess.QUEEN:
            return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        return []

    @staticmethod
    def _step_square(square: chess.Square, direction: Tuple[int, int]) -> Optional[chess.Square]:
        file_idx = chess.square_file(square) + direction[0]
        rank_idx = chess.square_rank(square) + direction[1]
        if file_idx < 0 or file_idx > 7 or rank_idx < 0 or rank_idx > 7:
            return None
        return chess.square(file_idx, rank_idx)

    def _detect_weak_square_move(
        self,
        board: chess.Board,
        move: chess.Move,
        attacker_color: Optional[bool],
    ) -> Optional[str]:
        if attacker_color is None:
            return None
        weak_squares = self._find_weak_squares(board, attacker_color)
        if move.to_square in weak_squares:
            return f"weak square on {chess.square_name(move.to_square)}"
        return None

    def _find_weak_squares(
        self,
        board: chess.Board,
        attacker_color: bool,
    ) -> List[chess.Square]:
        opponent_color = not attacker_color
        weak: List[chess.Square] = []
        for square in WEAK_SQUARE_CANDIDATES:
            rank = chess.square_rank(square)
            if attacker_color == chess.WHITE and rank < 4:
                continue
            if attacker_color == chess.BLACK and rank > 3:
                continue
            if self._pawn_defended(board, opponent_color, square):
                continue
            if not board.attackers(attacker_color, square):
                continue
            weak.append(square)
        return weak

    @staticmethod
    def _pawn_defended(board: chess.Board, color: bool, square: chess.Square) -> bool:
        for attacker in board.attackers(color, square):
            piece = board.piece_at(attacker)
            if piece and piece.piece_type == chess.PAWN:
                return True
        return False

    @staticmethod
    def _open_files(board: chess.Board) -> List[int]:
        pawn_files = set()
        for square in board.pieces(chess.PAWN, chess.WHITE):
            pawn_files.add(chess.square_file(square))
        for square in board.pieces(chess.PAWN, chess.BLACK):
            pawn_files.add(chess.square_file(square))
        return [idx for idx in range(8) if idx not in pawn_files]

    def _half_open_files(
        self,
        board: chess.Board,
        color: bool,
    ) -> List[int]:
        friendly_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
        enemy_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)}
        return [idx for idx in range(8) if idx not in friendly_files and idx in enemy_files]

    @staticmethod
    def _pawn_file_counts(
        board: chess.Board,
        color: bool,
    ) -> Dict[int, List[chess.Square]]:
        files: Dict[int, List[chess.Square]] = {}
        for square in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(square)
            files.setdefault(file_idx, []).append(square)
        return files

    def _find_isolated_pawn(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        pawn_files = self._pawn_file_counts(board, color)
        if not pawn_files:
            return None
        candidates = []
        for file_idx, squares in pawn_files.items():
            if (file_idx - 1) in pawn_files or (file_idx + 1) in pawn_files:
                continue
            for square in squares:
                candidates.append(square)
        if not candidates:
            return None
        if color == chess.WHITE:
            return max(candidates, key=chess.square_rank)
        return min(candidates, key=chess.square_rank)

    def _find_doubled_pawn_file(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[int]:
        pawn_files = self._pawn_file_counts(board, color)
        doubled = [file_idx for file_idx, squares in pawn_files.items() if len(squares) >= 2]
        if not doubled:
            return None
        return sorted(doubled)[0]

    def _find_backward_pawn(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        pawn_files = self._pawn_file_counts(board, color)
        if not pawn_files:
            return None
        direction = 1 if color == chess.WHITE else -1
        candidates = []
        for file_idx, squares in pawn_files.items():
            for square in squares:
                rank = chess.square_rank(square)
                adjacent_pawns = []
                for adj_file in (file_idx - 1, file_idx + 1):
                    if adj_file in pawn_files:
                        adjacent_pawns.extend(pawn_files[adj_file])
                if not adjacent_pawns:
                    continue
                if color == chess.WHITE:
                    if not all(chess.square_rank(p) > rank for p in adjacent_pawns):
                        continue
                else:
                    if not all(chess.square_rank(p) < rank for p in adjacent_pawns):
                        continue
                next_rank = rank + direction
                if next_rank < 0 or next_rank > 7:
                    continue
                forward_sq = chess.square(file_idx, next_rank)
                if board.piece_at(forward_sq) is not None:
                    continue
                if board.is_attacked_by(not color, forward_sq) and not board.is_attacked_by(color, forward_sq):
                    candidates.append(square)
        if not candidates:
            return None
        if color == chess.WHITE:
            return min(candidates, key=chess.square_rank)
        return max(candidates, key=chess.square_rank)

    @staticmethod
    def _pawn_majority(
        board: chess.Board,
        color: bool,
    ) -> Optional[str]:
        pawns = list(board.pieces(chess.PAWN, color))
        if not pawns:
            return None
        queenside = sum(1 for sq in pawns if chess.square_file(sq) <= 3)
        kingside = sum(1 for sq in pawns if chess.square_file(sq) >= 4)
        if abs(queenside - kingside) < 2:
            return None
        if queenside > kingside:
            return "queenside pawn majority"
        return "kingside pawn majority"

    def _find_passed_pawn(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        pawns = list(board.pieces(chess.PAWN, color))
        opponent_pawns = list(board.pieces(chess.PAWN, not color))
        for pawn_sq in pawns:
            file_idx = chess.square_file(pawn_sq)
            rank_idx = chess.square_rank(pawn_sq)
            passed = True
            for opp_sq in opponent_pawns:
                opp_file = chess.square_file(opp_sq)
                if abs(opp_file - file_idx) > 1:
                    continue
                opp_rank = chess.square_rank(opp_sq)
                if color == chess.WHITE and opp_rank > rank_idx:
                    passed = False
                    break
                if color == chess.BLACK and opp_rank < rank_idx:
                    passed = False
                    break
            if passed:
                return pawn_sq
        return None

    def _find_outpost(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[chess.Square]:
        minors = list(board.pieces(chess.KNIGHT, color)) + list(board.pieces(chess.BISHOP, color))
        for square in minors:
            rank = chess.square_rank(square)
            if color == chess.WHITE and rank < 4:
                continue
            if color == chess.BLACK and rank > 3:
                continue
            if not self._pawn_defended(board, color, square):
                continue
            if self._pawn_defended(board, not color, square):
                continue
            return square
        return None

    def _rook_on_seventh(
        self,
        board: chess.Board,
        color: bool,
    ) -> Optional[str]:
        target_rank = 6 if color == chess.WHITE else 1
        for square in board.pieces(chess.ROOK, color):
            if chess.square_rank(square) == target_rank:
                return "rook on the 7th rank" if color == chess.WHITE else "rook on the 2nd rank"
        return None

    def _is_exposed_king(
        self,
        board: chess.Board,
        color: bool,
    ) -> bool:
        king_sq = board.king(color)
        if king_sq is None:
            return False
        file_idx = chess.square_file(king_sq)
        rank_idx = chess.square_rank(king_sq)
        shield_rank = rank_idx + 1 if color == chess.WHITE else rank_idx - 1
        if shield_rank < 0 or shield_rank > 7:
            return False
        pawn_count = 0
        for delta in (-1, 0, 1):
            file_check = file_idx + delta
            if file_check < 0 or file_check > 7:
                continue
            square = chess.square(file_check, shield_rank)
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type == chess.PAWN:
                pawn_count += 1
        return pawn_count < 2

    @staticmethod
    def _plural_piece(name: str) -> str:
        if name.endswith("s"):
            return name
        if name.endswith("y"):
            return name[:-1] + "ies"
        return name + "s"

    def _motif_phrase(self, motifs: List[str], rng: random.Random) -> str:
        if not motifs:
            return ""
        prefix = rng.choice([
            "Motif:",
            "Tactical motif:",
            "Pattern:",
            "Notice:",
        ])
        return f"{prefix} {'; '.join(motifs)}."

    def _detect_trap_for_candidate(
        self,
        cand: Any,
        analysis: Any,
        cfg: Dict[str, Any],
    ) -> Optional[Tuple[str, str, str]]:
        shallow_cps = getattr(analysis, "shallow_move_cps", None) or {}
        shallow_win = getattr(analysis, "shallow_move_win_probs", None) or {}
        if not shallow_cps and not shallow_win:
            return None

        shallow_cp = shallow_cps.get(cand.uci)
        shallow_wp = shallow_win.get(cand.uci)
        best_shallow_cp = max(shallow_cps.values()) if shallow_cps else None
        best_shallow_wp = max(shallow_win.values()) if shallow_win else None

        trap_min_cp_swing = int(cfg.get("trap_min_cp_swing", 80))
        trap_min_wp_swing = float(cfg.get("trap_min_win_prob_swing", 0.12))
        trap_shallow_good_cp = int(cfg.get("trap_shallow_good_cp", 30))
        trap_shallow_bad_cp = int(cfg.get("trap_shallow_bad_cp", 120))
        trap_shallow_good_wp = float(cfg.get("trap_shallow_good_win_prob", 0.06))
        trap_shallow_bad_wp = float(cfg.get("trap_shallow_bad_win_prob", 0.18))

        deep_cp = getattr(cand, "centipawn", None)
        deep_wp = getattr(cand, "win_probability", None)

        if shallow_cp is not None and deep_cp is not None and best_shallow_cp is not None:
            cp_swing = abs(deep_cp - shallow_cp)
            shallow_good = shallow_cp >= best_shallow_cp - trap_shallow_good_cp
            shallow_bad = shallow_cp <= best_shallow_cp - trap_shallow_bad_cp
            deep_bad = deep_cp <= best_shallow_cp - trap_shallow_bad_cp
            deep_good = deep_cp >= best_shallow_cp - trap_shallow_good_cp

            if cp_swing >= trap_min_cp_swing and shallow_good and deep_bad:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "trap",
                    "looks strong at first glance but deeper analysis shows a problem",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )
            if cp_swing >= trap_min_cp_swing and shallow_bad and deep_good:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "resource",
                    "looks dubious at first glance but deeper analysis improves it",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )

        if shallow_wp is not None and deep_wp is not None and best_shallow_wp is not None:
            wp_swing = abs(deep_wp - shallow_wp)
            shallow_good = shallow_wp >= best_shallow_wp - trap_shallow_good_wp
            shallow_bad = shallow_wp <= best_shallow_wp - trap_shallow_bad_wp
            deep_bad = deep_wp <= best_shallow_wp - trap_shallow_bad_wp
            deep_good = deep_wp >= best_shallow_wp - trap_shallow_good_wp

            if wp_swing >= trap_min_wp_swing and shallow_good and deep_bad:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "trap",
                    "looks tempting at shallow depth but deeper analysis refutes it",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )
            if wp_swing >= trap_min_wp_swing and shallow_bad and deep_good:
                return self._confirm_trap(
                    cand,
                    analysis,
                    cfg,
                    "resource",
                    "looks quiet at shallow depth but hides a stronger idea",
                    deep_cp=deep_cp,
                    deep_wp=deep_wp,
                )

        return None

    def _trap_phrase(
        self,
        trap: Optional[Tuple[str, str, str]],
        rng: random.Random,
        board: chess.Board,
        cand: Any,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        if not trap:
            return ""
        kind, detail, confidence = trap
        if confidence != "confirmed":
            lead = rng.choice([
                "Possible trap:",
                "Likely trap:",
                "Looks tricky:",
            ]) if kind == "trap" else rng.choice([
                "Possible resource:",
                "Hidden resource:",
                "Looks better than it seems:",
            ])
            line = f"{lead} {detail}."
            refutation = self._trap_refutation_line(board, cand, notation, cfg)
            if refutation:
                return f"{line} {refutation}"
            return line
        if kind == "trap":
            lead = rng.choice([
                "Confirmed trap:",
                "Trap alert:",
                "Warning:",
                "Caution:",
            ])
        else:
            lead = rng.choice([
                "Confirmed resource:",
                "Hidden resource:",
                "Quiet resource:",
                "Surprise:",
            ])
        line = f"{lead} {detail}."
        refutation = self._trap_refutation_line(board, cand, notation, cfg)
        if refutation:
            return f"{line} {refutation}"
        return line

    def _confirm_trap(
        self,
        cand: Any,
        analysis: Any,
        cfg: Dict[str, Any],
        kind: str,
        detail: str,
        deep_cp: Optional[int] = None,
        deep_wp: Optional[float] = None,
    ) -> Tuple[str, str, str]:
        confirm_cps = getattr(analysis, "confirm_move_cps", None) or {}
        confirm_wps = getattr(analysis, "confirm_move_win_probs", None) or {}
        confirm_cp = confirm_cps.get(cand.uci)
        confirm_wp = confirm_wps.get(cand.uci)

        confidence = "uncertain"
        if confirm_cp is not None and deep_cp is not None:
            tol = int(cfg.get("trap_confirm_cp_tolerance", 30))
            confidence = "confirmed" if abs(confirm_cp - deep_cp) <= tol else "uncertain"
        elif confirm_wp is not None and deep_wp is not None:
            tol = float(cfg.get("trap_confirm_win_prob_tolerance", 0.05))
            confidence = "confirmed" if abs(confirm_wp - deep_wp) <= tol else "uncertain"

        return (kind, detail, confidence)

    def _apply_guardrails(
        self,
        text: str,
        tablebase: Optional[TablebaseInfo],
        cfg: Dict[str, Any],
    ) -> str:
        if not text:
            return text
        if not cfg.get("guardrail_enabled", True):
            return text

        if tablebase and tablebase.result == "drawing":
            text = self._replace_word(text, "winning", "drawn")
            text = self._replace_word(text, "losing", "drawn")
            text = self._replace_word(text, "win", "draw")
            text = self._replace_word(text, "loss", "draw")
        return text

    @staticmethod
    def _replace_word(text: str, target: str, replacement: str) -> str:
        pattern = re.compile(rf"\\b{re.escape(target)}\\b", flags=re.IGNORECASE)

        def repl(match: re.Match[str]) -> str:
            word = match.group(0)
            if word.isupper():
                return replacement.upper()
            if word[:1].isupper():
                return replacement.capitalize()
            return replacement

        return pattern.sub(repl, text)

    def _trap_refutation_line(
        self,
        board: chess.Board,
        cand: Any,
        notation: str,
        cfg: Dict[str, Any],
    ) -> str:
        max_len = int(cfg.get("trap_refutation_max_len", 4))
        if max_len <= 0:
            return ""
        pv_uci = list(getattr(cand, "pv_uci", None) or [])
        if not pv_uci:
            return ""

        start_board = board
        if pv_uci and pv_uci[0] == getattr(cand, "uci", ""):
            pv_uci = pv_uci[1:]
            try:
                start_board = board.copy()
                start_board.push(chess.Move.from_uci(getattr(cand, "uci", "")))
            except Exception:
                start_board = board

        if not pv_uci:
            return ""

        pruned_moves, _ = self._prune_pv_moves(start_board, pv_uci[:max_len], cfg)
        line = self._format_pv_line(start_board, pruned_moves, notation)
        if not line:
            return ""
        return f"Line: {line}."

    def _format_pv_line(
        self,
        board: chess.Board,
        pv_uci: List[str],
        notation: str,
    ) -> str:
        if not pv_uci:
            return ""
        if notation != "san":
            return " ".join(pv_uci)

        temp = board.copy()
        san_moves: List[str] = []
        for uci in pv_uci:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in temp.legal_moves:
                break
            try:
                san_moves.append(temp.san(move))
            except Exception:
                san_moves.append(uci)
            temp.push(move)
        return " ".join(san_moves)

    def _apply_budget(
        self,
        sections: List[Tuple[str, str]],
        exploration_paragraphs: List[str],
        cfg: Dict[str, Any],
    ) -> str:
        max_tokens = cfg.get("max_trace_tokens")
        if not max_tokens:
            return self._join_sections(sections)

        def build_text() -> str:
            return self._join_sections(sections)

        text = build_text()
        if self._count_tokens(text) <= max_tokens:
            return text

        # First, trim exploration paragraphs.
        if exploration_paragraphs:
            while exploration_paragraphs and self._count_tokens(text) > max_tokens:
                exploration_paragraphs.pop()
                for idx, (name, _) in enumerate(sections):
                    if name == "exploration":
                        sections[idx] = (
                            name,
                            "\n\n".join(exploration_paragraphs),
                        )
                        break
                text = build_text()
            if self._count_tokens(text) <= max_tokens:
                return text

        # Drop optional sections in priority order.
        drop_order = [
            "comparison",
            "dead_end",
            "reconsideration",
            "exploration",
            "candidates",
            "threat_scan",
            "opponent",
            "plan",
            "assessment",
            "orientation",
        ]
        for drop in drop_order:
            if self._count_tokens(text) <= max_tokens:
                break
            sections = [
                (name, body if name != drop else "")
                for name, body in sections
            ]
            text = build_text()

        if self._count_tokens(text) <= max_tokens:
            return text

        # Final fallback: return conclusion only.
        conclusion = ""
        for name, body in sections:
            if name == "conclusion":
                conclusion = body
                break
        return conclusion

    def _join_sections(self, sections: List[Tuple[str, str]]) -> str:
        return "\n\n".join(body for _, body in sections if body)

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, max(len(text.split()), len(text) // 4))

    @staticmethod
    def _default_opening_paths() -> List[Path]:
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "data" / "openings",
            root / "data" / "chess-openings",
            root / "data" / "chess_openings",
        ]
        paths: List[Path] = []
        for folder in candidates:
            if folder.exists():
                paths.extend(sorted(folder.glob("*.tsv")))
        return paths
