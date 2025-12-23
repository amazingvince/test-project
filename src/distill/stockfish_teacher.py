"""
Stockfish Teacher for Policy Distillation.

Provides a worker pool for efficient on-the-fly position analysis,
converting Stockfish evaluations to probability distributions for
knowledge distillation into LLMs.
"""

import chess
import chess.engine
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from collections import OrderedDict
import threading
import time
from pathlib import Path


@dataclass
class MoveAnalysis:
    """Analysis of a single move."""
    uci: str
    san: str
    centipawn: int
    cp_loss: int  # Loss from best move (0 for best)
    category: str  # Human-readable category
    mate_in: Optional[int] = None
    win_probability: float = 0.5  # Win probability (0.0 to 1.0)
    pv_uci: List[str] = field(default_factory=list)
    
    def format_for_thinking(self, is_best: bool = False) -> str:
        """Format move for inclusion in thinking tags using win probability."""
        win_pct = int(self.win_probability * 100)

        if self.mate_in is not None:
            if self.mate_in > 0:
                return f"- {self.san}: winning, mate in {self.mate_in}"
            else:
                return f"- {self.san}: losing, gets mated in {abs(self.mate_in)}"

        # Categorize by win probability
        if self.win_probability >= 0.80:
            category = "winning"
        elif self.win_probability >= 0.65:
            category = "better"
        elif self.win_probability >= 0.55:
            category = "slight edge"
        elif self.win_probability >= 0.45:
            category = "equal"
        elif self.win_probability >= 0.35:
            category = "slight disadvantage"
        elif self.win_probability >= 0.20:
            category = "worse"
        else:
            category = "losing"

        return f"- {self.san}: {category} ({win_pct}% win)"


@dataclass
class PositionAnalysis:
    """Complete analysis of a position with probability distribution."""
    fen: str
    move_analyses: List[MoveAnalysis]
    best_move_uci: str
    best_move_san: str
    best_score_cp: int

    # Principal variation (best line) in SAN notation
    best_pv: List[str] = field(default_factory=list)

    # Probability distribution over ALL legal moves
    move_probs: Dict[str, float] = field(default_factory=dict)

    # Top-k that were deeply analyzed
    top_k_moves: List[str] = field(default_factory=list)

    # Optional shallow pass scores (for trap detection)
    shallow_move_cps: Dict[str, int] = field(default_factory=dict)
    shallow_move_win_probs: Dict[str, float] = field(default_factory=dict)

    # Optional confirm (extra-deep) scores for trap confirmation
    confirm_move_cps: Dict[str, int] = field(default_factory=dict)
    confirm_move_win_probs: Dict[str, float] = field(default_factory=dict)
    
    def get_thinking_text(self, randomize_order: bool = True, max_display: int = 5) -> str:
        """
        Generate thinking text with move analyses.

        Args:
            randomize_order: Shuffle move order to prevent positional shortcuts
            max_display: Maximum moves to show in thinking
        """
        # Get moves to display (top by probability, but we'll shuffle)
        display_moves = self.move_analyses[:max_display]

        if randomize_order:
            # Shuffle but keep track of which is best
            import random
            display_moves = display_moves.copy()
            random.shuffle(display_moves)

        lines = ["Analyzing position..."]
        for ma in display_moves:
            is_best = (ma.uci == self.best_move_uci)
            lines.append(ma.format_for_thinking(is_best=is_best))

        # Show best line (PV)
        if self.best_pv:
            pv_str = ' '.join(self.best_pv[:5])
            lines.append(f"\nBest line: {pv_str}")
        else:
            lines.append(f"\nBest line: {self.best_move_san}")

        # Find best move's win probability
        best_win_pct = 50
        for ma in self.move_analyses:
            if ma.uci == self.best_move_uci:
                best_win_pct = int(ma.win_probability * 100)
                break

        lines.append(f"Playing {self.best_move_san} gives {best_win_pct}% win chance.")

        return "\n".join(lines)


def categorize_move(cp_loss: int, mate_in: Optional[int] = None) -> str:
    """
    Convert centipawn loss to human-readable category.
    
    Categories align with standard chess analysis terminology
    that appears in LLM pretraining data.
    """
    if mate_in is not None:
        if mate_in > 0:
            return "winning mate"
        else:
            return "blunder"  # Getting mated
    
    if cp_loss == 0:
        return "best move"
    elif cp_loss <= 10:
        return "excellent"
    elif cp_loss <= 30:
        return "good"
    elif cp_loss <= 50:
        return "slight inaccuracy"
    elif cp_loss <= 100:
        return "inaccuracy"
    elif cp_loss <= 200:
        return "mistake"
    else:
        return "blunder"


def cp_to_win_probability(cp: int) -> float:
    """
    Approximate win probability from centipawn score.

    Uses a logistic transform similar to common chess heuristics.
    """
    return 1.0 / (1.0 + 10 ** (-cp / 400.0))


def cp_to_probability_distribution(
    move_cps: Dict[str, int],
    all_legal_moves: List[str],
    temperature: float = 100.0,
    min_probability: float = 0.001,
) -> Dict[str, float]:
    """
    Convert centipawn evaluations to a probability distribution.
    
    Args:
        move_cps: Dict mapping analyzed UCI moves to centipawn evaluations
        all_legal_moves: List of ALL legal moves
        temperature: Softmax temperature (higher = softer distribution)
        min_probability: Minimum probability floor for every legal move
    
    Returns:
        Dict mapping all legal moves to probabilities (sums to 1.0)
    """
    num_moves = len(all_legal_moves)
    if num_moves == 0:
        return {}

    reserved_mass = min_probability * num_moves
    if reserved_mass >= 1.0:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    remaining_mass = 1.0 - reserved_mass

    if not move_cps:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    moves = list(move_cps.keys())
    cps = np.array([move_cps[m] for m in moves], dtype=np.float64)

    # Normalize to prevent overflow
    cps = cps - cps.max()

    exp_scores = np.exp(cps / temperature)
    softmax_probs = exp_scores / exp_scores.sum()

    probs = {m: min_probability for m in all_legal_moves}
    for move, prob in zip(moves, softmax_probs):
        probs[move] += float(prob) * remaining_mass

    # Normalize to exactly 1.0 (handle floating point errors)
    total = sum(probs.values())
    probs = {m: p / total for m, p in probs.items()}

    return probs


def win_prob_to_distribution(
    move_win_probs: Dict[str, float],
    all_legal_moves: List[str],
    temperature: float = 1.0,
    min_probability: float = 0.001,
) -> Dict[str, float]:
    """
    Convert win probabilities to a probability distribution.

    Args:
        move_win_probs: Dict mapping UCI moves to win probabilities [0, 1]
        all_legal_moves: List of ALL legal moves
        temperature: Softmax temperature for win-prob logits
        min_probability: Minimum probability floor for every legal move

    Returns:
        Dict mapping all legal moves to probabilities (sums to 1.0)
    """
    num_moves = len(all_legal_moves)
    if num_moves == 0:
        return {}

    reserved_mass = min_probability * num_moves
    if reserved_mass >= 1.0:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    remaining_mass = 1.0 - reserved_mass

    if not move_win_probs:
        uniform_prob = 1.0 / num_moves
        return {m: uniform_prob for m in all_legal_moves}

    moves = []
    scores = []
    for move in all_legal_moves:
        win_prob = move_win_probs.get(move, 0.5)
        win_prob = min(max(win_prob, 1e-6), 1.0 - 1e-6)
        logit = math.log(win_prob / (1.0 - win_prob))
        moves.append(move)
        scores.append(logit)

    scores = np.array(scores, dtype=np.float64)
    scores = scores - scores.max()
    exp_scores = np.exp(scores / max(temperature, 1e-6))
    softmax_probs = exp_scores / exp_scores.sum()

    probs = {m: min_probability for m in all_legal_moves}
    for move, prob in zip(moves, softmax_probs):
        probs[move] += float(prob) * remaining_mass

    total = sum(probs.values())
    probs = {m: p / total for m, p in probs.items()}

    return probs

class StockfishWorker:
    """Single Stockfish engine worker."""
    
    def __init__(
        self,
        stockfish_path: str,
        top_k: int = 5,
        depth: int = 12,
        hash_mb: int = 64,
        temperature: float = 100.0,
        min_probability: float = 0.001,
        threads: int = 1,
        time_limit_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        shallow_depth: int = 0,
        shallow_max_moves: Optional[int] = None,
        confirm_depth: int = 0,
        confirm_top_k: int = 0,
        prob_mode: str = "cp",
        wdl_temperature: float = 1.0,
        syzygy_path: Optional[str] = None,
    ):
        self.stockfish_path = stockfish_path
        self.top_k = top_k
        self.depth = depth
        self.hash_mb = hash_mb
        self.temperature = temperature
        self.min_probability = min_probability
        self.threads = max(1, threads)
        self.time_limit_ms = time_limit_ms
        self.nodes = nodes
        self.shallow_depth = shallow_depth
        self.shallow_max_moves = shallow_max_moves
        self.confirm_depth = confirm_depth
        self.confirm_top_k = confirm_top_k
        self.prob_mode = prob_mode
        self.wdl_temperature = 1.0 if wdl_temperature is None else wdl_temperature
        self.syzygy_path = syzygy_path
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._lock = threading.Lock()
    
    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            config = {
                "Threads": self.threads,
                "Hash": self.hash_mb,
            }
            # Add Syzygy tablebase path if provided
            if self.syzygy_path:
                config["SyzygyPath"] = self.syzygy_path
            self._engine.configure(config)
        return self._engine

    def _build_limit(
        self,
        depth: Optional[int],
        time_limit_ms: Optional[int],
        nodes: Optional[int],
    ) -> chess.engine.Limit:
        limit_kwargs: Dict[str, Any] = {}
        if depth is not None and depth > 0:
            limit_kwargs["depth"] = depth
        if time_limit_ms is not None and time_limit_ms > 0:
            limit_kwargs["time"] = time_limit_ms / 1000.0
        if nodes is not None and nodes > 0:
            limit_kwargs["nodes"] = nodes
        if not limit_kwargs:
            limit_kwargs["depth"] = self.depth
        return chess.engine.Limit(**limit_kwargs)

    @staticmethod
    def _normalize_analysis_result(result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            return [result]
        return result
    
    def analyze(
        self,
        board: chess.Board,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        shallow_depth: Optional[int] = None,
        shallow_max_moves: Optional[int] = None,
        confirm_depth: Optional[int] = None,
        confirm_top_k: Optional[int] = None,
        prob_mode: Optional[str] = None,
        wdl_temperature: Optional[float] = None,
    ) -> PositionAnalysis:
        """
        Analyze a position.
        
        Args:
            board: Chess position
            top_k: Number of moves for deep analysis
            depth: Search depth for deep analysis
            time_limit_ms: Time limit per analysis in milliseconds
            nodes: Node limit per analysis
            shallow_depth: Optional shallow depth for all-moves pass
            shallow_max_moves: Cap on shallow multipv moves
            prob_mode: "cp" or "wdl"
            wdl_temperature: Temperature for win-prob softmax
        
        Returns:
            PositionAnalysis with full probability distribution
        """
        with self._lock:
            engine = self._get_engine()

            legal_moves = list(board.legal_moves)
            num_legal = len(legal_moves)

            if num_legal == 0:
                raise ValueError("No legal moves in position")

            resolved_top_k = self.top_k if top_k is None else top_k
            resolved_depth = self.depth if depth is None else depth
            resolved_time_limit_ms = self.time_limit_ms if time_limit_ms is None else time_limit_ms
            resolved_nodes = self.nodes if nodes is None else nodes
            resolved_shallow_depth = self.shallow_depth if shallow_depth is None else shallow_depth
            resolved_shallow_max_moves = (
                self.shallow_max_moves if shallow_max_moves is None else shallow_max_moves
            )
            resolved_confirm_depth = (
                self.confirm_depth if confirm_depth is None else confirm_depth
            )
            resolved_confirm_top_k = (
                self.confirm_top_k if confirm_top_k is None else confirm_top_k
            )
            resolved_prob_mode = self.prob_mode if prob_mode is None else prob_mode
            resolved_wdl_temperature = (
                self.wdl_temperature if wdl_temperature is None else wdl_temperature
            )

            if resolved_depth is not None and resolved_depth <= 0:
                resolved_depth = None
            if resolved_shallow_depth is not None and resolved_shallow_depth <= 0:
                resolved_shallow_depth = 0
            if resolved_time_limit_ms is not None or resolved_nodes is not None:
                resolved_depth = None
            if resolved_confirm_depth is not None and resolved_confirm_depth <= 0:
                resolved_confirm_depth = 0
            resolved_confirm_top_k = max(int(resolved_confirm_top_k or 0), 0)

            if resolved_prob_mode not in {"cp", "wdl"}:
                raise ValueError(f"Unsupported prob_mode: {resolved_prob_mode}")

            all_legal_uci = [m.uci() for m in legal_moves]

            move_cps: Dict[str, int] = {}
            move_win_probs: Dict[str, float] = {}
            shallow_move_cps: Dict[str, int] = {}
            shallow_move_win_probs: Dict[str, float] = {}
            confirm_move_cps: Dict[str, int] = {}
            confirm_move_win_probs: Dict[str, float] = {}
            move_analyses: List[MoveAnalysis] = []
            best_cp = None
            best_move_uci = None
            best_move_san = None
            best_pv: List[str] = []

            # Optional shallow pass to score all moves cheaply
            if resolved_shallow_depth is not None and resolved_shallow_depth > 0:
                shallow_k = num_legal
                if resolved_shallow_max_moves is not None:
                    shallow_k = min(shallow_k, resolved_shallow_max_moves)
                if shallow_k > 0:
                    shallow_limit = self._build_limit(
                        resolved_shallow_depth,
                        resolved_time_limit_ms,
                        resolved_nodes,
                    )
                    shallow_result = engine.analyse(
                        board,
                        shallow_limit,
                        multipv=shallow_k,
                    )
                    for info in self._normalize_analysis_result(shallow_result):
                        if 'pv' not in info or len(info['pv']) == 0:
                            continue
                        pv_moves = info['pv']
                        move = pv_moves[0]
                        score = info.get('score')
                        if score is None:
                            continue
                        pov_score = score.pov(board.turn)
                        if pov_score.is_mate():
                            mate_in = pov_score.mate()
                            cp = 30000 if mate_in > 0 else -30000
                            win_prob = 1.0 if mate_in > 0 else 0.0
                        else:
                            cp = pov_score.score()
                            try:
                                wdl = pov_score.wdl()
                                win_prob = (wdl.wins + wdl.draws * 0.5) / 1000.0
                            except Exception:
                                win_prob = cp_to_win_probability(cp)

                        uci = move.uci()
                        shallow_move_cps[uci] = cp
                        shallow_move_win_probs[uci] = win_prob
                        move_cps[uci] = cp
                        move_win_probs[uci] = win_prob

            # Deep analysis of top-k moves
            actual_k = min(max(int(resolved_top_k), 1), num_legal)
            deep_limit = self._build_limit(
                resolved_depth,
                resolved_time_limit_ms,
                resolved_nodes,
            )
            deep_result = engine.analyse(
                board,
                deep_limit,
                multipv=actual_k
            )

            deep_entries = []
            for info in self._normalize_analysis_result(deep_result):
                if 'pv' not in info or len(info['pv']) == 0:
                    continue

                pv_moves = info['pv']
                pv_uci = [pv_move.uci() for pv_move in pv_moves[:10]]
                move = pv_moves[0]
                score = info.get('score')

                if score is None:
                    continue

                pov_score = score.pov(board.turn)

                if pov_score.is_mate():
                    mate_in = pov_score.mate()
                    cp = 30000 if mate_in > 0 else -30000
                    win_prob = 1.0 if mate_in > 0 else 0.0
                else:
                    cp = pov_score.score()
                    mate_in = None
                    try:
                        wdl = pov_score.wdl()
                        win_prob = (wdl.wins + wdl.draws * 0.5) / 1000.0
                    except Exception:
                        win_prob = cp_to_win_probability(cp)

                uci = move.uci()
                san = board.san(move)

                move_cps[uci] = cp
                move_win_probs[uci] = win_prob
                deep_entries.append((uci, san, cp, mate_in, win_prob, pv_moves, pv_uci))

                if best_cp is None or cp > best_cp:
                    best_cp = cp
                    best_move_uci = uci
                    best_move_san = san
                    best_pv = []
                    temp_board = board.copy()
                    for pv_move in pv_moves[:10]:
                        try:
                            best_pv.append(temp_board.san(pv_move))
                            temp_board.push(pv_move)
                        except Exception:
                            break

            # Fallback best move from shallow pass if deep pass failed
            if best_cp is None and move_cps:
                best_move_uci = max(move_cps.items(), key=lambda x: x[1])[0]
                best_cp = move_cps.get(best_move_uci, 0)
                try:
                    best_move_san = board.san(chess.Move.from_uci(best_move_uci))
                except Exception:
                    best_move_san = best_move_uci

            # Calculate CP loss and create MoveAnalysis objects for deep moves
            if best_cp is None:
                best_cp = 0

            for uci, san, cp, mate_in, win_prob, _, pv_uci in deep_entries:
                cp_loss = best_cp - cp
                category = categorize_move(cp_loss, mate_in)

                move_analyses.append(MoveAnalysis(
                    uci=uci,
                    san=san,
                    centipawn=cp,
                    cp_loss=cp_loss,
                    category=category,
                    mate_in=mate_in,
                    win_probability=win_prob,
                    pv_uci=pv_uci,
                ))

            move_analyses.sort(key=lambda x: x.centipawn, reverse=True)

            # Optional confirm pass at higher depth for trap detection
            if resolved_confirm_depth and resolved_confirm_top_k > 0:
                confirm_candidates: List[Tuple[float, str]] = []
                if shallow_move_cps:
                    for uci, san, cp, mate_in, win_prob, pv_moves, _ in deep_entries:
                        shallow_cp = shallow_move_cps.get(uci)
                        if shallow_cp is None:
                            continue
                        swing = abs(cp - shallow_cp)
                        confirm_candidates.append((swing, uci))
                    confirm_candidates.sort(key=lambda x: x[0], reverse=True)
                else:
                    confirm_candidates = [(0.0, uci) for uci, _, _, _, _, _, _ in deep_entries]

                confirm_candidates = confirm_candidates[:resolved_confirm_top_k]
                if confirm_candidates:
                    confirm_limit = self._build_limit(
                        resolved_confirm_depth,
                        None,
                        None,
                    )
                    for _, uci in confirm_candidates:
                        try:
                            move = chess.Move.from_uci(uci)
                        except ValueError:
                            continue
                        if move not in board.legal_moves:
                            continue
                        board_after = board.copy()
                        board_after.push(move)
                        try:
                            confirm_result = engine.analyse(
                                board_after,
                                confirm_limit,
                                multipv=1,
                            )
                        except Exception:
                            continue
                        info = self._normalize_analysis_result(confirm_result)[0]
                        score = info.get("score")
                        if score is None:
                            continue
                        pov_score = score.pov(board_after.turn)
                        if pov_score.is_mate():
                            mate_in = pov_score.mate()
                            cp_after = 30000 if mate_in > 0 else -30000
                            win_prob_opponent = 1.0 if mate_in > 0 else 0.0
                        else:
                            cp_after = pov_score.score()
                            try:
                                wdl = pov_score.wdl()
                                win_prob_opponent = (wdl.wins + wdl.draws * 0.5) / 1000.0
                            except Exception:
                                win_prob_opponent = cp_to_win_probability(cp_after)

                        cp_original = -cp_after
                        win_prob_original = 1.0 - win_prob_opponent
                        confirm_move_cps[uci] = cp_original
                        confirm_move_win_probs[uci] = win_prob_original

            # Convert to probability distribution
            if resolved_prob_mode == "wdl":
                full_win_probs: Dict[str, float] = {}
                for uci in all_legal_uci:
                    if uci in move_win_probs:
                        win_prob = move_win_probs[uci]
                    elif uci in move_cps:
                        win_prob = cp_to_win_probability(move_cps[uci])
                    else:
                        win_prob = 0.5
                    full_win_probs[uci] = win_prob

                move_probs = win_prob_to_distribution(
                    move_win_probs=full_win_probs,
                    all_legal_moves=all_legal_uci,
                    temperature=resolved_wdl_temperature,
                    min_probability=self.min_probability,
                )
            else:
                move_probs = cp_to_probability_distribution(
                    move_cps=move_cps,
                    all_legal_moves=all_legal_uci,
                    temperature=self.temperature,
                    min_probability=self.min_probability,
                )

            return PositionAnalysis(
                fen=board.fen(),
                move_analyses=move_analyses,
                best_move_uci=best_move_uci or "",
                best_move_san=best_move_san or "",
                best_score_cp=best_cp or 0,
                best_pv=best_pv,
                move_probs=move_probs,
                top_k_moves=[ma.uci for ma in move_analyses],
                shallow_move_cps=shallow_move_cps,
                shallow_move_win_probs=shallow_move_win_probs,
                confirm_move_cps=confirm_move_cps,
                confirm_move_win_probs=confirm_move_win_probs,
            )
    
    def close(self):
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
    
    def __del__(self):
        self.close()


class StockfishTeacher:
    """
    Pool of Stockfish workers for efficient streaming analysis.
    
    Provides async-friendly interface for on-the-fly position analysis
    during training.
    """
    
    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        num_workers: int = 4,
        depth: int = 12,
        top_k: int = 5,
        hash_mb_per_worker: int = 64,
        temperature: float = 100.0,
        min_probability: float = 0.001,
        threads_per_worker: int = 1,
        time_limit_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        shallow_depth: int = 0,
        shallow_max_moves: Optional[int] = None,
        confirm_depth: int = 0,
        confirm_top_k: int = 0,
        prob_mode: str = "cp",
        wdl_temperature: float = 1.0,
        cache_size: int = 0,
        syzygy_path: Optional[str] = None,
    ):
        """
        Initialize teacher with worker pool.

        Args:
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            num_workers: Number of parallel Stockfish instances
            depth: Search depth for analysis
            top_k: Number of moves for deep analysis
            hash_mb_per_worker: Hash table size per worker
            temperature: Softmax temperature for probability distribution
            min_probability: Floor probability for unanalyzed moves
            threads_per_worker: Engine threads per worker
            time_limit_ms: Time limit per analysis in milliseconds
            nodes: Node limit per analysis
            shallow_depth: Optional shallow depth for all-moves pass
            shallow_max_moves: Cap shallow multipv to avoid large fanout
            prob_mode: "cp" or "wdl" for probability conversion
            wdl_temperature: Temperature for win-prob softmax
            cache_size: LRU cache size for position analyses (0 = disabled)
            syzygy_path: Path to Syzygy tablebase files (enables perfect endgame play)
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.num_workers = num_workers
        self.depth = depth
        self.top_k = top_k
        self.hash_mb_per_worker = hash_mb_per_worker
        self.temperature = temperature
        self.min_probability = min_probability
        self.threads_per_worker = max(1, threads_per_worker)
        self.time_limit_ms = time_limit_ms
        self.nodes = nodes
        self.shallow_depth = shallow_depth
        self.shallow_max_moves = shallow_max_moves
        self.confirm_depth = confirm_depth
        self.confirm_top_k = confirm_top_k
        self.prob_mode = prob_mode
        self.wdl_temperature = 1.0 if wdl_temperature is None else wdl_temperature
        self.cache_size = max(cache_size or 0, 0)
        self.syzygy_path = syzygy_path

        if self.prob_mode not in {"cp", "wdl"}:
            raise ValueError(f"Unsupported prob_mode: {self.prob_mode}")
        
        # Worker pool
        self._workers: List[StockfishWorker] = []
        self._worker_queue: Queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        # Analysis cache + inflight tracking
        self._analysis_cache: "OrderedDict[Tuple[Any, ...], PositionAnalysis]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inflight: Dict[Tuple[Any, ...], Future] = {}
        self._inflight_lock = threading.Lock()
        
        self._initialized = False
    
    @staticmethod
    def _find_stockfish() -> str:
        """Find Stockfish binary."""
        import shutil
        
        paths = [
            shutil.which('stockfish'),
            '/usr/bin/stockfish',
            '/usr/games/stockfish',
            '/usr/local/bin/stockfish',
            '/opt/homebrew/bin/stockfish',
        ]
        
        for path in paths:
            if path and Path(path).exists():
                return path
        
        raise FileNotFoundError(
            "Stockfish not found. Install with: sudo apt install stockfish"
        )
    
    def _initialize_workers(self):
        """Lazily initialize workers."""
        if self._initialized:
            return
        
        for _ in range(self.num_workers):
            worker = StockfishWorker(
                stockfish_path=self.stockfish_path,
                top_k=self.top_k,
                depth=self.depth,
                hash_mb=self.hash_mb_per_worker,
                temperature=self.temperature,
                min_probability=self.min_probability,
                threads=self.threads_per_worker,
                time_limit_ms=self.time_limit_ms,
                nodes=self.nodes,
                shallow_depth=self.shallow_depth,
                shallow_max_moves=self.shallow_max_moves,
                confirm_depth=self.confirm_depth,
                confirm_top_k=self.confirm_top_k,
                prob_mode=self.prob_mode,
                wdl_temperature=self.wdl_temperature,
                syzygy_path=self.syzygy_path,
            )
            self._workers.append(worker)
            self._worker_queue.put(worker)
        
        self._initialized = True

    def _resolve_settings(
        self,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        shallow_depth: Optional[int] = None,
        shallow_max_moves: Optional[int] = None,
        confirm_depth: Optional[int] = None,
        confirm_top_k: Optional[int] = None,
        prob_mode: Optional[str] = None,
        wdl_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "top_k": self.top_k if top_k is None else top_k,
            "depth": self.depth if depth is None else depth,
            "time_limit_ms": self.time_limit_ms if time_limit_ms is None else time_limit_ms,
            "nodes": self.nodes if nodes is None else nodes,
            "shallow_depth": self.shallow_depth if shallow_depth is None else shallow_depth,
            "shallow_max_moves": (
                self.shallow_max_moves if shallow_max_moves is None else shallow_max_moves
            ),
            "confirm_depth": self.confirm_depth if confirm_depth is None else confirm_depth,
            "confirm_top_k": self.confirm_top_k if confirm_top_k is None else confirm_top_k,
            "prob_mode": self.prob_mode if prob_mode is None else prob_mode,
            "wdl_temperature": self.wdl_temperature if wdl_temperature is None else wdl_temperature,
        }

    def _make_cache_key(self, fen: str, settings: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            fen,
            settings["top_k"],
            settings["depth"],
            settings["time_limit_ms"],
            settings["nodes"],
            settings["shallow_depth"],
            settings["shallow_max_moves"],
            settings["confirm_depth"],
            settings["confirm_top_k"],
            settings["prob_mode"],
            settings["wdl_temperature"],
            self.temperature,
            self.min_probability,
        )

    def _get_cached(self, key: Tuple[Any, ...]) -> Optional[PositionAnalysis]:
        if self.cache_size <= 0:
            return None
        with self._cache_lock:
            cached = self._analysis_cache.get(key)
            if cached is not None:
                self._analysis_cache.move_to_end(key)
            return cached

    def _set_cached(self, key: Tuple[Any, ...], analysis: PositionAnalysis) -> None:
        if self.cache_size <= 0:
            return
        with self._cache_lock:
            self._analysis_cache[key] = analysis
            self._analysis_cache.move_to_end(key)
            while len(self._analysis_cache) > self.cache_size:
                self._analysis_cache.popitem(last=False)

    def _finalize_inflight(self, key: Tuple[Any, ...], future: Future) -> None:
        with self._inflight_lock:
            self._inflight.pop(key, None)
        if future.cancelled() or future.exception() is not None:
            return
        try:
            analysis = future.result()
        except Exception:
            return
        self._set_cached(key, analysis)

    def _analyze_with_worker(self, fen: str, settings: Dict[str, Any]) -> PositionAnalysis:
        self._initialize_workers()
        board = chess.Board(fen)
        worker = self._worker_queue.get()
        try:
            return worker.analyze(board, **settings)
        finally:
            self._worker_queue.put(worker)

    def _submit_analysis(self, fen: str, settings: Dict[str, Any]) -> Future:
        key = self._make_cache_key(fen, settings)
        cached = self._get_cached(key)
        if cached is not None:
            future: Future = Future()
            future.set_result(cached)
            return future

        with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return existing
            future = self._executor.submit(self._analyze_with_worker, fen, settings)
            self._inflight[key] = future
            future.add_done_callback(lambda f, k=key: self._finalize_inflight(k, f))
            return future
    
    def analyze_position(
        self,
        fen: str,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        shallow_depth: Optional[int] = None,
        shallow_max_moves: Optional[int] = None,
        confirm_depth: Optional[int] = None,
        confirm_top_k: Optional[int] = None,
        prob_mode: Optional[str] = None,
        wdl_temperature: Optional[float] = None,
    ) -> PositionAnalysis:
        """
        Analyze a single position (blocking).
        
        Args:
            fen: Position in FEN format
            top_k: Override default top_k
            depth: Override analysis depth
            time_limit_ms: Override time limit (ms)
            nodes: Override node limit
            shallow_depth: Override shallow depth
            shallow_max_moves: Override shallow multipv cap
            prob_mode: Override probability mode
            wdl_temperature: Override WDL temperature
        
        Returns:
            PositionAnalysis
        """
        settings = self._resolve_settings(
            top_k=top_k,
            depth=depth,
            time_limit_ms=time_limit_ms,
            nodes=nodes,
            shallow_depth=shallow_depth,
            shallow_max_moves=shallow_max_moves,
            confirm_depth=confirm_depth,
            confirm_top_k=confirm_top_k,
            prob_mode=prob_mode,
            wdl_temperature=wdl_temperature,
        )
        future = self._submit_analysis(fen, settings)
        return future.result()
    
    def analyze_position_async(
        self,
        fen: str,
        top_k: Optional[int] = None,
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        shallow_depth: Optional[int] = None,
        shallow_max_moves: Optional[int] = None,
        confirm_depth: Optional[int] = None,
        confirm_top_k: Optional[int] = None,
        prob_mode: Optional[str] = None,
        wdl_temperature: Optional[float] = None,
    ) -> Future:
        """
        Analyze a position asynchronously.
        
        Returns a Future that resolves to PositionAnalysis.
        """
        settings = self._resolve_settings(
            top_k=top_k,
            depth=depth,
            time_limit_ms=time_limit_ms,
            nodes=nodes,
            shallow_depth=shallow_depth,
            shallow_max_moves=shallow_max_moves,
            confirm_depth=confirm_depth,
            confirm_top_k=confirm_top_k,
            prob_mode=prob_mode,
            wdl_temperature=wdl_temperature,
        )
        return self._submit_analysis(fen, settings)
    
    def analyze_batch(
        self,
        fens: List[str],
        top_k: Optional[int] = None,
        analysis_overrides: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[Optional[PositionAnalysis]]:
        """
        Analyze multiple positions in parallel.
        
        Args:
            fens: List of FEN strings
            top_k: Override default top_k
            analysis_overrides: Optional per-position overrides
        
        Returns:
            List of `PositionAnalysis` (same order as input). If an individual
            analysis fails, the corresponding entry is `None`.
        """
        if analysis_overrides is not None and len(analysis_overrides) != len(fens):
            raise ValueError("analysis_overrides must match length of fens")

        allowed_keys = {
            "top_k",
            "depth",
            "time_limit_ms",
            "nodes",
            "shallow_depth",
            "shallow_max_moves",
            "confirm_depth",
            "confirm_top_k",
            "prob_mode",
            "wdl_temperature",
        }

        futures: List[Future] = []
        for idx, fen in enumerate(fens):
            overrides = None
            if analysis_overrides is not None:
                overrides = analysis_overrides[idx]

            if overrides:
                clean = {k: v for k, v in overrides.items() if k in allowed_keys}
                settings = self._resolve_settings(**clean)
            else:
                settings = self._resolve_settings(top_k=top_k)

            futures.append(self._submit_analysis(fen, settings))

        results: List[Optional[PositionAnalysis]] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                results.append(None)

        return results
    
    def close(self):
        """Shutdown all workers."""
        for worker in self._workers:
            worker.close()
        self._workers = []
        self._executor.shutdown(wait=False)
        self._initialized = False
        with self._cache_lock:
            self._analysis_cache.clear()
        with self._inflight_lock:
            self._inflight.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
