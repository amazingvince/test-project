"""
Stockfish evaluation utilities for chess position analysis.

Provides centipawn evaluations for all legal moves in a position,
enabling reward shaping and move reasoning in training data.
"""

import chess
import chess.engine
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path


@dataclass
class MoveEvaluation:
    """Evaluation of a single move."""
    uci: str
    san: str
    centipawn: int  # Centipawn score (positive = good for side to move)
    mate_in: Optional[int] = None  # Mate in N moves (positive = winning)
    
    def score_str(self) -> str:
        """Human-readable score string."""
        if self.mate_in is not None:
            return f"M{self.mate_in}" if self.mate_in > 0 else f"M{self.mate_in}"
        return f"{self.centipawn:+d}cp"
    
    def normalized_score(self, max_cp: int = 1000) -> float:
        """
        Normalize score to [-1, 1] range for reward shaping.
        Mate scores get ±1.0
        """
        if self.mate_in is not None:
            return 1.0 if self.mate_in > 0 else -1.0
        # Sigmoid-like clamping
        return max(-1.0, min(1.0, self.centipawn / max_cp))


@dataclass
class PositionAnalysis:
    """Complete analysis of a chess position."""
    fen: str
    move_evaluations: List[MoveEvaluation]
    best_move_uci: str
    best_move_san: str
    best_score_cp: int
    position_eval_cp: int  # Overall position evaluation
    
    def get_move_eval(self, uci: str) -> Optional[MoveEvaluation]:
        """Get evaluation for a specific move."""
        for mv in self.move_evaluations:
            if mv.uci == uci:
                return mv
        return None
    
    def rank_of_move(self, uci: str) -> int:
        """Get rank (1=best) of a move. Returns -1 if not found."""
        for i, mv in enumerate(self.move_evaluations):
            if mv.uci == uci:
                return i + 1
        return -1
    
    def centipawn_loss(self, uci: str) -> int:
        """Calculate centipawn loss for playing a move vs best move."""
        move_eval = self.get_move_eval(uci)
        if move_eval is None:
            return 9999
        return self.best_score_cp - move_eval.centipawn
    
    def format_thinking_content(
        self,
        top_n: int = 5,
        include_all: bool = False,
        compact: bool = False
    ) -> str:
        """
        Format move evaluations for <think> tag content.
        
        Args:
            top_n: Number of top moves to show (if not include_all)
            include_all: Whether to include all legal moves
            compact: Use compact single-line format
        
        Returns:
            Formatted string for thinking content
        """
        moves_to_show = self.move_evaluations if include_all else self.move_evaluations[:top_n]
        
        if compact:
            # Compact format: "e2e4(+35) d2d4(+30) g1f3(+15)"
            parts = [f"{mv.uci}({mv.score_str()})" for mv in moves_to_show]
            return " ".join(parts)
        
        # Detailed format
        lines = []
        lines.append(f"Position eval: {self.position_eval_cp:+d}cp")
        lines.append(f"Best: {self.best_move_san} ({self.best_score_cp:+d}cp)")
        lines.append("Top moves:")
        
        for i, mv in enumerate(moves_to_show, 1):
            loss = self.best_score_cp - mv.centipawn
            loss_str = f" (-{loss})" if loss > 0 else ""
            lines.append(f"  {i}. {mv.san} ({mv.score_str()}){loss_str}")
        
        if not include_all and len(self.move_evaluations) > top_n:
            lines.append(f"  ... and {len(self.move_evaluations) - top_n} more moves")
        
        return "\n".join(lines)


class StockfishEvaluator:
    """
    Wrapper for Stockfish engine to evaluate positions and moves.
    
    Supports multi-threaded analysis for batch processing.
    """
    
    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 12,
        threads: int = 1,
        hash_mb: int = 128
    ):
        """
        Initialize Stockfish evaluator.
        
        Args:
            stockfish_path: Path to Stockfish binary. Auto-detects if None.
            depth: Search depth for analysis
            threads: Number of threads for Stockfish to use
            hash_mb: Hash table size in MB
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None
    
    @staticmethod
    def _find_stockfish() -> str:
        """Find Stockfish binary."""
        possible_paths = [
            '/usr/bin/stockfish',
            '/usr/games/stockfish',
            '/usr/local/bin/stockfish',
            '/opt/homebrew/bin/stockfish',
            'stockfish',  # In PATH
        ]
        
        import shutil
        sf_in_path = shutil.which('stockfish')
        if sf_in_path:
            return sf_in_path
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(
            "Stockfish not found. Install with: sudo apt install stockfish"
        )
    
    def _get_engine(self) -> chess.engine.SimpleEngine:
        """Get or create engine instance."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self._engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb,
            })
        return self._engine
    
    def close(self):
        """Close engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def analyze_position(
        self,
        board: chess.Board,
        multipv: Optional[int] = None
    ) -> PositionAnalysis:
        """
        Analyze a position and return evaluations for all legal moves.
        
        Args:
            board: Chess position to analyze
            multipv: Number of principal variations (None = all legal moves)
        
        Returns:
            PositionAnalysis with all move evaluations sorted best to worst
        """
        engine = self._get_engine()
        
        legal_moves = list(board.legal_moves)
        num_moves = len(legal_moves)
        
        if num_moves == 0:
            raise ValueError("No legal moves in position")
        
        # Use multipv to get all moves if not specified
        if multipv is None:
            multipv = num_moves
        else:
            multipv = min(multipv, num_moves)
        
        # Analyze with MultiPV
        result = engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=multipv
        )
        
        # Convert to MoveEvaluation objects
        move_evals = []
        
        for info in result:
            if 'pv' not in info or len(info['pv']) == 0:
                continue
            
            move = info['pv'][0]
            score = info.get('score')
            
            if score is None:
                continue
            
            # Get centipawn score from white's perspective, then adjust
            pov_score = score.pov(board.turn)
            
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                # Use large centipawn value for mate
                cp = 30000 if mate_in > 0 else -30000
            else:
                cp = pov_score.score()
                mate_in = None
            
            move_evals.append(MoveEvaluation(
                uci=move.uci(),
                san=board.san(move),
                centipawn=cp,
                mate_in=mate_in
            ))
        
        # Sort by centipawn (best first)
        move_evals.sort(key=lambda x: x.centipawn, reverse=True)
        
        # Get position evaluation (before any move)
        pos_info = engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=1
        )[0]
        pos_score = pos_info['score'].pov(board.turn)
        pos_eval_cp = pos_score.score() if not pos_score.is_mate() else (30000 if pos_score.mate() > 0 else -30000)
        
        best = move_evals[0] if move_evals else None
        
        return PositionAnalysis(
            fen=board.fen(),
            move_evaluations=move_evals,
            best_move_uci=best.uci if best else "",
            best_move_san=best.san if best else "",
            best_score_cp=best.centipawn if best else 0,
            position_eval_cp=pos_eval_cp
        )
    
    def evaluate_move(
        self,
        board: chess.Board,
        move_uci: str
    ) -> Tuple[int, int]:
        """
        Quick evaluation of a single move.
        
        Returns:
            Tuple of (move_eval_cp, centipawn_loss)
        """
        analysis = self.analyze_position(board, multipv=None)
        move_eval = analysis.get_move_eval(move_uci)
        
        if move_eval is None:
            return (0, 9999)
        
        return (move_eval.centipawn, analysis.centipawn_loss(move_uci))


def analyze_positions_batch(
    positions: List[Tuple[str, str]],  # List of (fen, target_move_uci)
    stockfish_path: Optional[str] = None,
    depth: int = 12,
    num_workers: int = 4,
    multipv: Optional[int] = None
) -> List[Tuple[PositionAnalysis, str]]:
    """
    Analyze multiple positions in parallel.
    
    Args:
        positions: List of (FEN, target_move_uci) tuples
        stockfish_path: Path to Stockfish
        depth: Search depth
        num_workers: Number of parallel workers
        multipv: Moves to analyze per position (None = all)
    
    Returns:
        List of (PositionAnalysis, target_move_uci) tuples
    """
    def analyze_one(item):
        fen, target_move = item
        evaluator = StockfishEvaluator(
            stockfish_path=stockfish_path,
            depth=depth,
            threads=1
        )
        try:
            board = chess.Board(fen)
            analysis = evaluator.analyze_position(board, multipv=multipv)
            return (analysis, target_move)
        finally:
            evaluator.close()
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(analyze_one, positions))
    
    return results


def compute_move_reward(
    analysis: PositionAnalysis,
    played_move_uci: str,
    reward_type: str = "centipawn_loss",
    **kwargs
) -> float:
    """
    Compute reward for playing a specific move.
    
    Args:
        analysis: Position analysis with all move evaluations
        played_move_uci: The move that was played
        reward_type: Type of reward calculation:
            - "centipawn_loss": Penalize based on CP loss
            - "rank": Penalize based on move rank
            - "normalized": Use normalized [-1, 1] score
            - "binary": 1.0 if best move, 0.0 otherwise
            - "soft_rank": Smooth reward based on rank
    
    Returns:
        Reward value (higher = better)
    """
    move_eval = analysis.get_move_eval(played_move_uci)
    
    if move_eval is None:
        return 0.0  # Invalid move
    
    if reward_type == "centipawn_loss":
        # Reward = 1 - (cp_loss / max_cp_loss)
        max_loss = kwargs.get("max_cp_loss", 300)
        cp_loss = analysis.centipawn_loss(played_move_uci)
        reward = 1.0 - min(cp_loss / max_loss, 1.0)
        return max(0.0, reward)
    
    elif reward_type == "rank":
        # Reward based on rank (1st = 1.0, last = 0.0)
        rank = analysis.rank_of_move(played_move_uci)
        num_moves = len(analysis.move_evaluations)
        if num_moves <= 1:
            return 1.0
        return 1.0 - (rank - 1) / (num_moves - 1)
    
    elif reward_type == "normalized":
        # Use normalized score directly
        max_cp = kwargs.get("max_cp", 1000)
        return (move_eval.normalized_score(max_cp) + 1.0) / 2.0
    
    elif reward_type == "binary":
        # 1.0 if best move, 0.0 otherwise
        return 1.0 if analysis.rank_of_move(played_move_uci) == 1 else 0.0
    
    elif reward_type == "soft_rank":
        # Exponential decay based on rank
        decay = kwargs.get("decay", 0.5)
        rank = analysis.rank_of_move(played_move_uci)
        return decay ** (rank - 1)
    
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def compute_loss_weight_from_cp(
    analysis: PositionAnalysis,
    target_move_uci: str,
    weight_type: str = "inverse_loss",
    min_weight: float = 0.1,
    max_weight: float = 2.0,
    **kwargs
) -> float:
    """
    Compute loss weight for training based on move quality.
    
    Better moves (lower CP loss) get higher weight, encouraging
    the model to learn from good moves more.
    
    Args:
        analysis: Position analysis
        target_move_uci: The target move for training
        weight_type: How to compute weight:
            - "inverse_loss": Higher weight for lower CP loss
            - "rank_based": Weight based on move rank
            - "quality_gate": High weight if top N, low otherwise
        min_weight: Minimum weight
        max_weight: Maximum weight
    
    Returns:
        Loss weight for this training example
    """
    cp_loss = analysis.centipawn_loss(target_move_uci)
    rank = analysis.rank_of_move(target_move_uci)
    
    if weight_type == "inverse_loss":
        # Higher weight for lower CP loss
        max_loss = kwargs.get("max_cp_loss", 200)
        # Linear interpolation: 0 loss -> max_weight, max_loss -> min_weight
        ratio = min(cp_loss / max_loss, 1.0)
        return max_weight - ratio * (max_weight - min_weight)
    
    elif weight_type == "rank_based":
        # Higher weight for better ranked moves
        num_moves = len(analysis.move_evaluations)
        if num_moves <= 1:
            return max_weight
        ratio = (rank - 1) / (num_moves - 1)
        return max_weight - ratio * (max_weight - min_weight)
    
    elif weight_type == "quality_gate":
        # High weight if in top N moves, low otherwise
        top_n = kwargs.get("top_n", 3)
        return max_weight if rank <= top_n else min_weight
    
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Stockfish evaluator...")
    
    try:
        with StockfishEvaluator(depth=10) as evaluator:
            # Test starting position
            board = chess.Board()
            print("\nStarting position analysis:")
            analysis = evaluator.analyze_position(board)
            
            print(f"Best move: {analysis.best_move_san} ({analysis.best_score_cp:+d}cp)")
            print(f"\nTop 5 moves:")
            for i, mv in enumerate(analysis.move_evaluations[:5], 1):
                print(f"  {i}. {mv.san}: {mv.score_str()}")
            
            print("\n" + analysis.format_thinking_content(top_n=5))
            
            # Test move reward
            print("\n\nTesting reward computation for e2e4:")
            reward = compute_move_reward(analysis, "e2e4", "centipawn_loss")
            print(f"  Reward (centipawn_loss): {reward:.3f}")
            
            rank_reward = compute_move_reward(analysis, "e2e4", "rank")
            print(f"  Reward (rank): {rank_reward:.3f}")
            
            # Test loss weight
            weight = compute_loss_weight_from_cp(analysis, "e2e4", "inverse_loss")
            print(f"  Loss weight: {weight:.3f}")
            
            # Test a specific position
            print("\n\nTesting a tactical position:")
            board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
            print(board)
            
            analysis = evaluator.analyze_position(board)
            print(f"\nBest move: {analysis.best_move_san} ({analysis.best_score_cp:+d}cp)")
            print(analysis.format_thinking_content(top_n=5))
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Install Stockfish to run this test.")
