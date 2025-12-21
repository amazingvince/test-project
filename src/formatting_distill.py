"""
Formatting module for Chess Policy Distillation.

Generates training examples with Stockfish analysis in thinking tags,
using randomized move order and categorical labels for better LLM learning.
"""

import random
from typing import Dict, Any, Optional, List
import chess
import sys
from pathlib import Path

# Handle both relative imports (when used as module) and absolute imports (when run as script)
try:
    from .stockfish_teacher import PositionAnalysis, MoveAnalysis
    from .reasoning_trace import ReasoningTraceGenerator
except ImportError:
    # Add parent directory to path when running as script
    sys.path.insert(0, str(Path(__file__).parent))
    from stockfish_teacher import PositionAnalysis, MoveAnalysis
    from reasoning_trace import ReasoningTraceGenerator

# =============================================================================
# Prompt Templates for Distillation
# =============================================================================

DISTILLATION_PROMPT_TEMPLATE = """You are an expert chess player. Analyze this position and select the best move.

Position (FEN): {fen}

Legal moves: {legal_moves}

Board:
{board}

Analyze the candidate moves, evaluate their quality, then select the best one.

Output format:
<think>your analysis of the moves</think>
<uci_move>your_chosen_move</uci_move>"""

DISTILLATION_PROMPT_TEMPLATE_NO_BOARD = """You are an expert chess player. Analyze this position and select the best move.

Position (FEN): {fen}

Legal moves: {legal_moves}

Analyze the candidate moves, evaluate their quality, then select the best one.

Output format:
<think>your analysis of the moves</think>
<uci_move>your_chosen_move</uci_move>"""


DISTILLATION_RESPONSE_TEMPLATE = """<think>
{thinking}
</think>
<uci_move>{move}</uci_move>"""


# =============================================================================
# Thinking Text Generation
# =============================================================================

def categorize_by_win_prob(win_prob: float) -> str:
    """Categorize position based on win probability."""
    if win_prob >= 0.80:
        return "winning"
    elif win_prob >= 0.65:
        return "better"
    elif win_prob >= 0.55:
        return "slight edge"
    elif win_prob >= 0.45:
        return "equal"
    elif win_prob >= 0.35:
        return "slight disadvantage"
    elif win_prob >= 0.20:
        return "worse"
    else:
        return "losing"


def format_move_for_thinking(
    move_analysis: MoveAnalysis,
    is_best: bool = False,
) -> str:
    """
    Format a single move for the thinking section.

    Shows category based on win probability + win percentage.
    """
    win_prob = move_analysis.win_probability
    win_pct = int(win_prob * 100)

    if move_analysis.mate_in is not None:
        if move_analysis.mate_in > 0:
            return f"- {move_analysis.san}: winning, mate in {move_analysis.mate_in}"
        else:
            return f"- {move_analysis.san}: losing, gets mated in {abs(move_analysis.mate_in)}"

    category = categorize_by_win_prob(win_prob)
    return f"- {move_analysis.san}: {category} ({win_pct}% win)"


def generate_thinking_text(
    analysis: PositionAnalysis,
    target_move_uci: str,
    max_display: int = 5,
    randomize_order: bool = True,
    include_target_if_missing: bool = True,
    pv_length: int = 5,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Generate thinking text with move analysis.

    Args:
        analysis: Stockfish position analysis
        target_move_uci: The move we're training on
        max_display: Maximum moves to show
        randomize_order: Shuffle to prevent positional shortcuts
        include_target_if_missing: Ensure target move appears in analysis
        pv_length: Number of moves to show in best line
        rng: Random number generator for reproducibility

    Returns:
        Formatted thinking text
    """
    if rng is None:
        rng = random.Random()

    # Get moves to display
    display_moves = list(analysis.move_analyses[:max_display])

    # Ensure target move is included if it's in the analysis
    target_in_display = any(ma.uci == target_move_uci for ma in display_moves)

    if include_target_if_missing and not target_in_display:
        # Find target in full analysis
        for ma in analysis.move_analyses:
            if ma.uci == target_move_uci:
                # Replace the worst displayed move with target
                if len(display_moves) >= max_display:
                    display_moves[-1] = ma
                else:
                    display_moves.append(ma)
                break

    # Randomize order
    if randomize_order:
        rng.shuffle(display_moves)

    # Build thinking text
    lines = ["Analyzing position..."]

    for ma in display_moves:
        is_best = (ma.uci == analysis.best_move_uci)
        lines.append(format_move_for_thinking(ma, is_best=is_best))

    # Use PV (principal variation) instead of single move
    if analysis.best_pv:
        pv_str = ' '.join(analysis.best_pv[:pv_length])
        lines.append(f"\nBest line: {pv_str}")
    else:
        lines.append(f"\nBest line: {analysis.best_move_san}")

    # Find best move's win probability
    best_move_analysis = None
    for ma in analysis.move_analyses:
        if ma.uci == analysis.best_move_uci:
            best_move_analysis = ma
            break

    if best_move_analysis:
        best_win_pct = int(best_move_analysis.win_probability * 100)
        lines.append(f"Playing {analysis.best_move_san} gives {best_win_pct}% win chance.")

    return "\n".join(lines)


def generate_thinking_text_for_non_best_move(
    analysis: PositionAnalysis,
    target_move_uci: str,
    max_display: int = 5,
    randomize_order: bool = True,
    pv_length: int = 5,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Generate thinking text when target is NOT the best move.

    This teaches the model that multiple moves can be acceptable
    and how to reason about move quality.
    """
    if rng is None:
        rng = random.Random()

    # Find target move analysis
    target_analysis = None
    for ma in analysis.move_analyses:
        if ma.uci == target_move_uci:
            target_analysis = ma
            break

    # Get moves to display (ensure target and best are included)
    display_moves = []
    target_included = False
    best_included = False

    for ma in analysis.move_analyses:
        if ma.uci == target_move_uci:
            display_moves.append(ma)
            target_included = True
        elif ma.uci == analysis.best_move_uci:
            display_moves.append(ma)
            best_included = True
        elif len(display_moves) < max_display:
            display_moves.append(ma)

        if len(display_moves) >= max_display:
            break

    # Randomize
    if randomize_order:
        rng.shuffle(display_moves)

    # Build thinking text
    lines = ["Analyzing position..."]

    for ma in display_moves:
        is_best = (ma.uci == analysis.best_move_uci)
        lines.append(format_move_for_thinking(ma, is_best=is_best))

    # Show best line (PV) and best move's win probability
    if analysis.best_pv:
        pv_str = ' '.join(analysis.best_pv[:pv_length])
        lines.append(f"\nBest line: {pv_str}")
    else:
        lines.append(f"\nBest line: {analysis.best_move_san}")

    # Find best move's win probability
    best_move_analysis = None
    for ma in analysis.move_analyses:
        if ma.uci == analysis.best_move_uci:
            best_move_analysis = ma
            break

    if best_move_analysis:
        best_win_pct = int(best_move_analysis.win_probability * 100)
        lines.append(f"Playing {analysis.best_move_san} gives {best_win_pct}% win chance.")

    return "\n".join(lines)


# =============================================================================
# Position to Messages Conversion
# =============================================================================

def position_to_messages_distill(
    position: Dict[str, Any],
    analysis: PositionAnalysis,
    prompt_template: str = DISTILLATION_PROMPT_TEMPLATE,
    response_template: str = DISTILLATION_RESPONSE_TEMPLATE,
    max_display_moves: int = 5,
    randomize_order: bool = True,
    pv_length: int = 5,
    include_board: bool = True,
    reasoning_trace_generator: Optional[ReasoningTraceGenerator] = None,
    force_best_move: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Convert a chess position + analysis to chat message format.

    Args:
        position: Dict with position data (fen, legal_moves_uci, board_utf, target_move_uci)
        analysis: Stockfish analysis of the position
        prompt_template: User message template
        response_template: Assistant response template
        max_display_moves: Maximum moves to show in thinking
        randomize_order: Shuffle moves in thinking
        pv_length: Number of moves to show in best line
        rng: Random generator

    Returns:
        Dict with 'messages' key containing chat format
    """
    target_move_uci = position['target_move_uci']
    move_for_output = target_move_uci
    if force_best_move and analysis.best_move_uci:
        move_for_output = analysis.best_move_uci

    # Determine if target is best move or not
    is_best = (move_for_output == analysis.best_move_uci)

    if reasoning_trace_generator is not None:
        thinking = reasoning_trace_generator.generate(
            position=position,
            analysis=analysis,
            rng=rng,
        )
    else:
        if is_best:
            thinking = generate_thinking_text(
                analysis=analysis,
                target_move_uci=move_for_output,
                max_display=max_display_moves,
                randomize_order=randomize_order,
                pv_length=pv_length,
                rng=rng,
            )
        else:
            thinking = generate_thinking_text_for_non_best_move(
                analysis=analysis,
                target_move_uci=move_for_output,
                max_display=max_display_moves,
                randomize_order=randomize_order,
                pv_length=pv_length,
                rng=rng,
            )
    
    # Format user content
    if include_board:
        user_content = prompt_template.format(
            fen=position['fen'],
            legal_moves=position['legal_moves_uci'],
            board=position['board_utf'],
        )
    else:
        user_content = DISTILLATION_PROMPT_TEMPLATE_NO_BOARD.format(
            fen=position['fen'],
            legal_moves=position['legal_moves_uci'],
        )
    
    # Format assistant content
    assistant_content = response_template.format(
        thinking=thinking,
        move=move_for_output,
    )
    
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def create_distillation_example(
    fen: str,
    target_move_uci: str,
    analysis: PositionAnalysis,
    board_utf: Optional[str] = None,
    max_display_moves: int = 5,
    randomize_order: bool = True,
    pv_length: int = 5,
    include_board: bool = True,
    source: Optional[str] = None,
    reasoning_trace_generator: Optional[ReasoningTraceGenerator] = None,
    force_best_move: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Create a complete training example with distillation data.

    Returns a dict that includes:
    - messages: Chat format for SFT
    - soft_targets: Probability distribution from Stockfish
    - analysis metadata
    """
    board = chess.Board(fen)

    # Generate board UTF if not provided
    if board_utf is None:
        try:
            from .chess_utils import render_board_utf
        except ImportError:
            from chess_utils import render_board_utf
        board_utf = render_board_utf(board)

    # Get legal moves
    legal_moves_uci = ' '.join(m.uci() for m in board.legal_moves)

    move_for_output = target_move_uci
    if force_best_move and analysis.best_move_uci:
        move_for_output = analysis.best_move_uci

    position = {
        'fen': fen,
        'target_move_uci': move_for_output,
        'legal_moves_uci': legal_moves_uci,
        'board_utf': board_utf,
    }
    if source is not None:
        position['source'] = source

    messages_data = position_to_messages_distill(
        position=position,
        analysis=analysis,
        max_display_moves=max_display_moves,
        randomize_order=randomize_order,
        pv_length=pv_length,
        include_board=include_board,
        reasoning_trace_generator=reasoning_trace_generator,
        force_best_move=force_best_move,
        rng=rng,
    )
    
    def _find_move_info(move_uci: str) -> Dict[str, Any]:
        move_prob = analysis.move_probs.get(move_uci, 0.0)
        move_rank = -1
        move_cp_loss = 0
        for i, ma in enumerate(analysis.move_analyses):
            if ma.uci == move_uci:
                move_rank = i + 1
                move_cp_loss = ma.cp_loss
                break
        return {
            "prob": move_prob,
            "rank": move_rank,
            "cp_loss": move_cp_loss,
        }

    target_info = _find_move_info(move_for_output)
    source_info = _find_move_info(target_move_uci) if target_move_uci != move_for_output else None
    
    return {
        **messages_data,
        # Position data
        'fen': fen,
        'target_move_uci': move_for_output,
        'legal_moves_uci': legal_moves_uci,
        'board_utf': board_utf,
        
        # Analysis data
        'best_move_uci': analysis.best_move_uci,
        'best_move_san': analysis.best_move_san,
        'best_score_cp': analysis.best_score_cp,
        
        # Target move quality
        'target_move_rank': target_info['rank'],
        'target_move_cp_loss': target_info['cp_loss'],
        'target_move_prob': target_info['prob'],
        
        # Full probability distribution for distillation loss
        'move_probs': analysis.move_probs,

        # Optional shallow pass data (for trap detection)
        'shallow_move_cps': analysis.shallow_move_cps,
        'shallow_move_win_probs': analysis.shallow_move_win_probs,
        'confirm_move_cps': analysis.confirm_move_cps,
        'confirm_move_win_probs': analysis.confirm_move_win_probs,
        
        # Top-k analyzed moves (for reference)
        'top_k_moves': analysis.top_k_moves,
        **({
            'source_target_move_uci': target_move_uci,
            'source_target_move_rank': source_info['rank'],
            'source_target_move_cp_loss': source_info['cp_loss'],
            'source_target_move_prob': source_info['prob'],
        } if source_info is not None else {}),
    }


# =============================================================================
# Display Utilities
# =============================================================================

def format_example_for_display(example: Dict[str, Any]) -> str:
    """Format a distillation example for human-readable display."""
    output = []
    output.append("=" * 70)
    output.append("DISTILLATION TRAINING EXAMPLE")
    output.append("=" * 70)
    
    output.append(f"\nFEN: {example['fen']}")
    output.append(f"Target move: {example['target_move_uci']}")
    if 'source_target_move_uci' in example:
        output.append(f"Source move: {example['source_target_move_uci']}")
    output.append(f"Best move: {example['best_move_uci']} ({example['best_score_cp']:+d}cp)")
    output.append(f"Target rank: {example['target_move_rank']}")
    output.append(f"Target CP loss: {example['target_move_cp_loss']}")
    output.append(f"Target probability: {example['target_move_prob']:.4f}")
    
    output.append("\n" + "-" * 70)
    output.append("USER MESSAGE:")
    output.append("-" * 70)
    output.append(example['messages'][0]['content'])
    
    output.append("\n" + "-" * 70)
    output.append("ASSISTANT RESPONSE:")
    output.append("-" * 70)
    output.append(example['messages'][1]['content'])
    
    output.append("\n" + "-" * 70)
    output.append("TOP MOVE PROBABILITIES:")
    output.append("-" * 70)
    sorted_probs = sorted(example['move_probs'].items(), key=lambda x: -x[1])
    for move, prob in sorted_probs[:10]:
        marker = "★" if move == example['best_move_uci'] else ""
        target = "← target" if move == example['target_move_uci'] else ""
        output.append(f"  {move}: {prob:.4f} {marker} {target}")
    
    output.append("=" * 70)
    
    return "\n".join(output)


if __name__ == "__main__":
    # Test the formatting
    print("Testing distillation formatting...")
    
    try:
        from .stockfish_teacher import StockfishTeacher
        from .chess_utils import render_board_utf
    except ImportError:
        from stockfish_teacher import StockfishTeacher
        from chess_utils import render_board_utf
    import chess
    
    try:
        with StockfishTeacher(num_workers=1, depth=10, top_k=5) as teacher:
            # Test with starting position, e4 as target
            fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
            target_move = "e7e5"
            
            analysis = teacher.analyze_position(fen)
            
            example = create_distillation_example(
                fen=fen,
                target_move_uci=target_move,
                analysis=analysis,
                randomize_order=True,
            )
            
            print(format_example_for_display(example))
            
            # Test with a non-best move as target
            print("\n\n" + "=" * 70)
            print("TESTING WITH NON-BEST MOVE AS TARGET")
            print("=" * 70)
            
            # Find a suboptimal move
            for ma in analysis.move_analyses[1:]:  # Skip best
                if ma.cp_loss > 0:
                    target_move = ma.uci
                    break
            
            example2 = create_distillation_example(
                fen=fen,
                target_move_uci=target_move,
                analysis=analysis,
                randomize_order=True,
            )
            
            print(format_example_for_display(example2))
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
