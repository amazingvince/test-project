"""
Formatting module for Chess SFT training with Stockfish evaluations.

Converts chess positions to chat format suitable for SFT training,
with move evaluations embedded in thinking tags for distillation.
"""

from typing import Dict, Any, Optional, Callable, List
import chess


# =============================================================================
# Prompt Templates with Move Evaluation
# =============================================================================

DEFAULT_PROMPT_TEMPLATE = """You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves}

Board looks like this right now
{board}

Select the best move. Analyze the candidate moves and their values, then output your chosen move.

CRITICAL: Your UCI move must always be one of the moves from legal_moves_uci. With no spaces in between.

An example of a valid move for this position is <uci_move>{example_move}</uci_move>. 
Format:
<think>analyze moves and pick the best one</think>
<uci_move>your_move</uci_move>"""


# Response template with move analysis in thinking
DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL = """<think>
{move_analysis}
Best continuation: {best_move} ({best_score})
</think>
<uci_move>{move}</uci_move>"""


# Compact response template (less verbose)
COMPACT_RESPONSE_TEMPLATE_WITH_EVAL = """<think>{compact_analysis} → {move_san}({move_score})</think>
<uci_move>{move}</uci_move>"""


# Simple response (no thinking, for baseline)
SIMPLE_RESPONSE_TEMPLATE = "<think></think>\n<uci_move>{move}</uci_move>"


# =============================================================================
# Move Analysis Formatting
# =============================================================================

def format_move_analysis(
    move_evaluations: List[Dict[str, Any]],
    target_move_uci: str,
    top_n: int = 5,
    style: str = "detailed"
) -> Dict[str, str]:
    """
    Format move evaluations for inclusion in response.
    
    Args:
        move_evaluations: List of {uci, san, centipawn, mate_in} dicts
        target_move_uci: The move being played
        top_n: Number of top moves to include
        style: "detailed", "compact", or "minimal"
    
    Returns:
        Dict with formatted strings for template
    """
    if not move_evaluations:
        return {
            "move_analysis": "No analysis available",
            "compact_analysis": "?",
            "best_move": "?",
            "best_score": "?",
            "move_san": "?",
            "move_score": "?"
        }
    
    # Sort by centipawn (should already be sorted, but ensure)
    sorted_evals = sorted(move_evaluations, key=lambda x: x.get('centipawn', 0), reverse=True)
    
    best = sorted_evals[0]
    
    # Find target move evaluation
    target_eval = None
    target_rank = -1
    for i, ev in enumerate(sorted_evals):
        if ev['uci'] == target_move_uci:
            target_eval = ev
            target_rank = i + 1
            break
    
    if target_eval is None:
        # Fallback if target not found
        target_eval = {"san": target_move_uci, "centipawn": 0, "uci": target_move_uci}
        target_rank = len(sorted_evals) + 1
    
    def score_str(ev):
        if ev.get('mate_in') is not None:
            return f"M{ev['mate_in']}"
        return f"{ev.get('centipawn', 0):+d}cp"
    
    if style == "detailed":
        # Full analysis with commentary
        lines = []
        lines.append(f"Analyzing position with {len(sorted_evals)} legal moves...")
        lines.append("")
        lines.append("Top candidates:")
        
        for i, ev in enumerate(sorted_evals[:top_n], 1):
            cp_loss = best['centipawn'] - ev['centipawn']
            marker = "→" if ev['uci'] == target_move_uci else " "
            loss_str = f" (loses {cp_loss}cp)" if cp_loss > 0 else ""
            lines.append(f"{marker} {i}. {ev['san']}: {score_str(ev)}{loss_str}")
        
        if target_rank > top_n:
            # Show target move even if not in top N
            cp_loss = best['centipawn'] - target_eval['centipawn']
            lines.append(f"...")
            lines.append(f"→ {target_rank}. {target_eval['san']}: {score_str(target_eval)} (loses {cp_loss}cp)")
        
        move_analysis = "\n".join(lines)
        
    elif style == "compact":
        # Single line with scores
        parts = [f"{ev['san']}({score_str(ev)})" for ev in sorted_evals[:top_n]]
        move_analysis = " ".join(parts)
        
    else:  # minimal
        move_analysis = f"Best: {best['san']} {score_str(best)}"
    
    return {
        "move_analysis": move_analysis,
        "compact_analysis": " ".join(f"{ev['san']}({score_str(ev)})" for ev in sorted_evals[:3]),
        "best_move": best['san'],
        "best_score": score_str(best),
        "move_san": target_eval['san'],
        "move_score": score_str(target_eval)
    }


# =============================================================================
# Position to Messages Conversion
# =============================================================================

def position_to_messages(
    position: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    response_template: str = None,
    include_eval: bool = True,
    eval_style: str = "detailed",
    top_n_moves: int = 5
) -> Dict[str, Any]:
    """
    Convert a chess position to chat message format.
    
    Args:
        position: Dictionary with position data (from ChessPosition)
            Should include 'move_evaluations' for eval-based formatting
        prompt_template: Template for user message
        response_template: Template for assistant response (auto-selected if None)
        include_eval: Whether to include move evaluations in thinking
        eval_style: Style for move analysis ("detailed", "compact", "minimal")
        top_n_moves: Number of top moves to include in analysis
    
    Returns:
        Dictionary with 'messages' key containing chat format
    """
    # Format user content
    user_content = prompt_template.format(
        fen=position['fen'],
        legal_moves=position['legal_moves_uci'],
        board=position['board_utf'],
        example_move=position['first_legal_move']
    )
    
    # Determine response template
    if response_template is None:
        if include_eval and 'move_evaluations' in position and position['move_evaluations']:
            response_template = DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL
        else:
            response_template = SIMPLE_RESPONSE_TEMPLATE
    
    # Format assistant content
    if include_eval and 'move_evaluations' in position and position['move_evaluations']:
        # Format move analysis
        analysis = format_move_analysis(
            move_evaluations=position['move_evaluations'],
            target_move_uci=position['target_move_uci'],
            top_n=top_n_moves,
            style=eval_style
        )
        
        assistant_content = response_template.format(
            move=position['target_move_uci'],
            **analysis
        )
    else:
        # Fallback to simple format
        assistant_content = SIMPLE_RESPONSE_TEMPLATE.format(
            move=position['target_move_uci']
        )
    
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def position_to_text(
    position: Dict[str, Any],
    tokenizer: Any,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    response_template: str = None,
    include_eval: bool = True,
    eval_style: str = "detailed"
) -> str:
    """
    Convert position to formatted text using tokenizer's chat template.
    
    Args:
        position: Position dictionary
        tokenizer: HuggingFace tokenizer with chat template
        prompt_template: User message template
        response_template: Assistant response template
        include_eval: Whether to include evaluations
        eval_style: Style of evaluation format
    
    Returns:
        Formatted text string
    """
    messages = position_to_messages(
        position, 
        prompt_template, 
        response_template,
        include_eval=include_eval,
        eval_style=eval_style
    )["messages"]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def create_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    include_eval: bool = True,
    eval_style: str = "detailed"
) -> Callable[[Dict[str, Any]], str]:
    """
    Create a formatting function for SFTTrainer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_template: Custom user prompt template
        response_template: Custom response template
        include_eval: Whether to include move evaluations
        eval_style: Style of evaluation format
    
    Returns:
        Callable that converts examples to formatted text
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
    response = response_template  # Can be None for auto-selection
    
    def formatting_func(example: Dict[str, Any]) -> str:
        # Handle both raw positions and pre-formatted messages
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = position_to_messages(
                example, 
                prompt, 
                response,
                include_eval=include_eval,
                eval_style=eval_style
            )["messages"]
        
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    return formatting_func


def create_batch_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    include_eval: bool = True,
    eval_style: str = "detailed"
) -> Callable[[Dict[str, Any]], list]:
    """
    Create a batch formatting function for SFTTrainer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_template: Custom user prompt template
        response_template: Custom response template
        include_eval: Whether to include move evaluations
        eval_style: Style of evaluation format
    
    Returns:
        Callable that converts batched examples to formatted texts
    """
    single_func = create_formatting_func(
        tokenizer, 
        prompt_template, 
        response_template,
        include_eval=include_eval,
        eval_style=eval_style
    )
    
    def batch_formatting_func(examples: Dict[str, Any]) -> list:
        batch_size = len(examples[next(iter(examples.keys()))])
        
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(single_func(example))
        
        return results
    
    return batch_formatting_func


def add_messages_column(
    dataset: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    include_eval: bool = True,
    eval_style: str = "detailed",
    num_proc: int = 4
) -> Any:
    """
    Add a 'messages' column to a dataset.
    
    Args:
        dataset: HuggingFace Dataset
        prompt_template: Custom user prompt template
        response_template: Custom response template
        include_eval: Whether to include move evaluations
        eval_style: Style of evaluation format
        num_proc: Number of processes for mapping
    
    Returns:
        Dataset with added 'messages' column
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
    response = response_template
    
    def add_messages(example):
        return position_to_messages(
            example, 
            prompt, 
            response,
            include_eval=include_eval,
            eval_style=eval_style
        )
    
    return dataset.map(add_messages, num_proc=num_proc)


# =============================================================================
# Display Utilities
# =============================================================================

def format_example_for_display(
    position: Dict[str, Any],
    include_eval: bool = True
) -> str:
    """
    Format a position for human-readable display.
    
    Args:
        position: Position dictionary
        include_eval: Whether to show move evaluations
    
    Returns:
        Formatted string for display
    """
    messages = position_to_messages(
        position,
        include_eval=include_eval
    )["messages"]
    
    output = []
    output.append("=" * 70)
    output.append("USER MESSAGE:")
    output.append("-" * 70)
    output.append(messages[0]["content"])
    output.append("")
    output.append("=" * 70)
    output.append("ASSISTANT RESPONSE:")
    output.append("-" * 70)
    output.append(messages[1]["content"])
    output.append("=" * 70)
    
    # Add metadata if available
    if 'loss_weight' in position:
        output.append(f"\nLoss weight: {position['loss_weight']:.3f}")
    if 'move_evaluations' in position and position['move_evaluations']:
        output.append(f"Moves analyzed: {len(position['move_evaluations'])}")
    
    return "\n".join(output)


if __name__ == "__main__":
    # Test with a sample position with evaluations
    sample_position_with_eval = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "legal_moves_uci": "a7a6 a7a5 b7b6 b7b5 c7c6 c7c5 d7d6 d7d5 e7e6 e7e5 f7f6 f7f5 g7g6 g7g5 h7h6 h7h5 b8a6 b8c6 g8f6 g8h6",
        "target_move_uci": "e7e5",
        "first_legal_move": "a7a6",
        "board_utf": """   a  b  c  d  e  f  g  h  
   +------------------------+
8 | ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ | 8
7 | ♟  ♟  ♟  ♟  ♟  ♟  ♟  ♟ | 7
6 | ·  ·  ·  ·  ·  ·  ·  · | 6
5 | ·  ·  ·  ·  ·  ·  ·  · | 5
4 | ·  ·  ·  ·  ♙  ·  ·  · | 4
3 | ·  ·  ·  ·  ·  ·  ·  · | 3
2 | ♙  ♙  ♙  ♙  ·  ♙  ♙  ♙ | 2
1 | ♖  ♘  ♗  ♕  ♔  ♗  ♘  ♖ | 1
   +------------------------+
   a  b  c  d  e  f  g  h  """,
        "side_to_move": "Black",
        "white_elo": 1800,
        "black_elo": 1750,
        "move_number": 1,
        "source": "game",
        "loss_weight": 1.0,
        # Move evaluations from Stockfish
        "move_evaluations": [
            {"uci": "e7e5", "san": "e5", "centipawn": -20, "mate_in": None},
            {"uci": "c7c5", "san": "c5", "centipawn": -25, "mate_in": None},
            {"uci": "d7d5", "san": "d5", "centipawn": -30, "mate_in": None},
            {"uci": "g8f6", "san": "Nf6", "centipawn": -35, "mate_in": None},
            {"uci": "e7e6", "san": "e6", "centipawn": -40, "mate_in": None},
            {"uci": "c7c6", "san": "c6", "centipawn": -45, "mate_in": None},
            {"uci": "d7d6", "san": "d6", "centipawn": -50, "mate_in": None},
        ]
    }
    
    print("=" * 70)
    print("TESTING FORMATTING WITH MOVE EVALUATIONS")
    print("=" * 70)
    
    print("\n--- DETAILED STYLE ---")
    print(format_example_for_display(sample_position_with_eval, include_eval=True))
    
    print("\n\n--- COMPACT STYLE ---")
    messages = position_to_messages(
        sample_position_with_eval,
        response_template=COMPACT_RESPONSE_TEMPLATE_WITH_EVAL,
        eval_style="compact"
    )
    print(messages["messages"][1]["content"])
    
    print("\n\n--- WITHOUT EVAL ---")
    messages_no_eval = position_to_messages(
        sample_position_with_eval,
        include_eval=False
    )
    print(messages_no_eval["messages"][1]["content"])
