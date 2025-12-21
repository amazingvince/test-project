"""
Formatting module for Chess SFT training.

Converts chess positions to chat format suitable for SFT training.
"""

from typing import Dict, Any, Optional, Callable


DEFAULT_PROMPT_TEMPLATE = """You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves}

Board looks like this right now
{board}

Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.

CRITICAL: Your UCI move must always be one of the moves from legal_moves_uci. With no spaces in between.

An example of a valid move for this position is <uci_move>{example_move}</uci_move>. 
Format:
<think>brief thinking (2 sentences max)</think>
<uci_move>your_move</uci_move>"""

DEFAULT_RESPONSE_TEMPLATE = "<think></think>\n<uci_move>{move}</uci_move>"


def position_to_messages(
    position: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    response_template: str = DEFAULT_RESPONSE_TEMPLATE
) -> Dict[str, Any]:
    """
    Convert a chess position to chat message format.
    
    Args:
        position: Dictionary with position data (from ChessPosition)
        prompt_template: Template for user message
        response_template: Template for assistant response
    
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
    
    # Format assistant content (empty thinking for now)
    assistant_content = response_template.format(
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
    response_template: str = DEFAULT_RESPONSE_TEMPLATE
) -> str:
    """
    Convert position to formatted text using tokenizer's chat template.
    
    Args:
        position: Position dictionary
        tokenizer: HuggingFace tokenizer with chat template
        prompt_template: User message template
        response_template: Assistant response template
    
    Returns:
        Formatted text string
    """
    messages = position_to_messages(
        position, 
        prompt_template, 
        response_template
    )["messages"]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def create_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None
) -> Callable[[Dict[str, Any]], str]:
    """
    Create a formatting function for SFTTrainer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_template: Custom user prompt template
        response_template: Custom response template
    
    Returns:
        Callable that converts examples to formatted text
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
    response = response_template or DEFAULT_RESPONSE_TEMPLATE
    
    def formatting_func(example: Dict[str, Any]) -> str:
        # Handle both raw positions and pre-formatted messages
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = position_to_messages(example, prompt, response)["messages"]
        
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    return formatting_func


def create_batch_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None
) -> Callable[[Dict[str, Any]], list]:
    """
    Create a batch formatting function for SFTTrainer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_template: Custom user prompt template
        response_template: Custom response template
    
    Returns:
        Callable that converts batched examples to formatted texts
    """
    single_func = create_formatting_func(tokenizer, prompt_template, response_template)
    
    def batch_formatting_func(examples: Dict[str, Any]) -> list:
        # Get the number of examples in the batch
        batch_size = len(examples[next(iter(examples.keys()))])
        
        results = []
        for i in range(batch_size):
            # Extract single example
            example = {k: v[i] for k, v in examples.items()}
            results.append(single_func(example))
        
        return results
    
    return batch_formatting_func


def add_messages_column(
    dataset: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    num_proc: int = 4
) -> Any:
    """
    Add a 'messages' column to a dataset.
    
    Args:
        dataset: HuggingFace Dataset
        prompt_template: Custom user prompt template
        response_template: Custom response template
        num_proc: Number of processes for mapping
    
    Returns:
        Dataset with added 'messages' column
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
    response = response_template or DEFAULT_RESPONSE_TEMPLATE
    
    def add_messages(example):
        return position_to_messages(example, prompt, response)
    
    return dataset.map(add_messages, num_proc=num_proc)


# Convenience function for testing
def format_example_for_display(position: Dict[str, Any]) -> str:
    """
    Format a position for human-readable display.
    
    Args:
        position: Position dictionary
    
    Returns:
        Formatted string for display
    """
    messages = position_to_messages(position)["messages"]
    
    output = []
    output.append("=" * 60)
    output.append("USER MESSAGE:")
    output.append("-" * 60)
    output.append(messages[0]["content"])
    output.append("")
    output.append("=" * 60)
    output.append("ASSISTANT RESPONSE:")
    output.append("-" * 60)
    output.append(messages[1]["content"])
    output.append("=" * 60)
    
    return "\n".join(output)


if __name__ == "__main__":
    # Test with a sample position
    sample_position = {
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
        "source": "game"
    }
    
    print(format_example_for_display(sample_position))
