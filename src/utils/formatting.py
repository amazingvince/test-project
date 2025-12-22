"""
Formatting helpers for supervised fine-tuning (SFT).

This module converts a single chess position example into a two-message
chat-style conversation (`user`, `assistant`). The resulting `messages` can be
passed through a Hugging Face tokenizer chat template to produce the final
training text.

Expected position fields when using the default templates:
  - `fen`: FEN string
  - `legal_moves_uci`: space-separated UCI moves
  - `board_utf`: ASCII board rendering (see `src.utils.chess_utils.render_board_utf`)
  - `first_legal_move`: an example legal move in UCI
  - `target_move_uci`: the target move in UCI

Distillation uses a separate formatter in `src/distill/formatting_distill.py`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


DEFAULT_PROMPT_TEMPLATE = """You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves}

Board (ASCII):
{board}

Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.

CRITICAL: Your UCI move must always be one of the moves from legal_moves_uci, with no spaces.

Example:
<uci_move>{example_move}</uci_move>

Format:
<think>brief thinking (2 sentences max)</think>
<uci_move>your_move</uci_move>"""

DEFAULT_RESPONSE_TEMPLATE = "<think></think>\n<uci_move>{move}</uci_move>"


def position_to_messages(
    position: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    response_template: str = DEFAULT_RESPONSE_TEMPLATE,
) -> Dict[str, Any]:
    """
    Convert a position dict into the `messages` schema used by chat models.

    The returned dict contains a `messages` key with two entries:
      1. A `user` message containing the formatted prompt.
      2. An `assistant` message containing the formatted target response.
    """
    user_content = prompt_template.format(
        fen=position["fen"],
        legal_moves=position["legal_moves_uci"],
        board=position["board_utf"],
        example_move=position["first_legal_move"],
    )
    assistant_content = response_template.format(move=position["target_move_uci"])

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def position_to_text(
    position: Dict[str, Any],
    tokenizer: Any,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    response_template: str = DEFAULT_RESPONSE_TEMPLATE,
) -> str:
    """
    Render a position to a single training text using `tokenizer.apply_chat_template`.
    """
    messages = position_to_messages(position, prompt_template, response_template)["messages"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def create_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
) -> Callable[[Dict[str, Any]], str]:
    """
    Create a per-example formatting function compatible with HF Dataset `.map()`.

    This supports either:
      - raw examples (position dict fields), or
      - pre-formatted examples containing a `messages` field.
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
    response = response_template or DEFAULT_RESPONSE_TEMPLATE

    def formatting_func(example: Dict[str, Any]) -> str:
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = position_to_messages(example, prompt, response)["messages"]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return formatting_func


def create_batch_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
) -> Callable[[Dict[str, Any]], list[str]]:
    """
    Create a batched formatting function compatible with HF Dataset `.map(batched=True)`.
    """
    single_func = create_formatting_func(tokenizer, prompt_template, response_template)

    def batch_formatting_func(examples: Dict[str, Any]) -> list[str]:
        batch_size = len(examples[next(iter(examples.keys()))])
        return [single_func({k: v[i] for k, v in examples.items()}) for i in range(batch_size)]

    return batch_formatting_func


def add_messages_column(
    dataset: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    num_proc: int = 4,
) -> Any:
    """
    Add a `messages` column to a Hugging Face dataset.
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
    response = response_template or DEFAULT_RESPONSE_TEMPLATE

    def add_messages(example: Dict[str, Any]) -> Dict[str, Any]:
        return position_to_messages(example, prompt, response)

    return dataset.map(add_messages, num_proc=num_proc)


def format_example_for_display(position: Dict[str, Any]) -> str:
    """
    Render an example as a readable, two-section string (user / assistant).
    """
    messages = position_to_messages(position)["messages"]
    parts = [
        "=" * 60,
        "USER MESSAGE:",
        "-" * 60,
        messages[0]["content"],
        "",
        "=" * 60,
        "ASSISTANT RESPONSE:",
        "-" * 60,
        messages[1]["content"],
        "=" * 60,
    ]
    return "\n".join(parts)

