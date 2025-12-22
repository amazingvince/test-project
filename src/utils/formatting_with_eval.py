"""
Formatting helpers for SFT datasets that include Stockfish move evaluations.

This is similar to `src.utils.formatting`, but the assistant response can embed
top-move analysis inside `<think>...</think>`.

Expected extra fields (when `include_eval=True`):
  - `move_evaluations`: list of dicts with at least `uci`, `san`, and either
    `centipawn` or `mate_in`
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


DEFAULT_PROMPT_TEMPLATE = """You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves}

Board (ASCII):
{board}

Select the best move. Analyze candidate moves and their values, then output your chosen move.

CRITICAL: Your UCI move must always be one of the moves from legal_moves_uci, with no spaces.

Example:
<uci_move>{example_move}</uci_move>

Format:
<think>analyze moves and pick the best one</think>
<uci_move>your_move</uci_move>"""


DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL = """<think>
{move_analysis}
Best continuation: {best_move} ({best_score})
</think>
<uci_move>{move}</uci_move>"""


COMPACT_RESPONSE_TEMPLATE_WITH_EVAL = """<think>{compact_analysis} -> {move_san}({move_score})</think>
<uci_move>{move}</uci_move>"""


SIMPLE_RESPONSE_TEMPLATE = "<think></think>\n<uci_move>{move}</uci_move>"


def _score_str(ev: Dict[str, Any]) -> str:
    mate_in = ev.get("mate_in")
    if mate_in is not None:
        return f"M{mate_in}"
    return f"{int(ev.get('centipawn', 0)):+d}cp"


def format_move_analysis(
    move_evaluations: List[Dict[str, Any]],
    target_move_uci: str,
    top_n: int = 5,
    style: str = "detailed",
) -> Dict[str, str]:
    """
    Format Stockfish move evaluations for inclusion in a response template.

    Args:
        move_evaluations: List of evaluation dicts, each with `uci`, `san`, and
            either `centipawn` or `mate_in`.
        target_move_uci: Move being played (UCI).
        top_n: Number of candidate moves to list in detailed/compact output.
        style: One of `detailed`, `compact`, `minimal`.

    Returns:
        Mapping suitable for `DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL`.
    """
    if not move_evaluations:
        return {
            "move_analysis": "No analysis available",
            "compact_analysis": "?",
            "best_move": "?",
            "best_score": "?",
            "move_san": target_move_uci,
            "move_score": "?",
        }

    sorted_evals = sorted(move_evaluations, key=lambda x: x.get("centipawn", 0), reverse=True)
    best = sorted_evals[0]

    target_eval: Optional[Dict[str, Any]] = None
    target_rank = -1
    for i, ev in enumerate(sorted_evals):
        if ev.get("uci") == target_move_uci:
            target_eval = ev
            target_rank = i + 1
            break
    if target_eval is None:
        target_eval = {"uci": target_move_uci, "san": target_move_uci, "centipawn": 0}
        target_rank = len(sorted_evals) + 1

    if style == "detailed":
        lines: List[str] = [
            f"Analyzing position with {len(sorted_evals)} legal moves...",
            "",
            "Top candidates:",
        ]
        for i, ev in enumerate(sorted_evals[:top_n], 1):
            cp_loss = int(best.get("centipawn", 0)) - int(ev.get("centipawn", 0))
            marker = "*" if ev.get("uci") == target_move_uci else " "
            loss_str = f" (loses {cp_loss}cp)" if cp_loss > 0 else ""
            lines.append(f"{marker} {i}. {ev.get('san', ev.get('uci', '?'))}: {_score_str(ev)}{loss_str}")

        if target_rank > top_n:
            cp_loss = int(best.get("centipawn", 0)) - int(target_eval.get("centipawn", 0))
            lines.append("...")
            lines.append(
                f"* {target_rank}. {target_eval.get('san', target_eval.get('uci', '?'))}: "
                f"{_score_str(target_eval)} (loses {cp_loss}cp)"
            )

        move_analysis = "\n".join(lines)
    elif style == "compact":
        move_analysis = " ".join(
            f"{ev.get('san', ev.get('uci', '?'))}({_score_str(ev)})" for ev in sorted_evals[:top_n]
        )
    else:
        move_analysis = f"Best: {best.get('san', best.get('uci', '?'))} {_score_str(best)}"

    compact_analysis = " ".join(
        f"{ev.get('san', ev.get('uci', '?'))}({_score_str(ev)})" for ev in sorted_evals[:3]
    )

    return {
        "move_analysis": move_analysis,
        "compact_analysis": compact_analysis,
        "best_move": best.get("san", best.get("uci", "?")),
        "best_score": _score_str(best),
        "move_san": target_eval.get("san", target_eval.get("uci", "?")),
        "move_score": _score_str(target_eval),
    }


def position_to_messages(
    position: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    response_template: Optional[str] = None,
    *,
    include_eval: bool = True,
    eval_style: str = "detailed",
    top_n_moves: int = 5,
) -> Dict[str, Any]:
    """
    Convert a position into `{ "messages": [...] }` for chat-model fine-tuning.
    """
    user_content = prompt_template.format(
        fen=position["fen"],
        legal_moves=position["legal_moves_uci"],
        board=position["board_utf"],
        example_move=position["first_legal_move"],
    )

    if include_eval:
        template = response_template or DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL
        fmt = format_move_analysis(
            position.get("move_evaluations", []),
            target_move_uci=position["target_move_uci"],
            top_n=top_n_moves,
            style=eval_style,
        )
        assistant_content = template.format(move=position["target_move_uci"], **fmt)
    else:
        template = response_template or SIMPLE_RESPONSE_TEMPLATE
        assistant_content = template.format(move=position["target_move_uci"])

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def create_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    *,
    include_eval: bool = True,
    eval_style: str = "detailed",
    top_n_moves: int = 5,
) -> Callable[[Dict[str, Any]], str]:
    """
    Create a per-example formatting function for HF datasets.
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE

    def formatting_func(example: Dict[str, Any]) -> str:
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = position_to_messages(
                example,
                prompt_template=prompt,
                response_template=response_template,
                include_eval=include_eval,
                eval_style=eval_style,
                top_n_moves=top_n_moves,
            )["messages"]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return formatting_func


def create_batch_formatting_func(
    tokenizer: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    *,
    include_eval: bool = True,
    eval_style: str = "detailed",
    top_n_moves: int = 5,
) -> Callable[[Dict[str, Any]], list[str]]:
    """
    Create a batched formatting function for HF datasets (`batched=True`).
    """
    single = create_formatting_func(
        tokenizer,
        prompt_template=prompt_template,
        response_template=response_template,
        include_eval=include_eval,
        eval_style=eval_style,
        top_n_moves=top_n_moves,
    )

    def batch_formatting_func(examples: Dict[str, Any]) -> list[str]:
        batch_size = len(examples[next(iter(examples.keys()))])
        return [single({k: v[i] for k, v in examples.items()}) for i in range(batch_size)]

    return batch_formatting_func


def add_messages_column(
    dataset: Any,
    prompt_template: Optional[str] = None,
    response_template: Optional[str] = None,
    *,
    include_eval: bool = True,
    eval_style: str = "detailed",
    top_n_moves: int = 5,
    num_proc: int = 4,
) -> Any:
    """
    Add a `messages` column to a dataset using eval-aware templates.
    """
    prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE

    def add_messages(example: Dict[str, Any]) -> Dict[str, Any]:
        return position_to_messages(
            example,
            prompt_template=prompt,
            response_template=response_template,
            include_eval=include_eval,
            eval_style=eval_style,
            top_n_moves=top_n_moves,
        )

    return dataset.map(add_messages, num_proc=num_proc)


def format_example_for_display(position: Dict[str, Any], *, include_eval: bool = True) -> str:
    """
    Render an example as a readable, two-section string (user / assistant).
    """
    messages = position_to_messages(position, include_eval=include_eval)["messages"]
    parts = [
        "=" * 70,
        "USER MESSAGE:",
        "-" * 70,
        messages[0]["content"],
        "",
        "=" * 70,
        "ASSISTANT RESPONSE:",
        "-" * 70,
        messages[1]["content"],
        "=" * 70,
    ]

    if "loss_weight" in position:
        parts.append(f"\nLoss weight: {position['loss_weight']:.3f}")
    if include_eval and position.get("move_evaluations"):
        parts.append(f"Moves analyzed: {len(position['move_evaluations'])}")

    return "\n".join(parts)

