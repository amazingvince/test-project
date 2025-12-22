# Tokenizer Modes and Move Distillation

This repo supports two ways to represent the final chess move and two ways to
apply the distillation (KL/JSD) loss.

## Tokenizer Modes

Configured via `tokenizer.chess_mode`:

- `tags_only`
  - Adds only `<uci_move>` and `</uci_move>` as special tokens.
  - The move itself (e.g. `e2e4`) is generated as plain text (tokenized into 1+
    base-model tokens).
  - Recommended default for SFT: you avoid adding 8k new embeddings.

- `tags_and_moves`
  - Adds `<uci_move>`/`</uci_move>` plus an explicit vocabulary of all 8,064 UCI
    move strings (including promotions).
  - Every move becomes a single token, which is ideal for a direct KL loss over
    a move distribution.

## Distillation Modes (No 8k Move Vocab Option)

Configured via `distillation.move_distill_mode`:

- `move_token` (default)
  - Applies KL/JSD at the `<uci_move>` position over a single move token.
  - Requires `tokenizer.chess_mode: tags_and_moves`.
  - Fastest and closest to the “teacher distribution over legal moves” objective.

- `move_prefix`
  - Applies a token-level distillation loss over the move *string* after
    `<uci_move>`, using a prefix view of the teacher distribution:
    - step 0: teacher mass aggregated by the first token of each legal move
    - step 1+: teacher mass aggregated by the next token given the already-forced prefix
  - Works with `tokenizer.chess_mode: tags_only` (no 8k move vocab).
  - Trade-off: this is not identical to a full move-level KL, but it’s a
    practical approximation that avoids extra embeddings and still pressures the
    model toward the teacher’s move preferences.

## Recommended Setups

- SFT with reasoning traces + best-move targets:
  - `tokenizer.chess_mode: tags_only`
  - `reasoning_trace.enabled: true`
  - `data.target_move: best` (in `configs/sft/config_with_eval.yaml`)

- Distillation (highest throughput):
  - `tokenizer.chess_mode: tags_and_moves`
  - `distillation.move_distill_mode: move_token`

- Distillation without adding 8k move tokens:
  - `tokenizer.chess_mode: tags_only`
  - `distillation.move_distill_mode: move_prefix`

