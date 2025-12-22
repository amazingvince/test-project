#!/usr/bin/env bash
set -euo pipefail

# Example submission script for the AIcrowd Global Chess Challenge 2025.
#
# Edit the variables below, then run:
#   bash competition/global_chess_challenge_2025/aicrowd_submit_example.sh
#
# Notes:
# - Your model must be on Hugging Face (or a HF path you have access to).
# - Keep `--vllm-inference.max-tokens` small for faster move latency.

CHALLENGE="global-chess-challenge-2025"
HF_REPO="YOUR_HF_USERNAME/YOUR_MODEL_NAME"   # e.g. "amazi/chess-distill-qwen3-0.6b"
HF_REPO_TAG="main"

# Pick one of the templates in `competition/global_chess_challenge_2025/`
PROMPT_TEMPLATE="competition/global_chess_challenge_2025/prompt_with_board_ascii.jinja"

# Token budget for the model's response (<think> + <uci_move>).
VLLM_MAX_TOKENS="128"

aicrowd login

aicrowd submit-model \
  --challenge "$CHALLENGE" \
  --hf-repo "$HF_REPO" \
  --hf-repo-tag "$HF_REPO_TAG" \
  --prompt_template_path "$PROMPT_TEMPLATE" \
  --vllm-inference.max-tokens "$VLLM_MAX_TOKENS"
