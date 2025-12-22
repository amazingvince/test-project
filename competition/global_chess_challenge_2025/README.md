# Global Chess Challenge 2025 (AIcrowd) Submission Notes

This repo trains chess-capable chat LLMs that output:

```text
<think>...</think>
<uci_move>e2e4</uci_move>
```

The Global Chess Challenge 2025 starter kit expects the same output tags and
provides the input as FEN + a list of legal UCI moves, so the core format is
already compatible.

Template alignment:
- Competition eval uses `competition/global_chess_challenge_2025/prompt_minimal.jinja`.
- Training configs are aligned to the same wording (`configs/sft/config_sft.yaml`, distill prompts in `src/distill/formatting_distill.py`).
- If you have old preprocessed datasets with a baked `messages` column, regenerate them (or re-run preprocessing) to apply template changes.

## What You Need

- A trained model pushed to the Hugging Face Hub (recommended for submission).
- A prompt template (Jinja2) that the starter kit uses to format each turn.
- A short generation budget (keep outputs fast and predictable).

This folder includes:
- Prompt templates: `prompt_minimal.jinja`, `prompt_with_board_ascii.jinja`
- A submission command example: `aicrowd_submit_example.sh`

## Step 1: Upload Your Model to Hugging Face

From this repo:

```bash
python scripts/upload_to_hub.py --model_path ./outputs/chess-distill-final --repo_id <you>/<model-name>
```

Make sure your uploaded tokenizer includes the special tags used during training:
`<think>`, `</think>`, `<uci_move>`, `</uci_move>`.

## Step 2: Local Compatibility Test (Starter Kit)

This is the fastest way to confirm the prompt + output format matches what the
competition evaluator parses.

1) Prepare the starter kit + copy templates:

```bash
python scripts/global_chess_challenge_2025_setup.py
```

2) Install the starter kit requirements:

```bash
pip install -r tmp/global-chess-challenge-2025-starter-kit/requirements.txt
```

3) Start a vLLM server (separate terminal):

Linux/macOS (bash):
```bash
cd tmp/global-chess-challenge-2025-starter-kit/player_agents
pip install vllm
vllm serve "<your_hf_repo_or_local_path>" \
  --served-model-name aicrowd-chess-model \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --disable-log-stats \
  --host 0.0.0.0 \
  --port 5000
```

Windows (PowerShell):

```powershell
cd tmp/global-chess-challenge-2025-starter-kit/player_agents
pip install vllm
vllm serve "<your_hf_repo_or_local_path>" `
  --served-model-name aicrowd-chess-model `
  --dtype bfloat16 `
  --gpu-memory-utilization 0.9 `
  --enforce-eager `
  --disable-log-stats `
  --host 0.0.0.0 `
  --port 5000
```

4) Run the starter kit local evaluator:

```bash
cd tmp/global-chess-challenge-2025-starter-kit
python local_evaluation.py --template-file player_agents/chess_distill_prompt_minimal.jinja --endpoint http://localhost:5000/v1 --games-per-opponent 10
```

If you see warnings about illegal moves, the model is not reliably selecting
from the legal move list; tighten the prompt and/or reduce generation length.

## Step 3: Submit to AIcrowd

Edit and run:

```bash
bash competition/global_chess_challenge_2025/aicrowd_submit_example.sh
```

Recommended submission settings:
- Use `prompt_minimal.jinja` if you want lower latency.
- Set `--vllm-inference.max-tokens` to ~`64-256` (usually plenty).

## Notes / Gotchas

- The evaluator parses the move via a regex for `<uci_move>...</uci_move>`.
- Keep output short; long `<think>` traces increase move latency and raise the
  odds of producing extra junk after the move tag.
- If your HF repo is private/gated, you must grant AIcrowd access (see the
  starter kit doc `docs/huggingface-gated-models.md` in the AIcrowd repo).
