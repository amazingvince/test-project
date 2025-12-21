#!/usr/bin/env python3
"""
Upload trained chess model to Hugging Face Hub.

Usage:
    # Basic upload
    python scripts/upload_to_hub.py --model_path ./outputs/chess-distill-final --repo_id username/chess-llm

    # With custom commit message
    python scripts/upload_to_hub.py --model_path ./outputs/chess-distill-final --repo_id username/chess-llm --commit "v1.0 release"

    # Private repo
    python scripts/upload_to_hub.py --model_path ./outputs/chess-distill-final --repo_id username/chess-llm --private

Requirements:
    pip install huggingface_hub
    huggingface-cli login  # Or set HF_TOKEN environment variable
"""

import argparse
import os
from pathlib import Path


def create_model_card(
    repo_id: str,
    base_model: str = "Qwen/Qwen3-0.6B",
    training_type: str = "Policy Distillation",
) -> str:
    """Create a model card for the chess model."""
    return f"""---
license: mit
base_model: {base_model}
tags:
  - chess
  - game-playing
  - policy-distillation
  - stockfish
language:
  - en
pipeline_tag: text-generation
---

# Chess LLM - {training_type}

A language model fine-tuned to play chess using {training_type.lower()} from Stockfish.

## Model Description

This model was trained using a two-stage approach:
1. **SFT (Supervised Fine-Tuning)**: Learn basic chess patterns from human games
2. **Policy Distillation**: Refine with Stockfish's probability distribution over moves

## Training Details

- **Base Model**: {base_model}
- **Training Method**: {training_type}
- **Loss**: Cross-entropy on reasoning + Forward KL divergence on move distribution

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = '''You are an expert chess player. Analyze this position and select the best move.

Position (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

Legal moves: a7a6 a7a5 b7b6 b7b5 c7c6 c7c5 d7d6 d7d5 e7e6 e7e5 f7f6 f7f5 g7g6 g7g5 h7h6 h7h5 b8a6 b8c6 g8f6 g8h6

Analyze the candidate moves, evaluate their quality, then select the best one.

Output format:
<think>your analysis of the moves</think>
<uci_move>your_chosen_move</uci_move>'''

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Output Format

The model outputs analysis in this format:
```
<think>
Analyzing position...
- e5: equal (51% win)
- c5: equal (50% win)
- e6: equal (49% win)
- d5: slight disadvantage (47% win)
- Nf6: slight disadvantage (46% win)

Best line: e5 Nf3 Nc6 Bb5 a6
Playing e5 gives 51% win chance.
</think>
<uci_move>e7e5</uci_move>
```

## Limitations

- Trained primarily on standard chess positions
- May not perform optimally in unusual or highly tactical positions
- Output format must be parsed to extract the UCI move

## License

MIT License
"""


def upload_model(
    model_path: str,
    repo_id: str,
    commit_message: str = "Upload chess model",
    private: bool = False,
    create_card: bool = True,
    base_model: str = "Qwen/Qwen3-0.6B",
):
    """Upload model to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo, upload_folder

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Check for required files
    required_files = ["config.json"]
    has_safetensors = (model_path / "model.safetensors").exists()
    has_pytorch = (model_path / "pytorch_model.bin").exists()

    if not has_safetensors and not has_pytorch:
        # Check for sharded models
        has_sharded = any(model_path.glob("model-*.safetensors")) or any(model_path.glob("pytorch_model-*.bin"))
        if not has_sharded:
            raise FileNotFoundError(
                f"No model weights found in {model_path}. "
                "Expected model.safetensors, pytorch_model.bin, or sharded versions."
            )

    for f in required_files:
        if not (model_path / f).exists():
            raise FileNotFoundError(f"Required file not found: {model_path / f}")

    print(f"Uploading model from: {model_path}")
    print(f"Destination: https://huggingface.co/{repo_id}")
    print(f"Private: {private}")

    # Initialize API
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Create model card if requested
    if create_card:
        card_path = model_path / "README.md"
        if not card_path.exists():
            print("Creating model card...")
            card_content = create_model_card(
                repo_id=repo_id,
                base_model=base_model,
            )
            with open(card_path, "w") as f:
                f.write(card_content)
            print(f"Model card created: {card_path}")

    # Upload
    print(f"\nUploading files...")
    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message=commit_message,
    )

    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"Model URL: https://huggingface.co/{repo_id}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained chess model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="Upload chess model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--no-card",
        action="store_true",
        help="Skip creating model card",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name for model card",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )

    args = parser.parse_args()

    # Set token if provided
    if args.token:
        os.environ["HF_TOKEN"] = args.token

    upload_model(
        model_path=args.model_path,
        repo_id=args.repo_id,
        commit_message=args.commit,
        private=args.private,
        create_card=not args.no_card,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
