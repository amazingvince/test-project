#!/bin/bash
# train_sft.sh - Launch SFT (Supervised Fine-Tuning) training
#
# Stage 1 of the two-stage training workflow:
#   1. SFT: Learn basic chess from human games (this script)
#   2. Distillation: Refine with Stockfish policy (train_distill.sh)
#
# Usage:
#   ./scripts/train_sft.sh                    # Normal training
#   ./scripts/train_sft.sh --debug            # Debug mode (small dataset)
#   ./scripts/train_sft.sh --resume checkpoint # Resume from checkpoint
#   ./scripts/train_sft.sh --no-eval          # Skip post-training evaluation

set -e

# Parse arguments
DEBUG=""
RESUME=""
SKIP_EVAL=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --no-eval)
            SKIP_EVAL="1"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# Disable W&B if in debug mode
if [ -n "$DEBUG" ]; then
    export WANDB_MODE=disabled
fi

echo "=============================================="
echo "Chess LLM SFT - Training"
echo "=============================================="

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Show configuration
echo "Configuration:"
echo "  Config: configs/config_sft.yaml"
echo "  Debug: ${DEBUG:-disabled}"
echo "  Resume: ${RESUME:-none}"
echo "  Post-training eval: ${SKIP_EVAL:-enabled}"
echo ""

# Find Stockfish
STOCKFISH_PATH=""
if command -v stockfish &> /dev/null; then
    STOCKFISH_PATH=$(which stockfish)
elif [ -f "/usr/bin/stockfish" ]; then
    STOCKFISH_PATH="/usr/bin/stockfish"
elif [ -f "/usr/games/stockfish" ]; then
    STOCKFISH_PATH="/usr/games/stockfish"
elif [ -f "/usr/local/bin/stockfish" ]; then
    STOCKFISH_PATH="/usr/local/bin/stockfish"
fi

if [ -n "$STOCKFISH_PATH" ]; then
    echo "Stockfish: $STOCKFISH_PATH"
else
    echo "Stockfish: not found (ACPL metrics disabled)"
fi
echo ""

# Run training
echo "Starting training..."
echo ""

accelerate launch \
    --config_file configs/accelerate.yaml \
    train_sft.py \
    --config configs/config_sft.yaml \
    --streaming \
    $DEBUG \
    $RESUME \
    $EXTRA_ARGS

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

# Run post-training evaluation unless skipped or in debug mode
if [ -z "$SKIP_EVAL" ] && [ -z "$DEBUG" ]; then
    echo ""
    echo "=============================================="
    echo "Running post-training evaluation..."
    echo "=============================================="

    MODEL_PATH="./outputs/chess-sft-final"

    if [ -d "$MODEL_PATH" ]; then
        # Build evaluation command
        EVAL_CMD="python evaluate_fast.py --model $MODEL_PATH --num_positions 1000 --batch_size 32 --config configs/config_sft.yaml --source mixed --max_new_tokens 128"

        if [ -n "$STOCKFISH_PATH" ]; then
            EVAL_CMD="$EVAL_CMD --stockfish $STOCKFISH_PATH --workers 8 --stockfish_depth 12"
            echo "Running evaluation with ACPL metrics..."
        else
            echo "Running evaluation without Stockfish (no ACPL metrics)..."
        fi

        echo ""
        $EVAL_CMD

        echo ""
        echo "Results saved to: eval_results.json"
    else
        echo "Model not found at $MODEL_PATH"
        echo "Skipping evaluation."
    fi
fi

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="
echo ""
echo "Model saved to: ./outputs/chess-sft-final"
echo ""
echo "To run evaluation manually:"
echo "  ./scripts/evaluate.sh ./outputs/chess-sft-final"
