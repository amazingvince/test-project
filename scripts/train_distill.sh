#!/bin/bash
# train_distill.sh - Launch policy distillation training
#
# Usage:
#   ./scripts/train_distill.sh --streaming --max-steps 50000    # Streaming mode (recommended)
#   ./scripts/train_distill.sh --data ./data/chess_distill      # Preprocessed data
#   ./scripts/train_distill.sh --debug --max-steps 100          # Debug mode
#   ./scripts/train_distill.sh --resume checkpoint              # Resume from checkpoint

set -e

# Parse arguments
DEBUG=""
RESUME=""
DATA_PATH=""
STREAMING=""
MAX_STEPS=""
NO_LIGER=""
NO_CCE=""
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
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --streaming)
            STREAMING="--streaming"
            shift
            ;;
        --max-steps)
            MAX_STEPS="--max-steps $2"
            shift 2
            ;;
        --no-liger)
            NO_LIGER="--no-liger"
            shift
            ;;
        --no-cce)
            NO_CCE="--no-cce"
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

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

if [ -n "$DEBUG" ]; then
    export WANDB_MODE=disabled
fi

echo "=============================================="
echo "Chess LLM - Policy Distillation Training"
echo "=============================================="
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Find Stockfish
STOCKFISH_PATH=""
for path in "$(which stockfish 2>/dev/null)" "/usr/bin/stockfish" "/usr/games/stockfish" "/usr/local/bin/stockfish" "/opt/homebrew/bin/stockfish"; do
    if [ -f "$path" ]; then
        STOCKFISH_PATH="$path"
        break
    fi
done

# Validate mode and requirements
if [ -n "$STREAMING" ]; then
    echo "Mode: STREAMING (on-the-fly Stockfish analysis)"

    if [ -z "$STOCKFISH_PATH" ]; then
        echo "ERROR: Stockfish required for streaming mode"
        echo "  Install: sudo apt install stockfish"
        exit 1
    fi

    if [ -z "$MAX_STEPS" ]; then
        echo "ERROR: --max-steps required for streaming mode"
        echo "  Example: ./scripts/train_distill.sh --streaming --max-steps 50000"
        exit 1
    fi
else
    echo "Mode: PREPROCESSED"
    [ -z "$DATA_PATH" ] && DATA_PATH="./data/chess_distill"

    if [ ! -d "$DATA_PATH" ]; then
        echo "ERROR: Data not found at $DATA_PATH"
        echo "  Option 1: python distill/preprocess.py --output $DATA_PATH --size 100000"
        echo "  Option 2: ./scripts/train_distill.sh --streaming --max-steps 50000"
        exit 1
    fi
    echo "Data: $DATA_PATH"
fi

[ -n "$STOCKFISH_PATH" ] && echo "Stockfish: $STOCKFISH_PATH"
[ -n "$MAX_STEPS" ] && echo "Max steps: ${MAX_STEPS#--max-steps }"
[ -n "$DEBUG" ] && echo "Debug: enabled"
echo ""

# Build and run command
if [ -n "$STREAMING" ]; then
    CMD="python distill/train.py --config configs/distill/config_distill.yaml --streaming $MAX_STEPS $DEBUG $RESUME $NO_LIGER $NO_CCE $EXTRA_ARGS"
else
    CMD="python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path $DATA_PATH $MAX_STEPS $DEBUG $RESUME $NO_LIGER $NO_CCE $EXTRA_ARGS"
fi

echo "Command: $CMD"
echo ""
$CMD

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
