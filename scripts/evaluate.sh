#!/bin/bash
# evaluate.sh - Run fast evaluation with ACPL metrics
#
# Usage:
#   ./scripts/evaluate.sh MODEL_PATH                    # Full evaluation (1000 positions)
#   ./scripts/evaluate.sh MODEL_PATH --quick            # Quick evaluation (100 positions)
#   ./scripts/evaluate.sh MODEL_PATH --num 500          # Custom number of positions
#   ./scripts/evaluate.sh MODEL_PATH --no-stockfish     # Skip ACPL (faster)
#   ./scripts/evaluate.sh MODEL_PATH --source puzzles   # Evaluate only puzzles
#   ./scripts/evaluate.sh MODEL_PATH --config configs/config_sft.yaml

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./scripts/evaluate.sh MODEL_PATH [options]"
    echo ""
    echo "Options:"
    echo "  --quick           Quick eval (100 positions, depth 8)"
    echo "  --num N           Evaluate N positions (default: 1000)"
    echo "  --no-stockfish    Skip Stockfish ACPL evaluation"
    echo "  --workers N       Number of parallel Stockfish workers (default: 8)"
    echo "  --depth N         Stockfish search depth (default: 12)"
    echo "  --output FILE     Output file (default: eval_results.json)"
    echo "  --source NAME     Data source: mixed, games, puzzles (default: mixed)"
    echo "  --games-ratio R   Games ratio when source=mixed (overrides config)"
    echo "  --max-new-tokens N  Max tokens to generate per position (default: 128)"
    echo "  --config FILE     Optional config file for data settings"
    echo ""
    echo "Examples:"
    echo "  ./scripts/evaluate.sh ./outputs/chess-sft-final"
    echo "  ./scripts/evaluate.sh ./outputs/chess-sft-final --quick"
    echo "  ./scripts/evaluate.sh ./outputs/chess-sft-final --num 500 --workers 16"
    exit 1
fi

MODEL_PATH="$1"
shift

# Default values
NUM_POSITIONS=1000
WORKERS=8
DEPTH=12
OUTPUT="eval_results.json"
USE_STOCKFISH=1
BATCH_SIZE=32
SOURCE="mixed"
GAMES_RATIO=""
MAX_NEW_TOKENS=128
CONFIG_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            NUM_POSITIONS=100
            DEPTH=8
            shift
            ;;
        --num)
            NUM_POSITIONS="$2"
            shift 2
            ;;
        --no-stockfish)
            USE_STOCKFISH=0
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --games-ratio)
            GAMES_RATIO="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "=============================================="
echo "Chess LLM Evaluation"
echo "=============================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Positions: $NUM_POSITIONS"
echo "Batch size: $BATCH_SIZE"
echo "Source: $SOURCE"
echo "Max new tokens: $MAX_NEW_TOKENS"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config: $CONFIG_FILE"
fi

# Find Stockfish
STOCKFISH_PATH=""
STOCKFISH_ARGS=""

if [ "$USE_STOCKFISH" = "1" ]; then
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
        echo "Stockfish depth: $DEPTH"
        echo "Stockfish workers: $WORKERS"
        STOCKFISH_ARGS="--stockfish $STOCKFISH_PATH --stockfish_depth $DEPTH --workers $WORKERS"
    else
        echo "Stockfish: not found (ACPL metrics disabled)"
        echo ""
        echo "To install Stockfish:"
        echo "  Ubuntu/Debian: sudo apt install stockfish"
        echo "  macOS: brew install stockfish"
    fi
else
    echo "Stockfish: disabled"
fi

echo "Output: $OUTPUT"
echo ""

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Run evaluation
echo "Starting evaluation..."
echo ""

# Build optional args
CONFIG_ARGS=""
if [ -n "$CONFIG_FILE" ]; then
    CONFIG_ARGS="--config $CONFIG_FILE"
fi

GAMES_RATIO_ARGS=""
if [ -n "$GAMES_RATIO" ]; then
    GAMES_RATIO_ARGS="--games_ratio $GAMES_RATIO"
fi

python evaluate_fast.py \
    --model "$MODEL_PATH" \
    --num_positions $NUM_POSITIONS \
    --batch_size $BATCH_SIZE \
    --output "$OUTPUT" \
    --source "$SOURCE" \
    --max_new_tokens $MAX_NEW_TOKENS \
    $CONFIG_ARGS \
    $GAMES_RATIO_ARGS \
    $STOCKFISH_ARGS

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT"
echo ""
echo "To view results:"
echo "  cat $OUTPUT | python -m json.tool | head -50"
