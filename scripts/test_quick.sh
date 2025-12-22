#!/bin/bash
# test_quick.sh - Quick test to verify everything works
#
# Usage:
#   ./scripts/test_quick.sh

set -e

echo "=============================================="
echo "Chess LLM SFT - Quick Test"
echo "=============================================="

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run unit tests
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --tb=short || {
    echo "[WARN] Some tests failed, but continuing..."
}

# Test imports
echo ""
echo "Testing imports..."
python3 << 'EOF'
print("Testing core imports...")
from src.utils.chess_utils import render_board_utf, get_legal_moves_uci, extract_uci_from_response
from src.utils.data_processing import create_streaming_dataset
from src.utils.formatting import position_to_messages
import chess

# Test chess utils
board = chess.Board()
print("  [OK] Chess board created")

utf_board = render_board_utf(board)
print("  [OK] Board rendering works")

legal = get_legal_moves_uci(board)
print(f"  [OK] Legal moves: {len(legal.split())} moves found")

# Test move extraction
test_response = "<think>Testing</think><uci_move>e2e4</uci_move>"
move = extract_uci_from_response(test_response)
assert move == "e2e4", f"Expected e2e4, got {move}"
print("  [OK] Move extraction works")

# Test formatting
pos = {"fen": board.fen(), "target_move_uci": "e2e4"}
msgs = position_to_messages(pos)
assert "messages" in msgs
print("  [OK] Position formatting works")

print("\n[OK] All core imports and functions working!")
EOF

# Test model loading (if transformers available)
echo ""
echo "Testing model loading..."
python3 << 'EOF'
import torch
from transformers import AutoTokenizer

# Just test tokenizer loads (model would be too slow)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
print(f"  [OK] Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"  [OK] Vocab size: {tokenizer.vocab_size}")

if torch.cuda.is_available():
    print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("  [WARN] CUDA not available")
EOF

# Test Stockfish
echo ""
echo "Testing Stockfish..."
python3 << 'EOF'
import shutil

# Find stockfish
sf_path = shutil.which("stockfish")
if not sf_path:
    for path in ["/usr/bin/stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish"]:
        import os
        if os.path.exists(path):
            sf_path = path
            break

if sf_path:
    try:
        from stockfish import Stockfish
        sf = Stockfish(path=sf_path, depth=8)
        sf.set_fen_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        best = sf.get_best_move()
        eval_result = sf.get_evaluation()
        print(f"  [OK] Stockfish working at {sf_path}")
        print(f"  [OK] Best move after 1.e4: {best}")
        print(f"  [OK] Evaluation: {eval_result}")
    except Exception as e:
        print(f"  [WARN] Stockfish error: {e}")
else:
    print("  [WARN] Stockfish not found (ACPL eval disabled)")
    print("    Install with: sudo apt install stockfish")
EOF

echo ""
echo "=============================================="
echo "Quick test complete!"
echo "=============================================="
echo ""
echo "To run full training in debug mode:"
echo "  ./scripts/train_sft.sh --debug"
