#!/bin/bash
# setup.sh - Setup script for Chess LLM SFT
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh

set -e

echo "=============================================="
echo "Chess LLM SFT - Setup"
echo "=============================================="

# Check CUDA version
echo ""
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA $CUDA_VERSION found"

    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

    if [[ "$CUDA_MAJOR" -lt 12 ]] || [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -lt 8 ]]; then
        echo "Note: CUDA 12.8+ recommended for latest GPUs. Found CUDA $CUDA_VERSION"
    fi
else
    echo "nvcc not found. CUDA may not be installed."
fi

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support (optimal for modern GPUs)
echo ""
echo "Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA support..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Install Flash Attention 2
echo ""
echo "Installing Flash Attention 2..."
echo "Note: This may take a few minutes to compile..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation || {
    echo "Flash Attention installation failed."
    echo "Training will still work with eager attention."
}

# Install Cut Cross Entropy
echo ""
echo "Installing Cut Cross Entropy..."
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git" || {
    echo "Cut Cross Entropy installation failed (optional)."
}

# Install Stockfish for ACPL evaluation
echo ""
echo "=============================================="
echo "Installing Stockfish for ACPL evaluation..."
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/install_stockfish.sh" ]; then
    echo "Using optimized Stockfish installer..."
    chmod +x "$SCRIPT_DIR/install_stockfish.sh"
    sudo "$SCRIPT_DIR/install_stockfish.sh" || {
        echo "Stockfish installer failed. Trying apt/brew fallback..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y stockfish || echo "Could not install Stockfish via apt"
        elif command -v brew &> /dev/null; then
            brew install stockfish || echo "Could not install Stockfish via brew"
        else
            echo "Please install Stockfish manually from: https://stockfishchess.org/download/"
        fi
    }
else
    echo "Stockfish installer not found. Using package manager..."
    if command -v stockfish &> /dev/null; then
        echo "Stockfish already installed at $(which stockfish)"
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y stockfish || echo "Could not install Stockfish via apt"
    elif command -v brew &> /dev/null; then
        brew install stockfish || echo "Could not install Stockfish via brew"
    else
        echo "Please install Stockfish manually from: https://stockfishchess.org/download/"
    fi
fi

# Install Python stockfish package
pip install stockfish

# Verify Stockfish
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

if [ -n "$STOCKFISH_PATH" ] && [ -f "$STOCKFISH_PATH" ]; then
    echo ""
    echo "Testing Stockfish..."
    python3 << EOF
from stockfish import Stockfish
try:
    sf = Stockfish(path="$STOCKFISH_PATH", depth=10)
    sf.set_fen_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    best = sf.get_best_move()
    print(f"Stockfish working! Best move from starting position after e4: {best}")
except Exception as e:
    print(f"Stockfish test failed: {e}")
EOF
fi

# Verify installations
echo ""
echo "=============================================="
echo "Verifying installations..."
echo "=============================================="

python3 << 'EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        __import__(name)
        return True
    except ImportError:
        return False

checks = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("datasets", "HuggingFace Datasets"),
    ("accelerate", "Accelerate"),
    ("trl", "TRL"),
    ("liger_kernel", "Liger Kernel"),
    ("cut_cross_entropy", "Cut Cross Entropy"),
    ("flash_attn", "Flash Attention 2"),
    ("triton", "Triton"),
    ("stockfish", "Stockfish (Python)"),
    ("chess", "python-chess"),
]

print("")
for module, name in checks:
    status = "[x]" if check_import(module) else "[ ]"
    print(f"  {status} {name}")

# Check GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name} ({memory:.0f}GB)")

    cc = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
else:
    print("\nNo CUDA GPU available - training will be slow!")
EOF

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To start training:"
echo "  ./scripts/train.sh"
echo ""
echo "For debug mode (small dataset):"
echo "  ./scripts/train.sh --debug"
echo ""
echo "To evaluate a trained model:"
echo "  ./scripts/evaluate.sh ./outputs/chess-sft-final"
echo ""
echo "For help:"
echo "  python train.py --help"
echo "  python evaluate_fast.py --help"
