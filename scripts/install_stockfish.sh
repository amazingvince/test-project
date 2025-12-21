#!/bin/bash

# Install Latest Stockfish (17.1) to /usr/bin/stockfish
# Requires: curl, tar, and root privileges

# chmod +x scripts/install_stockfish.sh
# sudo ./scripts/install_stockfish.sh

set -e

STOCKFISH_VERSION="sf_17.1"
INSTALL_PATH="/usr/bin/stockfish"
TEMP_DIR=$(mktemp -d)
GITHUB_BASE="https://github.com/official-stockfish/Stockfish/releases/download/${STOCKFISH_VERSION}"

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Detect CPU capabilities for optimal binary selection
detect_cpu_features() {
    if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "arm64" ]]; then
        # ARM 64-bit
        if grep -q "asimddp" /proc/cpuinfo 2>/dev/null; then
            echo "armv8-dotprod"
        else
            echo "armv8"
        fi
    elif [[ "$(uname -m)" == "armv7"* ]] || [[ "$(uname -m)" == "armhf" ]]; then
        # ARM 32-bit
        if grep -q "neon" /proc/cpuinfo 2>/dev/null; then
            echo "armv7-neon"
        else
            echo "armv7"
        fi
    else
        # x86_64
        local flags=$(grep -m1 "^flags" /proc/cpuinfo 2>/dev/null || echo "")

        if echo "$flags" | grep -q "avx512" && echo "$flags" | grep -q "vnni"; then
            echo "x86-64-vnni512"
        elif echo "$flags" | grep -q "avx512"; then
            echo "x86-64-avx512"
        elif echo "$flags" | grep -q "bmi2"; then
            echo "x86-64-bmi2"
        elif echo "$flags" | grep -q "avx2"; then
            echo "x86-64-avx2"
        elif echo "$flags" | grep -q "sse4_1" && echo "$flags" | grep -q "popcnt"; then
            echo "x86-64-sse41-popcnt"
        else
            echo "x86-64"
        fi
    fi
}

# Determine OS type for download
get_os_type() {
    case "$(uname -s)" in
        Linux*)
            if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "armv7"* ]]; then
                echo "android"  # ARM binaries are under android
            else
                echo "ubuntu"
            fi
            ;;
        Darwin*)
            echo "macos"
            ;;
        *)
            echo "ubuntu"
            ;;
    esac
}

echo "=== Stockfish 17.1 Installer ==="
echo ""

# Check for root
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Detect system
CPU_FEATURES=$(detect_cpu_features)
OS_TYPE=$(get_os_type)

# For standard Linux x86_64, use ubuntu binaries
if [[ "$OS_TYPE" == "ubuntu" ]]; then
    DOWNLOAD_FILE="stockfish-ubuntu-${CPU_FEATURES}.tar"
elif [[ "$OS_TYPE" == "android" ]]; then
    DOWNLOAD_FILE="stockfish-android-${CPU_FEATURES}.tar"
elif [[ "$OS_TYPE" == "macos" ]]; then
    if [[ "$CPU_FEATURES" == "armv8"* ]]; then
        DOWNLOAD_FILE="stockfish-macos-m1-apple-silicon.tar"
    else
        DOWNLOAD_FILE="stockfish-macos-${CPU_FEATURES}.tar"
    fi
fi

DOWNLOAD_URL="${GITHUB_BASE}/${DOWNLOAD_FILE}"

echo "Detected CPU features: ${CPU_FEATURES}"
echo "OS type: ${OS_TYPE}"
echo "Download URL: ${DOWNLOAD_URL}"
echo ""

# Download
echo "Downloading Stockfish 17.1..."
cd "$TEMP_DIR"
if ! curl -L -o stockfish.tar "$DOWNLOAD_URL" 2>/dev/null; then
    echo "Error: Failed to download from $DOWNLOAD_URL"
    echo "Falling back to basic x86-64 binary..."
    DOWNLOAD_URL="${GITHUB_BASE}/stockfish-ubuntu-x86-64.tar"
    curl -L -o stockfish.tar "$DOWNLOAD_URL"
fi

# Extract
echo "Extracting..."
tar -xf stockfish.tar

# Find the stockfish binary (it's usually in a subdirectory)
STOCKFISH_BIN=$(find . -type f -name "stockfish*" ! -name "*.tar" ! -name "*.nnue" | head -1)

if [[ -z "$STOCKFISH_BIN" ]]; then
    echo "Error: Could not find stockfish binary in archive"
    exit 1
fi

# Install
echo "Installing to ${INSTALL_PATH}..."
cp "$STOCKFISH_BIN" "$INSTALL_PATH"
chmod 755 "$INSTALL_PATH"

# Verify installation
echo ""
echo "=== Installation Complete ==="
echo "Installed to: ${INSTALL_PATH}"
echo ""
echo "Version info:"
"$INSTALL_PATH" --version 2>/dev/null || "$INSTALL_PATH" uci <<< "quit" | head -1

echo ""
echo "Testing with benchmark..."
"$INSTALL_PATH" bench 16 1 13 2>&1 | tail -3
