#!/bin/bash

# **************** DOWNLOAD & INSTALL SFPI ****************************
SFPI_PATH="sfpi-rel" 
FIRMWARE_PATH="firmware/riscv"

# Check if the folder does not exist
if [ ! -d "$SFPI_PATH" ]; then
    git submodule add https://github.com/tenstorrent/sfpi-rel/
else
    echo "SFPI already installed."
    git submodule init
    git submodule update --remote
fi

# TODO: GET FIRMWARE

# **************** DOWNLOAD & INSTALL DEBUDA ****************************
# TODO: NEWEST DEBUDA WHEEL
FILE_URL="https://github.com/tenstorrent/tt-debuda/releases/debuda-0.1.241015+dev.1fa63bf-cp310-cp310-linux_x86_64.whl"
OUTPUT_DIR="./downloads"
OUTPUT_FILE="${OUTPUT_DIR}/debuda-0.1.241015+dev.1fa63bf-cp310-cp310-linux_x86_64.whl"

mkdir -p "$OUTPUT_DIR"
curl -o "$OUTPUT_FILE" "$FILE_URL" || { echo "Failed to download Debuda."; exit 1; }
echo "File downloaded successfully to $OUTPUT_FILE."

pip install downloads/debuda-0.1.241016+dev.62b602b-cp310-cp310-linux_x86_64.whl || { echo "Failed to install Debuda."; exit 1; }

# **************** SETUP PYTHON VENV **********************************

sudo apt install -y python3.10-venv || { echo "Failed to install python3.10-venv."; exit 1; }
set -eo pipefail

if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR="$(pwd)/.venv"
fi

echo "Creating virtual env in: $PYTHON_ENV_DIR"
python3 -m venv "$PYTHON_ENV_DIR"

source "$PYTHON_ENV_DIR/bin/activate"

echo "Ensuring pip is installed and up-to-date"
python3 -m ensurepip
pip install --upgrade pip

# needed packages
pip install -U pytest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "Failed to install PyTorch packages."; exit 1; }
