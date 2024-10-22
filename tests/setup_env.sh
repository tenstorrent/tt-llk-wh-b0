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
pip install git+https://github.com/tenstorrent/tt-debuda.git
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
