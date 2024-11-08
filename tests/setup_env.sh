#!/bin/bash

# **************** DOWNLOAD & INSTALL SFPI ****************************
git submodule update --init --recursive
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
pip install numpy
pip install -U pytest
pip install pytest-cov
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "Failed to install PyTorch packages."; exit 1; }

# reset the board
/home/software/syseng/wh/tt-smi -wr 0 
