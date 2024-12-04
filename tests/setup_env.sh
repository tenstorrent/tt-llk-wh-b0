#!/bin/bash

sudo apt update
sudo apt install gawk

# **************** DOWNLOAD & INSTALL SFPI ****************************
#git submodule add https://github.com/tenstorrent/sfpi sfpi
#git submodule sync
#git submodule update --init --recursive
wget https://github.com/tenstorrent/sfpi/releases/download/v6.0.0/sfpi-release.tgz
tar -xzvf sfpi-release.tgz 
rm -rf sfpi-release.tgz 
# **************** DOWNLOAD & INSTALL TT-LENS ****************************
pip install git+https://github.com/tenstorrent/tt-debuda.git@d4ce04c3d4e68cccdf0f53b0b5748680a8a573ed
# **************** DOWNLOAD & INSTALL TT_SMI ****************************
git clone https://github.com/tenstorrent/tt-smi
cd tt-smi
python3 -m venv .venv
source .venv/bin/activate
pip3 install .
pip3 install --upgrade pip
# Find on which architecture code will be tested
arch_options=("Wormhole" "Blackhole" "Grayskull")
file="arch.dump"
result=$(python3 helpers/find_arch.py "${arch_options[@]}" "$file")
export CHIP_ARCH=$result
echo "Chip architecture is $CHIP_ARCH"
# **************** SETUP PYTHON VENV **********************************

# Try to install python3.10-venv first, fallback to python3.8-venv if it fails
sudo apt install -y python3.10-venv || {
    echo "Failed to install python3.10-venv, trying python3.8-venv...";
    sudo apt install -y python3.8-venv || { echo "Failed to install python3.8-venv."; exit 1; }
}

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
pip install pytest-cov
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "Failed to install PyTorch packages."; exit 1; }

