#!/bin/bash

# Update packages and install gawk (if necessary)
echo "Updating system packages..."
sudo apt update
sudo apt install -y gawk

# **************** DOWNLOAD & INSTALL TT-LENS ****************************
wget https://github.com/tenstorrent/tt-lens/releases/download/latest/ttlens-0.1.241202+dev.d4ce04c-cp38-cp38-linux_x86_64.whl
pip install ttlens-0.1.241202+dev.d4ce04c-cp38-cp38-linux_x86_64.whl

# **************** DOWNLOAD & INSTALL SFPI ****************************
echo "Downloading SFPI release..."
wget https://github.com/tenstorrent/sfpi/releases/download/v6.0.0/sfpi-release.tgz -O sfpi-release.tgz
if [ ! -f "sfpi-release.tgz" ]; then
    echo "SFPI release not found!"
    exit 1
fi
echo "Extracting SFPI release..."
tar -xzvf sfpi-release.tgz
rm -f sfpi-release.tgz

# **************** DOWNLOAD & INSTALL TT-SMI ****************************
echo "Cloning tt-smi repository..."
git clone https://github.com/tenstorrent/tt-smi
cd tt-smi

echo "Creating Python virtual environment..."
python3 -m venv .venv
if [ ! -d ".venv" ]; then
    echo "Failed to create virtual environment"
    exit 1
fi
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install .
pip install pytest pytest-cov

# Detect architecture for chip
echo "Running tt-smi -ls to detect architecture..."
tt-smi -ls > ../arch.dump
echo "tt-smi -ls completed. Running find_arch.py..."
cd ..
result=$(python3 helpers/find_arch.py "Wormhole" "Blackhole" "Grayskull" arch.dump)
echo "Detected architecture: $result"

if [ -z "$result" ]; then
    echo "Error: Architecture detection failed!"
    exit 1
fi

echo "Setting CHIP_ARCH variable..."
export CHIP_ARCH="$result"
echo "CHIP_ARCH is: $CHIP_ARCH"

# Install torch and related packages
echo "Installing PyTorch and related packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# **************** SETUP PYTHON VENV **********************************
# Ensure python3.10-venv is installed, fallback to python3.8-venv
echo "Checking python3.10-venv..."
if ! dpkg -l | grep -q python3.10-venv; then
    echo "python3.10-venv not found, attempting to install python3.8-venv..."
    sudo apt install -y python3.8-venv || { echo "Failed to install python3.8-venv."; exit 1; }
else
    sudo apt install -y python3.10-venv
fi

# Set up Python virtual environment if not already set
echo "Ensuring virtual environment is set up..."
python3 -m ensurepip
pip install --upgrade pip

# Install needed packages
pip install -U pytest pytest-cov
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Script completed successfully!"
