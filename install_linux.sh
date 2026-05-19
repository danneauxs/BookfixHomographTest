#!/bin/bash
set -e

echo "=========================================="
echo "HomographFix Linux Installer"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ $MAJOR -lt 3 ] || ([ $MAJOR -eq 3 ] && [ $MINOR -lt 10 ]); then
    echo "WARNING: Python 3.10+ recommended, found $PYTHON_VERSION"
fi

# Create venv
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Detect CUDA
echo ""
echo "Detecting CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu124
else
    echo "CUDA not detected. Installing PyTorch CPU version (slower)..."
    pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other packages
echo ""
echo "Installing transformers, spacy, and spacy-transformers..."
pip install transformers==4.53.2 spacy==3.8.7 spacy-transformers==1.4.0

# Download spaCy model
echo ""
echo "Downloading spaCy transformer model (en_core_web_trf)..."
python -m spacy download en_core_web_trf

# Create folders
echo ""
echo "Creating InputText and OutputText folders..."
mkdir -p InputText OutputText

# Done
echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Run the program: python word_hybrid_test.py"
echo ""
echo "NOTE: First run will download roberta-large-mnli (~1.4 GB)"
echo "      from HuggingFace — this is normal and one-time only."
echo ""
