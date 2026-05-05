#!/bin/bash
set -e

RESEARCH=/research/shared-rdenn-anaralikatti
PROJECT=/home/anaralikatti/research/682/Neural-Audio-Codec-Interpretability

# Redirect all caches out of home directory
export PIP_CACHE_DIR=$RESEARCH/.pip-cache
export TORCH_HOME=$RESEARCH/.torch-cache
export HF_HOME=$RESEARCH/.hf-cache

cd "$PROJECT"

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# PyTorch (check your CUDA version with: nvidia-smi)
pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Project dependencies
pip install --no-cache-dir \
    encodec \
    speechtokenizer \
    "librosa>=0.10.0" \
    "scikit-learn>=1.5.0" \
    "matplotlib>=3.7.0" \
    "numpy>=1.24.0" \
    tqdm \
    soundfile \
    scipy \
    beartype \
    tensorboard \
    huggingface_hub

echo ""
echo "Setup complete. To use this environment:"
echo "  source $PROJECT/.venv/bin/activate"
echo "  export TORCH_HOME=$RESEARCH/.torch-cache"
echo "  export HF_HOME=$RESEARCH/.hf-cache"
