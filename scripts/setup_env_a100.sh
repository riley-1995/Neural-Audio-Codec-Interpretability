#!/usr/bin/env bash
set -euo pipefail

# A100 setup helper: creates .venv and installs CUDA 12.4 PyTorch + project deps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# STORAGE is intentionally hardcoded for the project A100 environment.
# For a different cluster/location, edit this value directly.
STORAGE="/research/shared-rdenn-anaralikatti"
mkdir -p "${STORAGE}" "${STORAGE}/.uv-cache" "${STORAGE}/.torch-cache" "${STORAGE}/.hf-cache"

# Redirect caches out of $HOME to shared/scratch storage.
export UV_CACHE_DIR="${STORAGE}/.uv-cache"
export TORCH_HOME="${STORAGE}/.torch-cache"
export HF_HOME="${STORAGE}/.hf-cache"

cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation finished, but uv is still not on PATH."
    echo "Add ${HOME}/.local/bin to PATH and re-run this script."
    exit 1
fi

uv venv --python 3.12
source "${PROJECT_ROOT}/.venv/bin/activate"

# Install PyTorch from the CUDA 12.4 index first.
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies from pyproject.toml.
uv pip install -e .

echo ""
echo "Setup complete."
echo "Project root: ${PROJECT_ROOT}"
echo "To use this environment:"
echo "  source ${PROJECT_ROOT}/.venv/bin/activate"
echo "Exported cache vars for this shell:"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  HF_HOME=${HF_HOME}"