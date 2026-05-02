#!/usr/bin/env bash
set -euo pipefail

# Example single-node A100 launcher for probe parallelism.
# All probe execution behavior is controlled by main.py CLI flags.

: "${LIBRISPEECH_ROOT:?Set LIBRISPEECH_ROOT (e.g. data/LibriSpeech)}"
: "${ALIGNMENTS_ROOT:?Set ALIGNMENTS_ROOT (e.g. data/alignments/train-clean-100)}"
: "${ST_CKPT:?Set ST_CKPT path}"
: "${ST_CONFIG:?Set ST_CONFIG path}"

OUTPUT_DIR="${OUTPUT_DIR:-results_a100}"
SPLIT="${SPLIT:-train-clean-100}"
EVAL_FRAC="${EVAL_FRAC:-0.1}"
MAX_UTTERANCES="${MAX_UTTERANCES:-0}"
DEVICE="${DEVICE:-cuda}"
PROBE_EXEC_PROFILE="${PROBE_EXEC_PROFILE:-a100}"
PROBE_WORKERS="${PROBE_WORKERS:-0}"
PROBE_BLAS_THREADS="${PROBE_BLAS_THREADS:-1}"
PROBE_MAX_ITER="${PROBE_MAX_ITER:-1000}"

# Keep per-worker BLAS thread counts low to avoid severe oversubscription
# when running many probe jobs in parallel.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Expected repository-local virtualenv at .venv/bin/python" >&2
  echo "Create it with: uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ." >&2
  exit 1
fi

.venv/bin/python main.py \
  --librispeech_root "${LIBRISPEECH_ROOT}" \
  --alignments_root "${ALIGNMENTS_ROOT}" \
  --st_ckpt "${ST_CKPT}" \
  --st_config "${ST_CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --eval_frac "${EVAL_FRAC}" \
  --max_utterances "${MAX_UTTERANCES}" \
  --device "${DEVICE}" \
  --probe_exec_profile "${PROBE_EXEC_PROFILE}" \
  --probe_workers "${PROBE_WORKERS}" \
  --probe_blas_threads "${PROBE_BLAS_THREADS}" \
  --probe_max_iter "${PROBE_MAX_ITER}" \
  "$@"
