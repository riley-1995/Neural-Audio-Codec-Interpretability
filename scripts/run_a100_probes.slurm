#!/usr/bin/env bash
#SBATCH --job-name=codec-probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Example Slurm wrapper. Configure required environment variables before submit:
#   export LIBRISPEECH_ROOT=/path/to/LibriSpeech
#   export ALIGNMENTS_ROOT=/path/to/alignments/train-clean-100
#   export ST_CKPT=/path/to/speechtokenizer.pt
#   export ST_CONFIG=/path/to/config.json
#   export OUTPUT_DIR=/path/to/results_a100
# Optional:
#   export PROBE_WORKERS=0
#   export PROBE_BLAS_THREADS=1
#   export PROBE_MAX_ITER=1000
#   export MAX_UTTERANCES=0
# Submit with:
#   sbatch scripts/run_a100_probes.slurm

mkdir -p logs

if [[ -z "${PROBE_WORKERS:-}" ]]; then
  # Conservative default for 32 CPU cores assigned to this job.
  export PROBE_WORKERS=16
fi

export PROBE_EXEC_PROFILE="${PROBE_EXEC_PROFILE:-a100}"

bash scripts/run_a100_probes.sh
