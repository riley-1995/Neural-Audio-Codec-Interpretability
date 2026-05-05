# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Linear probing study comparing how phoneme identity, speaker identity, and pitch (F0) distribute across the 8 RVQ layers of two neural audio codecs: **EnCodec** (Meta) and **SpeechTokenizer**. Extends Sadok et al. 2025 by applying quantifiable linear probing and direct cross-codec comparison. The central question: does designing RVQ-1 for semantic alignment (SpeechTokenizer) produce stronger linear decodability of phonemes than emergent compression (EnCodec)?

## Setup

PyTorch must be installed separately before project dependencies:

```bash
uv venv --python 3.11 && source .venv/bin/activate

# macOS (Apple Silicon MPS)
uv pip install torch torchaudio
# Linux/CUDA 12.4
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

uv pip install -e .
```

**Required external data** (not included in repo):
1. LibriSpeech `train-clean-100` → `data/LibriSpeech/train-clean-100/`
2. MFA TextGrid alignments → `data/alignments/train-clean-100/`
3. SpeechTokenizer checkpoint (`speechtokenizer.pt` + `config.json`) from ZhangXInFD/SpeechTokenizer

EnCodec downloads automatically (~100 MB) on first use.

## Running Experiments

Quick sanity check (50 utterances, ~5–10 min):
```bash
python main.py \
    --librispeech_root data/LibriSpeech \
    --alignments_root  data/alignments/train-clean-100 \
    --st_ckpt          /path/to/speechtokenizer.pt \
    --st_config        /path/to/config.json \
    --output_dir       results \
    --max_utterances   50
```

Full run (all ~28k utterances, `--max_utterances 0`). Key flags:
- `--device auto|cpu|cuda|mps` (default: auto, prefers CUDA > MPS > CPU)
- `--force_resplit` — recompute train/eval split (normally loads saved `split.json`)
- `--eval_frac 0.1` — held-out fraction (default 10%)

To re-probe without re-encoding (fast, ~1–2 min):
```bash
# Embeddings are cached in results/cache/; just rerun main.py
python main.py ...
```

To force re-encoding:
```bash
rm -rf results/cache/ && python main.py ...
```

## Running Tests

```bash
python tests/test_load_alignments.py
```

Tests cover correctness of `phoneme_labels_for_tokens()` (edge cases, boundary conditions, both token rates) and benchmark the O(n+m) sweep against a reference O(n²) implementation.

## Architecture

```
main.py                         # Orchestrates full pipeline: split → encode → probe → eval → plot
data/
  load_librispeech.py           # LibriSpeech FLAC iterator; yields Utterance dataclasses
  load_alignments.py            # Parses MFA TextGrid files; O(n+m) sweep maps tokens → phoneme labels
  extract_pitch.py              # Per-token F0 via librosa.yin; unvoiced frames → NaN (excluded from regression)
  split.py                      # Utterance-level stratified 90/10 split saved to split.json
encode/
  encode_encodec.py             # EnCodec inference at 24 kHz → 8 × (N_tokens, 128) embeddings
  encode_speechtokenizer.py     # SpeechTokenizer inference at 16 kHz → 8 × (N_tokens, D) embeddings
  collect.py                    # collect_bundle(): load/cache embeddings, align labels, extract pitch
probe/
  train_probes.py               # Trains 24 probes per codec (8 layers × 3 tasks); saves .pkl files
  evaluate_probes.py            # Evaluates saved probes; returns accuracy/macro-F1/MAE/R² per layer
visualize/
  plot_curves.py                # 3 figures comparing EnCodec vs SpeechTokenizer across RVQ layers
tests/
  test_load_alignments.py       # Unit tests + benchmark for phoneme alignment sweep
```

**Pipeline flow:** scan LibriSpeech → stratified split → encode both codecs (cached to NPZ) → fit label encoders → train 24 probes per codec (phoneme/speaker: `LogisticRegression(class_weight="balanced")`; pitch: `LinearRegression` on voiced frames only) → evaluate on held-out set → plot 3 figures.

## Key Design Decisions

- **Utterance-level split** (not token-level) avoids label leakage
- **Frozen probes only** — codec parameters never updated; purely diagnostic
- **Macro-F1 as primary metric** — equal weight across phoneme/speaker classes regardless of frequency
- **Pitch as regression** (continuous F0 Hz), not classification; R² is the headline metric
- **Shared label encoders** — fitted once on training data, reused for both codecs to ensure comparability
- **Seed = 42** throughout; `split.json` ensures identical train/eval partition across re-runs

## Outputs (`results/`)

| File | Contents |
|------|----------|
| `split.json` | Train/eval utterance IDs for reproducibility |
| `cache/{encodec,speechtokenizer}/*.npz` | Per-utterance embeddings (8 layers) |
| `probes/probe_{codec}_layer{N}_{task}.pkl` | 48 trained probe files |
| `figures/{phoneme,speaker,pitch}_probing.png` | Layer-wise metric curves, both codecs |
| `results.pkl` | Raw metrics dict with 6 lists of 8 floats per codec |
