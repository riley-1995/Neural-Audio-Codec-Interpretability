# Neural Audio Codec Interpretability

A linear probing study of residual vector quantization (RVQ) layers in two neural audio codecs — **EnCodec** and **SpeechTokenizer** — to quantify how phoneme identity, speaker identity, and pitch (F0) distribute across codec depth.

This project extends [Sadok et al. 2025](https://arxiv.org/abs/2506.04492) (*Bringing Interpretability to Neural Audio Codecs*, Interspeech 2025), which used mutual information and t-SNE. We apply **linear probing** as a complementary and more directly quantifiable methodology, and introduce a **comparative codec analysis** — identical probing pipelines applied to both codecs side-by-side.

---

## Research Question

> How are phoneme identity, speaker identity, and pitch distributed across the 8 RVQ layers of EnCodec vs SpeechTokenizer? Does explicitly designing RVQ-1 for semantic alignment (SpeechTokenizer) produce stronger linear decodability than emergent compression structure (EnCodec)?

**Hypothesis:** Phoneme information will be more strongly linearly decodable from earlier RVQ layers in SpeechTokenizer (by design — RVQ-1 is aligned to HuBERT semantic tokens), while speaker identity will concentrate in middle-to-deeper layers for both codecs, and pitch (F0) will be weakly decodable across all layers.

---

## Codecs Compared

| | EnCodec | SpeechTokenizer |
|---|---|---|
| **Paper** | Défossez et al. 2022 | Zhang et al. 2023 |
| **Sample rate** | 24 kHz | 16 kHz |
| **RVQ layers** | 8 | 8 |
| **Token rate** | 75 tokens/sec | 50 tokens/sec |
| **Codebook size** | 1024 | 1024 |
| **Embedding dim** | 128 | varies |
| **Design intent** | General audio compression | RVQ-1 aligned to HuBERT semantic tokens |

---

## Probing Tasks

| Task | Probe type | Labels | Primary metric |
|---|---|---|---|
| Phoneme identity | Logistic regression | MFA forced alignment TextGrids | Macro-F1 |
| Speaker identity | Logistic regression | LibriSpeech speaker metadata | Macro-F1 |
| Pitch (F0) | Linear regression | YIN algorithm (librosa) | R² |

**48 total probes:** 2 codecs × 8 layers × 3 tasks. Codec parameters are never updated — all probes train on frozen embeddings.

---

## Project Structure

```
.
├── main.py                        # Entry point — orchestrates the full pipeline
│
├── data/
│   ├── LibriSpeech/               # Audio data (gitignored — download separately)
│   │   ├── train-clean-100/       # 251 speakers, 28 539 utterances (~100h)
│   │   ├── train-clean-360/       # 921 speakers (~360h) — optional ablation
│   │   ├── dev-clean/             # Reserved
│   │   └── test-clean/            # Reserved
│   ├── alignments/                # TextGrid files (gitignored — download separately)
│   │   └── train-clean-100/       # speaker/chapter/utterance.TextGrid layout
│   ├── split.py                   # Utterance-level stratified train/eval split
│   ├── load_librispeech.py        # FLAC loader → Utterance dataclass
│   ├── load_alignments.py         # TextGrid parser → phoneme intervals
│   └── extract_pitch.py           # YIN F0 extraction per token
│
├── encode/
│   ├── encode_encodec.py          # EnCodec inference → 8 layer embeddings
│   ├── encode_speechtokenizer.py  # SpeechTokenizer inference → 8 layer embeddings
│   └── collect.py                 # collect_bundle(): encode + align + cache
│
├── probe/
│   ├── train_probes.py            # fit_label_encoders() + train_probes()
│   └── evaluate_probes.py         # evaluate_probes() → layer-wise metrics
│
├── visualize/
│   └── plot_curves.py             # plot_all() → 3 comparison figures
│
├── results/                       # Generated outputs (gitignored)
│   ├── split.json                 # Saved train/eval utterance IDs
│   ├── cache/                     # Per-utterance NPZ embedding cache
│   ├── probes/                    # Saved probe .pkl files + label encoders
│   ├── figures/                   # phoneme_probing.png, speaker_probing.png, pitch_probing.png
│   └── results.pkl                # Raw metrics dict
│
├── proposal/                      # Project proposal (LaTeX + PDF)
├── pyproject.toml                 # Dependencies (uv-compatible)
└── .gitignore
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Neural-Audio-Codec-Interpretability
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create a virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 4. Install PyTorch (platform-specific)

PyTorch must be installed before the rest of the dependencies because the wheel differs by platform.

**macOS (Apple Silicon — MPS):**
```bash
uv pip install torch torchaudio
```

**Linux / NVIDIA A100 (CUDA 12.4):**
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 5. Install project dependencies

```bash
uv pip install -e .
```

---

## Data Setup

### LibriSpeech audio

Download `train-clean-100` from [OpenSLR](https://www.openslr.org/12/). Extract so the directory structure is:

```
data/LibriSpeech/train-clean-100/<speaker_id>/<chapter_id>/<utterance_id>.flac
```

`dev-clean`, `test-clean`, and `train-clean-360` can optionally be placed under the same `data/LibriSpeech/` root.

### Forced alignment TextGrids

Download pre-aligned TextGrid files for `train-clean-100` from the [CorentinJ/librispeech-alignments](https://github.com/CorentinJ/librispeech-alignments) GitHub Releases page. Extract so the structure is:

```
data/alignments/train-clean-100/<speaker_id>/<chapter_id>/<utterance_id>.TextGrid
```

These files map each millisecond of audio to a phoneme label (e.g., `/K/ 0.05–0.12s`), which is used to assign phoneme labels to codec token positions.

### SpeechTokenizer checkpoint

Download the pretrained SpeechTokenizer weights from the [ZhangXInFD/SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer) repository. You need two files:

- `speechtokenizer.pt` — model weights
- `config.json` — model configuration

---

## Running the Pipeline

### Sanity check (50 utterances — runs in a few minutes)

```bash
python main.py \
    --librispeech_root data/LibriSpeech \
    --alignments_root  data/alignments/train-clean-100 \
    --st_ckpt          /path/to/speechtokenizer.pt \
    --st_config        /path/to/config.json \
    --output_dir       results \
    --max_utterances   50
```

### Full run (embeddings cached after first pass)

```bash
python main.py \
    --librispeech_root data/LibriSpeech \
    --alignments_root  data/alignments/train-clean-100 \
    --st_ckpt          /path/to/speechtokenizer.pt \
    --st_config        /path/to/config.json \
    --output_dir       results \
    --max_utterances   0
```

### Optional: train-clean-360 ablation

```bash
python main.py \
    --split            train-clean-360 \
    --alignments_root  data/alignments/train-clean-360 \
    --librispeech_root data/LibriSpeech \
    --st_ckpt          /path/to/speechtokenizer.pt \
    --st_config        /path/to/config.json \
    --output_dir       results_360 \
    --max_utterances   0
```

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--librispeech_root` | *(required)* | Path to `data/LibriSpeech/` |
| `--alignments_root` | *(required)* | Path to TextGrid alignment root for the chosen split |
| `--st_ckpt` | *(required)* | Path to `speechtokenizer.pt` |
| `--st_config` | *(required)* | Path to SpeechTokenizer `config.json` |
| `--output_dir` | `results` | Directory for all outputs |
| `--split` | `train-clean-100` | LibriSpeech split to use |
| `--eval_frac` | `0.1` | Fraction of utterances held out for evaluation |
| `--max_utterances` | `500` | Cap on utterances (0 = no cap) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--force_resplit` | off | Ignore cached `split.json` and recompute |

---

## Pipeline Details

```
Scan train-clean-100 utterance paths
         │
         ▼
Utterance-level 90/10 split (stratified by speaker)
  → saved to results/split.json for reproducibility
         │
         ▼
For each training utterance:
  ├─ Load forced alignment TextGrid → phoneme labels per token
  ├─ Encode with EnCodec  → 8 × (N_enc_tokens, 128) embeddings   [cached to .npz]
  ├─ Encode with ST       → 8 × (N_st_tokens, D)   embeddings    [cached to .npz]
  └─ Extract pitch (YIN)  → F0 in Hz per token (NaN if unvoiced)
         │
         ▼
Fit label encoders on training data (shared across both codecs)
  → label_encoder_phoneme.pkl, label_encoder_speaker.pkl
         │
         ▼
Train 24 probes per codec (8 layers × 3 tasks):
  ├─ Phoneme → LogisticRegression (class_weight="balanced")
  ├─ Speaker → LogisticRegression (class_weight="balanced")
  └─ Pitch   → LinearRegression  (voiced frames only)
         │
         ▼
Evaluate on held-out utterances → metrics per layer per codec
         │
         ▼
Plot 3 figures with chance-level reference lines
  → results/figures/{phoneme,speaker,pitch}_probing.png
```

**Embedding cache:** After a codec encodes an utterance for the first time, the 8-layer embeddings are saved to `results/cache/{codec}/{utterance_id}.npz`. Subsequent runs load from disk, skipping the encoding step entirely. This makes re-running with different probe hyperparameters fast.

**Reproducibility:** The train/eval split is saved to `results/split.json` on first run and reloaded on subsequent runs. The random seed is fixed at 42.

---

## Outputs

| File | Description |
|---|---|
| `results/split.json` | Train/eval utterance IDs — guarantees reproducible splits |
| `results/cache/encodec/*.npz` | Cached EnCodec embeddings per utterance |
| `results/cache/speechtokenizer/*.npz` | Cached SpeechTokenizer embeddings per utterance |
| `results/probes/probe_{codec}_layer{N}_{task}.pkl` | 48 trained probe models |
| `results/probes/label_encoder_{task}.pkl` | Fitted label encoders |
| `results/figures/phoneme_probing.png` | Phoneme accuracy + macro-F1 by layer |
| `results/figures/speaker_probing.png` | Speaker accuracy + macro-F1 by layer |
| `results/figures/pitch_probing.png` | Pitch MAE (Hz) + R² by layer |
| `results/results.pkl` | Raw metrics dict for downstream analysis |

---

## Methodology Notes

**Why utterance-level split?** Splitting at the token level leaks: tokens from the same utterance appear in both train and eval. Utterance-level splitting ensures no utterance is split across sets, which is standard practice in NLP probing studies.

**Why not dev-clean for evaluation?** LibriSpeech is designed so speakers never overlap between splits. Dev-clean's 40 speakers are entirely distinct from train-clean-100's 251 speakers. A speaker probe trained to classify 251 speakers cannot evaluate on unseen speakers — `LabelEncoder.transform()` raises `ValueError`. Using a single utterance-level hold-out keeps the methodology consistent across all three tasks.

**Why macro-F1?** English phoneme and speaker distributions are skewed. Macro-F1 weights all classes equally regardless of frequency, making it the appropriate primary metric for an imbalanced multi-class problem. `class_weight="balanced"` in the probe training likewise compensates for class imbalance.

**What is F0 / pitch?** The YIN algorithm extracts the **fundamental frequency (F0)** — the single "base pitch" of a voiced speech frame in Hz (e.g., 120 Hz for a low typical male voice, 220 Hz for a higher typical female voice). Unvoiced frames (consonants, silence) have no fundamental frequency and are excluded from regression. A high R² means the codec embedding linearly predicts the speaker's fundamental frequency.

---

## Background Literature

| Paper | Role |
|---|---|
| Sadok et al. 2025 — [arXiv:2506.04492](https://arxiv.org/abs/2506.04492) | Primary reference; this project extends their methodology |
| Défossez et al. 2022 — [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) | EnCodec architecture |
| Zhang et al. 2023 — [arXiv:2308.16692](https://arxiv.org/abs/2308.16692) | SpeechTokenizer architecture |
| Park et al. 2025 — [arXiv:2509.01390](https://arxiv.org/abs/2509.01390) | Statistical analysis of codec token structure |
| Belinkov 2022 — [arXiv:2102.12452](https://arxiv.org/abs/2102.12452) | Probing classifier methodology and best practices |

---

## Expected Results

Based on codec design and the prior literature:

| Task | EnCodec (expected) | SpeechTokenizer (expected) |
|---|---|---|
| Phoneme | Gradual rise in early layers, then plateau | Sharp rise at layer 1 (HuBERT alignment by design) |
| Speaker | Weak early, stronger in mid/deep layers | Similar pattern but possibly different depth |
| Pitch | Low R² across all layers | Low R² across all layers (replicates Sadok et al.) |

---

## Citation

If you use this code or build on this analysis:

```
Riley Denn and Akshay Aralikatti. Neural Audio Codec Interpretability: 
A Linear Probing Study of RVQ Layers in EnCodec and SpeechTokenizer. 
CSCI 682 Final Project, California State University, Chico, Spring 2026.
```