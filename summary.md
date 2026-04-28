# Project Summary: Neural Audio Codec Interpretability

## What This Project Is

A linear probing study that quantifies how **phoneme identity**, **speaker identity**, and **pitch (F0)**
distribute across the 8 RVQ layers of two neural audio codecs:

- **EnCodec** (Meta) — general-purpose codec, no explicit linguistic design
- **SpeechTokenizer** — RVQ-1 is explicitly trained to encode HuBERT semantic content

This extends Sadok et al. 2025 ("Bringing Interpretability to Neural Audio Codecs", Interspeech),
which used mutual information and t-SNE but **did not apply linear probing**.
The comparative codec design (EnCodec vs SpeechTokenizer) is the key novelty.

---

## Research Question

> How are phoneme identity, speaker identity, and pitch distributed across RVQ layers
> of EnCodec vs SpeechTokenizer? Does designing RVQ-1 for semantic alignment produce
> stronger linear decodeability than emergent compression structure?

---

## Hypothesis

| Attribute | Expected Finding |
|-----------|-----------------|
| Phoneme   | Concentrated in early RVQ layers; stronger in SpeechTokenizer RVQ-1 (by design) |
| Speaker   | Stronger in middle/deeper RVQ layers for both codecs |
| Pitch     | Weak/diffuse across all layers for both codecs (replicates Sadok et al. using a new method) |

---

## Method

**Data:** LibriSpeech train-clean-100, 90/10 train/eval split at the token level.

**Labels:**
- Phoneme — from MFA forced alignment TextGrids (timestamp → token index mapping)
- Speaker — from LibriSpeech metadata
- Pitch — extracted per token using `librosa.yin` (unvoiced frames excluded)

**Probes (48 total = 2 codecs × 8 layers × 3 tasks):**
- Phoneme → `LogisticRegression` → accuracy + macro-F1
- Speaker → `LogisticRegression` → accuracy + macro-F1
- Pitch   → `LinearRegression` on voiced frames only → MAE (Hz) + R²

Codec parameters are **never updated** — all probes use frozen embeddings.

---

## Codec Technical Details

| | EnCodec | SpeechTokenizer |
|---|---|---|
| Sample rate | 24 kHz | 16 kHz |
| RVQ layers | 8 | 8 |
| Token rate | 75 tokens/sec | 50 tokens/sec |
| Codebook size | 1024 | 1024 |
| Install | `pip install encodec` | `pip install speechtokenizer` + checkpoint |
| Design intent | General audio compression | RVQ-1 aligned to HuBERT semantic tokens |

---

## Code Structure

```
Neural-Audio-Codec-Interpretability/
  requirements.txt
  main.py                          ← single entrypoint, runs full pipeline
  data/
    load_librispeech.py            ← iterates .flac files, yields Utterance objects
    load_alignments.py             ← parses TextGrid files, maps timestamps to token indices
    extract_pitch.py               ← librosa YIN pitch extraction per token
  encode/
    encode_encodec.py              ← EnCodec inference → list of 8 embedding arrays
    encode_speechtokenizer.py      ← SpeechTokenizer inference → list of 8 embedding arrays
  probe/
    train_probes.py                ← trains & saves 24 probes per codec
    evaluate_probes.py             ← evaluates saved probes, returns metrics dict
  visualize/
    plot_curves.py                 ← 3 figures (phoneme / speaker / pitch), EnCodec vs ST
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download required data
- **LibriSpeech train-clean-100** — from https://openslr.org/12
- **Forced alignment TextGrids** — from https://github.com/CorentinJ/librispeech-alignments
- **SpeechTokenizer checkpoint** — `speechtokenizer.pt` + `config.json` from https://github.com/zhangxinfd/speechtokenizer

### 3. Quick sanity-check run (100 utterances)
```bash
python main.py \
  --librispeech_root /path/to/LibriSpeech \
  --alignments_root  /path/to/alignments \
  --st_ckpt          /path/to/speechtokenizer.pt \
  --st_config        /path/to/config.json \
  --output_dir       results \
  --max_utterances   100
```

### 4. Full run
```bash
python main.py \
  --librispeech_root /path/to/LibriSpeech \
  --alignments_root  /path/to/alignments \
  --st_ckpt          /path/to/speechtokenizer.pt \
  --st_config        /path/to/config.json \
  --output_dir       results \
  --max_utterances   0
```

### 5. Outputs
```
results/
  probes/              ← 48 .pkl probe files + 2 label encoders
  figures/
    phoneme_probing.png
    speaker_probing.png
    pitch_probing.png
  results.pkl          ← raw metrics dict for further analysis
```

---

## Sanity Checks
1. Encode 1 utterance with each codec → confirm 8 layers of embeddings returned
2. Verify token count ≈ `duration × token_rate`
3. SpeechTokenizer RVQ-1 phoneme accuracy should be noticeably higher than EnCodec RVQ-1
4. Pitch R² should be low across all layers for both codecs

---

## Key Differentiators from Sadok et al. 2025

1. **Linear probing** — not used in the original paper (they used mutual information + co-occurrence)
2. **Comparative codec analysis** — identical methodology applied to both codecs side-by-side
3. **Pitch as regression** — continuous Hz prediction (MAE + R²) vs. MI estimation
4. **Unified evaluation framework** — same probing pipeline across all 3 attributes

---

## Plan File
Full implementation plan: `/Users/akshaya/.claude/plans/so-we-are-working-virtual-hummingbird.md`
