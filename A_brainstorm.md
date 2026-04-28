# Neural Audio Codec Interpretability — Linear Probing Study

## Context
Based on Sadok et al. 2025 ("Bringing Interpretability to Neural Audio Codecs", Interspeech 2025),
which used mutual information and t-SNE to analyze codec token structure. This project extends
that work by applying systematic linear probing — a method not used in the paper — to quantify
how phoneme identity, speaker identity, and pitch distribute across RVQ layers in two codecs.

## Research Question
How are phoneme identity, speaker identity, and pitch distributed across the RVQ layers of
EnCodec vs SpeechTokenizer? Does designing RVQ-1 for semantic alignment (SpeechTokenizer)
produce stronger linear decodeability than emergent compression structure (EnCodec)?

## Hypothesis
- Phoneme: concentrated in early RVQ layers; stronger in SpeechTokenizer RVQ-1 (by design)
- Speaker: stronger in middle/deeper RVQ layers for both codecs
- Pitch: weak/diffuse across all layers for both codecs (replicating Sadok et al. finding via new method)

## Data
- **Dataset:** LibriSpeech train-clean-100
- **Phoneme labels:** publicly available forced alignments (timestamp → token index mapping)
- **Speaker labels:** LibriSpeech metadata directly
- **Pitch labels:** extracted from raw audio using `librosa.yin` or `parselmouth`

## Models
- **EnCodec** (Meta): 24kHz, 6kbps, 8 RVQ layers, ~75 tokens/sec. Install: `pip install encodec`
- **SpeechTokenizer**: 16kHz, 8 RVQ layers, ~50 tokens/sec. Install: `pip install speechtokenizer` + download checkpoint. Repo: https://github.com/zhangxinfd/speechtokenizer

Both used frozen (inference only). No codec parameters updated.

## Pipeline

### Stage 1 — Encode Audio
- Encode LibriSpeech audio with EnCodec → extract per-layer codebook embedding vectors (not just indices)
- Repeat with SpeechTokenizer at 16kHz
- Token rate differs: EnCodec = sampling_rate/320 = 75 tok/s; SpeechTokenizer = 16000/320 = 50 tok/s

### Stage 2 — Align Labels to Token Positions
- Phoneme: convert forced alignment timestamps (seconds) → token index via `t * token_rate`
- Speaker: broadcast speaker ID across all tokens in an utterance
- Pitch: extract per-frame pitch (Hz) from raw audio, align to token positions; unvoiced frames = NaN (excluded from pitch regression)

### Stage 3 — Train Probes
- For each codec × each RVQ layer × each task:
  - Phoneme → `sklearn.linear_model.LogisticRegression` on frozen layer embeddings
  - Speaker → `sklearn.linear_model.LogisticRegression` on frozen layer embeddings
  - Pitch   → `sklearn.linear_model.LinearRegression` on voiced frames only
- Total: 2 codecs × 8 layers × 3 tasks = 48 probes
- Use train split for fitting, held-out split for evaluation

### Stage 4 — Evaluate
- Phoneme + Speaker: accuracy + macro-F1
- Pitch: MAE (Hz) + R²

### Stage 5 — Visualize
- 3 plots total (one per task)
- Each plot: x-axis = RVQ layer (1–8), y-axis = metric
- Two lines per plot: EnCodec vs SpeechTokenizer

## Key Differentiators from Sadok et al.
1. **Linear probing** — not used in the original paper; they used mutual information and co-occurrence
2. **Comparative codec analysis** — both codecs probed with identical methodology
3. **Pitch via regression** — continuous Hz prediction vs. MI estimation
4. **Same evaluation framework across all 3 attributes** — unified probing pipeline

## File Structure (to build)
```
project/
  data/
    download_librispeech.py
    extract_alignments.py
    extract_pitch.py
  encode/
    encode_encodec.py
    encode_speechtokenizer.py
  probe/
    train_probes.py
    evaluate_probes.py
  visualize/
    plot_curves.py
  main.py
```

## Dependencies
```
encodec
speechtokenizer
librosa          # pitch extraction
praat-parselmouth  # alternative pitch extractor
scikit-learn
torch
torchaudio
matplotlib
numpy
```

## Verification
1. Encode 1 utterance with each codec, confirm 8 layers of embeddings returned
2. Verify token count matches expected rate (utterance_duration × token_rate)
3. Confirm phoneme label alignment: plot token-level phoneme labels and verify boundaries look correct
4. Train probes on small subset (100 utterances), confirm training runs without error
5. SpeechTokenizer RVQ-1 phoneme accuracy should be noticeably higher than EnCodec RVQ-1 (sanity check — this is by design)
6. Pitch R² should be low across all layers for both codecs (aligns with Sadok et al.)
