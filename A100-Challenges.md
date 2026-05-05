# A100 Server — Challenges, Bottlenecks, and Resolutions

**Project:** CSCI 682 Final — Linear Probing of RVQ Layers: EnCodec vs SpeechTokenizer
**Server:** `cscigpu` (NVIDIA A100 80 GB PCIe, 582 GB CPU RAM, 16 CPU cores)
**Goal:** Train 48 linear probes (8 RVQ layers × 3 tasks × 2 codecs) on LibriSpeech train-clean-100 (~25,672 utterances)

---

## System Scale — Why This Was Hard

Before diving into individual problems, understanding the data scale is essential.

| Codec | Sample Rate | Token Rate | Embed Dim | Avg Tokens/Utt | Total Tokens (25k utts) | Per-Layer RAM (float32) |
|---|---|---|---|---|---|---|
| EnCodec | 24 kHz | 75 tok/s | 128 | ~953 | ~24.4 M | ~12.5 GB |
| SpeechTokenizer | 16 kHz | 50 tok/s | 1024 | ~635 | ~16.3 M | ~66.7 GB |

- **8 layers × 2 codecs = 16 embedding arrays**
- **All layers in RAM simultaneously:** 8 × 12.5 + 8 × 66.7 = **634 GB** — exceeds the server's 582 GB
- **Server RAM:** 582 GB total, ~575 GB available
- **Server GPU:** A100 80 GB — used only during encoding (codec inference); sklearn probes run on CPU

---

## Challenge 1 — Disk Quota Exceeded on `~/.cache/torch`

### What Happened
EnCodec downloads its weights (~100 MB) to `~/.cache/torch` by default. The home directory had a tight quota. On first run:

```
OSError: [Errno 122] Disk quota exceeded: '/home/anaralikatti/.cache/torch/hub/...'
```

### Resolution
Set `TORCH_HOME` to a research partition with no quota before every run:

```bash
export TORCH_HOME=/research/shared-rdenn-anaralikatti/.cache/torch
```

This must be set **before** launching `nohup`. It was added permanently to the startup command.

---

## Challenge 2 — `ModuleNotFoundError: No module named 'numpy'`

### What Happened
Launching with `python main.py` after activating the venv worked interactively, but `nohup python ...` resolved to the system Python (which had no project dependencies installed).

```
ModuleNotFoundError: No module named 'numpy'
```

### Resolution
Always use the **absolute path** to the venv Python in `nohup` commands:

```bash
# Wrong
nohup python main.py ... &

# Correct
nohup /research/anaralikatti/682/Neural-Audio-Codec-Interpretability/venv/bin/python main.py ... &
```

---

## Challenge 3 — SyntaxError After Copy-Pasting Code

### What Happened
When pasting large code blocks into remote files via `cat > file.py << 'ENDOFFILE'`, a shell command accidentally ended up as line 2 of `collect.py`:

```
  File "encode/collect.py", line 2
    cat > encode/collect.py << 'ENDOFFILE'
        ^
SyntaxError: invalid syntax
```

### Resolution
```bash
sed -i '/^cat > /d' encode/collect.py
```

**Lesson:** Always verify pasted files with `head -5 filename.py` before running.

---

## Challenge 4 — Silent SIGKILL at 98% Collection (First Major OOM)

### What Happened
The original `collect_bundle()` accumulated **all 8 layers of both codecs** in RAM as Python lists throughout the entire collection loop:

```python
# BEFORE — accumulates everything in memory
encdc_layers: List[List[np.ndarray]] = [[] for _ in range(NUM_LAYERS)]
st_layers:    List[List[np.ndarray]] = [[] for _ in range(NUM_LAYERS)]

for flac_path, speaker_id, utterance_id in tqdm(entries):
    encdc_embeddings, _, _ = encode_encodec(...)
    st_embeddings, _, _    = encode_speechtokenizer(...)

    for i in range(NUM_LAYERS):
        encdc_layers[i].append(encdc_embeddings[i])   # accumulates forever
        st_layers[i].append(st_embeddings[i])          # accumulates forever

# After the loop — concatenate all at once
embeddings = [np.concatenate(layers[i], axis=0) for i in range(NUM_LAYERS)]
```

At ~25,000 utterances, the `_bundle()` list comprehension tried to allocate 8 output arrays **while the 8 input arrays were still alive** — effectively doubling peak RAM:

| Stage | RAM Used |
|---|---|
| Input chunks (all 8 layers, both codecs) | ~634 GB |
| Output concatenated arrays being allocated | ~634 GB |
| **Peak during concatenation** | **~816 GB** |

The OS kernel sent SIGKILL with no traceback. The tqdm bar showed 98% and then silence.

### First Partial Fix (Insufficient)
Free each layer's input list immediately after concatenating it:

```python
# PARTIAL FIX — frees inputs one layer at a time
embeddings = []
for i in range(NUM_LAYERS):
    embeddings.append(np.concatenate(layers[i], axis=0))
    layers[i].clear()   # release input chunks before next layer
```

This reduced peak RAM but still kept all 16 output arrays (8 per codec) alive simultaneously — still ~634 GB, still over limit at 25k utterances.

### Final Fix — Layer-by-Layer Streaming Architecture

**Core idea:** Never hold embeddings in RAM after collection. Store only metadata. Load one RVQ layer at a time during probe training.

```python
# AFTER — collect.py returns metadata only, no embeddings in RAM

class CollectedMetadata(NamedTuple):
    utterance_ids: List[str]
    phonemes:      np.ndarray
    speakers:      np.ndarray
    pitches:       np.ndarray
    codec:         str
    cache_dir:     Path

def collect_bundle(...) -> Tuple[CollectedMetadata, CollectedMetadata]:
    utt_ids, phonemes, speakers, pitches = [], [], [], []

    for flac_path, speaker_id, utterance_id in tqdm(entries):
        # encode → saved to NPZ cache → immediately freed
        embeddings, _, _ = encode_fn(model, audio, sr)
        _save_cache(cache_dir, codec, utterance_id, embeddings, ...)
        del embeddings   # <-- freed immediately; never accumulates

        utt_ids.append(utterance_id)
        phonemes.append(phone_labels)
        # ...

    return CollectedMetadata(utt_ids, phonemes, speakers, pitches, ...)
```

**RAM during collection:**

| Before | After |
|---|---|
| ~634 GB (all layers accumulating) | ~340 MB (one utterance at a time) |

---

## Challenge 5 — Probe Training OOM with 6 Concurrent Workers

### What Happened
After the streaming refactor, probe training was launched with `--probe_exec_profile a100` (6 concurrent sklearn workers). Each worker received the full SpeechTokenizer layer (~67 GB, float32). sklearn's `lbfgs` solver internally converts the input to **float64**, doubling the size:

| Item | Size |
|---|---|
| ST layer float32 (shared) | 67 GB |
| float64 copy per worker | 134 GB |
| 6 workers × 134 GB | **804 GB** |

The process was killed immediately after collection completed — label encoders were saved but zero probe `.pkl` files existed.

### Resolution
Limit concurrent workers to 2, keeping float64 copies within budget:

```bash
# Before (too many workers)
--probe_exec_profile a100        # 6 workers → 804 GB → OOM

# After (safe)
--probe_workers 2                # 2 workers → 2 × 134 GB = 268 GB → safe
```

**Peak RAM with 2 workers:**

| Component | RAM |
|---|---|
| ST layer float32 (shared, loaded once) | 67 GB |
| EnCodec layer float32 (shared) | 12.5 GB |
| 2 × sklearn float64 copies | 268 GB |
| Labels, OS, other | ~30 GB |
| **Total** | **~378 GB** (within 582 GB) |

---

## Challenge 6 — Ctrl+C in Monitoring Terminal Killing the Background Process

### What Happened
While monitoring with `tail -f run.log`, pressing **Ctrl+C** sent SIGINT to the entire process group, terminating the background pipeline.

### Resolution
- **Never press Ctrl+C** in a terminal running `tail -f`.
- To stop monitoring: close the terminal window or open a new SSH session.
- Use `disown` when launching to detach from the shell:

```bash
nohup python main.py ... > run.log 2>&1 &
echo $! > run.pid && disown $(cat run.pid)
```

To safely check if the process is alive from any terminal:
```bash
kill -0 $(cat run.pid) && echo "ALIVE" || echo "DEAD"
```

---

## Challenge 7 — lbfgs With 24 Million Tokens (The Final Bottleneck — 9 Hours, Zero Probes)

### What Happened
This was the critical failure. After all the memory fixes, probe training launched successfully with 2 workers and the streaming architecture. The process ran for **9+ hours** and saved **0 probe files**. The log had only 13 lines — all from startup and collection. No probe output at all.

**Root cause:** `_fit_one_probe()` called sklearn's `LogisticRegression` directly on the **entire token dataset** with no subsampling:

```python
# BEFORE — no subsampling; intractable at dataset scale
def _fit_one_probe(codec_name, layer_num, task, X, y, voiced_mask, probe_dir, max_iter):
    ...
    else:
        probe = LogisticRegression(
            max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        ).fit(X, y)   # X has 24 million rows
```

**Why lbfgs fails at this scale:**

lbfgs is a quasi-Newton method. Each iteration requires a **full gradient computation** over all N samples:

```
Cost per lbfgs iteration = O(N × d × C)

EnCodec: 24.4M × 128 × 40  =  124.9 billion ops  ≈  12–20 sec/iter
ST:      16.3M × 1024 × 251 = 4.19 trillion ops   ≈  400+ sec/iter

max_iter = 1000

Worst case for ST speaker probe:
1000 iterations × 400 sec = 400,000 sec = 111 hours
```

sklearn only writes the `.pkl` file **after** `fit()` returns. After 9 hours inside a single `fit()` call, no file had been written.

**htop during the stuck run:**
```
PID    USER       CPU%   MEM%   TIME+     COMMAND
804709 anaralikatti 100.1  51.1   9h34:40  /research/anaralikatti/682/...
804708 anaralikatti  99.5  51.1   9h18:22  /research/anaralikatti/682/...
```
Two workers pegged at 100% CPU, 297 GB RES each. The process would never have printed anything or saved any file until lbfgs converged — which for ST speaker probes would take days, if it converged at all.

### Why FP16 Doesn't Help
FP16 (half-precision) was considered as a potential fix. It was ruled out because:
- sklearn's lbfgs internally converts inputs to **float64** regardless of input dtype
- The float64 copy is the same size (134 GB) whether input is FP16 or FP32
- Storing embeddings in FP16 saves disk/transfer but does not reduce sklearn's working memory

### Resolution — Token Subsampling

**Linear probes do not benefit from training on millions of tokens.** The probe is a single weight matrix — it saturates with 50k–200k samples. Research papers in this area routinely subsample. 200,000 tokens was chosen as the cap:

```python
# AFTER — subsample to 200k tokens before fitting
def _fit_one_probe(codec_name, layer_num, task, X, y, voiced_mask, probe_dir, max_iter):
    ...
    else:
        if len(X) > 200_000:
            rng = np.random.RandomState(42)       # reproducible
            idx = rng.choice(len(X), 200_000, replace=False)
            X, y = X[idx], y[idx]
        probe = LogisticRegression(
            max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        ).fit(X, y)
```

**Cost per lbfgs iteration after subsampling:**

```
EnCodec: 200k × 128 × 40  =  1.02 billion ops  ≈  0.1 sec/iter
ST:      200k × 1024 × 251 = 51.4 billion ops   ≈  5 sec/iter

Convergence typically in 100–300 iterations:
EnCodec probe: 10–30 sec
ST probe:      500–1500 sec (~8–25 min)

48 probes / 2 workers ≈ 1–3 hours total
```

**RAM per sklearn fit after subsampling:**

| | Before | After |
|---|---|---|
| Input X (float32) | 67 GB (full ST layer) | 1.6 GB (200k × 1024 × 4) |
| sklearn float64 copy | 134 GB | 3.2 GB |
| **Total per worker** | **134 GB** | **3.2 GB** |

The 200k subsample represents:
- 0.82% of EnCodec training tokens (200k / 24.4M)
- 1.23% of ST training tokens (200k / 16.3M)
- ~800 samples per speaker class on average (251 speakers)
- ~5,000 samples per phoneme class on average (40 phonemes)

All three are well above the minimum needed for a stable linear probe.

---

## Challenge 8 — Monitoring Without Disrupting the Run

### Problem
The `tail -f run.log` command in a terminal, when interrupted with Ctrl+C, kills the foreground process group and can propagate signals to the background job if `disown` was not used.

Additionally, htop and nvtop showed conflicting information:
- **htop**: confirmed process alive with expected CPU/RAM usage
- **nvtop**: showed 0% GPU — this was **correct and expected** (encoding phase done; sklearn is CPU-only)

### Safe Monitoring Pattern
```bash
# Check if alive (safe, no side effects)
kill -0 $(cat run.pid) && echo "ALIVE" || echo "DEAD"

# Check probe count
ls results_capfull_20260503_071048/probes/probe_*.pkl | wc -l

# Read last 30 lines of log (safe, not tail -f)
tail -30 run.log

# If you must follow the log live, open a NEW SSH window; never Ctrl+C it
tail -f run.log   # close the WINDOW to stop, do NOT press Ctrl+C
```

---

## Final Working Configuration

### Launch Command
```bash
export TORCH_HOME=/research/shared-rdenn-anaralikatti/.cache/torch

nohup /research/anaralikatti/682/Neural-Audio-Codec-Interpretability/venv/bin/python main.py \
    --librispeech_root /research/shared-rdenn-anaralikatti/LibriSpeech \
    --alignments_root  /research/shared-rdenn-anaralikatti/LibriSpeech/train-clean-100 \
    --st_ckpt          /research/shared-rdenn-anaralikatti/SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
    --st_config        /research/shared-rdenn-anaralikatti/SpeechTokenizer/speechtokenizer_hubert_avg/config.json \
    --output_dir       results_capfull_20260503_071048 \
    --max_utterances   0 \
    --probe_workers    2 \
    > run.log 2>&1 &

echo $! > run.pid && disown $(cat run.pid) && echo "Started PID $(cat run.pid)"
```

### Expected Timeline (With All Fixes Applied)

| Phase | Time |
|---|---|
| Model loading | ~2 min |
| Collection (all 25k cached) | ~15 min |
| Label encoder fitting | ~1 min |
| Probe training (48 probes, 2 workers, 200k subsample) | ~1–3 hours |
| Evaluation (2,867 eval utterances) | ~20 min |
| Plotting | ~1 min |
| **Total** | **~2–4 hours** |

### Expected Outputs
```
results_capfull_20260503_071048/
├── split.json                              # reproducible train/eval split
├── cache/
│   ├── encodec/*.npz                       # 25,672 files, ~100 GB
│   └── speechtokenizer/*.npz              # 25,672 files, ~600 GB
├── probes/
│   ├── label_encoder_phoneme.pkl
│   ├── label_encoder_speaker.pkl
│   ├── probe_encodec_layer{1-8}_{phoneme,speaker,pitch}.pkl   # 24 files
│   └── probe_speechtokenizer_layer{1-8}_{phoneme,speaker,pitch}.pkl  # 24 files
├── results.pkl                             # metrics dict
└── figures/
    ├── phoneme_probing.png
    ├── speaker_probing.png
    └── pitch_probing.png
```

---

## Summary of All Fixes Applied

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 1 | Disk quota exceeded | `~/.cache/torch` in home dir | `export TORCH_HOME=/research/shared-rdenn-...` |
| 2 | ModuleNotFoundError | `nohup python` used system Python | Use absolute venv path in nohup |
| 3 | SyntaxError in .py file | Shell heredoc command pasted into file | `sed -i '/^cat > /d' collect.py` |
| 4 | SIGKILL at 98% collection | All 8 layers × 2 codecs in RAM (~816 GB peak) | Layer-by-layer streaming; `CollectedMetadata` returns no embeddings |
| 5 | OOM after collection (0 probes) | 6 workers × 134 GB float64 copies = 804 GB | Limit to `--probe_workers 2` (~268 GB) |
| 6 | Ctrl+C killed background job | `tail -f` Ctrl+C sends SIGINT to process group | `disown` at launch; never Ctrl+C a monitoring terminal |
| 7 | 9 hours, 0 probes saved | lbfgs on 24M tokens; each iteration takes minutes | Subsample to 200k tokens in `_fit_one_probe` before sklearn fit |
