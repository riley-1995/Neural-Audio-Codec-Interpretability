"""
Encode utterances with both codecs, align phoneme/speaker/pitch labels,
and cache per-utterance embeddings as NPZ files to avoid re-encoding on
subsequent runs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torchaudio
from tqdm import tqdm

from data.extract_pitch import extract_pitch_hz
from data.load_alignments import load_textgrid_phones, phoneme_labels_for_tokens
from encode.encode_encodec import encode as encodec_encode
from encode.encode_speechtokenizer import encode as st_encode

NUM_LAYERS         = 8
ENCODEC_TOKEN_RATE = 75.0   # 24 000 Hz / 320 samples per frame
ST_TOKEN_RATE      = 50.0   # 16 000 Hz / 320 samples per frame

UttEntry = Tuple[str, str, str]   # (flac_path, speaker_id, utterance_id)


class Bundle(TypedDict):
    """Per-codec data bundle returned by collect_bundle."""
    embeddings: List[np.ndarray]   # 8 arrays, each shape (N_tokens, embed_dim)
    phonemes:   np.ndarray         # (N_tokens,) string phoneme labels
    speakers:   np.ndarray         # (N_tokens,) string speaker IDs
    pitch:      np.ndarray         # (N_tokens,) float32 Hz; NaN for unvoiced


# ── Alignment ──────────────────────────────────────────────────────────────────

def _alignment_path(alignments_root: str, utterance_id: str) -> Path:
    """
    CorentinJ/librispeech-alignments directory layout:
      <root>/<speaker_id>/<chapter_id>/<utterance_id>.TextGrid
    """
    parts = utterance_id.split("-")   # e.g. ["1069", "133699", "0001"]
    return Path(alignments_root) / parts[0] / parts[1] / f"{utterance_id}.TextGrid"


# ── Embedding cache ────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, codec: str, utterance_id: str) -> Path:
    return cache_dir / codec / f"{utterance_id}.npz"


def _load_cached(
    cache_dir: Path, codec: str, utterance_id: str
) -> Optional[List[np.ndarray]]:
    """Return 8 layer-embedding arrays from disk, or None if not cached yet."""
    p = _cache_path(cache_dir, codec, utterance_id)
    if not p.exists():
        return None
    data = np.load(p)
    return [data[f"layer_{i}"] for i in range(NUM_LAYERS)]


def _save_cache(
    cache_dir: Path, codec: str, utterance_id: str, embeddings: List[np.ndarray]
) -> None:
    p = _cache_path(cache_dir, codec, utterance_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{f"layer_{i}": emb for i, emb in enumerate(embeddings)})


# ── Main collection function ───────────────────────────────────────────────────

def collect_bundle(
    utterance_entries: List[UttEntry],
    encodec_model,
    st_model,
    alignments_root: str,
    cache_dir: Path,
    max_utterances: int = 0,
) -> "Tuple[Bundle, Bundle]":
    """
    Encode utterances with EnCodec and SpeechTokenizer; align labels.

    Each utterance is skipped if:
      - Its TextGrid alignment file is missing.
      - Either codec raises an exception during encoding.

    Embeddings are saved to cache_dir/{codec}/{utterance_id}.npz after encoding
    so re-runs load from disk instead of re-encoding.

    Args:
        utterance_entries: List of (flac_path, speaker_id, utterance_id).
        encodec_model:     Loaded EnCodec model.
        st_model:          Loaded SpeechTokenizer model.
        alignments_root:   Root of TextGrid alignment files.
        cache_dir:         Directory for NPZ embedding cache.
        max_utterances:    Stop after this many utterances (0 = no limit).

    Returns:
        (enc_bundle, st_bundle) — dicts with keys:
            "embeddings": List of 8 ndarrays, each shape (N_tokens, embed_dim)
            "phonemes":   ndarray of shape (N_tokens,), dtype str
            "speakers":   ndarray of shape (N_tokens,), dtype str
            "pitch":      ndarray of shape (N_tokens,), float32 (NaN = unvoiced)
    """
    enc_layers = [[] for _ in range(NUM_LAYERS)]
    st_layers  = [[] for _ in range(NUM_LAYERS)]
    enc_phones: List[str] = []
    st_phones:  List[str] = []
    enc_spk:    List[str] = []
    st_spk:     List[str] = []
    enc_pitch:  List[np.ndarray] = []
    st_pitch:   List[np.ndarray] = []

    count = 0
    for flac_path, speaker_id, utterance_id in tqdm(utterance_entries, desc="Collecting"):
        if max_utterances > 0 and count >= max_utterances:
            break

        # Alignment must exist before we bother loading audio
        tg_path = _alignment_path(alignments_root, utterance_id)
        phone_intervals = load_textgrid_phones(str(tg_path))
        if not phone_intervals:
            continue

        audio, sr = torchaudio.load(flac_path)

        # ── EnCodec ────────────────────────────────────────────────────────
        enc_embs = _load_cached(cache_dir, "encodec", utterance_id)
        if enc_embs is None:
            try:
                enc_embs, _ = encodec_encode(encodec_model, audio, sr)
            except Exception as e:
                print(f"  [EnCodec] {utterance_id}: {e}")
                continue
            _save_cache(cache_dir, "encodec", utterance_id, enc_embs)

        # ── SpeechTokenizer ────────────────────────────────────────────────
        st_embs = _load_cached(cache_dir, "speechtokenizer", utterance_id)
        if st_embs is None:
            try:
                st_embs, _ = st_encode(st_model, audio, sr)
            except Exception as e:
                print(f"  [SpeechTokenizer] {utterance_id}: {e}")
                continue   # skip whole utterance if either codec fails
            _save_cache(cache_dir, "speechtokenizer", utterance_id, st_embs)

        enc_ntok = enc_embs[0].shape[0]
        st_ntok  = st_embs [0].shape[0]

        # Align phoneme labels and extract pitch for each codec's token grid
        enc_phones.extend(phoneme_labels_for_tokens(phone_intervals, ENCODEC_TOKEN_RATE, enc_ntok))
        st_phones .extend(phoneme_labels_for_tokens(phone_intervals, ST_TOKEN_RATE,      st_ntok))
        enc_spk   .extend([speaker_id] * enc_ntok)
        st_spk    .extend([speaker_id] * st_ntok)
        enc_pitch .append(extract_pitch_hz(audio, sr, ENCODEC_TOKEN_RATE, enc_ntok))
        st_pitch  .append(extract_pitch_hz(audio, sr, ST_TOKEN_RATE,      st_ntok))

        for i in range(NUM_LAYERS):
            enc_layers[i].append(enc_embs[i])
            st_layers [i].append(st_embs [i])

        count += 1

    def _bundle(layers, phones, spk, pitch_parts) -> Bundle:
        if not layers[0]:
            raise ValueError(
                "No utterances were collected. Check alignment files and codec errors before training probes."
            )
        return Bundle(
            embeddings=[np.concatenate(layers[i], axis=0) for i in range(NUM_LAYERS)],
            phonemes=np.array(phones),
            speakers=np.array(spk),
            pitch=(np.concatenate(pitch_parts) if pitch_parts
                   else np.array([], dtype=np.float32)),
        )

    return (
        _bundle(enc_layers, enc_phones, enc_spk, enc_pitch),
        _bundle(st_layers,  st_phones,  st_spk,  st_pitch),
    )
