"""
Encode utterances with both codecs, align phoneme/speaker/pitch labels,
and cache per-utterance embeddings, phonemes, and pitches as NPZ files to
avoid re-encoding and re-extracting features on subsequent runs.
"""

from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

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
    pitches:    np.ndarray         # (N_tokens,) float32 Hz; NaN for unvoiced


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


def _failed_audio_path(cache_dir: Path, utterance_id: str) -> Path:
    return cache_dir / "_failed_audio" / f"{utterance_id}.txt"


def _has_failed_audio(cache_dir: Path, utterance_id: str) -> bool:
    return _failed_audio_path(cache_dir, utterance_id).exists()


def _mark_failed_audio(cache_dir: Path, utterance_id: str, error: BaseException) -> None:
    marker = _failed_audio_path(cache_dir, utterance_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")


def _load_cached(
    cache_dir: Path, codec: str, utterance_id: str,
) -> Optional[Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """Return (embeddings, phonemes, pitches) from disk, or None if not cached.
    phonemes and pitches are None when absent from an older cache file."""
    path = _cache_path(cache_dir, codec, utterance_id)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            embeddings = [data[f"layer_{i}"] for i in range(NUM_LAYERS)]
            phonemes   = data["phonemes"] if "phonemes" in data else None
            pitches    = data["pitches"] if "pitches" in data else None
    except Exception as e:
        print(f"  [cache] {codec}/{utterance_id}: {type(e).__name__}: {e}; rebuilding")
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return embeddings, phonemes, pitches


def _cache_is_complete(
    cached: Optional[Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]],
) -> bool:
    return cached is not None and cached[1] is not None and cached[2] is not None


def _save_cache(
    cache_dir: Path, codec: str, utterance_id: str,
    embeddings: List[np.ndarray],
    phonemes: np.ndarray,
    pitches: np.ndarray,
) -> None:
    path = _cache_path(cache_dir, codec, utterance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{i}": emb for i, emb in enumerate(embeddings)}
    payload["phonemes"] = phonemes
    payload["pitches"] = pitches
    np.savez(file=path, **payload)  # pyright: ignore[reportArgumentType]


# ── Per-utterance feature helpers ─────────────────────────────────────────────

def _extract_features(
    embeddings: List[np.ndarray],
    audio,
    sr: int,
    phone_intervals,
    token_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (phonemes, pitches) aligned to the codec's token grid."""
    num_tokens = embeddings[0].shape[0]
    phones = np.array(phoneme_labels_for_tokens(phone_intervals, token_rate, num_tokens))
    pitches = extract_pitch_hz(audio, sr, token_rate, num_tokens)
    return phones, pitches


def _get_utterance_data(
    codec: str,
    cached: Optional[Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]],
    cache_dir: Path,
    utterance_id: str,
    encode_fn,
    model,
    audio,
    sr,
    phone_intervals,
    token_rate: float,
) -> Optional[Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:
    """Resolve (embeddings, phonemes, pitches) for one utterance and codec.

    Handles three cases:
        - No cache: encode audio, extract features, save everything.
        - Partial cache (old file missing phonemes or pitches): reuse embeddings,
          extract features from audio, re-save the updated cache.
        - Full cache: return cached data immediately without touching audio.

    Returns None if encoding fails; the caller should skip the utterance.
    """
    if cached is None:
        try:
            embeddings, _ = encode_fn(model, audio, sr)
        except Exception as e:
            print(f"  [{codec}] {utterance_id}: {e}")
            return None
        phones, pitches = _extract_features(embeddings, audio, sr, phone_intervals, token_rate)
        _save_cache(cache_dir, codec, utterance_id, embeddings, phones, pitches)
        return embeddings, phones, pitches

    embeddings, phones, pitches = cached
    if phones is None or pitches is None:
        phones, pitches = _extract_features(embeddings, audio, sr, phone_intervals, token_rate)
        _save_cache(cache_dir, codec, utterance_id, embeddings, phones, pitches)
    return embeddings, phones, pitches


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
            - Its TextGrid alignment file is missing and a cache entry still needs
                aligned phoneme or pitch features.
      - Either codec raises an exception during encoding.

    Embeddings, phonemes, and pitches are saved to
    cache_dir/{codec}/{utterance_id}.npz after the first run so subsequent
    runs load everything from disk without re-encoding or re-extracting.

    Args:
        utterance_entries: List of (flac_path, speaker_id, utterance_id).
        encodec_model:     Loaded EnCodec model.
        st_model:          Loaded SpeechTokenizer model.
        alignments_root:   Root of TextGrid alignment files.
        cache_dir:         Directory for NPZ cache.
        max_utterances:    Process at most this many utterance entries (0 = no limit).

    Returns:
        (enc_bundle, st_bundle) — dicts with keys:
            "embeddings": List of 8 ndarrays, each shape (N_tokens, embed_dim)
            "phonemes":   ndarray of shape (N_tokens,), dtype str
            "speakers":   ndarray of shape (N_tokens,), dtype str
            "pitches":    ndarray of shape (N_tokens,), float32 (NaN = unvoiced)
    """
    encdc_layers:   List[List[np.ndarray]] = [[] for _ in range(NUM_LAYERS)]
    st_layers:      List[List[np.ndarray]] = [[] for _ in range(NUM_LAYERS)]
    encdc_phonemes: List[np.ndarray] = []
    st_phonemes:    List[np.ndarray] = []
    encdc_speakers: List[str] = []
    st_speakers:    List[str] = []
    encdc_pitches:  List[np.ndarray] = []
    st_pitches:     List[np.ndarray] = []

    entries = utterance_entries[:max_utterances] if max_utterances > 0 else utterance_entries
    for flac_path, speaker_id, utterance_id in tqdm(entries, desc="Collecting"):

        encdc_cached = _load_cached(cache_dir, "encodec", utterance_id)
        st_cached = _load_cached(cache_dir, "speechtokenizer", utterance_id)
        needs_alignment = not (_cache_is_complete(encdc_cached) and _cache_is_complete(st_cached))

        # If this utterance has previously failed audio decoding, skip it
        # immediately on retries rather than spending tens of seconds failing again.
        if needs_alignment and _has_failed_audio(cache_dir, utterance_id):
            continue

        phone_intervals = None
        if needs_alignment:
            tg_path = _alignment_path(alignments_root, utterance_id)
            phone_intervals = load_textgrid_phones(str(tg_path))
            if not phone_intervals:
                continue

        need_audio = needs_alignment
        if need_audio:
            try:
                audio, sr = torchaudio.load(flac_path)
            except Exception as e:
                _mark_failed_audio(cache_dir, utterance_id, e)
                print(f"  [audio] {utterance_id}: {type(e).__name__}: {e}")
                continue
        else:
            audio, sr = None, None

        encdc_result = _get_utterance_data(
            "encodec", encdc_cached, cache_dir, utterance_id,
            encodec_encode, encodec_model, audio, sr, phone_intervals, ENCODEC_TOKEN_RATE,
        )
        if encdc_result is None:
            continue

        st_result = _get_utterance_data(
            "speechtokenizer", st_cached, cache_dir, utterance_id,
            st_encode, st_model, audio, sr, phone_intervals, ST_TOKEN_RATE,
        )
        if st_result is None:
            continue

        encdc_embeddings, encdc_phone_labels, encdc_pitch_values = encdc_result
        st_embeddings, st_phone_labels, st_pitch_values = st_result

        encdc_phonemes.append(encdc_phone_labels)
        st_phonemes.append(st_phone_labels)
        encdc_speakers.extend([speaker_id] * encdc_embeddings[0].shape[0])
        st_speakers.extend([speaker_id] * st_embeddings[0].shape[0])
        encdc_pitches.append(encdc_pitch_values)
        st_pitches.append(st_pitch_values)

        for i in range(NUM_LAYERS):
            encdc_layers[i].append(encdc_embeddings[i])
            st_layers[i].append(st_embeddings[i])

    def _bundle(layers, phonemes, speakers, pitches) -> Bundle:
        if not layers[0]:
            raise ValueError(
                "No utterances were collected. Check alignment files and codec errors before training probes."
            )
        # Clear each layer's list immediately after concatenation so the input
        # chunks are freed before the next layer is allocated.  Without this,
        # the list comprehension keeps all 8 layers' input arrays alive alongside
        # all 8 output arrays, doubling peak RAM (~816 GB for ST at 25 k utts).
        embeddings = []
        for i in range(NUM_LAYERS):
            embeddings.append(np.concatenate(layers[i], axis=0))
            layers[i].clear()
        return Bundle(
            embeddings=embeddings,
            phonemes=np.concatenate(phonemes),
            speakers=np.array(speakers),
            pitches=(np.concatenate(pitches) if pitches else np.array([], dtype=np.float32)),
        )

    return (
        _bundle(encdc_layers, encdc_phonemes, encdc_speakers, encdc_pitches),
        _bundle(st_layers, st_phonemes, st_speakers, st_pitches),
    )
