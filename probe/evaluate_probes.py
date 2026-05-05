"""
Evaluate saved probes on held-out embeddings and return layer-wise metrics.

Metrics:
  phoneme / speaker : accuracy, macro-F1
  pitch             : MAE (Hz), R²
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

NUM_LAYERS = 8


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _encode_known_labels(
    encoder: LabelEncoder,
    labels: np.ndarray,
    task_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode only labels observed during training and return (encoded, mask)."""
    known_mask = np.isin(labels, encoder.classes_)
    skipped = int((~known_mask).sum())
    if skipped:
        print(f"  [{task_name}] Skipping {skipped} eval tokens with unseen labels")

    if not known_mask.any():
        return np.array([], dtype=np.int64), known_mask

    return np.asarray(encoder.transform(labels[known_mask])), known_mask


def evaluate_probes(
    embeddings_by_layer: List[np.ndarray],
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    pitch_values: np.ndarray,
    codec_name: str,
    probe_dir: str,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Dict[str, List[float]]:
    """
    Evaluate all 24 probes for one codec on the held-out split.

    Args:
        embeddings_by_layer: List of 8 arrays, shape (N_tokens, embed_dim).
        phoneme_labels:      String phoneme labels for eval tokens.
        speaker_labels:      String speaker IDs for eval tokens.
        pitch_values:        Float Hz array (NaN for unvoiced) for eval tokens.
        codec_name:          "encodec" or "speechtokenizer".
        probe_dir:           Directory containing saved .pkl probe files.
        label_encoders:      Pre-fitted encoders from fit_label_encoders().
                             If None, loads from probe_dir (backward compat).

    Returns:
        Dict with keys:
            "phoneme_acc", "phoneme_f1"   — list of 8 floats per layer
            "speaker_acc", "speaker_f1"
            "pitch_mae",   "pitch_r2"
    """
    probe_path = Path(probe_dir)

    if label_encoders is None:
        label_encoders = {
            "phoneme": _load(probe_path / "label_encoder_phoneme.pkl"),
            "speaker": _load(probe_path / "label_encoder_speaker.pkl"),
        }

    phoneme_enc, phoneme_mask = _encode_known_labels(
        label_encoders["phoneme"],
        phoneme_labels,
        "phoneme",
    )
    speaker_enc, speaker_mask = _encode_known_labels(
        label_encoders["speaker"],
        speaker_labels,
        "speaker",
    )
    voiced_mask = ~np.isnan(pitch_values)

    results: Dict[str, List[float]] = {
        "phoneme_acc": [], "phoneme_f1": [],
        "speaker_acc": [], "speaker_f1": [],
        "pitch_mae":   [], "pitch_r2":   [],
    }

    for layer_idx in range(NUM_LAYERS):
        X         = embeddings_by_layer[layer_idx]
        layer_num = layer_idx + 1

        # Phoneme
        probe  = _load(probe_path / f"probe_{codec_name}_layer{layer_num}_phoneme.pkl")
        if phoneme_enc.size > 0:
            y_pred = probe.predict(X[phoneme_mask])
            results["phoneme_acc"].append(accuracy_score(phoneme_enc, y_pred))
            results["phoneme_f1"].append(
                f1_score(phoneme_enc, y_pred, average="macro", zero_division=0)
            )
        else:
            results["phoneme_acc"].append(np.nan)
            results["phoneme_f1"].append(np.nan)

        # Speaker
        probe  = _load(probe_path / f"probe_{codec_name}_layer{layer_num}_speaker.pkl")
        if speaker_enc.size > 0:
            y_pred = probe.predict(X[speaker_mask])
            results["speaker_acc"].append(accuracy_score(speaker_enc, y_pred))
            results["speaker_f1"].append(
                f1_score(speaker_enc, y_pred, average="macro", zero_division=0)
            )
        else:
            results["speaker_acc"].append(np.nan)
            results["speaker_f1"].append(np.nan)

        # Pitch regression on voiced frames only
        probe = _load(probe_path / f"probe_{codec_name}_layer{layer_num}_pitch.pkl")
        if probe is not None and voiced_mask.sum() > 0:
            y_pred = probe.predict(X[voiced_mask])
            y_true = pitch_values[voiced_mask]
            results["pitch_mae"].append(mean_absolute_error(y_true, y_pred))
            results["pitch_r2"] .append(r2_score(y_true, y_pred))
        else:
            results["pitch_mae"].append(np.nan)
            results["pitch_r2"] .append(np.nan)

        print(f"  [{codec_name}] layer {layer_num}/8 evaluated")

    return results
