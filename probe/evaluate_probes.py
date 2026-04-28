"""
Evaluate saved probes on held-out embeddings and return layer-wise metrics.

Metrics:
  - phoneme/speaker:  accuracy, macro-F1
  - pitch:            MAE (Hz), R²
"""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

NUM_LAYERS = 8


def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_probes(
    embeddings_by_layer: List[np.ndarray],
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    pitch_values: np.ndarray,
    codec_name: str,
    probe_dir: str,
) -> Dict[str, List[float]]:
    """
    Evaluate all 24 probes for one codec on the held-out split.

    Args:
        embeddings_by_layer: List of 8 arrays, shape (N_tokens, embed_dim) — eval set.
        phoneme_labels:      String phoneme labels for eval tokens.
        speaker_labels:      String speaker IDs for eval tokens.
        pitch_values:        Float Hz array (NaN for unvoiced) for eval tokens.
        codec_name:          "encodec" or "speechtokenizer".
        probe_dir:           Directory containing saved .pkl probe files.

    Returns:
        results: Dict with keys:
            "phoneme_acc"  → list of 8 floats (accuracy per layer)
            "phoneme_f1"   → list of 8 floats (macro-F1 per layer)
            "speaker_acc"  → list of 8 floats
            "speaker_f1"   → list of 8 floats
            "pitch_mae"    → list of 8 floats (MAE in Hz)
            "pitch_r2"     → list of 8 floats
    """
    probe_path = Path(probe_dir)

    le_phoneme = _load(probe_path / "label_encoder_phoneme.pkl")
    le_speaker = _load(probe_path / "label_encoder_speaker.pkl")

    phoneme_encoded = le_phoneme.transform(phoneme_labels)
    speaker_encoded = le_speaker.transform(speaker_labels)
    voiced_mask = ~np.isnan(pitch_values)

    results = {
        "phoneme_acc": [], "phoneme_f1": [],
        "speaker_acc": [], "speaker_f1": [],
        "pitch_mae":   [], "pitch_r2":   [],
    }

    for layer_idx in range(NUM_LAYERS):
        X = embeddings_by_layer[layer_idx]
        layer_num = layer_idx + 1

        # Phoneme
        phoneme_probe = _load(probe_path / f"probe_{codec_name}_layer{layer_num}_phoneme.pkl")
        y_pred = phoneme_probe.predict(X)
        results["phoneme_acc"].append(accuracy_score(phoneme_encoded, y_pred))
        results["phoneme_f1"].append(f1_score(phoneme_encoded, y_pred, average="macro", zero_division=0))

        # Speaker
        speaker_probe = _load(probe_path / f"probe_{codec_name}_layer{layer_num}_speaker.pkl")
        y_pred = speaker_probe.predict(X)
        results["speaker_acc"].append(accuracy_score(speaker_encoded, y_pred))
        results["speaker_f1"].append(f1_score(speaker_encoded, y_pred, average="macro", zero_division=0))

        # Pitch
        pitch_probe = _load(probe_path / f"probe_{codec_name}_layer{layer_num}_pitch.pkl")
        if pitch_probe is not None and voiced_mask.sum() > 0:
            X_voiced = X[voiced_mask]
            y_true = pitch_values[voiced_mask]
            y_pred = pitch_probe.predict(X_voiced)
            results["pitch_mae"].append(mean_absolute_error(y_true, y_pred))
            results["pitch_r2"].append(r2_score(y_true, y_pred))
        else:
            results["pitch_mae"].append(np.nan)
            results["pitch_r2"].append(np.nan)

        print(f"  [{codec_name}] layer {layer_num}/8 evaluated")

    return results
