"""
Train one probe per (codec, RVQ layer, task) on frozen token embeddings.

Tasks:
  - phoneme  → LogisticRegression  → saved as probe_<codec>_layer<N>_phoneme.pkl
  - speaker  → LogisticRegression  → saved as probe_<codec>_layer<N>_speaker.pkl
  - pitch    → LinearRegression    → saved as probe_<codec>_layer<N>_pitch.pkl

All codec parameters stay frozen — we never update them.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

NUM_LAYERS = 8


def _fit_classification_probe(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def _fit_regression_probe(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


def train_probes(
    embeddings_by_layer: List[np.ndarray],
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    pitch_values: np.ndarray,
    codec_name: str,
    output_dir: str,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Dict[str, LabelEncoder]:
    """
    Train and save 3 probes × 8 layers = 24 probes for one codec.

    Args:
        embeddings_by_layer: List of 8 arrays, each shape (N_tokens, embed_dim).
                             These are the TRAINING set embeddings.
        phoneme_labels:      Array of shape (N_tokens,) with string phoneme labels.
        speaker_labels:      Array of shape (N_tokens,) with string speaker IDs.
        pitch_values:        Array of shape (N_tokens,) with float Hz or NaN for unvoiced.
        codec_name:          "encodec" or "speechtokenizer" — used in output filenames.
        output_dir:          Directory to save .pkl probe files.
        label_encoders:      Optional pre-fitted encoders (pass in for reuse across splits).

    Returns:
        label_encoders: Dict with keys "phoneme" and "speaker", each a fitted LabelEncoder.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fit label encoders on training data
    if label_encoders is None:
        label_encoders = {}
        for task, labels in [("phoneme", phoneme_labels), ("speaker", speaker_labels)]:
            le = LabelEncoder()
            le.fit(labels)
            label_encoders[task] = le
            # Save label encoder alongside probes
            with open(out_path / f"label_encoder_{task}.pkl", "wb") as f:
                pickle.dump(le, f)

    phoneme_encoded = label_encoders["phoneme"].transform(phoneme_labels)
    speaker_encoded = label_encoders["speaker"].transform(speaker_labels)

    # Voiced-frame mask for pitch regression
    voiced_mask = ~np.isnan(pitch_values)

    for layer_idx in range(NUM_LAYERS):
        X = embeddings_by_layer[layer_idx]   # (N, D)
        layer_num = layer_idx + 1            # 1-indexed for readability

        # --- Phoneme probe ---
        phoneme_probe = _fit_classification_probe(X, phoneme_encoded)
        with open(out_path / f"probe_{codec_name}_layer{layer_num}_phoneme.pkl", "wb") as f:
            pickle.dump(phoneme_probe, f)

        # --- Speaker probe ---
        speaker_probe = _fit_classification_probe(X, speaker_encoded)
        with open(out_path / f"probe_{codec_name}_layer{layer_num}_speaker.pkl", "wb") as f:
            pickle.dump(speaker_probe, f)

        # --- Pitch probe (voiced frames only) ---
        if voiced_mask.sum() > 0:
            pitch_probe = _fit_regression_probe(X[voiced_mask], pitch_values[voiced_mask])
        else:
            pitch_probe = None
        with open(out_path / f"probe_{codec_name}_layer{layer_num}_pitch.pkl", "wb") as f:
            pickle.dump(pitch_probe, f)

        print(f"  [{codec_name}] layer {layer_num}/8 probes trained")

    return label_encoders


def load_probe(probe_path: str):
    with open(probe_path, "rb") as f:
        return pickle.load(f)
