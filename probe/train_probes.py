"""
Train one linear probe per (codec, RVQ layer, task) on frozen token embeddings.

Tasks:
  phoneme → LogisticRegression  (class_weight="balanced" for skewed phoneme dist.)
  speaker → LogisticRegression
  pitch   → LinearRegression    (voiced frames only; unvoiced tokens are NaN)

Codec parameters are never updated — embeddings are treated as fixed features.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

NUM_LAYERS = 8


# ── Label encoder fitting ──────────────────────────────────────────────────────

def fit_label_encoders(
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    probe_dir: Path,
) -> Dict[str, LabelEncoder]:
    """
    Fit LabelEncoders for phoneme and speaker on training data and save to disk.

    Must be called once before train_probes or evaluate_probes so that both
    codecs and both splits share identical integer mappings.
    """
    encoders: Dict[str, LabelEncoder] = {}
    for task, labels in [("phoneme", phoneme_labels), ("speaker", speaker_labels)]:
        le = LabelEncoder()
        le.fit(labels)
        encoders[task] = le
        with open(probe_dir / f"label_encoder_{task}.pkl", "wb") as f:
            pickle.dump(le, f)
    return encoders


# ── Probe fitting helpers ──────────────────────────────────────────────────────

def _fit_classification_probe(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    # class_weight="balanced" compensates for skewed phoneme/speaker distributions
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def _fit_regression_probe(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


# ── Main training function ─────────────────────────────────────────────────────

def train_probes(
    embeddings_by_layer: List[np.ndarray],
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    pitch_values: np.ndarray,
    codec_name: str,
    output_dir: str,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> None:
    """
    Train and save 3 probes × 8 layers = 24 probes for one codec.

    Args:
        embeddings_by_layer: List of 8 arrays, shape (N_tokens, embed_dim).
        phoneme_labels:      String phoneme labels, shape (N_tokens,).
        speaker_labels:      String speaker IDs, shape (N_tokens,).
        pitch_values:        Float Hz values, shape (N_tokens,); NaN = unvoiced.
        codec_name:          "encodec" or "speechtokenizer" — used in filenames.
        output_dir:          Directory for saved .pkl probe files.
        label_encoders:      Pre-fitted encoders from fit_label_encoders().
                             If None, loads from output_dir (backward compat).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if label_encoders is None:
        with open(out / "label_encoder_phoneme.pkl", "rb") as f:
            label_encoders = {"phoneme": pickle.load(f)}
        with open(out / "label_encoder_speaker.pkl", "rb") as f:
            label_encoders["speaker"] = pickle.load(f)

    phoneme_enc = label_encoders["phoneme"].transform(phoneme_labels)
    speaker_enc = label_encoders["speaker"].transform(speaker_labels)
    voiced_mask = ~np.isnan(pitch_values)

    for layer_idx in range(NUM_LAYERS):
        X        = embeddings_by_layer[layer_idx]
        layer_num = layer_idx + 1   # 1-indexed for readability in filenames

        phoneme_probe = _fit_classification_probe(X, phoneme_enc)
        with open(out / f"probe_{codec_name}_layer{layer_num}_phoneme.pkl", "wb") as f:
            pickle.dump(phoneme_probe, f)

        speaker_probe = _fit_classification_probe(X, speaker_enc)
        with open(out / f"probe_{codec_name}_layer{layer_num}_speaker.pkl", "wb") as f:
            pickle.dump(speaker_probe, f)

        # Pitch regression only on voiced frames
        pitch_probe = (
            _fit_regression_probe(X[voiced_mask], pitch_values[voiced_mask])
            if voiced_mask.sum() > 0 else None
        )
        with open(out / f"probe_{codec_name}_layer{layer_num}_pitch.pkl", "wb") as f:
            pickle.dump(pitch_probe, f)

        print(f"  [{codec_name}] layer {layer_num}/8 trained")
