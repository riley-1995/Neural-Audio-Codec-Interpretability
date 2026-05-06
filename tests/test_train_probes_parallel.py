"""Tests for parallel probe training dispatch and artifact compatibility."""

import numpy as np
import pytest

from probe.evaluate_probes import evaluate_probes
from probe.train_probes import (
    MAX_TRAIN_TOKENS,
    NUM_LAYERS,
    ProbeTrainingBundle,
    ProbeTrainingError,
    _fit_classification_probe,
    fit_label_encoders,
    train_probes,
    train_probes_for_codecs,
)


def _make_labels(num_tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phonemes = np.array(["AA" if i % 2 == 0 else "BB" for i in range(num_tokens)])
    speakers = np.array(["S1" if i % 3 == 0 else "S2" for i in range(num_tokens)])
    pitches = np.array([120.0 + i for i in range(num_tokens)], dtype=np.float32)
    pitches[::3] = np.nan
    return phonemes, speakers, pitches


def _make_bundle(codec_name: str, num_tokens: int = 12, embed_dim: int = 5) -> ProbeTrainingBundle:
    rng = np.random.default_rng(123)
    embeddings = [
        rng.normal(size=(num_tokens, embed_dim)).astype(np.float32)
        for _ in range(NUM_LAYERS)
    ]
    phonemes, speakers, pitches = _make_labels(num_tokens)
    return ProbeTrainingBundle(
        codec_name=codec_name,
        embeddings_by_layer=embeddings,
        phoneme_labels=phonemes,
        speaker_labels=speakers,
        pitch_values=pitches,
    )


def test_train_probes_for_codec_writes_expected_artifacts(tmp_path, monkeypatch):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    def _fake_classification(X, y, max_iter=1000):
        return {"kind": "classification", "rows": int(X.shape[0]), "classes": int(np.unique(y).size)}

    def _fake_regression(X, y):
        return {"kind": "regression", "rows": int(X.shape[0]), "targets": int(y.shape[0])}

    monkeypatch.setattr("probe.train_probes._fit_classification_probe", _fake_classification)
    monkeypatch.setattr("probe.train_probes._fit_regression_probe", _fake_regression)

    train_probes_for_codecs(
        bundles=[bundle],
        output_dir=str(tmp_path),
        label_encoders=encoders,
        max_workers=4,
    )

    for layer_num in range(1, NUM_LAYERS + 1):
        for task in ("phoneme", "speaker", "pitch"):
            assert (tmp_path / f"probe_encodec_layer{layer_num}_{task}.pkl").exists()


def test_train_probes_for_codecs_writes_48_probe_files(tmp_path, monkeypatch):
    enc_bundle = _make_bundle("encodec")
    st_bundle = _make_bundle("speechtokenizer")
    all_phonemes = np.concatenate([enc_bundle.phoneme_labels, st_bundle.phoneme_labels])
    all_speakers = np.concatenate([enc_bundle.speaker_labels, st_bundle.speaker_labels])
    encoders = fit_label_encoders(all_phonemes, all_speakers, tmp_path)

    monkeypatch.setattr(
        "probe.train_probes._fit_classification_probe",
        lambda X, y, max_iter=1000: {"ok": True, "n": int(X.shape[0]), "max_iter": int(max_iter)},
    )
    monkeypatch.setattr("probe.train_probes._fit_regression_probe", lambda X, y: {"ok": True, "n": int(X.shape[0])})

    train_probes_for_codecs(
        bundles=[enc_bundle, st_bundle],
        output_dir=str(tmp_path),
        label_encoders=encoders,
        max_workers=6,
    )

    files = list(tmp_path.glob("probe_*_layer*_*.pkl"))
    assert len(files) == 48


def test_train_probes_for_codecs_skips_existing_probe_files(tmp_path, monkeypatch):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    call_counts = {"classification": 0, "regression": 0}

    def _fake_classification(X, y, max_iter=1000):
        call_counts["classification"] += 1
        return {"ok": True, "rows": int(X.shape[0]), "max_iter": int(max_iter)}

    def _fake_regression(X, y):
        call_counts["regression"] += 1
        return {"ok": True, "rows": int(X.shape[0])}

    monkeypatch.setattr("probe.train_probes._fit_classification_probe", _fake_classification)
    monkeypatch.setattr("probe.train_probes._fit_regression_probe", _fake_regression)

    train_probes_for_codecs(
        bundles=[bundle],
        output_dir=str(tmp_path),
        label_encoders=encoders,
        max_workers=3,
    )
    assert call_counts["classification"] == NUM_LAYERS * 2
    assert call_counts["regression"] == NUM_LAYERS

    call_counts["classification"] = 0
    call_counts["regression"] = 0

    train_probes_for_codecs(
        bundles=[bundle],
        output_dir=str(tmp_path),
        label_encoders=encoders,
        max_workers=3,
    )
    assert call_counts["classification"] == 0
    assert call_counts["regression"] == 0


def test_train_probes_for_codecs_surfaces_failures(tmp_path, monkeypatch):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    def _fake_run_probe_job(job):
        if job.layer_num == 2 and job.task == "speaker":
            raise RuntimeError("intentional failure")
        return "ok"

    monkeypatch.setattr("probe.train_probes._run_probe_job", _fake_run_probe_job)

    with pytest.raises(ProbeTrainingError, match=r"layer 2/8 speaker"):
        train_probes_for_codecs(
            bundles=[bundle],
            output_dir=str(tmp_path),
            label_encoders=encoders,
            max_workers=3,
        )


def test_train_probes_artifacts_are_compatible_with_evaluate(tmp_path):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    train_probes(
        embeddings_by_layer=bundle.embeddings_by_layer,
        phoneme_labels=bundle.phoneme_labels,
        speaker_labels=bundle.speaker_labels,
        pitch_values=bundle.pitch_values,
        codec_name=bundle.codec_name,
        output_dir=str(tmp_path),
        label_encoders=encoders,
        max_workers=2,
    )

    results = evaluate_probes(
        embeddings_by_layer=bundle.embeddings_by_layer,
        phoneme_labels=bundle.phoneme_labels,
        speaker_labels=bundle.speaker_labels,
        pitch_values=bundle.pitch_values,
        codec_name=bundle.codec_name,
        probe_dir=str(tmp_path),
        label_encoders=encoders,
    )

    expected_keys = {
        "phoneme_acc",
        "phoneme_f1",
        "speaker_acc",
        "speaker_f1",
        "pitch_mae",
        "pitch_r2",
    }
    assert set(results.keys()) == expected_keys
    assert all(len(values) == NUM_LAYERS for values in results.values())


def test_train_probes_for_codecs_rejects_invalid_worker_count(tmp_path):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        train_probes_for_codecs(
            bundles=[bundle],
            output_dir=str(tmp_path),
            label_encoders=encoders,
            max_workers=0,
        )


def test_train_probes_for_codecs_rejects_invalid_blas_threads(tmp_path):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    with pytest.raises(ValueError, match="blas_threads must be >= 0"):
        train_probes_for_codecs(
            bundles=[bundle],
            output_dir=str(tmp_path),
            label_encoders=encoders,
            max_workers=2,
            blas_threads=-1,
        )


def test_train_probes_for_codecs_rejects_invalid_max_iter(tmp_path):
    bundle = _make_bundle("encodec")
    encoders = fit_label_encoders(bundle.phoneme_labels, bundle.speaker_labels, tmp_path)

    with pytest.raises(ValueError, match="classification_max_iter must be >= 1"):
        train_probes_for_codecs(
            bundles=[bundle],
            output_dir=str(tmp_path),
            label_encoders=encoders,
            max_workers=2,
            classification_max_iter=0,
        )


def test_fit_classification_probe_subsamples_to_cap_deterministically(monkeypatch):
    total_rows = MAX_TRAIN_TOKENS + 137
    X = np.arange(total_rows * 3, dtype=np.float32).reshape(total_rows, 3)
    y = (np.arange(total_rows) % 11).astype(np.int32)
    fit_calls = []

    class _FakeLogisticRegression:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X_fit, y_fit):
            fit_calls.append((X_fit.copy(), y_fit.copy()))
            return self

    monkeypatch.setattr("probe.train_probes.LogisticRegression", _FakeLogisticRegression)

    _fit_classification_probe(X, y, max_iter=321)
    _fit_classification_probe(X, y, max_iter=321)

    assert len(fit_calls) == 2
    for X_fit, y_fit in fit_calls:
        assert X_fit.shape == (MAX_TRAIN_TOKENS, 3)
        assert y_fit.shape == (MAX_TRAIN_TOKENS,)

    np.testing.assert_array_equal(fit_calls[0][0], fit_calls[1][0])
    np.testing.assert_array_equal(fit_calls[0][1], fit_calls[1][1])
