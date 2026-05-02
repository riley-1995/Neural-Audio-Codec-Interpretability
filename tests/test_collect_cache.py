"""
Tests for the NPZ cache helpers in encode/collect.py.

Run: pytest tests/test_collect_cache.py -v

Note: tests/test_load_alignments.py is a plain Python script (not pytest);
migrating it is tracked separately.
"""

import numpy as np
import pytest

from encode.collect import NUM_LAYERS, _get_utterance_data, _load_cached, _save_cache


def _make_embeddings(num_tokens: int = 3, embed_dim: int = 4) -> list:
    return [np.random.rand(num_tokens, embed_dim).astype(np.float32) for _ in range(NUM_LAYERS)]


def test_round_trip(tmp_path):
    embeddings = _make_embeddings()
    phonemes   = np.array(["AH", "B", "SIL"])
    pitches    = np.array([120.0, np.nan, 200.0], dtype=np.float32)

    _save_cache(tmp_path, "encodec", "utt-001", embeddings, phonemes, pitches)
    result = _load_cached(tmp_path, "encodec", "utt-001")

    assert result is not None
    loaded_embeddings, loaded_phonemes, loaded_pitches = result
    assert loaded_phonemes is not None
    assert loaded_pitches is not None

    for i in range(NUM_LAYERS):
        np.testing.assert_array_equal(loaded_embeddings[i], embeddings[i])
    np.testing.assert_array_equal(loaded_phonemes, phonemes)
    np.testing.assert_array_equal(loaded_pitches[~np.isnan(pitches)],
                                  pitches[~np.isnan(pitches)])
    np.testing.assert_array_equal(np.isnan(loaded_pitches), np.isnan(pitches))


def test_backward_compat(tmp_path):
    embeddings = _make_embeddings()
    npz_path = tmp_path / "encodec" / "utt-002.npz"
    npz_path.parent.mkdir(parents=True)
    np.savez(npz_path, **{f"layer_{i}": emb for i, emb in enumerate(embeddings)})

    result = _load_cached(tmp_path, "encodec", "utt-002")

    assert result is not None
    loaded_embeddings, loaded_phonemes, loaded_pitches = result
    for i in range(NUM_LAYERS):
        np.testing.assert_array_equal(loaded_embeddings[i], embeddings[i])
    assert loaded_phonemes is None
    assert loaded_pitches is None


def test_missing_file(tmp_path):
    assert _load_cached(tmp_path, "encodec", "utt-999") is None


def test_partial_cache_missing_pitches_loads_as_none(tmp_path):
    embeddings = _make_embeddings()
    phonemes = np.array(["AH", "B", "SIL"])
    npz_path = tmp_path / "encodec" / "utt-003.npz"
    npz_path.parent.mkdir(parents=True)
    np.savez(
        npz_path,
        **{f"layer_{i}": emb for i, emb in enumerate(embeddings)},
        phonemes=phonemes,
    )

    result = _load_cached(tmp_path, "encodec", "utt-003")

    assert result is not None
    loaded_embeddings, loaded_phonemes, loaded_pitches = result
    for i in range(NUM_LAYERS):
        np.testing.assert_array_equal(loaded_embeddings[i], embeddings[i])
    np.testing.assert_array_equal(loaded_phonemes, phonemes)
    assert loaded_pitches is None


def test_get_utterance_data_backfills_missing_pitches(tmp_path, monkeypatch):
    embeddings = _make_embeddings()
    cached = (embeddings, np.array(["AH", "B", "SIL"]), None)
    expected_phones = np.array(["AA", "BB", "CC"])
    expected_pitches = np.array([101.0, np.nan, 205.0], dtype=np.float32)

    def _fake_extract_features(*_args, **_kwargs):
        return expected_phones, expected_pitches

    monkeypatch.setattr("encode.collect._extract_features", _fake_extract_features)

    result = _get_utterance_data(
        "encodec",
        cached,
        tmp_path,
        "utt-004",
        encode_fn=None,
        model=None,
        audio=np.array([0.0], dtype=np.float32),
        sr=16000,
        phone_intervals=[],
        token_rate=75.0,
    )

    assert result is not None
    loaded_embeddings, loaded_phones, loaded_pitches = result
    for i in range(NUM_LAYERS):
        np.testing.assert_array_equal(loaded_embeddings[i], embeddings[i])
    np.testing.assert_array_equal(loaded_phones, expected_phones)
    np.testing.assert_array_equal(np.isnan(loaded_pitches), np.isnan(expected_pitches))
    np.testing.assert_array_equal(
        loaded_pitches[~np.isnan(expected_pitches)],
        expected_pitches[~np.isnan(expected_pitches)],
    )

    cached_after = _load_cached(tmp_path, "encodec", "utt-004")
    assert cached_after is not None
    _, phones_after, pitches_after = cached_after
    assert phones_after is not None
    assert pitches_after is not None
    np.testing.assert_array_equal(phones_after, expected_phones)
    np.testing.assert_array_equal(np.isnan(pitches_after), np.isnan(expected_pitches))
