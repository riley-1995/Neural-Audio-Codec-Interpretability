"""
Tests for the NPZ cache helpers in encode/collect.py.

Run: pytest tests/test_collect_cache.py -v

Note: tests/test_load_alignments.py is a plain Python script (not pytest);
migrating it is tracked separately.
"""

import numpy as np
import pytest

from encode.collect import NUM_LAYERS, _get_utterance_data, _load_cached, _save_cache, collect_bundle


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


def test_corrupt_cache_file_is_dropped_and_rebuilt(tmp_path):
    npz_path = tmp_path / "encodec" / "utt-corrupt.npz"
    npz_path.parent.mkdir(parents=True)
    npz_path.write_bytes(b"")

    result = _load_cached(tmp_path, "encodec", "utt-corrupt")

    assert result is None
    assert not npz_path.exists()


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


def test_collect_bundle_uses_full_cache_without_alignment_or_audio(tmp_path, monkeypatch):
    enc_embeddings = _make_embeddings(num_tokens=3)
    st_embeddings = _make_embeddings(num_tokens=2)
    enc_phonemes = np.array(["AH", "B", "SIL"])
    st_phonemes = np.array(["D", "EH"])
    enc_pitches = np.array([120.0, np.nan, 200.0], dtype=np.float32)
    st_pitches = np.array([90.0, 95.0], dtype=np.float32)

    _save_cache(tmp_path, "encodec", "100-200-0001", enc_embeddings, enc_phonemes, enc_pitches)
    _save_cache(tmp_path, "speechtokenizer", "100-200-0001", st_embeddings, st_phonemes, st_pitches)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("warm-cache path should not access uncached inputs")

    monkeypatch.setattr("encode.collect.load_textgrid_phones", _should_not_run)
    monkeypatch.setattr("encode.collect.torchaudio.load", _should_not_run)
    monkeypatch.setattr("encode.collect.encodec_encode", _should_not_run)
    monkeypatch.setattr("encode.collect.st_encode", _should_not_run)

    enc_bundle, st_bundle = collect_bundle(
        [("missing.flac", "speaker-1", "100-200-0001")],
        encodec_model=None,
        st_model=None,
        alignments_root="missing-alignments",
        cache_dir=tmp_path,
    )

    for i in range(NUM_LAYERS):
        np.testing.assert_array_equal(enc_bundle["embeddings"][i], enc_embeddings[i])
        np.testing.assert_array_equal(st_bundle["embeddings"][i], st_embeddings[i])
    np.testing.assert_array_equal(enc_bundle["phonemes"], enc_phonemes)
    np.testing.assert_array_equal(st_bundle["phonemes"], st_phonemes)
    np.testing.assert_array_equal(np.isnan(enc_bundle["pitches"]), np.isnan(enc_pitches))
    np.testing.assert_array_equal(enc_bundle["pitches"][~np.isnan(enc_pitches)], enc_pitches[~np.isnan(enc_pitches)])
    np.testing.assert_array_equal(st_bundle["pitches"], st_pitches)
    np.testing.assert_array_equal(enc_bundle["speakers"], np.array(["speaker-1", "speaker-1", "speaker-1"]))
    np.testing.assert_array_equal(st_bundle["speakers"], np.array(["speaker-1", "speaker-1"]))


def test_collect_bundle_respects_attempt_cap_even_on_codec_failures(tmp_path, monkeypatch):
    entries = [
        (f"fake-{i}.flac", "speaker-1", f"100-200-{i:04d}")
        for i in range(5)
    ]
    enc_embeddings = _make_embeddings(num_tokens=1)
    attempts = 0

    def _fake_torchaudio_load(_path):
        nonlocal attempts
        attempts += 1
        return np.zeros((1, 320), dtype=np.float32), 16000

    def _fake_load_textgrid_phones(_path):
        return [("AA", 0.0, 0.02)]

    def _fake_extract_features(*_args, **_kwargs):
        return np.array(["AA"]), np.array([110.0], dtype=np.float32)

    def _fake_encodec(*_args, **_kwargs):
        return enc_embeddings, 1

    def _fake_st(*_args, **_kwargs):
        raise RuntimeError("forced SpeechTokenizer failure")

    monkeypatch.setattr("encode.collect.torchaudio.load", _fake_torchaudio_load)
    monkeypatch.setattr("encode.collect.load_textgrid_phones", _fake_load_textgrid_phones)
    monkeypatch.setattr("encode.collect._extract_features", _fake_extract_features)
    monkeypatch.setattr("encode.collect.encodec_encode", _fake_encodec)
    monkeypatch.setattr("encode.collect.st_encode", _fake_st)

    with pytest.raises(ValueError, match="No utterances were collected"):
        collect_bundle(
            entries,
            encodec_model=None,
            st_model=None,
            alignments_root="unused",
            cache_dir=tmp_path,
            max_utterances=2,
        )

    assert attempts == 2


def test_collect_bundle_skips_audio_decode_failures(tmp_path, monkeypatch):
    entries = [
        ("bad.flac", "speaker-1", "100-200-0001"),
        ("good.flac", "speaker-1", "100-200-0002"),
    ]
    embeddings = _make_embeddings(num_tokens=1)

    def _fake_torchaudio_load(path):
        if path == "bad.flac":
            raise RuntimeError("decode failed")
        return np.zeros((1, 320), dtype=np.float32), 16000

    def _fake_load_textgrid_phones(_path):
        return [("AA", 0.0, 0.02)]

    def _fake_extract_features(*_args, **_kwargs):
        return np.array(["AA"]), np.array([110.0], dtype=np.float32)

    def _fake_encodec(*_args, **_kwargs):
        return embeddings, 1

    def _fake_st(*_args, **_kwargs):
        return embeddings, 1

    monkeypatch.setattr("encode.collect.torchaudio.load", _fake_torchaudio_load)
    monkeypatch.setattr("encode.collect.load_textgrid_phones", _fake_load_textgrid_phones)
    monkeypatch.setattr("encode.collect._extract_features", _fake_extract_features)
    monkeypatch.setattr("encode.collect.encodec_encode", _fake_encodec)
    monkeypatch.setattr("encode.collect.st_encode", _fake_st)

    enc_bundle, st_bundle = collect_bundle(
        entries,
        encodec_model=None,
        st_model=None,
        alignments_root="unused",
        cache_dir=tmp_path,
        max_utterances=2,
    )

    assert enc_bundle["embeddings"][0].shape[0] == 1
    assert st_bundle["embeddings"][0].shape[0] == 1
    np.testing.assert_array_equal(enc_bundle["speakers"], np.array(["speaker-1"]))
    np.testing.assert_array_equal(st_bundle["speakers"], np.array(["speaker-1"]))

    assert (tmp_path / "_failed_audio" / "100-200-0001.txt").exists()

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("failed-audio cache should skip retrying unreadable utterances")

    monkeypatch.setattr("encode.collect.load_textgrid_phones", _should_not_run)
    monkeypatch.setattr("encode.collect.torchaudio.load", _should_not_run)
    monkeypatch.setattr("encode.collect.encodec_encode", _should_not_run)
    monkeypatch.setattr("encode.collect.st_encode", _should_not_run)

    enc_bundle_2, st_bundle_2 = collect_bundle(
        entries,
        encodec_model=None,
        st_model=None,
        alignments_root="unused",
        cache_dir=tmp_path,
        max_utterances=2,
    )

    assert enc_bundle_2["embeddings"][0].shape[0] == 1
    assert st_bundle_2["embeddings"][0].shape[0] == 1
