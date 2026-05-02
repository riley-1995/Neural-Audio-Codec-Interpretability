import json

import pytest
import torch

from data.load_librispeech import iter_librispeech
from data.split import scan_utterance_paths, split_utterances


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_scan_utterance_paths_sorted_and_speaker_extracted(tmp_path):
    split_root = tmp_path / "train-clean-100"
    _touch(split_root / "100" / "001" / "100-001-0003.flac")
    _touch(split_root / "100" / "001" / "100-001-0001.flac")
    _touch(split_root / "200" / "007" / "200-007-0001.flac")

    entries = scan_utterance_paths(str(tmp_path), "train-clean-100")

    assert [e[2] for e in entries] == ["100-001-0001", "100-001-0003", "200-007-0001"]
    assert [e[1] for e in entries] == ["100", "100", "200"]


def test_scan_utterance_paths_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scan_utterance_paths(str(tmp_path), "missing-split")


def test_scan_utterance_paths_not_directory_raises(tmp_path):
    split_file = tmp_path / "train-clean-100"
    split_file.touch()

    with pytest.raises(NotADirectoryError, match="not a directory"):
        scan_utterance_paths(str(tmp_path), "train-clean-100")


def test_split_utterances_deterministic_no_overlap_and_conservation(tmp_path):
    entries = []
    for speaker in ["100", "200", "300"]:
        for i in range(10):
            uid = f"{speaker}-001-{i:04d}"
            entries.append((f"/fake/{uid}.flac", speaker, uid))

    save_path_1 = tmp_path / "split_a.json"
    train_1, eval_1 = split_utterances(entries, eval_frac=0.2, seed=42, save_path=save_path_1)

    save_path_2 = tmp_path / "split_b.json"
    train_2, eval_2 = split_utterances(entries, eval_frac=0.2, seed=42, save_path=save_path_2)

    train_uids_1 = [e[2] for e in train_1]
    eval_uids_1 = [e[2] for e in eval_1]
    train_uids_2 = [e[2] for e in train_2]
    eval_uids_2 = [e[2] for e in eval_2]

    assert train_uids_1 == train_uids_2
    assert eval_uids_1 == eval_uids_2

    train_set = set(train_uids_1)
    eval_set = set(eval_uids_1)
    all_set = {e[2] for e in entries}

    assert train_set.isdisjoint(eval_set)
    assert train_set.union(eval_set) == all_set


def test_split_utterances_speaker_distribution_within_tolerance(tmp_path):
    entries = []
    for speaker in ["100", "200"]:
        for i in range(20):
            uid = f"{speaker}-001-{i:04d}"
            entries.append((f"/fake/{uid}.flac", speaker, uid))

    train, eval_ = split_utterances(
        entries,
        eval_frac=0.2,
        seed=123,
        save_path=tmp_path / "split.json",
    )

    for speaker in ["100", "200"]:
        total = sum(1 for _, spk, _ in entries if spk == speaker)
        eval_count = sum(1 for _, spk, _ in eval_ if spk == speaker)
        observed = eval_count / total
        assert abs(observed - 0.2) <= 0.1

    assert len(train) + len(eval_) == len(entries)


def test_split_utterances_single_utterance_speaker_stays_in_train(tmp_path):
    entries = [
        ("/fake/100-001-0001.flac", "100", "100-001-0001"),
        ("/fake/200-001-0001.flac", "200", "200-001-0001"),
        ("/fake/300-001-0001.flac", "300", "300-001-0001"),
        ("/fake/300-001-0002.flac", "300", "300-001-0002"),
    ]

    train, eval_ = split_utterances(
        entries,
        eval_frac=0.5,
        seed=7,
        save_path=tmp_path / "split.json",
    )

    train_uids = {e[2] for e in train}
    eval_uids = {e[2] for e in eval_}

    assert "100-001-0001" in train_uids
    assert "200-001-0001" in train_uids
    assert "100-001-0001" not in eval_uids
    assert "200-001-0001" not in eval_uids


def test_split_utterances_eval_frac_zero(tmp_path):
    entries = [(f"/fake/100-001-{i:04d}.flac", "100", f"100-001-{i:04d}") for i in range(8)]

    train, eval_ = split_utterances(
        entries,
        eval_frac=0.0,
        seed=5,
        save_path=tmp_path / "split.json",
    )

    assert len(train) == len(entries)
    assert len(eval_) == 0


def test_split_utterances_eval_frac_out_of_bounds(tmp_path):
    entries = [("/fake/100-001-0001.flac", "100", "100-001-0001")]

    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        split_utterances(entries, eval_frac=-0.1, seed=0, save_path=tmp_path / "neg.json")

    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        split_utterances(entries, eval_frac=1.0, seed=0, save_path=tmp_path / "one.json")


def test_split_utterances_reload_and_force_behavior(tmp_path):
    entries = [(f"/fake/100-001-{i:04d}.flac", "100", f"100-001-{i:04d}") for i in range(20)]
    save_path = tmp_path / "split.json"

    train_a, eval_a = split_utterances(entries, eval_frac=0.1, seed=42, save_path=save_path)
    train_b, eval_b = split_utterances(entries, eval_frac=0.4, seed=42, save_path=save_path, force=False)

    assert [e[2] for e in train_a] == [e[2] for e in train_b]
    assert [e[2] for e in eval_a] == [e[2] for e in eval_b]

    train_c, eval_c = split_utterances(entries, eval_frac=0.4, seed=42, save_path=save_path, force=True)
    assert len(eval_c) > len(eval_a)
    assert len(train_c) < len(train_a)


def test_split_utterances_malformed_saved_json_raises(tmp_path):
    entries = [("/fake/100-001-0001.flac", "100", "100-001-0001")]
    save_path = tmp_path / "split.json"
    save_path.write_text("{ malformed json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed split JSON"):
        split_utterances(entries, eval_frac=0.2, seed=1, save_path=save_path)


def test_split_utterances_missing_json_keys_raises(tmp_path):
    entries = [("/fake/100-001-0001.flac", "100", "100-001-0001")]
    save_path = tmp_path / "split.json"
    save_path.write_text(json.dumps({"train": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected keys 'train' and 'eval'"):
        split_utterances(entries, eval_frac=0.2, seed=1, save_path=save_path)


def test_iter_librispeech_yields_expected_fields(tmp_path, monkeypatch):
    split_root = tmp_path / "train-clean-100"
    _touch(split_root / "123" / "456" / "123-456-0001.flac")

    def fake_load(_path):
        return torch.zeros((1, 1600), dtype=torch.float32), 16000

    monkeypatch.setattr("data.load_librispeech.torchaudio.load", fake_load)

    utterances = list(iter_librispeech(str(tmp_path), "train-clean-100"))

    assert len(utterances) == 1
    u = utterances[0]
    assert u.speaker_id == "123"
    assert u.utterance_id == "123-456-0001"
    assert u.sample_rate == 16000
    assert tuple(u.audio.shape) == (1, 1600)


def test_iter_librispeech_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="split not found"):
        list(iter_librispeech(str(tmp_path), "missing"))


def test_iter_librispeech_malformed_filename_raises(tmp_path, monkeypatch):
    split_root = tmp_path / "train-clean-100"
    _touch(split_root / "123" / "456" / "badname.flac")

    def fake_load(_path):
        return torch.zeros((1, 1600), dtype=torch.float32), 16000

    monkeypatch.setattr("data.load_librispeech.torchaudio.load", fake_load)

    with pytest.raises(ValueError, match="Malformed LibriSpeech utterance filename"):
        list(iter_librispeech(str(tmp_path), "train-clean-100"))


def test_iter_librispeech_load_failure_has_context(tmp_path, monkeypatch):
    split_root = tmp_path / "train-clean-100"
    _touch(split_root / "123" / "456" / "123-456-0001.flac")

    def fake_load(_path):
        raise OSError("decode failure")

    monkeypatch.setattr("data.load_librispeech.torchaudio.load", fake_load)

    with pytest.raises(RuntimeError, match="Failed to load audio file"):
        list(iter_librispeech(str(tmp_path), "train-clean-100"))


def test_scan_split_pipeline_consistency(tmp_path):
    split_root = tmp_path / "train-clean-100"
    for speaker in ["100", "200"]:
        for i in range(5):
            _touch(split_root / speaker / "001" / f"{speaker}-001-{i:04d}.flac")

    entries = scan_utterance_paths(str(tmp_path), "train-clean-100")
    train, eval_ = split_utterances(
        entries,
        eval_frac=0.3,
        seed=11,
        save_path=tmp_path / "split.json",
    )

    assert len(entries) == 10
    assert len(train) + len(eval_) == 10
    assert set(e[2] for e in train).isdisjoint(set(e[2] for e in eval_))
