"""
Utterance-level stratified train/eval split for LibriSpeech.

Splits are saved to a JSON file on first run and reloaded on subsequent
runs so the exact same utterances are always in train vs eval.
"""

import json
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Tuple

from data._librispeech_layout import validate_librispeech_entry

import numpy as np

# (flac_path, speaker_id, utterance_id)
UttEntry = Tuple[str, str, str]


def scan_utterance_paths(librispeech_root: str, split: str) -> List[UttEntry]:
    """Return a sorted list of (flac_path, speaker_id, utterance_id) for a split."""
    split_root = Path(librispeech_root) / split
    if not split_root.exists():
        raise FileNotFoundError(f"LibriSpeech split path does not exist: {split_root}")
    if not split_root.is_dir():
        raise NotADirectoryError(f"LibriSpeech split path is not a directory: {split_root}")

    entries: List[UttEntry] = []

    for flac_path in split_root.rglob("*.flac"):
        speaker_id, _chapter_id, utterance_id = validate_librispeech_entry(flac_path, split_root)
        entries.append((str(flac_path), speaker_id, utterance_id))
    entries.sort(key=lambda x: x[2])   # deterministic order by utterance ID
    return entries


def split_utterances(
    entries: List[UttEntry],
    eval_frac: float,
    seed: int,
    save_path: Path,
    force: bool = False,
) -> Tuple[List[UttEntry], List[UttEntry]]:
    """
    Stratified utterance-level split: each speaker contributes ~eval_frac
    of their utterances to the eval set.

    The split is saved to save_path (JSON) on first call so that results are
    reproducible across runs. Pass force=True to recompute from scratch.
    """
    if not 0 <= eval_frac < 1:
        raise ValueError(f"eval_frac must be in [0, 1), current value: {eval_frac}")

    if save_path.exists() and not force:
        try:
            with open(save_path) as f:
                saved = json.load(f)
        except JSONDecodeError as e:
            raise ValueError(f"Malformed split JSON at {save_path}: {e}") from e

        if not isinstance(saved, dict) or "train" not in saved or "eval" not in saved:
            raise ValueError(
                f"Invalid split JSON at {save_path}: expected keys 'train' and 'eval'"
            )
        if not isinstance(saved["train"], list) or not isinstance(saved["eval"], list):
            raise ValueError(
                f"Invalid split JSON at {save_path}: 'train' and 'eval' must be lists"
            )

        for uid in saved["train"] + saved["eval"]:
            if not isinstance(uid, str):
                raise ValueError(
                    f"Invalid split JSON at {save_path}: all utterance IDs must be strings"
                )

        combined = saved["train"] + saved["eval"]
        if len(combined) != len(set(combined)):
            raise ValueError(
                f"Invalid split JSON at {save_path}: utterance IDs must be unique"
            )

        overlap = set(saved["train"]) & set(saved["eval"])
        if overlap:
            raise ValueError(
                f"Invalid split JSON at {save_path}: 'train' and 'eval' must be disjoint"
            )

        by_id = {e[2]: e for e in entries}
        saved_ids = set(saved["train"]) | set(saved["eval"])
        current_ids = set(by_id.keys())
        missing_ids = sorted(current_ids - saved_ids)
        unknown_ids = sorted(saved_ids - current_ids)
        if missing_ids or unknown_ids:
            raise ValueError(
                "Invalid split JSON at "
                f"{save_path}: cached IDs do not match current entries "
                f"(missing={len(missing_ids)}, unknown={len(unknown_ids)}). "
                "Re-run with force=True (--force_resplit) to recompute the split."
            )

        train = [by_id[uid] for uid in saved["train"]]
        eval_ = [by_id[uid] for uid in saved["eval"]]
        print(f"Loaded split from {save_path}  (train={len(train)}, eval={len(eval_)})")
        return train, eval_

    rng = np.random.default_rng(seed)

    by_speaker: Dict[str, List[UttEntry]] = defaultdict(list)
    for entry in entries:
        by_speaker[entry[1]].append(entry)

    train: List[UttEntry] = []
    eval_: List[UttEntry] = []
    for spk_entries in by_speaker.values():
        shuffled = list(spk_entries)
        rng.shuffle(shuffled)
        n_utt = len(shuffled)
        # Keep approximately eval_frac in eval while preserving at least one
        # training utterance for speakers that have multiple utterances.
        if n_utt <= 1 or eval_frac == 0.0:
            n_eval = 0
        else:
            n_eval = max(1, round(n_utt * eval_frac))
            n_eval = min(n_eval, n_utt - 1)
        eval_.extend(shuffled[:n_eval])
        train.extend(shuffled[n_eval:])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(
            {
                "seed":      seed,
                "eval_frac": eval_frac,
                "train":     [e[2] for e in train],
                "eval":      [e[2] for e in eval_],
            },
            f,
            indent=2,
        )
    print(f"Split saved to {save_path}  (train={len(train)}, eval={len(eval_)})")
    return train, eval_
