"""
Utterance-level stratified train/eval split for LibriSpeech.

Splits are saved to a JSON file on first run and reloaded on subsequent
runs so the exact same utterances are always in train vs eval.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# (flac_path, speaker_id, utterance_id)
UttEntry = Tuple[str, str, str]


def scan_utterance_paths(librispeech_root: str, split: str) -> List[UttEntry]:
    """Return a sorted list of (flac_path, speaker_id, utterance_id) for a split."""
    split_root = Path(librispeech_root) / split
    entries: List[UttEntry] = []

    for flac_path in split_root.rglob("*.flac"):
        rel_path = flac_path.relative_to(split_root)
        speaker_id = rel_path.parts[0]
        utterance_id = flac_path.stem
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
    if save_path.exists() and not force:
        with open(save_path) as f:
            saved = json.load(f)
        by_id = {e[2]: e for e in entries}
        train = [by_id[uid] for uid in saved["train"] if uid in by_id]
        eval_ = [by_id[uid] for uid in saved["eval"]  if uid in by_id]
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
        if n_utt <= 1:
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
