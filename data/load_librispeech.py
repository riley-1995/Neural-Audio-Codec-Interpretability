"""
Load LibriSpeech utterances and return (audio, sample_rate, speaker_id, utterance_id).
Expects LibriSpeech data at LIBRISPEECH_ROOT in the standard directory layout:
  <root>/<split>/<speaker_id>/<chapter_id>/<utterance>.flac
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torchaudio


@dataclass
class Utterance:
    audio: torch.Tensor      # shape (1, T) — mono, original sample rate
    sample_rate: int
    speaker_id: str
    utterance_id: str        # e.g. "1234-5678-0001"
    flac_path: str


def iter_librispeech(root: str, split: str = "train-clean-100") -> Iterator[Utterance]:
    """Yield Utterance objects for every .flac file under root/split."""
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"LibriSpeech split not found at {split_dir}")

    for flac_path in sorted(split_dir.rglob("*.flac")):
        parts = flac_path.stem.split("-")
        speaker_id = parts[0]
        utterance_id = flac_path.stem

        audio, sr = torchaudio.load(str(flac_path))
        yield Utterance(
            audio=audio,
            sample_rate=sr,
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            flac_path=str(flac_path),
        )
