"""
Load LibriSpeech utterances and return (audio, sample_rate, speaker_id, utterance_id).
Expects LibriSpeech data at LIBRISPEECH_ROOT in the standard directory layout:
  <root>/<split>/<speaker_id>/<chapter_id>/<utterance>.flac
"""

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
    if not split_dir.is_dir():
        raise NotADirectoryError(f"LibriSpeech split is not a directory: {split_dir}")

    for flac_path in sorted(split_dir.rglob("*.flac")):
        parts = flac_path.stem.split("-")
        if len(parts) < 3:
            raise ValueError(
                f"Malformed LibriSpeech utterance filename: {flac_path.name}. "
                "Expected <speaker>-<chapter>-<utterance>.flac"
            )
        speaker_id = parts[0]
        utterance_id = flac_path.stem

        try:
            audio, sr = torchaudio.load(str(flac_path))
        except Exception as e:  # pragma: no cover - defensive context wrapper
            raise RuntimeError(f"Failed to load audio file {flac_path}: {e}") from e

        yield Utterance(
            audio=audio,
            sample_rate=sr,
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            flac_path=str(flac_path),
        )
