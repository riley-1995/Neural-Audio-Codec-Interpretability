"""
Shared helpers for validating the LibriSpeech directory layout.

Expected on-disk structure::

    <root>/<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utterance>.flac

Both ``data.split`` and ``data.load_librispeech`` rely on this contract;
keeping the checks in one place ensures both code paths stay in sync.
"""

from pathlib import Path
from typing import Tuple


def validate_librispeech_entry(flac_path: Path, split_dir: Path) -> Tuple[str, str, str]:
    """Validate a single ``.flac`` path and return ``(speaker_id, chapter_id, utterance_id)``.

    Parameters
    ----------
    flac_path:
        Absolute path to the ``.flac`` file.
    split_dir:
        Root of the split directory (``<root>/<split>``).

    Raises
    ------
    ValueError
        If the path depth, filename format, or path/filename prefix mismatch
        does not conform to the LibriSpeech layout.
    """
    rel_path = flac_path.relative_to(split_dir)
    if len(rel_path.parts) != 3:
        raise ValueError(
            f"Malformed LibriSpeech path: {flac_path}. "
            "Expected <split>/<speaker>/<chapter>/<utterance>.flac"
        )

    speaker_id = rel_path.parts[0]
    chapter_id = rel_path.parts[1]

    parts = flac_path.stem.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Malformed LibriSpeech utterance filename: {flac_path.name}. "
            "Expected <speaker>-<chapter>-<utterance>.flac"
        )
    if parts[0] != speaker_id or parts[1] != chapter_id:
        raise ValueError(
            f"LibriSpeech path/filename mismatch for {flac_path}: "
            f"directory speaker/chapter is {speaker_id}/{chapter_id}, "
            f"but filename starts with {parts[0]}/{parts[1]}"
        )

    return speaker_id, chapter_id, flac_path.stem
