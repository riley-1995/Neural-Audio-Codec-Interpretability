"""
Load Montreal Forced Aligner (MFA) TextGrid alignments for LibriSpeech.

Expected alignment file layout (MFA default output):
  <alignments_root>/<speaker_id>/<chapter_id>/<utterance_id>.TextGrid

Each TextGrid has a phoneme tier named "phones" with intervals like:
  xmin = 0.0, xmax = 0.05, text = "SIL"
  xmin = 0.05, xmax = 0.12, text = "HH"
  ...

Download pre-aligned LibriSpeech TextGrids from:
  https://github.com/CorentinJ/librispeech-alignments
or run MFA yourself on the dataset.
"""

import re
from pathlib import Path
from typing import List, Tuple


# (start_sec, end_sec, phoneme_label)
PhonemeInterval = Tuple[float, float, str]

SILENCE_LABELS = {"SIL", "SP", "", "<eps>"}


def load_textgrid_phones(textgrid_path: str) -> List[PhonemeInterval]:
    """Parse a .TextGrid file and return a list of (start, end, phone) tuples."""
    intervals = []
    path = Path(textgrid_path)
    if not path.exists():
        return intervals

    text = path.read_text(encoding="utf-8")

    # Find the "phones" tier block
    tier_match = re.search(
        r'name\s*=\s*"phones".*?intervals:\s*size\s*=\s*\d+(.*?)(?=item \[|\Z)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not tier_match:
        return intervals

    tier_text = tier_match.group(1)
    entries = re.findall(
        r"xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+).*?text\s*=\s*\"(.*?)\"",
        tier_text,
        re.DOTALL,
    )

    for xmin, xmax, label in entries:
        label = label.strip()
        if label not in SILENCE_LABELS:
            intervals.append((float(xmin), float(xmax), label))

    return intervals


def phoneme_labels_for_tokens(
    intervals: List[PhonemeInterval],
    token_rate: float,
    num_tokens: int,
) -> List[str]:
    """
    Map each codec token position to a phoneme label.

    Each token at index i covers the time window [i/token_rate, (i+1)/token_rate].
    The phoneme assigned is the one with maximum overlap with that window.
    Tokens with no phoneme coverage are labeled "<SIL>".

    Uses a moving-pointer sweep over sorted intervals — O(num_tokens + num_intervals).
    """
    labels = ["<SIL>"] * num_tokens
    if not intervals or num_tokens == 0:
        return labels

    # MFA outputs are already sorted by start time; sort defensively.
    sorted_intervals = sorted(intervals, key=lambda iv: iv[0])
    num_intervals = len(sorted_intervals)

    # first_live: index of the earliest interval whose end time is still after
    # the current token's start. Advances forward, never resets to 0.
    first_live = 0

    for token_idx in range(num_tokens):
        token_start = token_idx / token_rate
        token_end = (token_idx + 1) / token_rate

        # Drop intervals that ended at or before this token's start time.
        while first_live < num_intervals and sorted_intervals[first_live][1] <= token_start:
            first_live += 1

        # Scan forward for all intervals that could overlap this token window.
        best_label = "<SIL>"
        best_overlap = 0.0
        scan = first_live
        while scan < num_intervals and sorted_intervals[scan][0] < token_end:
            phone_start, phone_end, phoneme = sorted_intervals[scan]
            overlap = max(0.0, min(token_end, phone_end) - max(token_start, phone_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = phoneme
            scan += 1

        labels[token_idx] = best_label

    return labels
