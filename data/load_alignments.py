"""
Load Montreal Forced Aligner (MFA) TextGrid alignments for LibriSpeech.

Expected alignment file layout (MFA default output):
  <alignments_root>/<speaker_id>/<utterance_id>.TextGrid

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
    """
    labels = ["<SIL>"] * num_tokens

    for i in range(num_tokens):
        t_start = i / token_rate
        t_end = (i + 1) / token_rate
        best_label = "<SIL>"
        best_overlap = 0.0

        for p_start, p_end, phone in intervals:
            overlap = max(0.0, min(t_end, p_end) - max(t_start, p_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = phone

        labels[i] = best_label

    return labels
