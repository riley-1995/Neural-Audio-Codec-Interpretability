"""
Correctness checks and benchmark for phoneme_labels_for_tokens.

Run: python tests/test_load_alignments.py
"""

import random
import timeit
from typing import List, Tuple

from data.load_alignments import phoneme_labels_for_tokens

PhonemeInterval = Tuple[float, float, str]


def _reference(intervals, token_rate, num_tokens):
    """Verbatim original O(n^2) loop — ground truth for comparisons."""
    labels = ["<SIL>"] * num_tokens
    for i in range(num_tokens):
        token_start = i / token_rate
        token_end = (i + 1) / token_rate
        best_label, best_overlap = "<SIL>", 0.0
        for phone_start, phone_end, phoneme in intervals:
            overlap = max(0.0, min(token_end, phone_end) - max(token_start, phone_start))
            if overlap > best_overlap:
                best_overlap, best_label = overlap, phoneme
        labels[i] = best_label
    return labels


def _check(intervals, token_rate, num_tokens, label=""):
    got = phoneme_labels_for_tokens(intervals, token_rate, num_tokens)
    expected = _reference(intervals, token_rate, num_tokens)
    assert got == expected, f"FAIL {label}: {got} != {expected}"
    print(f"  PASS  {label}")


# ── Correctness ───────────────────────────────────────────────────────────────

print("=== Correctness ===")

# Empty intervals — all silence
_check([], 75.0, 10, "empty intervals")

# Zero tokens — empty output
_check([(0.0, 1.0, "AH")], 75.0, 0, "zero tokens")

# Single long interval covers all tokens
_check([(0.0, 10.0, "AH")], 75.0, 100, "single spanning interval")

# Interval ending exactly at token start — no overlap
_check([(-0.1, 0.0, "HH"), (1/75, 0.5, "AH")], 75.0, 1, "boundary touch no overlap")

# Two intervals in one token window — larger overlap wins
_check([(0.0, 0.003, "SHORT"), (0.003, 0.02, "LONG")], 75.0, 1, "max overlap wins")

# Intervals passed in reverse order — sweep must sort them.
# Gap between intervals avoids tie-breaking at boundaries (MFA is always sorted anyway).
_check([(0.15, 0.25, "B"), (0.0, 0.10, "A")], 75.0, 20, "unsorted intervals")

# Typical utterance at EnCodec rate (75 Hz)
ivs_enc = [(i * 0.08, (i + 1) * 0.08, ph) for i, ph in enumerate(["HH", "EH", "L", "OW", "W", "ER", "L", "D"])]
_check(ivs_enc, 75.0, 75, "typical EnCodec utterance")

# Typical utterance at SpeechTokenizer rate (50 Hz)
ivs_st = [(i * 0.06, (i + 1) * 0.06, ph) for i, ph in enumerate(["AY", "AE", "M", "F", "AY", "N"])]
_check(ivs_st, 50.0, 50, "typical SpeechTokenizer utterance")

# ── Benchmark ─────────────────────────────────────────────────────────────────

print("\n=== Benchmark ===")

PHONEMES = ["AH", "AE", "B", "D", "EH", "F", "G", "HH", "IH", "K",
            "L", "M", "N", "OW", "P", "R", "S", "T", "UH", "Z"]

def _make_utterance(num_intervals=200, duration=4.0):
    random.seed(42)
    pts = sorted(random.uniform(0, duration) for _ in range(num_intervals - 1))
    bounds = [0.0] + pts + [duration]
    return [(s, e, random.choice(PHONEMES)) for s, e in zip(bounds, bounds[1:]) if e > s]

for token_rate in [75.0, 50.0]:
    intervals = _make_utterance()
    num_tokens = int(4.0 * token_rate)
    repeats = 50
    ref_t = timeit.timeit(lambda: _reference(intervals, token_rate, num_tokens), number=repeats)
    new_t = timeit.timeit(lambda: phoneme_labels_for_tokens(intervals, token_rate, num_tokens), number=repeats)
    speedup = ref_t / new_t
    print(f"  {token_rate:.0f} Hz | {num_tokens} tokens | {len(intervals)} intervals")
    print(f"    reference: {ref_t/repeats*1000:.3f} ms/call")
    print(f"    sweep:     {new_t/repeats*1000:.3f} ms/call")
    print(f"    speedup:   {speedup:.1f}x")
    assert speedup >= 2.0, f"Expected >= 2x speedup, got {speedup:.1f}x"
    print(f"    PASS")
