"""
Plot layer-wise probing results for EnCodec vs SpeechTokenizer.

Produces 3 figures (one per task), each with two side-by-side subplots:
  phoneme_probing.png  — accuracy and macro-F1 across RVQ layers
  speaker_probing.png  — accuracy and macro-F1 across RVQ layers
  pitch_probing.png    — MAE (Hz) and R² across RVQ layers

A dashed horizontal chance-level line is drawn for classification tasks so
readers can immediately see how much above random each probe performs.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

LAYERS = list(range(1, 9))
COLORS = {"encodec": "#1f77b4", "speechtokenizer": "#ff7f0e"}
LABELS = {"encodec": "EnCodec", "speechtokenizer": "SpeechTokenizer"}


def _plot_two_metrics(
    results: Dict[str, Dict],
    metric_a: str,
    metric_b: str,
    ylabel_a: str,
    ylabel_b: str,
    title: str,
    output_path: str,
    chance_a: Optional[float] = None,
    chance_b: Optional[float] = None,
    invert_a: bool = False,
) -> None:
    """Render two metric subplots side-by-side for both codecs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, metric, ylabel, chance, invert in [
        (axes[0], metric_a, ylabel_a, chance_a, invert_a),
        (axes[1], metric_b, ylabel_b, chance_b, False),
    ]:
        for codec, codec_results in results.items():
            ax.plot(
                LAYERS,
                codec_results[metric],
                marker="o",
                color=COLORS[codec],
                label=LABELS[codec],
                linewidth=2,
                markersize=6,
            )

        # Dashed chance-level reference line
        if chance is not None:
            ax.axhline(
                chance,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"Chance ({chance:.3f})",
            )

        ax.set_xlabel("RVQ Layer", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(LAYERS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if invert:
            ax.invert_yaxis()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_phoneme(
    results: Dict[str, Dict],
    output_dir: str,
    n_phonemes: int = 0,
) -> None:
    chance = 1.0 / n_phonemes if n_phonemes > 0 else None
    _plot_two_metrics(
        results,
        metric_a="phoneme_acc", ylabel_a="Accuracy",
        metric_b="phoneme_f1",  ylabel_b="Macro-F1",
        title="Phoneme Identity — Linear Probe by RVQ Layer",
        output_path=str(Path(output_dir) / "phoneme_probing.png"),
        chance_a=chance,
        chance_b=chance,
    )


def plot_speaker(
    results: Dict[str, Dict],
    output_dir: str,
    n_speakers: int = 0,
) -> None:
    chance = 1.0 / n_speakers if n_speakers > 0 else None
    _plot_two_metrics(
        results,
        metric_a="speaker_acc", ylabel_a="Accuracy",
        metric_b="speaker_f1",  ylabel_b="Macro-F1",
        title="Speaker Identity — Linear Probe by RVQ Layer",
        output_path=str(Path(output_dir) / "speaker_probing.png"),
        chance_a=chance,
        chance_b=chance,
    )


def plot_pitch(results: Dict[str, Dict], output_dir: str) -> None:
    _plot_two_metrics(
        results,
        metric_a="pitch_mae", ylabel_a="MAE (Hz) ↓ lower is better",
        metric_b="pitch_r2",  ylabel_b="R²  ↑ higher is better",
        title="Pitch (F0) — Linear Regression by RVQ Layer",
        output_path=str(Path(output_dir) / "pitch_probing.png"),
        invert_a=True,
    )


def plot_all(
    results: Dict[str, Dict],
    output_dir: str = "results/figures",
    n_phonemes: int = 0,
    n_speakers: int = 0,
) -> None:
    """
    Generate all 3 probing plots.

    Args:
        results:     Dict keyed by codec name ("encodec", "speechtokenizer"),
                     each value is the dict returned by evaluate_probes().
        output_dir:  Directory to write .png files.
        n_phonemes:  Number of unique phoneme classes (for chance-level line).
        n_speakers:  Number of unique speaker classes (for chance-level line).
    """
    plot_phoneme(results, output_dir, n_phonemes=n_phonemes)
    plot_speaker(results, output_dir, n_speakers=n_speakers)
    plot_pitch(results, output_dir)
