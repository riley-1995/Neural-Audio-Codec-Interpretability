"""
Plot layer-wise probing results for EnCodec vs SpeechTokenizer.

Produces 3 figures:
  1. Phoneme identity — accuracy and macro-F1 across RVQ layers
  2. Speaker identity — accuracy and macro-F1 across RVQ layers
  3. Pitch            — MAE (Hz) and R² across RVQ layers
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

LAYERS = list(range(1, 9))   # 1 through 8
COLORS = {"encodec": "#1f77b4", "speechtokenizer": "#ff7f0e"}
LABELS = {"encodec": "EnCodec", "speechtokenizer": "SpeechTokenizer"}


def _plot_two_metrics(
    results: Dict[str, Dict],
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    ylabel_a: str,
    ylabel_b: str,
    title: str,
    output_path: str,
    invert_metric_a: bool = False,
):
    """Plot two metrics side-by-side for both codecs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, metric, ylabel, invert in [
        (axes[0], metric_a, ylabel_a, invert_metric_a),
        (axes[1], metric_b, ylabel_b, False),
    ]:
        for codec_name, codec_results in results.items():
            values = codec_results[metric]
            ax.plot(
                LAYERS,
                values,
                marker="o",
                color=COLORS[codec_name],
                label=LABELS[codec_name],
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("RVQ Layer", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(LAYERS)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if invert:
            ax.invert_yaxis()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_phoneme(results: Dict[str, Dict], output_dir: str):
    _plot_two_metrics(
        results,
        metric_a="phoneme_acc",
        metric_b="phoneme_f1",
        label_a="Accuracy",
        label_b="Macro-F1",
        ylabel_a="Accuracy",
        ylabel_b="Macro-F1",
        title="Phoneme Identity — Linear Probe Accuracy by RVQ Layer",
        output_path=str(Path(output_dir) / "phoneme_probing.png"),
    )


def plot_speaker(results: Dict[str, Dict], output_dir: str):
    _plot_two_metrics(
        results,
        metric_a="speaker_acc",
        metric_b="speaker_f1",
        label_a="Accuracy",
        label_b="Macro-F1",
        ylabel_a="Accuracy",
        ylabel_b="Macro-F1",
        title="Speaker Identity — Linear Probe Accuracy by RVQ Layer",
        output_path=str(Path(output_dir) / "speaker_probing.png"),
    )


def plot_pitch(results: Dict[str, Dict], output_dir: str):
    _plot_two_metrics(
        results,
        metric_a="pitch_mae",
        metric_b="pitch_r2",
        label_a="MAE (Hz)",
        label_b="R²",
        ylabel_a="MAE (Hz) ↓ lower is better",
        ylabel_b="R²  ↑ higher is better",
        title="Pitch (F0) — Linear Regression by RVQ Layer",
        output_path=str(Path(output_dir) / "pitch_probing.png"),
        invert_metric_a=False,
    )


def plot_all(results: Dict[str, Dict], output_dir: str = "results/figures"):
    """
    Generate all 3 plots.

    Args:
        results: Dict keyed by codec name ("encodec", "speechtokenizer"),
                 each value is the dict returned by evaluate_probes().
        output_dir: Directory to save .png files.
    """
    plot_phoneme(results, output_dir)
    plot_speaker(results, output_dir)
    plot_pitch(results, output_dir)
