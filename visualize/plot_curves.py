"""
Plot layer-wise probing results for EnCodec vs SpeechTokenizer.

Produces 3 figures (one per task), each with two side-by-side subplots:
  phoneme_probing.png  — accuracy and macro-F1 across RVQ layers
  speaker_probing.png  — accuracy and macro-F1 across RVQ layers
  pitch_probing.png    — MAE (Hz) and R² across RVQ layers

A dashed horizontal chance-level line is drawn for classification tasks so
readers can immediately see how much above random each probe performs.

Can be run as a script to visualize results from a saved results.pkl:
    python -m visualize.plot_curves --results_dir <results_dir>
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import argparse
import pickle

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
    baseline_a: Optional[float] = None,
    baseline_b: Optional[float] = None,
    baseline_name_a: str = "Baseline",
    baseline_name_b: str = "Baseline",
    invert_a: bool = False,
) -> None:
    """Render two metric subplots side-by-side for both codecs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, metric, ylabel, baseline, baseline_name, invert in [
        (axes[0], metric_a, ylabel_a, baseline_a, baseline_name_a, invert_a),
        (axes[1], metric_b, ylabel_b, baseline_b, baseline_name_b, False),
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

        # Dashed baseline reference line
        if baseline is not None:
            ax.axhline(
                baseline,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"{baseline_name} ({baseline:.3f})",
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
    baseline = 1.0 / n_phonemes if n_phonemes > 0 else None
    _plot_two_metrics(
        results,
        metric_a="phoneme_acc", ylabel_a="Accuracy",
        metric_b="phoneme_f1",  ylabel_b="Macro-F1",
        title="Phoneme Identity — Linear Probe by RVQ Layer",
        output_path=str(Path(output_dir) / "phoneme_probing.png"),
        baseline_a=baseline,
        baseline_b=baseline,
        baseline_name_a="Chance",
        baseline_name_b="Chance",
    )


def plot_speaker(
    results: Dict[str, Dict],
    output_dir: str,
    n_speakers: int = 0,
) -> None:
    baseline = 1.0 / n_speakers if n_speakers > 0 else None
    _plot_two_metrics(
        results,
        metric_a="speaker_acc", ylabel_a="Accuracy",
        metric_b="speaker_f1",  ylabel_b="Macro-F1",
        title="Speaker Identity — Linear Probe by RVQ Layer",
        output_path=str(Path(output_dir) / "speaker_probing.png"),
        baseline_a=baseline,
        baseline_b=baseline,
        baseline_name_a="Chance",
        baseline_name_b="Chance",
    )


def plot_pitch(results: Dict[str, Dict], output_dir: str) -> None:
    _plot_two_metrics(
        results,
        metric_a="pitch_mae", ylabel_a="MAE (Hz) ↓ lower is better",
        metric_b="pitch_r2",  ylabel_b="R²  ↑ higher is better",
        title="Pitch (F0) — Linear Regression by RVQ Layer",
        output_path=str(Path(output_dir) / "pitch_probing.png"),
        baseline_a=None,
        baseline_b=0.0,
        baseline_name_b="Mean predictor baseline",
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

def main():
    p = argparse.ArgumentParser(description="Visualize probing results from results.pkl")
    p.add_argument("--results_dir", required=True,
                   help="Path to results directory containing results.pkl")
    p.add_argument("--n_phonemes", type=int, default=71,
                   help="Number of phoneme classes (for chance line)")
    p.add_argument("--n_speakers", type=int, default=251,
                   help="Number of speaker classes (for chance line)")
    args = p.parse_args()

    results_path = Path(args.results_dir) / "results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.pkl found at {results_path}")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    fig_dir = Path(args.results_dir) / "figures"
    print(f"Visualizing results in {fig_dir}...")
    plot_all(results, str(fig_dir), n_phonemes=args.n_phonemes, n_speakers=args.n_speakers)
    print("Done!")


if __name__ == "__main__":
    main()