"""
RVQ Linear Probing Study: EnCodec vs SpeechTokenizer

Compares how phoneme identity, speaker identity, and pitch (F0) are
distributed across the 8 RVQ layers of two neural audio codecs using
frozen linear probes. Extends Sadok et al. 2025 (Interspeech) which used
mutual information and t-SNE but did not apply linear probing.

Usage (sanity check — 50 utterances):
    python main.py \\
        --librispeech_root data/LibriSpeech \\
        --alignments_root  data/alignments/train-clean-100 \\
        --st_ckpt          /path/to/speechtokenizer.pt \\
        --st_config        /path/to/config.json \\
        --output_dir       results \\
        --max_utterances   50

Usage (full run — embeddings cached after first pass):
    python main.py ... --max_utterances 0
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data.split import scan_utterance_paths, split_utterances
from encode.collect import collect_bundle
from encode.encode_encodec import load_encodec
from encode.encode_speechtokenizer import load_speechtokenizer
from probe.evaluate_probes import evaluate_probes
from probe.train_probes import ProbeTrainingBundle, fit_label_encoders, train_probes_for_codecs
from visualize.plot_curves import plot_all

SEED = 42
PROBE_EXEC_PROFILES = ("sequential", "local", "local-fast", "a100")
DEFAULT_PROBE_MAX_ITER = 1000
FAST_LOCAL_PROBE_MAX_ITER = 300


@dataclass(frozen=True)
class ProbeExecutionSettings:
    workers: int
    blas_threads: int
    max_iter: int
    skip_eval: bool


def _resolve_device(requested: str) -> str:
    """Return the best available device, or the explicitly requested one."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_probe_workers(profile: str) -> int:
    cpu_count = os.cpu_count() or 1
    if profile == "sequential":
        return 1
    if profile == "local":
        reserve = 4 if cpu_count >= 12 else 2
        return max(1, min(32, cpu_count - reserve))
    if profile == "local-fast":
        reserve = 2 if cpu_count >= 8 else 1
        return max(1, min(32, cpu_count - reserve))
    if profile == "a100":
        reserve = 4 if cpu_count >= 16 else 2
        return max(1, min(48, cpu_count - reserve))
    raise ValueError(f"Unknown probe execution profile: {profile}")


def _resolve_probe_workers(profile: str, override: int) -> int:
    if override < 0:
        raise ValueError("--probe_workers must be >= 0")
    if override == 0:
        return _default_probe_workers(profile)
    return override


def _resolve_probe_presets(args) -> ProbeExecutionSettings:
    probe_workers = _resolve_probe_workers(args.probe_exec_profile, args.probe_workers)
    probe_blas_threads = args.probe_blas_threads
    probe_max_iter = args.probe_max_iter
    skip_eval = args.skip_eval

    if args.probe_exec_profile == "local-fast":
        if args.probe_blas_threads == 0:
            probe_blas_threads = 1
        if args.probe_max_iter == DEFAULT_PROBE_MAX_ITER:
            probe_max_iter = FAST_LOCAL_PROBE_MAX_ITER
        if not args.run_eval:
            skip_eval = True

    return ProbeExecutionSettings(
        workers=probe_workers,
        blas_threads=probe_blas_threads,
        max_iter=probe_max_iter,
        skip_eval=skip_eval,
    )


def _make_probe_training_bundle(codec_name: str, collected: Mapping[str, Any]) -> ProbeTrainingBundle:
    return ProbeTrainingBundle(
        codec_name=codec_name,
        embeddings_by_layer=collected["embeddings"],
        phoneme_labels=collected["phonemes"],
        speaker_labels=collected["speakers"],
        pitch_values=collected["pitches"],
    )


def _parse_args():
    p = argparse.ArgumentParser(description="RVQ Linear Probing Study")
    p.add_argument("--librispeech_root", required=True,
                   help="Path containing train-clean-100/, dev-clean/, etc.")
    p.add_argument("--alignments_root", required=True,
                   help="Root of TextGrid alignments for the training split")
    p.add_argument("--st_ckpt",   required=True, help="SpeechTokenizer .pt checkpoint")
    p.add_argument("--st_config", required=True, help="SpeechTokenizer config.json")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: results_cap{N}_{TIMESTAMP})")
    p.add_argument("--split", default="train-clean-100",
                   help="LibriSpeech split to use (e.g. train-clean-100)")
    p.add_argument("--eval_frac", type=float, default=0.1,
                   help="Fraction of utterances held out for evaluation (default 0.1)")
    p.add_argument("--max_utterances", type=int, default=500,
                   help="Cap on utterances processed per split (0 = all)")
    p.add_argument("--device", default="auto",
                   help="Compute device: auto | cpu | cuda | mps")
    p.add_argument("--force_resplit", action="store_true",
                   help="Ignore cached split.json and recompute the train/eval split")
    p.add_argument(
        "--probe_exec_profile",
        choices=PROBE_EXEC_PROFILES,
        default="local",
        help=(
            "Probe training execution profile: "
            "sequential (1 worker), local, local-fast (one-flag warm-up), "
            "a100 (higher concurrency)"
        ),
    )
    p.add_argument(
        "--probe_workers",
        type=int,
        default=0,
        help="Maximum concurrent probe jobs (0 = profile default)",
    )
    p.add_argument(
        "--probe_blas_threads",
        type=int,
        default=0,
        help=(
            "BLAS/OpenMP threads per probe worker (0 = auto: 1 for parallel runs, "
            "unconstrained for sequential)"
        ),
    )
    p.add_argument(
        "--probe_max_iter",
        type=int,
        default=DEFAULT_PROBE_MAX_ITER,
        help="Max iterations for phoneme/speaker logistic probes",
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Stop after training probes (skip eval, results.pkl, and plots)",
    )
    p.add_argument(
        "--run_eval",
        action="store_true",
        help="Force evaluation even when execution profile presets skip it",
    )
    return p.parse_args()


def main():
    args   = _parse_args()
    if args.skip_eval and args.run_eval:
        raise ValueError("--skip_eval and --run_eval are mutually exclusive")

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    if args.output_dir is None:
        cap_str = str(args.max_utterances) if args.max_utterances > 0 else "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_cap{cap_str}_{timestamp}"

    out        = Path(args.output_dir)
    probe_dir  = out / "probes"
    fig_dir    = out / "figures"
    cache_dir  = out / "cache"
    split_path = out / "split.json"
    probe_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading EnCodec...")
    enc_model = load_encodec(device=device)
    print("Loading SpeechTokenizer...")
    st_model  = load_speechtokenizer(args.st_ckpt, args.st_config, device=device)

    # ── Scan and split utterances ──────────────────────────────────────────────
    print(f"Scanning {args.split}...")
    all_entries = scan_utterance_paths(args.librispeech_root, args.split)
    train_entries, eval_entries = split_utterances(
        all_entries, args.eval_frac, SEED, split_path, force=args.force_resplit
    )

    # ── Collect training data (embeddings, phonemes, and pitches all cached) ──────
    print(f"Collecting training data (cap={args.max_utterances or 'none'})...")
    enc_train, st_train = collect_bundle(
        train_entries, enc_model, st_model,
        args.alignments_root, cache_dir, args.max_utterances,
    )

    # ── Fit label encoders once; shared across both codecs and both splits ──────
    # Speaker encoder is fit on ALL training speaker IDs (not just the capped
    # subset) so eval tokens from unseen-in-cap speakers aren't discarded.
    print("Fitting label encoders...")
    all_train_speakers = np.asarray([entry[1] for entry in train_entries])
    label_encoders = fit_label_encoders(
        enc_train["phonemes"], all_train_speakers, probe_dir
    )

    # ── Train all probes through a shared dispatcher ───────────────────────────
    probe_settings = _resolve_probe_presets(args)
    print(
        "Training probes "
        f"(profile={args.probe_exec_profile}, workers={probe_settings.workers}, "
        f"blas_threads={probe_settings.blas_threads or 'auto'}, max_iter={probe_settings.max_iter})..."
    )
    train_probes_for_codecs(
        bundles=[
            _make_probe_training_bundle("encodec", enc_train),
            _make_probe_training_bundle("speechtokenizer", st_train),
        ],
        output_dir=str(probe_dir),
        label_encoders=label_encoders,
        max_workers=probe_settings.workers,
        blas_threads=probe_settings.blas_threads,
        classification_max_iter=probe_settings.max_iter,
    )

    if probe_settings.skip_eval:
        reason = "local-fast preset" if args.probe_exec_profile == "local-fast" and not args.skip_eval else "--skip_eval set"
        print(f"Skipping evaluation/plotting ({reason}).")
        return

    # ── Collect eval data (honor the configured utterance cap) ─────────────────
    print("Collecting eval data...")
    enc_eval, st_eval = collect_bundle(
        eval_entries, enc_model, st_model,
        args.alignments_root, cache_dir, max_utterances=args.max_utterances,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("Evaluating EnCodec probes...")
    enc_results = evaluate_probes(
        enc_eval["embeddings"], enc_eval["phonemes"],
        enc_eval["speakers"],   enc_eval["pitches"],
        codec_name="encodec", probe_dir=str(probe_dir),
        label_encoders=label_encoders,
    )
    print("Evaluating SpeechTokenizer probes...")
    st_results = evaluate_probes(
        st_eval["embeddings"], st_eval["phonemes"],
        st_eval["speakers"],   st_eval["pitches"],
        codec_name="speechtokenizer", probe_dir=str(probe_dir),
        label_encoders=label_encoders,
    )

    # ── Save and visualize ─────────────────────────────────────────────────────
    results = {"encodec": enc_results, "speechtokenizer": st_results}
    with open(out / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {out / 'results.pkl'}")

    n_phonemes = len(label_encoders["phoneme"].classes_)
    n_speakers = len(label_encoders["speaker"].classes_)
    plot_all(results, str(fig_dir), n_phonemes=n_phonemes, n_speakers=n_speakers)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n=== Results Summary (macro-F1 / R²) ===")
    print(f"{'Layer':<6} {'ENC-Ph':>8} {'ST-Ph':>8} "
          f"{'ENC-Sp':>8} {'ST-Sp':>8} {'ENC-Pit':>8} {'ST-Pit':>8}")
    for i in range(8):
        print(
            f"  {i+1:<4} "
            f"{enc_results['phoneme_f1'][i]:>8.3f} "
            f"{st_results ['phoneme_f1'][i]:>8.3f} "
            f"{enc_results['speaker_f1'][i]:>8.3f} "
            f"{st_results ['speaker_f1'][i]:>8.3f} "
            f"{enc_results['pitch_r2'][i]:>8.3f} "
            f"{st_results ['pitch_r2'][i]:>8.3f}"
        )


if __name__ == "__main__":
    main()
