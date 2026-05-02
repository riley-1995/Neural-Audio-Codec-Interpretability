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
import pickle
from pathlib import Path

import torch

from data.split import scan_utterance_paths, split_utterances
from encode.collect import collect_bundle
from encode.encode_encodec import load_encodec
from encode.encode_speechtokenizer import load_speechtokenizer
from probe.evaluate_probes import evaluate_probes
from probe.train_probes import fit_label_encoders, train_probes
from visualize.plot_curves import plot_all

SEED = 42


def _resolve_device(requested: str) -> str:
    """Return the best available device, or the explicitly requested one."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_args():
    p = argparse.ArgumentParser(description="RVQ Linear Probing Study")
    p.add_argument("--librispeech_root", required=True,
                   help="Path containing train-clean-100/, dev-clean/, etc.")
    p.add_argument("--alignments_root", required=True,
                   help="Root of TextGrid alignments for the training split")
    p.add_argument("--st_ckpt",   required=True, help="SpeechTokenizer .pt checkpoint")
    p.add_argument("--st_config", required=True, help="SpeechTokenizer config.json")
    p.add_argument("--output_dir", default="results")
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
    return p.parse_args()


def main():
    args   = _parse_args()
    device = _resolve_device(args.device)
    print(f"Device: {device}")

    out        = Path(args.output_dir)
    probe_dir  = out / "probes"
    fig_dir    = out / "figures"
    cache_dir  = out / "cache"
    split_path = out / "split.json"
    probe_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Collect training data (embedding encode cache; align + pitch recomputed) ─
    print(f"Collecting training data (cap={args.max_utterances or 'none'})...")
    enc_train, st_train = collect_bundle(
        train_entries, enc_model, st_model,
        args.alignments_root, cache_dir, args.max_utterances,
    )

    # ── Fit label encoders once; shared across both codecs and both splits ──────
    print("Fitting label encoders...")
    label_encoders = fit_label_encoders(
        enc_train["phonemes"], enc_train["speakers"], probe_dir
    )

    # ── Train 24 probes per codec ──────────────────────────────────────────────
    print("Training EnCodec probes...")
    train_probes(
        enc_train["embeddings"], enc_train["phonemes"],
        enc_train["speakers"],   enc_train["pitch"],
        codec_name="encodec", output_dir=str(probe_dir),
        label_encoders=label_encoders,
    )
    print("Training SpeechTokenizer probes...")
    train_probes(
        st_train["embeddings"], st_train["phonemes"],
        st_train["speakers"],   st_train["pitch"],
        codec_name="speechtokenizer", output_dir=str(probe_dir),
        label_encoders=label_encoders,
    )

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
        enc_eval["speakers"],   enc_eval["pitch"],
        codec_name="encodec", probe_dir=str(probe_dir),
        label_encoders=label_encoders,
    )
    print("Evaluating SpeechTokenizer probes...")
    st_results = evaluate_probes(
        st_eval["embeddings"], st_eval["phonemes"],
        st_eval["speakers"],   st_eval["pitch"],
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
