"""
Main pipeline: encode → label → probe → evaluate → plot.

Usage:
    python main.py \
        --librispeech_root /path/to/LibriSpeech \
        --alignments_root  /path/to/alignments \
        --st_ckpt          /path/to/speechtokenizer.pt \
        --st_config        /path/to/config.json \
        --output_dir       results \
        --max_utterances   500

    Set --max_utterances to a small number (e.g. 100) for a quick sanity-check run.
    Set to 0 to process the full dataset.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.extract_pitch import extract_pitch_hz
from data.load_alignments import load_textgrid_phones, phoneme_labels_for_tokens
from data.load_librispeech import iter_librispeech
from encode.encode_encodec import encode as encodec_encode
from encode.encode_encodec import load_encodec
from encode.encode_speechtokenizer import encode as st_encode
from encode.encode_speechtokenizer import load_speechtokenizer
from probe.evaluate_probes import evaluate_probes
from probe.train_probes import train_probes
from visualize.plot_curves import plot_all

ENCODEC_TOKEN_RATE = 75.0
ST_TOKEN_RATE = 50.0
TRAIN_RATIO = 0.9   # 90% train, 10% eval


def parse_args():
    p = argparse.ArgumentParser(description="RVQ Linear Probing Study")
    p.add_argument("--librispeech_root", required=True)
    p.add_argument("--alignments_root",  required=True)
    p.add_argument("--st_ckpt",          required=True, help="SpeechTokenizer .pt checkpoint")
    p.add_argument("--st_config",        required=True, help="SpeechTokenizer config.json")
    p.add_argument("--output_dir",       default="results")
    p.add_argument("--split",            default="train-clean-100")
    p.add_argument("--max_utterances",   type=int, default=500,
                   help="Cap number of utterances (0 = no cap). Use 100 for quick test.")
    p.add_argument("--device",           default="cpu")
    return p.parse_args()


def collect_data(args, encodec_model, st_model):
    """
    Iterate LibriSpeech, encode with both codecs, align labels.
    Returns two dicts (one per codec) each containing:
        embeddings_by_layer, phoneme_labels, speaker_labels, pitch_values
    as lists to be stacked later.
    """
    enc_data = {i: [] for i in range(8)}
    st_data  = {i: [] for i in range(8)}
    phonemes_enc, phonemes_st = [], []
    speakers_enc, speakers_st = [], []
    pitch_enc,    pitch_st    = [], []

    utterances = iter_librispeech(args.librispeech_root, args.split)
    count = 0

    for utt in tqdm(utterances, desc="Encoding utterances"):
        if args.max_utterances > 0 and count >= args.max_utterances:
            break

        # Load forced alignment
        align_path = (
            Path(args.alignments_root)
            / utt.speaker_id
            / f"{utt.utterance_id}.TextGrid"
        )
        phone_intervals = load_textgrid_phones(str(align_path))
        if not phone_intervals:
            continue   # skip utterances with no alignment

        # --- EnCodec ---
        try:
            enc_embs, enc_ntok = encodec_encode(encodec_model, utt.audio, utt.sample_rate)
        except Exception as e:
            print(f"  EnCodec encode failed for {utt.utterance_id}: {e}")
            continue

        enc_phones = phoneme_labels_for_tokens(phone_intervals, ENCODEC_TOKEN_RATE, enc_ntok)
        enc_pitch  = extract_pitch_hz(utt.audio, utt.sample_rate, ENCODEC_TOKEN_RATE, enc_ntok)

        for layer_idx in range(8):
            enc_data[layer_idx].append(enc_embs[layer_idx])
        phonemes_enc.extend(enc_phones)
        speakers_enc.extend([utt.speaker_id] * enc_ntok)
        pitch_enc.append(enc_pitch)

        # --- SpeechTokenizer (needs 16kHz audio — encode() handles resampling) ---
        try:
            st_embs, st_ntok = st_encode(st_model, utt.audio, utt.sample_rate)
        except Exception as e:
            print(f"  SpeechTokenizer encode failed for {utt.utterance_id}: {e}")
            continue

        st_phones = phoneme_labels_for_tokens(phone_intervals, ST_TOKEN_RATE, st_ntok)
        st_pitch  = extract_pitch_hz(utt.audio, utt.sample_rate, ST_TOKEN_RATE, st_ntok)

        for layer_idx in range(8):
            st_data[layer_idx].append(st_embs[layer_idx])
        phonemes_st.extend(st_phones)
        speakers_st.extend([utt.speaker_id] * st_ntok)
        pitch_st.append(st_pitch)

        count += 1

    # Stack into arrays
    enc_embs_stacked = [np.concatenate(enc_data[i], axis=0) for i in range(8)]
    st_embs_stacked  = [np.concatenate(st_data[i],  axis=0) for i in range(8)]

    enc_bundle = dict(
        embeddings=enc_embs_stacked,
        phonemes=np.array(phonemes_enc),
        speakers=np.array(speakers_enc),
        pitch=np.concatenate(pitch_enc),
    )
    st_bundle = dict(
        embeddings=st_embs_stacked,
        phonemes=np.array(phonemes_st),
        speakers=np.array(speakers_st),
        pitch=np.concatenate(pitch_st),
    )
    return enc_bundle, st_bundle


def split_bundle(bundle, train_ratio=TRAIN_RATIO):
    """Split token-level data into train and eval sets."""
    n = len(bundle["phonemes"])
    n_train = int(n * train_ratio)
    idx = np.random.permutation(n)
    train_idx, eval_idx = idx[:n_train], idx[n_train:]

    def select(arr, idx):
        if isinstance(arr, list):
            return [a[idx] for a in arr]
        return arr[idx]

    train = dict(
        embeddings=[e[train_idx] for e in bundle["embeddings"]],
        phonemes=bundle["phonemes"][train_idx],
        speakers=bundle["speakers"][train_idx],
        pitch=bundle["pitch"][train_idx],
    )
    eval_ = dict(
        embeddings=[e[eval_idx] for e in bundle["embeddings"]],
        phonemes=bundle["phonemes"][eval_idx],
        speakers=bundle["speakers"][eval_idx],
        pitch=bundle["pitch"][eval_idx],
    )
    return train, eval_


def main():
    args = parse_args()
    np.random.seed(42)

    out = Path(args.output_dir)
    probe_dir = out / "probes"
    fig_dir   = out / "figures"
    probe_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    encodec_model = load_encodec(device=args.device)
    st_model      = load_speechtokenizer(args.st_ckpt, args.st_config, device=args.device)

    print(f"Collecting data (max {args.max_utterances} utterances)...")
    enc_bundle, st_bundle = collect_data(args, encodec_model, st_model)

    print("Splitting train/eval...")
    enc_train, enc_eval = split_bundle(enc_bundle)
    st_train,  st_eval  = split_bundle(st_bundle)

    print("Training probes...")
    enc_le = train_probes(
        enc_train["embeddings"], enc_train["phonemes"],
        enc_train["speakers"],   enc_train["pitch"],
        codec_name="encodec", output_dir=str(probe_dir),
    )
    st_le = train_probes(
        st_train["embeddings"], st_train["phonemes"],
        st_train["speakers"],   st_train["pitch"],
        codec_name="speechtokenizer", output_dir=str(probe_dir),
    )

    print("Evaluating probes...")
    enc_results = evaluate_probes(
        enc_eval["embeddings"], enc_eval["phonemes"],
        enc_eval["speakers"],   enc_eval["pitch"],
        codec_name="encodec", probe_dir=str(probe_dir),
    )
    st_results = evaluate_probes(
        st_eval["embeddings"], st_eval["phonemes"],
        st_eval["speakers"],   st_eval["pitch"],
        codec_name="speechtokenizer", probe_dir=str(probe_dir),
    )

    # Save raw results for later analysis
    results_path = out / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"encodec": enc_results, "speechtokenizer": st_results}, f)
    print(f"Results saved to {results_path}")

    print("Plotting...")
    plot_all({"encodec": enc_results, "speechtokenizer": st_results}, str(fig_dir))

    # Print summary table
    print("\n=== Results Summary ===")
    print(f"{'Layer':<8} {'ENC-Ph-Acc':>12} {'ST-Ph-Acc':>12} {'ENC-Sp-Acc':>12} {'ST-Sp-Acc':>12} {'ENC-Pitch-R2':>14} {'ST-Pitch-R2':>13}")
    for i in range(8):
        print(
            f"  {i+1:<6} "
            f"{enc_results['phoneme_acc'][i]:>12.3f} "
            f"{st_results['phoneme_acc'][i]:>12.3f} "
            f"{enc_results['speaker_acc'][i]:>12.3f} "
            f"{st_results['speaker_acc'][i]:>12.3f} "
            f"{enc_results['pitch_r2'][i]:>14.3f} "
            f"{st_results['pitch_r2'][i]:>13.3f}"
        )


if __name__ == "__main__":
    main()
