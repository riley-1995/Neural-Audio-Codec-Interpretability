"""
t-SNE visualizations of codec RVQ layer embeddings, colored by phoneme
group, speaker identity, and pitch — inspired by Sadok et al. 2025.

Layout for each figure: 2 rows (EnCodec / SpeechTokenizer) × 4 columns
(RVQ layers 1, 2, 4, 8).

Run on the server:
    python visualize/plot_tsne.py \\
        --cache_dir  results_capfull_20260503_071048/cache \\
        --output_dir results_capfull_20260503_071048/figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Constants ─────────────────────────────────────────────────────────────────

SHOW_LAYERS    = [1, 2, 4, 8]
NUM_LAYERS     = 8
N_TOKENS       = 10_000
MAX_PER_UTT    = 40
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 30

_CODECS       = ("encodec", "speechtokenizer")
_CODEC_LABELS = {"encodec": "EnCodec", "speechtokenizer": "SpeechTokenizer"}

# dot style — larger and slightly more opaque than the previous version
_DOT = dict(s=14, alpha=0.72, linewidths=0, rasterized=True)

# ── Phoneme grouping ──────────────────────────────────────────────────────────

_PH_MAP = {
    **{p: "Vowel"     for p in ["AA","AE","AH","AO","AW","AY","EH","ER",
                                 "EY","IH","IY","OW","OY","UH","UW"]},
    **{p: "Stop"      for p in ["B","D","G","P","T","K"]},
    **{p: "Fricative" for p in ["DH","F","HH","S","SH","TH","V","Z","ZH"]},
    **{p: "Nasal"     for p in ["M","N","NG"]},
    **{p: "Approx."   for p in ["L","R","W","Y"]},
    **{p: "Affricate" for p in ["CH","JH"]},
    **{p: "Silence"   for p in ["SIL","SP","SPN","sil","sp","spn","pau",
                                  "<SIL>","<sil>"]},
}

# draw silence first (background), interesting categories on top
_PH_DRAW_ORDER = ["Silence", "Other", "Affricate", "Approx.", "Nasal",
                  "Fricative", "Stop", "Vowel"]

# high-contrast ColorBrewer-style palette
_PH_COLORS = {
    "Vowel":     "#d73027",   # vivid red
    "Stop":      "#4575b4",   # deep blue
    "Fricative": "#1a9850",   # forest green
    "Nasal":     "#7b2d8b",   # deep purple
    "Approx.":   "#f46d43",   # coral-orange
    "Affricate": "#74add1",   # sky blue
    "Silence":   "#d9d9d9",   # light gray
    "Other":     "#fdae61",   # pale orange
}

# legend order (same as draw but reversed so top-drawn items appear first)
_PH_LEGEND_ORDER = ["Vowel","Stop","Fricative","Nasal",
                    "Approx.","Affricate","Silence","Other"]


def _to_group(ph: str) -> str:
    base = re.sub(r"\d+$", "", str(ph)).upper()
    return _PH_MAP.get(base, "Other")


# ── Speaker palette: 8 maximally distinct colors ──────────────────────────────

_SP_PALETTE = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#a65628",  # brown
    "#f781bf",  # pink
    "#000000",  # black
]
TOP_SPEAKERS = 8


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sample(cache_dir: Path, codec: str, seed: int = 42):
    rng      = np.random.RandomState(seed)
    cdir     = cache_dir / codec
    files    = sorted(cdir.glob("*.npz"))
    shuffled_idx = rng.permutation(len(files))
    files = [files[i] for i in shuffled_idx]

    lbuf = [[] for _ in range(NUM_LAYERS)]
    pbuf, sbuf, pitbuf = [], [], []
    total = 0

    for path in files:
        if total >= N_TOKENS:
            break
        speaker = path.stem.split("-")[0]
        try:
            with np.load(path, allow_pickle=False) as d:
                if "phonemes" not in d or "pitches" not in d:
                    continue
                n    = d["layer_0"].shape[0]
                take = min(MAX_PER_UTT, n, N_TOKENS - total)
                idx  = rng.choice(n, take, replace=False)
                for i in range(NUM_LAYERS):
                    lbuf[i].append(d[f"layer_{i}"][idx])
                pbuf.append(d["phonemes"][idx])
                pitbuf.append(d["pitches"][idx])
                sbuf.extend([speaker] * take)
                total += take
        except Exception:
            continue

    if total == 0:
        raise RuntimeError(f"No valid NPZ files in {cdir}")

    embeddings = [np.concatenate(lbuf[i], axis=0) for i in range(NUM_LAYERS)]
    print(f"  [{codec}] {total} tokens loaded")
    return embeddings, np.concatenate(pbuf), np.array(sbuf), np.concatenate(pitbuf)


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(X: np.ndarray, seed: int = 42) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"run_tsne expects a 2D array, got shape {X.shape}")

    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError(
            "run_tsne requires at least 2 samples; "
            f"got {n_samples}. Increase sampled tokens or lower filtering."
        )

    n_pca = min(PCA_COMPONENTS, n_features, n_samples - 1)
    if n_features > n_pca:
        X = PCA(n_components=n_pca, random_state=seed).fit_transform(X)

    perplexity = min(TSNE_PERPLEXITY, n_samples - 1)
    return TSNE(
        n_components=2, perplexity=perplexity, max_iter=1000,
        init="pca", learning_rate="auto", random_state=seed,
    ).fit_transform(X)


def compute_tsne_all_layers(embeddings: list, seed: int = 42) -> dict:
    out = {}
    for layer in SHOW_LAYERS:
        print(f"    t-SNE layer {layer}...", flush=True)
        out[layer] = run_tsne(embeddings[layer - 1], seed=seed)
    return out


# ── Layout helpers ────────────────────────────────────────────────────────────

def _make_grid(title: str):
    """Return (fig, axes[nrows, ncols]) with codec × layer layout."""
    nrows = len(_CODECS)
    ncols = len(SHOW_LAYERS)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    # Column headers: layer numbers
    for col, layer in enumerate(SHOW_LAYERS):
        axes[0, col].set_title(f"Layer {layer}", fontsize=11,
                               fontweight="bold", pad=6)

    # Row labels: codec names rotated 90°, left of first column
    for row, codec in enumerate(_CODECS):
        axes[row, 0].annotate(
            _CODEC_LABELS[codec],
            xy=(-0.18, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", rotation=90,
            ha="center", va="center",
        )
    return fig, axes


def _clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#aaaaaa")


# ── Phoneme ───────────────────────────────────────────────────────────────────

def plot_phoneme(tsne_per_codec, phonemes_per_codec, out_dir: Path):
    fig, axes = _make_grid("Phoneme Identity — RVQ Layer t-SNE")

    for row, codec in enumerate(_CODECS):
        groups = np.array([_to_group(p) for p in phonemes_per_codec[codec]])
        for col, layer in enumerate(SHOW_LAYERS):
            ax = axes[row, col]
            Z  = tsne_per_codec[codec][layer]
            for g in _PH_DRAW_ORDER:
                mask = groups == g
                if not mask.any():
                    continue
                ax.scatter(Z[mask, 0], Z[mask, 1],
                           c=_PH_COLORS[g], **_DOT)
            _clean_ax(ax)

    # single shared legend below the figure
    present = [g for g in _PH_LEGEND_ORDER
               if any((np.array([_to_group(p) for p in phonemes_per_codec[c]]) == g).any()
                      for c in _CODECS)]
    handles = [mpatches.Patch(color=_PH_COLORS[g], label=g) for g in present]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        fontsize=9, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    path = out_dir / "tsne_phoneme.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Speaker ───────────────────────────────────────────────────────────────────

def plot_speaker(tsne_per_codec, speakers_per_codec, out_dir: Path):
    # Select top-N speakers by token count (consistent across both codecs)
    all_sp           = np.concatenate(list(speakers_per_codec.values()))
    uniq, cnt        = np.unique(all_sp, return_counts=True)
    top_sp_ids       = set(uniq[np.argsort(cnt)[-TOP_SPEAKERS:]])
    sp_list          = sorted(top_sp_ids)
    sp_color         = {sp: _SP_PALETTE[i] for i, sp in enumerate(sp_list)}

    fig, axes = _make_grid(
        f"Speaker Identity — RVQ Layer t-SNE  (top-{TOP_SPEAKERS} speakers)"
    )

    for row, codec in enumerate(_CODECS):
        speakers = speakers_per_codec[codec]
        for col, layer in enumerate(SHOW_LAYERS):
            ax = axes[row, col]
            Z  = tsne_per_codec[codec][layer]

            # background: all non-top speakers in very light gray
            bg = np.array([s not in top_sp_ids for s in speakers])
            if bg.any():
                ax.scatter(Z[bg, 0], Z[bg, 1], c="#e0e0e0",
                           s=8, alpha=0.3, linewidths=0, rasterized=True)

            # foreground: top speakers with full color and larger dots
            for sp in sp_list:
                mask = speakers == sp
                if not mask.any():
                    continue
                ax.scatter(Z[mask, 0], Z[mask, 1],
                           c=sp_color[sp],
                           s=22, alpha=0.85, linewidths=0, rasterized=True,
                           zorder=3)
            _clean_ax(ax)

    handles = [mpatches.Patch(color=sp_color[sp], label=f"spk {sp}")
               for sp in sp_list]
    fig.legend(
        handles=handles, loc="lower center", ncol=TOP_SPEAKERS,
        fontsize=8, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    path = out_dir / "tsne_speaker.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Pitch ─────────────────────────────────────────────────────────────────────

def plot_pitch(tsne_per_codec, pitches_per_codec, out_dir: Path):
    all_voiced = np.concatenate([p[~np.isnan(p)] for p in pitches_per_codec.values()])
    if len(all_voiced) == 0:
        print("  No voiced tokens; skipping pitch t-SNE.")
        return

    p_lo  = np.percentile(all_voiced, 5)
    p_hi  = np.percentile(all_voiced, 95)
    norm  = Normalize(vmin=float(p_lo), vmax=float(p_hi))
    cmap  = plt.get_cmap("coolwarm")   # blue=low pitch, red=high pitch

    fig, axes = _make_grid("Pitch F0 (Hz) — RVQ Layer t-SNE")

    sc_last = None
    for row, codec in enumerate(_CODECS):
        pitches = pitches_per_codec[codec]
        voiced  = ~np.isnan(pitches)
        for col, layer in enumerate(SHOW_LAYERS):
            ax = axes[row, col]
            Z  = tsne_per_codec[codec][layer]

            # unvoiced tokens: neutral gray background
            if (~voiced).any():
                ax.scatter(Z[~voiced, 0], Z[~voiced, 1],
                           c="#ececec", s=8, alpha=0.35,
                           linewidths=0, rasterized=True)

            # voiced tokens: coolwarm colored by F0
            sc = ax.scatter(
                Z[voiced, 0], Z[voiced, 1],
                c=pitches[voiced], cmap=cmap, norm=norm,
                s=16, alpha=0.78, linewidths=0, rasterized=True, zorder=3,
            )
            sc_last = sc
            _clean_ax(ax)

    # shared colorbar: reserve space on the right
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    if sc_last is not None:
        cbar_ax = fig.add_axes((0.94, 0.12, 0.018, 0.75))
        cb = fig.colorbar(sc_last, cax=cbar_ax)
        cb.set_label("F0 (Hz)", fontsize=10)
        cb.ax.tick_params(labelsize=8)

    path = out_dir / "tsne_pitch.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsne_per_codec     = {}
    phonemes_per_codec = {}
    speakers_per_codec = {}
    pitches_per_codec  = {}

    for codec in _CODECS:
        print(f"\n=== {codec}: loading ===")
        emb, ph, sp, pit = load_sample(cache_dir, codec, seed=args.seed)
        phonemes_per_codec[codec] = ph
        speakers_per_codec[codec] = sp
        pitches_per_codec[codec]  = pit
        print(f"=== {codec}: t-SNE ===")
        tsne_per_codec[codec] = compute_tsne_all_layers(emb, seed=args.seed)

    print("\n=== Plotting ===")
    plot_phoneme(tsne_per_codec, phonemes_per_codec, out_dir)
    plot_speaker(tsne_per_codec, speakers_per_codec, out_dir)
    plot_pitch(tsne_per_codec,   pitches_per_codec,  out_dir)
    print(f"\nDone — written to {out_dir}")


if __name__ == "__main__":
    main()
