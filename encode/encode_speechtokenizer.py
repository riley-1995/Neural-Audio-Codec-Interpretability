"""
Encode audio with SpeechTokenizer and return per-RVQ-layer codebook embeddings.

SpeechTokenizer: 16 kHz, 8 RVQ layers, ~50 tokens/second.
RVQ-1 is explicitly trained to encode HuBERT semantic content.
Token rate: 16000 / 320 = 50 tokens/second.

Setup:
    pip install speechtokenizer
    Download checkpoint files from the SpeechTokenizer repo:
      - speechtokenizer.pt   (model weights)
      - config.json          (model config)

Usage:
    model = load_speechtokenizer(ckpt_path, config_path)
    embeddings, num_tokens = encode(model, audio_tensor, original_sr)
    # embeddings: list of 8 arrays, each shape (num_tokens, 1024)
"""

import numpy as np
import torch
import torchaudio

TARGET_SR = 16000
TOKEN_RATE = 50.0       # tokens per second
NUM_LAYERS = 8


def _rvq_layers(model):
    """Return the per-layer RVQ modules across SpeechTokenizer API variants."""
    quantizer = getattr(model, "quantizer", None)
    if quantizer is None:
        raise AttributeError("SpeechTokenizer model has no 'quantizer' attribute")

    if hasattr(quantizer, "quantizers"):
        return quantizer.quantizers

    vq = getattr(quantizer, "vq", None)
    if vq is not None and hasattr(vq, "layers"):
        return vq.layers

    if hasattr(quantizer, "layers"):
        return quantizer.layers

    raise AttributeError("SpeechTokenizer quantizer does not expose RVQ layers")


def _codebook_embed(layer):
    """Return the codebook embedding tensor for one RVQ layer."""
    if hasattr(layer, "_codebook") and hasattr(layer._codebook, "embed"):
        return layer._codebook.embed

    codebook = getattr(layer, "codebook", None)
    if codebook is None:
        raise AttributeError("RVQ layer does not expose a codebook")

    if hasattr(codebook, "embed"):
        return codebook.embed
    return codebook


def load_speechtokenizer(ckpt_path: str, config_path: str, device: str = "cpu"):
    """
    Load pretrained SpeechTokenizer model.

    Args:
        ckpt_path:   Path to speechtokenizer.pt weights file
        config_path: Path to config.json
        device:      "cpu" or "cuda"
    """
    from speechtokenizer import SpeechTokenizer

    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval()
    model.to(device)
    return model


def _resample(audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Resample to 16kHz mono."""
    if orig_sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, orig_sr, TARGET_SR)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


def encode(model, audio: torch.Tensor, sample_rate: int):
    """
    Encode one utterance and return layer embeddings.

    Args:
        model:       Loaded SpeechTokenizer model from load_speechtokenizer()
        audio:       Waveform tensor, shape (1, T) or (2, T)
        sample_rate: Original sample rate of the audio

    Returns:
        embeddings:  List of 8 np.float32 arrays, each shape (num_tokens, embed_dim)
        num_tokens:  Number of codec frames in this utterance
    """
    device = next(model.parameters()).device

    audio = _resample(audio, sample_rate)
    audio = audio.unsqueeze(0).to(device)   # (1, 1, T)

    with torch.no_grad():
        # SpeechTokenizer.encode returns codes: (num_layers, 1, num_tokens)
        codes = model.encode(audio)

    codes = codes.squeeze(1)        # (num_layers, num_tokens)
    if codes.shape[0] < NUM_LAYERS:
        raise ValueError(
            f"SpeechTokenizer returned {codes.shape[0]} layers; expected at least {NUM_LAYERS}"
        )

    layers = _rvq_layers(model)
    if len(layers) < NUM_LAYERS:
        raise ValueError(
            f"SpeechTokenizer quantizer exposes {len(layers)} layers; expected at least {NUM_LAYERS}"
        )

    num_tokens = codes.shape[1]

    # Extract per-layer codebook embeddings
    embeddings = []
    for layer_idx in range(NUM_LAYERS):
        codebook = _codebook_embed(layers[layer_idx])
        token_ids = codes[layer_idx].long()    # (num_tokens,)
        layer_emb = torch.index_select(codebook, dim=0, index=token_ids)
        layer_emb = layer_emb.detach().cpu().numpy().astype(np.float32)
        embeddings.append(layer_emb)

    return embeddings, num_tokens
