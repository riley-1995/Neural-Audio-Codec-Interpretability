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
    Set SPEECHTOKENIZER_CKPT and SPEECHTOKENIZER_CONFIG env vars or pass paths directly.

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
    num_tokens = codes.shape[1]

    # Extract per-layer codebook embeddings
    embeddings = []
    for layer_idx in range(NUM_LAYERS):
        # Access the RVQ quantizer for this layer
        quantizer = model.quantizer.quantizers[layer_idx]
        codebook = quantizer._codebook.embed   # (codebook_size, embed_dim)
        token_ids = codes[layer_idx]           # (num_tokens,)
        layer_emb = codebook[token_ids].cpu().numpy().astype(np.float32)
        embeddings.append(layer_emb)

    return embeddings, num_tokens
