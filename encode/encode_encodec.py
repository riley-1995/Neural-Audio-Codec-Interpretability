"""
Encode audio with EnCodec and return per-RVQ-layer codebook embeddings.

EnCodec produces 8 RVQ layers at 24 kHz / 6 kbps.
Token rate: 24000 / 320 = 75 tokens/second.

Usage:
    model = load_encodec()
    embeddings, num_tokens = encode(model, audio_tensor, original_sr)
    # embeddings: list of 8 arrays, each shape (num_tokens, 128)
"""

import numpy as np
import torch
import torchaudio

TARGET_SR = 24000
TOKEN_RATE = 75.0       # tokens per second
NUM_LAYERS = 8
EMBED_DIM = 128         # EnCodec codebook embedding dimension


def load_encodec(bandwidth: float = 6.0, device: str = "cpu"):
    """Load pretrained EnCodec model (downloads on first call ~100MB)."""
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()
    model.to(device)
    # store convert_audio helper on the model for convenience
    model._convert_audio = convert_audio
    return model


def encode(model, audio: torch.Tensor, sample_rate: int):
    """
    Encode one utterance and return layer embeddings.

    Args:
        model:       Loaded EnCodec model from load_encodec()
        audio:       Waveform tensor, shape (1, T) or (2, T)
        sample_rate: Original sample rate of the audio

    Returns:
        embeddings:  List of 8 np.float32 arrays, each shape (num_tokens, 128)
        num_tokens:  Number of codec frames in this utterance
    """
    from encodec.utils import convert_audio

    device = next(model.parameters()).device

    # Resample and convert to mono if needed
    audio = convert_audio(audio, sample_rate, TARGET_SR, model.channels)
    audio = audio.unsqueeze(0).to(device)   # (1, channels, T)

    with torch.no_grad():
        encoded_frames = model.encode(audio)

    # encoded_frames is a list of (codes, scale) tuples; one frame for short audio
    # codes shape: (1, num_layers, num_tokens)
    codes = torch.cat([f[0] for f in encoded_frames], dim=-1)  # (1, 8, T)
    codes = codes.squeeze(0)                                    # (8, T)
    num_tokens = codes.shape[1]

    # Look up codebook embedding vectors for each layer
    embeddings = []
    for layer_idx in range(NUM_LAYERS):
        # Each quantizer has an embedding table: (codebook_size, embed_dim)
        codebook = model.quantizer.vq.layers[layer_idx]._codebook.embed  # (1024, 128)
        token_ids = codes[layer_idx]                                       # (T,)
        layer_emb = codebook[token_ids].cpu().numpy().astype(np.float32)  # (T, 128)
        embeddings.append(layer_emb)

    return embeddings, num_tokens
