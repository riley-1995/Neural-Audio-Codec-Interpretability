"""
Extract per-token pitch (F0 in Hz) from raw audio using librosa's YIN algorithm.

Unvoiced frames are returned as NaN and excluded from linear regression training.
"""

from typing import List

import librosa
import numpy as np
import torch


def extract_pitch_hz(
    audio: torch.Tensor,
    sample_rate: int,
    token_rate: float,
    num_tokens: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
) -> np.ndarray:
    """
    Return a float32 array of shape (num_tokens,) with F0 in Hz.
    Unvoiced / silence tokens are NaN.

    Args:
        audio:       Mono waveform tensor, shape (1, T) or (T,)
        sample_rate: Audio sample rate in Hz
        token_rate:  Codec tokens per second (e.g. 75 for EnCodec, 50 for SpeechTokenizer)
        num_tokens:  Number of codec tokens in this utterance
        fmin:        Minimum expected F0 (default 50 Hz)
        fmax:        Maximum expected F0 (default 500 Hz, covers most speech)
    """
    wav = audio.squeeze().numpy().astype(np.float32)

    # hop_length: how many audio samples per codec token
    hop_length = int(sample_rate / token_rate)

    # YIN frame-level F0 — returns array of length ~num_tokens
    f0 = librosa.yin(
        wav,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length,
    )

    # Resize to exactly num_tokens (librosa may return ±1 frame)
    f0_out = np.full(num_tokens, np.nan, dtype=np.float32)
    n = min(len(f0), num_tokens)
    f0_out[:n] = f0[:n]

    # YIN marks unvoiced frames as 0.0 — replace with NaN
    f0_out[f0_out <= 0.0] = np.nan

    return f0_out
