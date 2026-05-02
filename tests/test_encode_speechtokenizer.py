import types

import numpy as np
import torch

from encode.encode_speechtokenizer import NUM_LAYERS, encode


def _fake_codes(num_tokens: int = 3) -> torch.Tensor:
    base = torch.arange(num_tokens, dtype=torch.long)
    codes = torch.stack([base for _ in range(NUM_LAYERS)], dim=0)
    return codes.unsqueeze(1)  # (num_layers, 1, num_tokens)


class _ModelWithVqLayers(torch.nn.Module):
    def __init__(self, codebook: torch.Tensor):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        layers = [
            types.SimpleNamespace(_codebook=types.SimpleNamespace(embed=codebook))
            for _ in range(NUM_LAYERS)
        ]
        self.quantizer = types.SimpleNamespace(vq=types.SimpleNamespace(layers=layers))

    def encode(self, _audio: torch.Tensor) -> torch.Tensor:
        return _fake_codes()


class _ModelWithLegacyQuantizers(torch.nn.Module):
    def __init__(self, codebook: torch.Tensor):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        quantizers = [
            types.SimpleNamespace(_codebook=types.SimpleNamespace(embed=codebook))
            for _ in range(NUM_LAYERS)
        ]
        self.quantizer = types.SimpleNamespace(quantizers=quantizers)

    def encode(self, _audio: torch.Tensor) -> torch.Tensor:
        return _fake_codes()


def test_encode_supports_current_vq_layers_layout():
    codebook = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    model = _ModelWithVqLayers(codebook)

    embeddings, num_tokens = encode(model, torch.zeros(1, 320), 16000)

    assert num_tokens == 3
    assert len(embeddings) == NUM_LAYERS
    expected = codebook[[0, 1, 2]].numpy().astype(np.float32)
    for emb in embeddings:
        np.testing.assert_array_equal(emb, expected)


def test_encode_supports_legacy_quantizers_layout():
    codebook = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    model = _ModelWithLegacyQuantizers(codebook)

    embeddings, num_tokens = encode(model, torch.zeros(1, 320), 16000)

    assert num_tokens == 3
    assert len(embeddings) == NUM_LAYERS
    expected = codebook[[0, 1, 2]].numpy().astype(np.float32)
    for emb in embeddings:
        np.testing.assert_array_equal(emb, expected)
