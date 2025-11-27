import pytest
import torch
from hamha.core import HexagonalMultiHeadAttention

class TestSpectralHAMHAIntegration:
    """Integration tests for spectral HAMHA."""

    def test_hamha_with_spectral_attention(self):
        """Test full HAMHA with spectral attention."""
        hamha = HexagonalMultiHeadAttention(
            d_model=128,
            grid_radius=2,
            use_spectral=True
        )

        x = torch.randn(2, 19, 128)
        output = hamha(x)

        assert output.shape == (2, 19, 128)
        assert not torch.isnan(output).any()

    def test_hamha_without_spectral_attention(self):
        """Test full HAMHA without spectral attention."""
        hamha = HexagonalMultiHeadAttention(
            d_model=128,
            grid_radius=2,
            use_spectral=False
        )

        x = torch.randn(2, 19, 128)
        output = hamha(x)

        assert output.shape == (2, 19, 128)
        assert not torch.isnan(output).any()
