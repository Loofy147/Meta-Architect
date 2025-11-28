import pytest
import torch
from hamha.spectral.attention import SpectralAttentionLayer

class TestSpectralAttention:
    """Unit tests for spectral attention layer."""

    @pytest.fixture
    def layer(self):
        """Fixture: Create spectral attention layer."""
        return SpectralAttentionLayer(
            d_model=128,
            num_heads=4,
            d_head=32,
            k_eigenvectors=16
        )

    @pytest.fixture
    def graph_spectrum(self):
        """Fixture: Dummy graph spectrum."""
        k = 16
        N = 19
        eigenvalues = torch.linspace(0, 1, k)
        eigenvectors = torch.randn(N, k)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)  # Orthonormalize
        return eigenvectors, eigenvalues

    def test_forward_shape(self, layer, graph_spectrum):
        """Test output shape is correct."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        B, N, D = 2, 19, 128
        x = torch.randn(B, N, D)
        output = layer(x)

        assert output.shape == (B, N, 4 * 32), \
            f"Expected shape ({B}, {N}, {4 * 32}), got {output.shape}"

    def test_forward_no_nan(self, layer, graph_spectrum):
        """Test output contains no NaN values."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, 19, 128)
        output = layer(x)

        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_backward_pass(self, layer, graph_spectrum):
        """Test gradients flow correctly."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, 19, 128, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
