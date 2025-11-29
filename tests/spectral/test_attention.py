import pytest
import torch
from hamha.spectral.attention import SpectralAttentionLayer

class TestSpectralAttention:
    """Unit tests for spectral attention layer."""

    @pytest.fixture
    def layer_params(self):
        """Parameters for the layer."""
        return {
            "d_model": 128,
            "num_heads": 19,
            "d_head": 32,
            "k_eigenvectors": 16
        }

    @pytest.fixture
    def layer(self, layer_params):
        """Fixture: Create spectral attention layer."""
        return SpectralAttentionLayer(**layer_params)

    @pytest.fixture
    def graph_spectrum(self, layer_params):
        """Fixture: Dummy graph spectrum."""
        k = layer_params["k_eigenvectors"]
        num_heads = layer_params["num_heads"]

        # Ensure k <= num_heads
        assert k <= num_heads, "k_eigenvectors cannot be greater than num_heads"

        eigenvalues = torch.linspace(0, 1, k)
        eigenvectors = torch.randn(num_heads, k)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)  # Orthonormalize
        return eigenvectors, eigenvalues

    def test_forward_shape(self, layer, layer_params, graph_spectrum):
        """Test output shape is correct."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        B, N, D = 2, layer_params["num_heads"], layer_params["d_model"]
        x = torch.randn(B, N, D)
        output = layer(x)

        expected_shape = (B, N, layer_params["num_heads"] * layer_params["d_head"])
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"

    def test_forward_no_nan(self, layer, layer_params, graph_spectrum):
        """Test output contains no NaN values."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, layer_params["num_heads"], layer_params["d_model"])
        output = layer(x)

        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_backward_pass(self, layer, layer_params, graph_spectrum):
        """Test gradients flow correctly."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, layer_params["num_heads"], layer_params["d_model"], requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
