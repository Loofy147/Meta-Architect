import pytest
import torch
from lma.protocols import EmergencyProtocols
from hamha.core import HexagonalMultiHeadAttention

class TestEmergencyProtocols:
    """Unit tests for the EmergencyProtocols class."""

    def test_weight_transfer_preserves_weights(self):
        """
        Tests that the _transfer_weights function correctly preserves weights
        for overlapping heads between two different architectures.
        """
        # 1. Create two models with different architectures but some overlap
        model_a = HexagonalMultiHeadAttention(d_model=128, grid_radius=2, d_head=32)
        model_b = HexagonalMultiHeadAttention(d_model=128, grid_radius=1, d_head=32)

        # Create a protocols instance to access the transfer function
        protocols = EmergencyProtocols(model_a)

        # 2. Call the weight transfer function
        protocols._transfer_weights(model_a, model_b)

        # 3. Verify that the weights of overlapping heads are identical
        old_coords = set(model_a.coord_to_idx.keys())
        new_coords = set(model_b.coord_to_idx.keys())
        overlapping_coords = old_coords.intersection(new_coords)

        assert len(overlapping_coords) > 0, "Test requires overlapping heads."

        for coord in overlapping_coords:
            idx_a = model_a.coord_to_idx[coord]
            idx_b = model_b.coord_to_idx[coord]

            head_a = model_a.heads[idx_a]
            head_b = model_b.heads[idx_b]

            # Check that the base projection weights are identical
            assert torch.equal(head_a.W_Q_base.data, head_b.W_Q_base.data)
            assert torch.equal(head_a.W_K_base.data, head_b.W_K_base.data)
            assert torch.equal(head_a.W_V_base.data, head_b.W_V_base.data)

        # 4. Verify that the final projection layer (W_O) is partially identical
        d_head = model_a.d_head
        for coord in overlapping_coords:
            idx_a = model_a.coord_to_idx[coord]
            idx_b = model_b.coord_to_idx[coord]

            slice_a = model_a.W_O.data[idx_a * d_head : (idx_a + 1) * d_head, :]
            slice_b = model_b.W_O.data[idx_b * d_head : (idx_b + 1) * d_head, :]
            assert torch.equal(slice_a, slice_b)

    def test_weight_transfer_spectral_models(self):
        """Tests weight transfer for spectral attention models."""
        model_a = HexagonalMultiHeadAttention(d_model=128, grid_radius=2, use_spectral=True)
        model_b = HexagonalMultiHeadAttention(d_model=128, grid_radius=1, use_spectral=True)
        protocols = EmergencyProtocols(model_a)
        protocols._transfer_weights(model_a, model_b)

        overlapping_coords = set(model_a.coord_to_idx.keys()).intersection(set(model_b.coord_to_idx.keys()))
        for coord in overlapping_coords:
            idx_a = model_a.coord_to_idx[coord]
            idx_b = model_b.coord_to_idx[coord]
            filter_a = model_a.spectral_attention.filters[idx_a]
            filter_b = model_b.spectral_attention.filters[idx_b]
            assert torch.equal(filter_a.band_boundaries, filter_b.band_boundaries)
            assert torch.equal(filter_a.filter_weights, filter_b.filter_weights)

    def test_weight_transfer_gnn_mixing(self):
        """Tests weight transfer for the GNN mixing layer."""
        model_a = HexagonalMultiHeadAttention(d_model=128, grid_radius=2, d_head=32)
        model_b = HexagonalMultiHeadAttention(d_model=128, grid_radius=1, d_head=32)
        protocols = EmergencyProtocols(model_a)
        protocols._transfer_weights(model_a, model_b)

        assert torch.equal(model_a.gnn_mixing.W_self.data, model_b.gnn_mixing.W_self.data)
        assert torch.equal(model_a.gnn_mixing.W_neighbor.data, model_b.gnn_mixing.W_neighbor.data)

        overlapping_coords = set(model_a.coord_to_idx.keys()).intersection(set(model_b.coord_to_idx.keys()))
        for coord in overlapping_coords:
            idx_a = model_a.coord_to_idx[coord]
            idx_b = model_b.coord_to_idx[coord]
            assert model_a.gnn_mixing.lambda_self.data[idx_a] == model_b.gnn_mixing.lambda_self.data[idx_b]

    def test_weight_transfer_incompatible_d_head(self):
        """Tests that weight transfer is skipped for incompatible d_head."""
        model_a = HexagonalMultiHeadAttention(d_model=128, grid_radius=2, d_head=32)
        model_b = HexagonalMultiHeadAttention(d_model=128, grid_radius=1, d_head=64)
        protocols = EmergencyProtocols(model_a)
        protocols._transfer_weights(model_a, model_b)

        # Verify that the weights are NOT equal
        head_b = model_b.heads[0]
        are_weights_equal = torch.equal(model_a.heads[0].W_Q_base.data, head_b.W_Q_base.data)
        assert not are_weights_equal

    def test_weight_transfer_incompatible_hypernet(self):
        """Tests that weight transfer is skipped for incompatible hypernet configs."""
        model_a = HexagonalMultiHeadAttention(d_model=128, grid_radius=2, use_hypernet=True)
        model_b = HexagonalMultiHeadAttention(d_model=128, grid_radius=1, use_hypernet=False)
        protocols = EmergencyProtocols(model_a)
        protocols._transfer_weights(model_a, model_b)

        # Verify that the weights are NOT equal
        head_b = model_b.heads[0]
        are_weights_equal = torch.equal(model_a.heads[0].W_Q_base.data, head_b.W_Q_base.data)
        assert not are_weights_equal
