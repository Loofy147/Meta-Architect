import pytest
import torch
from lma.task_encoder import TaskEncoder
from lma.meta_nas import MetaNASController
from lma.search_space import is_valid_architecture, SEARCH_SPACE
from lma.protocols import EmergencyProtocols
from hamha.core import HexagonalMultiHeadAttention

class TestMetaNAS:
    """Unit tests for the Meta-NAS components."""

    def test_task_encoder_forward(self):
        """Test the forward pass of the TaskEncoder."""
        encoder = TaskEncoder(d_in=128, embedding_dim=64)
        sample_data = torch.randn(10, 20, 128)
        embedding = encoder(sample_data)
        assert embedding.shape == (64,)

    def test_meta_nas_controller_forward(self):
        """Test the forward pass of the MetaNASController."""
        controller = MetaNASController(task_embedding_dim=64)
        task_embedding = torch.randn(64)
        arch_desc = controller(task_embedding)
        assert isinstance(arch_desc, dict)
        assert is_valid_architecture(arch_desc)

    def test_search_space_validation(self):
        """Test the search space validation function."""
        valid_arch = {"d_head": 64, "use_spectral": True}
        invalid_arch = {"d_head": 99, "use_spectral": False}
        assert is_valid_architecture(valid_arch)
        assert not is_valid_architecture(invalid_arch)
        assert not is_valid_architecture({"invalid_param": True})

    def test_meta_nas_pipeline_integration(self):
        """Test the full Meta-NAS pipeline integration."""
        hamha_model = HexagonalMultiHeadAttention(d_model=128)
        task_encoder = TaskEncoder(d_in=128, embedding_dim=64)
        meta_nas_controller = MetaNASController(task_embedding_dim=64)
        protocols = EmergencyProtocols(hamha_model, task_encoder, meta_nas_controller)

        sample_data = torch.randn(10, 20, 128)
        new_model, new_arch = protocols.adapt_architecture(sample_data)

        assert new_model is not None
        assert new_arch is not None
        assert isinstance(new_model, HexagonalMultiHeadAttention)
        assert is_valid_architecture(new_arch)
