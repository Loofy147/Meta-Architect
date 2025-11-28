import pytest
import torch
from lma.task_encoder import TaskEncoder
from lma.meta_nas import MetaNASController

class TestMetaNAS:
    """Unit tests for the Meta-NAS components."""

    def test_task_encoder_forward(self):
        """Test the forward pass of the TaskEncoder."""
        encoder = TaskEncoder(embedding_dim=64)
        task_description = {"task": "test"}
        embedding = encoder(task_description)
        assert embedding.shape == (1, 64)

    def test_meta_nas_controller_forward(self):
        """Test the forward pass of the MetaNASController."""
        controller = MetaNASController(task_embedding_dim=64)
        task_embedding = torch.randn(1, 64)
        arch_desc = controller(task_embedding)
        assert isinstance(arch_desc, dict)
        assert "d_head" in arch_desc
        assert "use_spectral" in arch_desc
