import pytest
import torch
import torch.optim as optim
from lma.meta_nas import MetaNASController
from lma.task_sampler import generate_synthetic_task_batch
from lma.meta_training import MetaTrainingLoop

class TestMetaLearning:
    """Unit tests for the meta-learning components."""

    @pytest.fixture
    def controller(self):
        return MetaNASController(task_embedding_dim=64)

    def test_meta_training_loop(self, controller):
        """Tests a full REINFORCE step of the meta-training loop."""
        meta_optimizer = optim.Adam(controller.parameters(), lr=1e-3)
        meta_training_loop = MetaTrainingLoop(controller, meta_optimizer)

        task_batch = generate_synthetic_task_batch(
            num_tasks=4,
            k_shot=5,
            n_way=5,
            d_model=32,
            seq_len=10,
            task_embedding_dim=64
        )

        # Store the original weights of the controller's main layers
        original_weights = controller.layers[2].weight.clone()

        avg_loss = meta_training_loop.meta_train_step(task_batch)

        # Check that the controller's main layer weights have been updated
        updated_weights = controller.layers[2].weight
        assert not torch.equal(original_weights, updated_weights)
        assert isinstance(avg_loss, float)
