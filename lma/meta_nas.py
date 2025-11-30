import torch
import torch.nn as nn
import torch.nn.functional as F
from lma.search_space import SEARCH_SPACE
from hamha.core import HexagonalMultiHeadAttention

class MetaNASController(nn.Module):
    """A meta-learned controller for HAMHA architecture adaptation.

    This controller is a neural network that takes a task embedding as input
    and outputs a probability distribution over the architectural choices defined
    in the `SEARCH_SPACE`. It is trained with reinforcement learning (REINFORCE)
    to select architectures that perform well on given tasks.

    Attributes:
        task_embedding_dim (int): The dimensionality of the input task embedding.
        hidden_dim (int): The size of the hidden layer in the controller network.
        meta_lr (nn.Parameter): A learnable parameter representing the meta
            learning rate (though not directly used in the REINFORCE sampling logic).
        layers (nn.Sequential): The neural network that maps task embeddings to
            logits for the architecture search space.
    """
    def __init__(self, task_embedding_dim: int = 64, hidden_dim: int = 128, meta_lr: float = 1e-3):
        """Initializes the MetaNASController.

        Args:
            task_embedding_dim (int, optional): The dimensionality of the task
                embedding. Defaults to 64.
            hidden_dim (int, optional): The size of the controller's hidden layer.
                Defaults to 128.
            meta_lr (float, optional): The initial meta learning rate.
                Defaults to 1e-3.
        """
        super().__init__()
        self.task_embedding_dim = task_embedding_dim
        self.hidden_dim = hidden_dim
        self.meta_lr = nn.Parameter(torch.tensor(meta_lr))

        self.layers = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.get_total_search_space_size()),
        )

    def get_total_search_space_size(self) -> int:
        """Calculates the total size of the architectural search space.

        This is the sum of the number of options for each hyperparameter.

        Returns:
            int: The total size of the search space.
        """
        return sum(len(v) for v in SEARCH_SPACE.values())

    def sample_architecture(self, task_embedding: torch.Tensor) -> (dict, torch.Tensor):
        """Samples an architecture based on a task embedding.

        This method uses the controller network to produce logits, which are then
        used to create a categorical probability distribution for each
        hyperparameter in the search space. An architecture is sampled from these
        distributions.

        Args:
            task_embedding (torch.Tensor): The embedding of the target task.

        Returns:
            tuple[dict, torch.Tensor]: A tuple containing:
                - A dictionary representing the sampled architecture.
                - The total log probability of the sampled architecture, which
                  is needed for the REINFORCE update.
        """
        logits = self.layers(task_embedding)
        arch_params = {}
        total_log_prob = 0
        current_idx = 0

        for key, values in SEARCH_SPACE.items():
            num_values = len(values)
            param_logits = logits[current_idx : current_idx + num_values]

            # Create a categorical distribution and sample an action
            dist = torch.distributions.Categorical(logits=param_logits)
            action = dist.sample()

            # Store the chosen parameter and its log probability
            arch_params[key] = values[action.item()]
            total_log_prob += dist.log_prob(action)
            current_idx += num_values

        return arch_params, total_log_prob
