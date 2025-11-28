import torch
import torch.nn as nn
from lma.search_space import SEARCH_SPACE

class MetaNASController(nn.Module):
    """
    ## Meta-Neural Architecture Search (NAS) Controller

    This module learns to generate neural network architectures that are
    optimized for a specific task. It takes a task embedding as input and
    outputs a description of a HAMHA architecture.

    The controller is trained using reinforcement learning, where the reward
    is the performance of the generated architecture on the task.

    ### Forward Pass:

    -   `task_embedding`: A tensor representing the task embedding from the TaskEncoder.

    ### Returns:

    -   An architectural description (e.g., a dictionary of hyperparameters).
    """
    def __init__(self, task_embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.task_embedding_dim = task_embedding_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.get_total_search_space_size()),
        )

    def get_total_search_space_size(self) -> int:
        """
        Calculates the total number of output dimensions required to represent the
        entire search space.
        """
        return sum(len(v) for v in SEARCH_SPACE.values())

    def forward(self, task_embedding: torch.Tensor) -> dict:
        """
        Generates an architectural description from a task embedding.
        """
        logits = self.layers(task_embedding)

        arch_params = {}
        current_idx = 0
        for key, values in SEARCH_SPACE.items():
            num_values = len(values)
            param_logits = logits[current_idx : current_idx + num_values]
            chosen_idx = torch.argmax(param_logits).item()
            arch_params[key] = values[chosen_idx]
            current_idx += num_values

        return arch_params
