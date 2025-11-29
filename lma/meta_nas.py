import torch
import torch.nn as nn
import torch.nn.functional as F
from lma.search_space import SEARCH_SPACE
from hamha.core import HexagonalMultiHeadAttention

class MetaNASController(nn.Module):
    """
    Meta-learned controller for HAMHA architecture adaptation.
    """
    def __init__(self, task_embedding_dim: int = 64, hidden_dim: int = 128, meta_lr: float = 1e-3):
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
        return sum(len(v) for v in SEARCH_SPACE.values())

    def sample_architecture(self, task_embedding: torch.Tensor) -> (dict, torch.Tensor):
        """
        Samples an architecture based on the task embedding using a categorical
        distribution, returning the architecture and the log probability of the
        choice.
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
