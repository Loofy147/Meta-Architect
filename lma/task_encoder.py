import torch
import torch.nn as nn

class TaskEncoder(nn.Module):
    """
    ## Task Encoder

    This module is responsible for generating a numerical representation (embedding)
    of a given task. The task embedding is a low-dimensional vector that captures
    the essential characteristics of a task, such as its complexity, domain, and
    objective.

    This embedding will be used by the Meta-NAS Controller to generate a
    specialized architecture for the task.

    ### Forward Pass:

    -   `sample_data`: A sample batch of data from the task.

    ### Returns:

    -   A tensor representing the task embedding.
    """
    def __init__(self, d_in: int, embedding_dim: int = 64):
        super().__init__()
        self.d_in = d_in
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, sample_data: torch.Tensor) -> torch.Tensor:
        """
        Generates a task embedding from a sample of task data.
        """
        # For now, we'll use the mean of the sample data as the input to the encoder.
        # This is a simple way to get a representation of the data's distribution.
        task_representation = sample_data.mean(dim=0).mean(dim=0)
        return self.encoder(task_representation)
