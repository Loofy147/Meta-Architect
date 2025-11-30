import torch
import torch.nn as nn

class TaskEncoder(nn.Module):
    """Generates a numerical embedding for a given task.

    This module creates a low-dimensional vector representation (embedding)
    that captures the essential characteristics of a task. This embedding is
    then used by the `MetaNASController` to propose a specialized architecture.

    The current implementation generates the embedding by computing the mean
    of a sample batch of data and passing it through a simple feed-forward
    network.

    Attributes:
        d_in (int): The dimensionality of the input data features.
        embedding_dim (int): The dimensionality of the output task embedding.
        encoder (nn.Sequential): The neural network used to generate the
            embedding.
    """
    def __init__(self, d_in: int, embedding_dim: int = 64):
        """Initializes the TaskEncoder.

        Args:
            d_in (int): The dimensionality of the input features of the task data.
            embedding_dim (int, optional): The desired dimensionality of the
                task embedding. Defaults to 64.
        """
        super().__init__()
        self.d_in = d_in
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, sample_data: torch.Tensor) -> torch.Tensor:
        """Generates a task embedding from a sample of task data.

        Args:
            sample_data (torch.Tensor): A sample batch of data from the task,
                typically of shape [batch_size, seq_len, d_in].

        Returns:
            torch.Tensor: The task embedding vector of shape [embedding_dim].
        """
        # For now, we'll use the mean of the sample data as the input to the encoder.
        # This is a simple way to get a representation of the data's distribution.
        task_representation = sample_data.mean(dim=0).mean(dim=0)
        return self.encoder(task_representation)
