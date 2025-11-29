import torch
from typing import Dict, List

def generate_synthetic_task_batch(
    num_tasks: int,
    k_shot: int,
    n_way: int,
    d_model: int,
    seq_len: int,
    task_embedding_dim: int
) -> List[Dict[str, torch.Tensor]]:
    """
    Generates a batch of synthetic few-shot learning tasks for meta-training.

    Each task includes a random embedding for the controller and a simple
    regression problem for the HAMHA model.
    """
    task_batch = []
    for _ in range(num_tasks):
        # Each task has a unique, simple linear transformation
        task_transform = torch.randn(d_model, d_model)

        # Generate support set
        support_input = torch.randn(k_shot, seq_len, d_model)
        support_target = torch.einsum("bsd,dd->bsd", support_input, task_transform)

        # Generate query set
        query_input = torch.randn(n_way, seq_len, d_model)
        query_target = torch.einsum("bsd,dd->bsd", query_input, task_transform)

        # Generate a random task embedding
        task_embedding = torch.randn(task_embedding_dim)

        task = {
            'support': {'input': support_input, 'target': support_target},
            'query': {'input': query_input, 'target': query_target},
            'embedding': task_embedding
        }
        task_batch.append(task)

    return task_batch
