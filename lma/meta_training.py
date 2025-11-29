import torch
import torch.nn.functional as F
from typing import List, Dict
from lma.meta_nas import MetaNASController
from hamha.core import HexagonalMultiHeadAttention

class MetaTrainingLoop:
    """
    REINFORCE-based meta-training for the MetaNASController.
    """

    def __init__(self, controller: MetaNASController, meta_optimizer: torch.optim.Optimizer):
        self.controller = controller
        self.meta_optimizer = meta_optimizer

    def _adapt_model(self, model, support_set, num_adapt_steps=5, adapt_lr=1e-3):
        """
        Adapts the model to a task's support set using a few steps of gradient
        descent (MAML inner loop).
        """
        # Create a temporary optimizer for the inner loop
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=adapt_lr)

        for _ in range(num_adapt_steps):
            # Forward pass on the support set
            output = model(support_set['input'])
            loss = F.mse_loss(output, support_set['target'])

            # Inner loop update
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return model

    def meta_train_step(self, task_batch: List[Dict[str, torch.Tensor]]) -> float:
        """
        Performs one REINFORCE step across a batch of tasks.
        """
        self.meta_optimizer.zero_grad()
        total_query_loss = 0.0
        total_log_probs = 0.0

        for task in task_batch:
            # 1. Sample architecture from the controller
            arch_params, log_prob = self.controller.sample_architecture(task['embedding'])

            # 2. Instantiate the model with the sampled architecture
            hamha_model = HexagonalMultiHeadAttention(d_model=32, **arch_params)

            # 3. Fast adaptation using the support set (MAML inner loop)
            adapted_model = self._adapt_model(
                hamha_model,
                task['support'],
                num_adapt_steps=5
            )

            # 4. Evaluate on the query set to get the reward signal
            query_output = adapted_model(task['query']['input'])
            query_loss = F.mse_loss(query_output, task['query']['target'])

            total_query_loss += query_loss.item()
            total_log_probs += log_prob

        # 5. Calculate policy loss (REINFORCE)
        # We use the negative loss as the reward
        reward = -total_query_loss / len(task_batch)
        policy_loss = -total_log_probs * reward

        # 6. Backpropagate and update the controller
        policy_loss.backward()
        self.meta_optimizer.step()

        return total_query_loss / len(task_batch)
