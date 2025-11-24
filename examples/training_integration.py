"""
Example: Integrating HAMHA + LMA into a training loop
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from hamha.core import HexagonalMultiHeadAttention
from lma.architect import LeadMetaArchitect
from config import SystemConfig


class LanguageModelWithHAMHA(nn.Module):
    """Example language model using HAMHA."""

    def __init__(self, vocab_size: int, d_model: int, grid_radius: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.hamha = HexagonalMultiHeadAttention(d_model, grid_radius)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.hamha(x)
        x = self.feedforward(x)
        return self.output_proj(x)


def train_with_lma_monitoring():
    """Training loop with LMA monitoring."""

    # Configuration
    config = SystemConfig()
    vocab_size = 10000
    d_model = 512
    grid_radius = 2

    # Model
    model = LanguageModelWithHAMHA(vocab_size, d_model, grid_radius)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Initialize LMA
    lma = LeadMetaArchitect(model.hamha)
    lma.command_activate_module("GNN_OPT", {"target_t_mix": 35})

    # Training loop
    for epoch in range(10):
        for batch_idx in range(100):
            # Generate dummy data
            x = torch.randint(0, vocab_size, (4, 128))
            targets = torch.randint(0, vocab_size, (4, 128))

            # Forward pass
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LMA monitoring (every 10 steps)
            if batch_idx % 10 == 0:
                result = lma.process_step()
                status = result["status"]

                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Health: {status['health']}")
                print(f"  Avg Entropy: {status['avg_entropy']:.3f}")

                # Check for interventions
                if result["interventions"]:
                    print(f"  Interventions: {result['interventions']}")

        # Epoch summary
        print("\n" + lma.generate_report())
