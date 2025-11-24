"""
═══════════════════════════════════════════════════════════════════
CONFIGURATION, UTILITIES, AND TESTING FRAMEWORK
═══════════════════════════════════════════════════════════════════
"""

#═══════════════════════════════════════════════════════════════════
# FILE: config.py
#═══════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HAMHAConfig:
    """Configuration for HAMHA system."""
    d_model: int = 512
    d_head: int = 64
    grid_radius: int = 2
    use_hypernet: bool = False
    dropout: float = 0.1


@dataclass
class LMAConfig:
    """Configuration for Lead Meta-Architect."""
    # Telemetry
    telemetry_window_size: int = 20
    telemetry_collection_frequency: int = 1  # Every N steps
    
    # Alert Thresholds
    kappa_threshold: float = 100.0
    fixation_threshold: float = 0.3
    drift_threshold: float = 0.9
    entropy_derivative_threshold: float = -0.005
    vanishing_gradient_threshold: float = 1e-6
    exploding_gradient_threshold: float = 1e3
    
    # Intervention Parameters
    entropy_reg_increment: float = 0.01
    self_mixing_decay_factor: float = 0.95
    neighbor_boost_factor: float = 1.05
    
    # Prediction
    prediction_horizon: int = 20
    prediction_confidence_threshold: float = 0.7
    
    # CMCG
    cmcg_edge_confidence_threshold: float = 0.7
    
    # Evolutionary Modules
    gnn_opt_target_t_mix: float = 35.0
    adapt_bias_mode: str = "exploration"


@dataclass
class SystemConfig:
    """Complete system configuration."""
    hamha: HAMHAConfig = None
    lma: LMAConfig = None
    
    def __post_init__(self):
        if self.hamha is None:
            self.hamha = HAMHAConfig()
        if self.lma is None:
            self.lma = LMAConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hamha': self.hamha.__dict__,
            'lma': self.lma.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
        hamha_config = HAMHAConfig(**config_dict.get('hamha', {}))
        lma_config = LMAConfig(**config_dict.get('lma', {}))
        return cls(hamha=hamha_config, lma=lma_config)


#═══════════════════════════════════════════════════════════════════
# FILE: utils/metrics.py
#═══════════════════════════════════════════════════════════════════

import torch
import numpy as np
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Utility functions for computing telemetry metrics."""
    
    @staticmethod
    def compute_condition_number(matrix: torch.Tensor) -> float:
        """Compute condition number κ = σ_max / σ_min."""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (S.max() / (S.min() + 1e-8)).item()
    
    @staticmethod
    def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # attention_weights: [batch, seq_len, seq_len]
        entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1)
        return entropy.mean().item()
    
    @staticmethod
    def compute_gradient_norm(parameters: List[torch.nn.Parameter]) -> float:
        """Compute global gradient norm."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    @staticmethod
    def compute_spectral_radius(matrix: torch.Tensor) -> float:
        """Compute spectral radius (largest singular value)."""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return S.max().item()
    
    @staticmethod
    def compute_rank_estimate(matrix: torch.Tensor, threshold: float = 1e-5) -> int:
        """Estimate rank of matrix."""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (S > threshold).sum().item()


#═══════════════════════════════════════════════════════════════════
# FILE: utils/visualization.py
#═══════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np


class TelemetryVisualizer:
    """Visualization utilities for LMA telemetry."""
    
    def __init__(self, telemetry_history: List):
        self.history = telemetry_history
        
    def plot_entropy_evolution(self, coord_str: str = None):
        """Plot entropy evolution over time."""
        if coord_str is None:
            # Plot all heads
            fig, ax = plt.subplots(figsize=(12, 6))
            for coord_str in self.history[0].attention_entropy.keys():
                entropies = [s.attention_entropy.get(coord_str, 0) for s in self.history]
                steps = [s.step for s in self.history]
                ax.plot(steps, entropies, label=coord_str, alpha=0.7)
            
            ax.axhline(y=0.3, color='r', linestyle='--', label='Fixation Threshold')
            ax.axhline(y=0.9, color='orange', linestyle='--', label='Drift Threshold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Attention Entropy')
            ax.set_title('Attention Entropy Evolution (All Heads)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Plot single head
            entropies = [s.attention_entropy.get(coord_str, 0) for s in self.history]
            steps = [s.step for s in self.history]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, entropies, 'b-', linewidth=2)
            ax.axhline(y=0.3, color='r', linestyle='--', label='Fixation')
            ax.axhline(y=0.9, color='orange', linestyle='--', label='Drift')
            ax.set_xlabel('Step')
            ax.set_ylabel('Entropy')
            ax.set_title(f'Entropy Evolution: {coord_str}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_condition_numbers(self):
        """Plot condition number evolution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique keys
        all_keys = set()
        for s in self.history:
            all_keys.update(s.condition_numbers.keys())
        
        for key in sorted(all_keys):
            kappas = [s.condition_numbers.get(key, 1) for s in self.history]
            steps = [s.step for s in self.history]
            ax.plot(steps, kappas, label=key, alpha=0.6)
        
        ax.axhline(y=100, color='r', linestyle='--', label='Instability Threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Condition Number (κ)')
        ax.set_title('Spectral Stability (Condition Numbers)')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_computational_profile(self):
        """Plot computational time breakdown."""
        t_proj = [s.t_proj for s in self.history if s.t_proj > 0]
        t_attn = [s.t_attn for s in self.history if s.t_attn > 0]
        t_mix = [s.t_mix for s in self.history if s.t_mix > 0]
        t_grad = [s.t_grad for s in self.history if s.t_grad > 0]
        steps = [s.step for s in self.history if s.t_proj > 0]
        
        if not t_proj:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time series
        ax1.plot(steps, t_proj, label='T_proj', alpha=0.8)
        ax1.plot(steps, t_attn, label='T_attn', alpha=0.8)
        ax1.plot(steps, t_mix, label='T_mix', alpha=0.8)
        ax1.plot(steps, t_grad, label='T_grad', alpha=0.8)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Time (µs)')
        ax1.set_title('Computational Profile Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pie chart of average distribution
        avg_times = [np.mean(t_proj), np.mean(t_attn), np.mean(t_mix), np.mean(t_grad)]
        labels = ['Projection', 'Attention', 'Mixing', 'Gradient']
        ax2.pie(avg_times, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Average Time Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_gradient_flow(self):
        """Plot gradient norm evolution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for coord in self.history[0].gradient_norms.keys():
            grads = [s.gradient_norms.get(coord, 0) for s in self.history]
            steps = [s.step for s in self.history]
            ax.plot(steps, grads, label=coord, alpha=0.6)
        
        ax.axhline(y=1e-6, color='b', linestyle='--', label='Vanishing Threshold')
        ax.axhline(y=1e3, color='r', linestyle='--', label='Exploding Threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Flow (Per Head)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


#═══════════════════════════════════════════════════════════════════
# FILE: tests/test_hamha.py
#═══════════════════════════════════════════════════════════════════

import unittest
import torch


class TestHAMHA(unittest.TestCase):
    """Unit tests for HAMHA core functionality."""
    
    def setUp(self):
        self.d_model = 128
        self.grid_radius = 1
        self.batch_size = 2
        self.seq_len = 32
        
    def test_hamha_forward(self):
        """Test basic forward pass."""
        from hamha.core import HexagonalMultiHeadAttention
        model = HexagonalMultiHeadAttention(self.d_model, self.grid_radius)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_hexagonal_topology(self):
        """Test hexagonal grid generation."""
        from hamha.topology import generate_hex_grid, HexCoordinate
        coords = generate_hex_grid(radius=2)
        self.assertEqual(len(coords), 19)  # 1 + 6 + 12 for radius 2
        
        # Check neighbors
        center = HexCoordinate(0, 0)
        neighbors = center.neighbors()
        self.assertEqual(len(neighbors), 6)
    
    def test_attention_head(self):
        """Test single attention head."""
        from hamha.heads import AttentionHead
        from hamha.topology import HexCoordinate
        coord = HexCoordinate(0, 0)
        head = AttentionHead(coord, self.d_model, 64)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = head(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))


class TestLMA(unittest.TestCase):
    """Unit tests for LMA components."""
    
    def setUp(self):
        from hamha.core import HexagonalMultiHeadAttention
        self.model = HexagonalMultiHeadAttention(128, 1)
        
    def test_telemetry_collection(self):
        """Test telemetry collector."""
        from lma.telemetry import TelemetryCollector
        collector = TelemetryCollector(self.model)
        snapshot = collector.collect()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.step, 0)
    
    def test_cmcg_creation(self):
        """Test CMCG initialization."""
        from lma.cmcg import CrossModalCausalGraph
        cmcg = CrossModalCausalGraph()
        self.assertGreater(cmcg.graph.number_of_nodes(), 0)
        self.assertGreater(cmcg.graph.number_of_edges(), 0)
    
    def test_hypothesis_generation(self):
        """Test HGE."""
        from lma.hge import HypothesisGenerationEngine
        from lma.cmcg import CrossModalCausalGraph
        from lma.telemetry import TelemetrySnapshot
        
        cmcg = CrossModalCausalGraph()
        hge = HypothesisGenerationEngine(cmcg)
        snapshot = TelemetrySnapshot(step=0, timestamp=0.0)
        snapshot.condition_numbers = {'H(0,0)_Q': 150.0}
        snapshot.gradient_norms = {'H(0,0)': 1e-7}
        
        hypotheses = hge.generate_from_snapshot(snapshot)
        self.assertGreater(len(hypotheses), 0)
    
    def test_lma_integration(self):
        """Test full LMA integration."""
        from lma.architect import LeadMetaArchitect
        lma = LeadMetaArchitect(self.model)
        result = lma.process_step()
        self.assertIn('snapshot', result)
        self.assertIn('status', result)


#═══════════════════════════════════════════════════════════════════
# FILE: examples/training_integration.py
#═══════════════════════════════════════════════════════════════════

"""
Example: Integrating HAMHA + LMA into a training loop
"""

import torch
import torch.nn as nn
from torch.optim import AdamW


class LanguageModelWithHAMHA(nn.Module):
    """Example language model using HAMHA."""
    
    def __init__(self, vocab_size: int, d_model: int, grid_radius: int):
        super().__init__()
        from hamha.core import HexagonalMultiHeadAttention
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.hamha = HexagonalMultiHeadAttention(d_model, grid_radius)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.hamha(x)
        x = self.feedforward(x)
        return self.output_proj(x)


def train_with_lma_monitoring():
    """Training loop with LMA monitoring."""
    from lma.architect import LeadMetaArchitect
    from config import SystemConfig
    
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
    lma.command_activate_module('GNN_OPT', {'target_t_mix': 35})
    
    # Training loop
    for epoch in range(10):
        for batch_idx in range(100):
            # Generate dummy data
            x = torch.randint(0, vocab_size, (4, 128))
            targets = torch.randint(0, vocab_size, (4, 128))
            
            # Forward pass
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # LMA monitoring (every 10 steps)
            if batch_idx % 10 == 0:
                result = lma.process_step()
                status = result['status']
                
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Health: {status['health']}")
                print(f"  Avg Entropy: {status['avg_entropy']:.3f}")
                
                # Check for interventions
                if result['interventions']:
                    print(f"  Interventions: {result['interventions']}")
        
        # Epoch summary
        print("\n" + lma.generate_report())


#═══════════════════════════════════════════════════════════════════
# FILE: cli.py - Command Line Interface
#═══════════════════════════════════════════════════════════════════

import argparse
import json


def main_cli():
    """Command-line interface for HAMHA + LMA."""
    parser = argparse.ArgumentParser(
        description='HAMHA with Lead Meta-Architect Governance'
    )
    
    parser.add_argument('--mode', choices=['demo', 'train', 'analyze'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--d-model', type=int, default=512, 
                       help='Model dimension')
    parser.add_argument('--grid-radius', type=int, default=2, 
                       help='Hexagonal grid radius')
    parser.add_argument('--steps', type=int, default=100, 
                       help='Number of training steps')
    parser.add_argument('--checkpoint', type=str, 
                       help='Load from checkpoint')
    parser.add_argument('--save-telemetry', action='store_true', 
                       help='Save telemetry to file')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        from main import main_demo
        main_demo()
    
    elif args.mode == 'train':
        from examples.training_integration import train_with_lma_monitoring
        train_with_lma_monitoring()
    
    elif args.mode == 'analyze':
        print("Analysis mode not yet implemented")
    
    print("\n✓ Execution complete")


if __name__ == "__main__":
    main_cli()
