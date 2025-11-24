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
