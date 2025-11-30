import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np


class TelemetryVisualizer:
    """Provides methods for visualizing LMA telemetry data.

    This class takes a history of `TelemetrySnapshot` objects and uses
    `matplotlib` and `seaborn` to generate various plots for analyzing the
    model's behavior over time.

    Attributes:
        history (List[TelemetrySnapshot]): A list of telemetry snapshots to be
            visualized.
    """

    def __init__(self, telemetry_history: List):
        """Initializes the TelemetryVisualizer.

        Args:
            telemetry_history (List[TelemetrySnapshot]): The historical telemetry
                data to be used for plotting.
        """
        self.history = telemetry_history

    def plot_entropy_evolution(self, coord_str: str = None):
        """Plots the evolution of attention entropy over time.

        If `coord_str` is provided, it plots the entropy for that specific head.
        Otherwise, it plots the entropy for all heads on a single graph.

        Args:
            coord_str (str, optional): The coordinate string of a specific head
                (e.g., "H(0,0)"). If None, all heads are plotted. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The generated Matplotlib figure object.
        """
        if coord_str is None:
            # Plot all heads
            fig, ax = plt.subplots(figsize=(12, 6))
            for coord_str in self.history[0].attention_entropy.keys():
                entropies = [
                    s.attention_entropy.get(coord_str, 0) for s in self.history
                ]
                steps = [s.step for s in self.history]
                ax.plot(steps, entropies, label=coord_str, alpha=0.7)

            ax.axhline(y=0.3, color="r", linestyle="--", label="Fixation Threshold")
            ax.axhline(y=0.9, color="orange", linestyle="--", label="Drift Threshold")
            ax.set_xlabel("Step")
            ax.set_ylabel("Attention Entropy")
            ax.set_title("Attention Entropy Evolution (All Heads)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Plot single head
            entropies = [s.attention_entropy.get(coord_str, 0) for s in self.history]
            steps = [s.step for s in self.history]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, entropies, "b-", linewidth=2)
            ax.axhline(y=0.3, color="r", linestyle="--", label="Fixation")
            ax.axhline(y=0.9, color="orange", linestyle="--", label="Drift")
            ax.set_xlabel("Step")
            ax.set_ylabel("Entropy")
            ax.set_title(f"Entropy Evolution: {coord_str}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def plot_hexagonal_grid(hamha_model, title: str):
    """Visualizes the hexagonal grid topology of a HAMHA model.

    This function creates a 3D scatter plot to represent the hexagonal grid,
    with each point corresponding to an attention head. The axial coordinates
    are converted to cube coordinates for a more intuitive 3D visualization.

    Args:
        hamha_model (HexagonalMultiHeadAttention): The HAMHA model instance whose
            grid is to be plotted.
        title (str): The title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """
    coords = hamha_model.grid_coords
    q_coords = [c.q for c in coords]
    r_coords = [c.r for c in coords]

    # Convert axial to cube coordinates for plotting
    x = [c.q for c in coords]
    z = [c.r for c in coords]
    y = [-q - r for q, r in zip(x, z)]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the heads
    ax.scatter(x, y, z, s=500, c="skyblue", depthshade=False)

    # Label the heads
    for i, c in enumerate(coords):
        ax.text(c.q, -c.q - c.r, c.r + 0.1, f"H({c.q},{c.r})",
                ha='center', va='bottom', fontsize=8)

    # Set plot properties
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Q")
    ax.set_ylabel("Y")
    ax.set_zlabel("R")

    # Hide the grid planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Set aspect ratio to equal
    ax.set_box_aspect([np.ptp(i) if np.ptp(i) > 0 else 1 for i in [x, y, z]])

    plt.tight_layout()
    return fig

def plot_condition_numbers(self):
    """Plots the evolution of condition numbers over time.

    This plot is used to monitor the spectral stability of the projection
    matrices. High condition numbers can indicate an approaching rank
    collapse. The y-axis is on a logarithmic scale to better visualize
    large variations.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique keys
    all_keys = set()
    for s in self.history:
        all_keys.update(s.condition_numbers.keys())

    for key in sorted(all_keys):
        kappas = [s.condition_numbers.get(key, 1) for s in self.history]
        steps = [s.step for s in self.history]
        ax.plot(steps, kappas, label=key, alpha=0.6)

    ax.axhline(y=100, color="r", linestyle="--", label="Instability Threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Condition Number (κ)")
    ax.set_title("Spectral Stability (Condition Numbers)")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_computational_profile(self):
    """Plots the computational time breakdown of the model.

    This method generates two subplots:
    1. A time-series plot showing the evolution of the time spent in
       different phases (projection, attention, mixing, gradient).
    2. A pie chart showing the average distribution of computational time
       among these phases.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object,
            or None if there is no data to plot.
    """
    t_proj = [s.t_proj for s in self.history if s.t_proj > 0]
    t_attn = [s.t_attn for s in self.history if s.t_attn > 0]
    t_mix = [s.t_mix for s in self.history if s.t_mix > 0]
    t_grad = [s.t_grad for s in self.history if s.t_grad > 0]
    steps = [s.step for s in self.history if s.t_proj > 0]

    if not t_proj:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time series
    ax1.plot(steps, t_proj, label="T_proj", alpha=0.8)
    ax1.plot(steps, t_attn, label="T_attn", alpha=0.8)
    ax1.plot(steps, t_mix, label="T_mix", alpha=0.8)
    ax1.plot(steps, t_grad, label="T_grad", alpha=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Time (µs)")
    ax1.set_title("Computational Profile Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pie chart of average distribution
    avg_times = [np.mean(t_proj), np.mean(t_attn), np.mean(t_mix), np.mean(t_grad)]
    labels = ["Projection", "Attention", "Mixing", "Gradient"]
    ax2.pie(avg_times, labels=labels, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Average Time Distribution")

    plt.tight_layout()
    return fig

def plot_gradient_flow(self):
    """Plots the evolution of gradient norms over time for each head.

    This plot helps in diagnosing issues like vanishing or exploding
    gradients. The y-axis is on a logarithmic scale to accommodate a wide
    range of gradient norm values.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for coord in self.history[0].gradient_norms.keys():
        grads = [s.gradient_norms.get(coord, 0) for s in self.history]
        steps = [s.step for s in self.history]
        ax.plot(steps, grads, label=coord, alpha=0.6)

    ax.axhline(y=1e-6, color="b", linestyle="--", label="Vanishing Threshold")
    ax.axhline(y=1e3, color="r", linestyle="--", label="Exploding Threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Flow (Per Head)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
