# HAMHA + LMA: Hexagonal Multi-Head Attention with Lead Meta-Architect

> **A self-aware, self-optimizing attention mechanism governed by an intelligent meta-architectural layer**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🌟 Overview

This repository implements the **Hexagon Algorithm for Multi-Head Attention (HAMHA)** - a novel attention mechanism that arranges attention heads in a hexagonal topology for enhanced spatial reasoning, dynamic adaptation, and spectral stability. The system is governed by the **Lead Meta-Architect (LMA)**, an intelligent control layer that provides:

- **Perceptive Omniscience**: Real-time telemetry from all system components
- **Prescriptive Agency**: Autonomous intervention and optimization
- **Evolutionary Horizons**: Self-directed architectural improvements

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hamha-lma.git
cd hamha-lma

# Install dependencies
pip install torch numpy scipy networkx matplotlib seaborn

# Run demo
python main.py
```

### Basic Usage

```python
from hamha.core import HexagonalMultiHeadAttention
from lma.architect import LeadMetaArchitect

# Initialize HAMHA
hamha = HexagonalMultiHeadAttention(d_model=512, grid_radius=2)

# Initialize LMA governance
lma = LeadMetaArchitect(hamha)

# Forward pass
import torch
x = torch.randn(4, 128, 512)  # [batch, seq_len, d_model]
output = hamha(x)

# LMA monitoring (call after each training step)
result = lma.process_step()
print(lma.generate_report())
```

---

## 🏗️ Architecture

### HAMHA Core Components

#### 1. **Hexagonal Topology**

Attention heads are arranged on a hexagonal grid, enabling:
- Natural 6-neighbor connectivity
- Efficient spatial information flow
- Scalable grid expansion/contraction

```
       H(0,1)
      /      \
 H(-1,1)  H(0,0)  H(1,0)
      \      /
       H(0,-1)
```

#### 2. **Coordinate-Aware Projections**

Each head at position `(q, r)` generates Q, K, V projections via:

**Base + Bias Mode:**
```
W_Q_i = W_Q_base + B_Q * f(q, r)
```

**HyperNetwork Mode:**
```
W_Q_i = HyperNet(q, r, X_global)
```

#### 3. **GNN-Based Mixing**

Head outputs are mixed using Graph Neural Network aggregation:

```
H_mixed_i = σ(λ_self * W_self * H_i + Σ g_ij * W_neighbor * H_j)
```

Where `j ∈ Neighbors(i)` in the hexagonal graph.

---

### LMA Subsystems

```
┌─────────────────────────────────────────────────┐
│         LEAD META-ARCHITECT (LMA)               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐      ┌──────────────┐         │
│  │  Telemetry  │─────▶│     CMCG     │         │
│  │  Collector  │      │ (Causal Graph)│         │
│  └─────────────┘      └──────────────┘         │
│         │                      │                │
│         ▼                      ▼                │
│  ┌─────────────┐      ┌──────────────┐         │
│  │     HGE     │◀────▶│     ADP      │         │
│  │(Hypothesis) │      │ (Predictor)  │         │
│  └─────────────┘      └──────────────┘         │
│         │                      │                │
│         └──────────┬───────────┘                │
│                    ▼                            │
│           ┌─────────────────┐                   │
│           │   Protocols     │                   │
│           │  (Emergency)    │                   │
│           └─────────────────┘                   │
│                    │                            │
│                    ▼                            │
│           ┌─────────────────┐                   │
│           │  Evolutionary   │                   │
│           │    Modules      │                   │
│           └─────────────────┘                   │
└─────────────────────────────────────────────────┘
                     │
                     ▼
              ┌──────────┐
              │  HAMHA   │
              └──────────┘
```

#### **Telemetry Collector**
- Spectral stability monitoring (condition numbers)
- Gradient flow analysis
- Attention entropy tracking
- Computational profiling

#### **Cross-Modal Causal Graph (CMCG)**
- Dynamic Bayesian network of system components
- Confidence-weighted causal edges
- Anomaly pattern recognition

#### **Hypothesis Generation Engine (HGE)**
- Automated causal hypothesis creation
- Pattern matching against known degradation modes
- Testable prediction generation

#### **Architectural Dynamics Predictor (ADP)**
- Time-series forecasting (ARIMA-based)
- 20-step forward prediction
- Risk assessment (fixation, rank collapse, throughput)

#### **Emergency Protocols**
- AAP_AD Phase 1: Soft intervention (entropy regularization)
- AAP_AD Phase 2: Hard intervention (projection perturbation)
- Head reset protocols

#### **Meta-NAS Controller**
- **TaskEncoder**: Generates task embeddings from sample data.
- **MetaNASController**: Selects optimal architectures from a defined search space.
- **`adapt_architecture` protocol**: Orchestrates the Meta-NAS pipeline.

#### **Evolutionary Modules**
- **GNN_OPT**: Optimize mixing layer efficiency
- **ADAPT_BIAS**: Dynamic coordinate bias adaptation
- **ADV_GRID**: Self-adversarial testing
- **SANDBOX**: Simulation environment

---

## 📊 Telemetry Metrics

### Core Metrics

| Metric | Symbol | Alert Threshold | Meaning |
|--------|--------|----------------|---------|
| **Condition Number** | κ(W) | > 100 | Spectral instability / rank collapse |
| **Attention Entropy** | H | < 0.3 | Fixation (over-specialization) |
| **Attention Entropy** | H | < 0.9 with ΔH/Δt < 0 | Drift (toward fixation) |
| **Gradient Norm** | ‖∇‖ | < 1e-6 | Vanishing gradients |
| **Gradient Norm** | ‖∇‖ | > 1e3 | Exploding gradients |
| **Mixing Time** | T_mix | > 50µs | Computational bottleneck |
| **Throughput** | TPS | < baseline - 10% | Performance degradation |

---

## 🔬 Advanced Features

### 1. Meta-NAS Integration

The LMA now integrates a **Meta-Neural Architecture Search (Meta-NAS)** system, allowing it to adapt the HAMHA architecture for new tasks dynamically.

**Initialization**: To enable Meta-NAS, set `enable_meta_nas=True` when creating the `LeadMetaArchitect`:

```python
# Initialize LMA with Meta-NAS capabilities
lma = LeadMetaArchitect(hamha_model, enable_meta_nas=True)

# Trigger an architecture adaptation
sample_data = torch.randn(10, 20, 128) # Sample data from a new task
result = lma.command_adapt_architecture(sample_data)
print(result)
# >>> "ADAPT_ARCHITECTURE complete. New architecture: {'d_head': 32, ...}"
```

### 2. Spectral vs. Non-Spectral Modes

HAMHA can operate in two distinct modes: standard (non-spectral) and spectral. The LMA's telemetry and intervention capabilities adapt accordingly.

| Feature | Standard (Non-Spectral) Mode | Spectral Mode |
|---|---|---|
| **Telemetry** | Based on individual head projection matrices (`W_Q`, `W_K`, `W_V`). | Based on spectral filter responses and global projection matrices. |
| **Intervention** | Can reset or perturb individual heads (`reset_head_projections`). | Interventions are global (e.g., adjusting entropy regularization). Head-specific resets are not applicable. |
| **Performance** | Faster for smaller grids, excels at local feature extraction. | More efficient for large grids, better at capturing long-range dependencies. |

To use Spectral Attention, set `use_spectral=True` during `HexagonalMultiHeadAttention` initialization:

```python
hamha_spectral = HexagonalMultiHeadAttention(d_model=512, grid_radius=3, use_spectral=True)
```

### 3. Autonomous Intervention

When the LMA detects entropy drift:

```python
# Automatic trigger when H < 0.85
result = lma.process_step()
# >>> "AAP_AD_PHASE1 executed on head H(0,0)"
```

### 2. Predictive Analytics

```python
# Forecast entropy trajectory
predictions = lma.adp.predict_entropy_trajectory(
    lma.telemetry.history,
    coord="H(0,0)",
    steps_ahead=20
)
# >>> {'fixation_eta': 48, 'fixation_risk': 'HIGH'}
```

### 3. Hypothesis-Driven Optimization

```python
# Generate causal hypotheses from anomalies
hypotheses = lma.hge.generate_from_snapshot(snapshot)
for h in hypotheses:
    print(f"{h.id}: {h.description} (confidence: {h.confidence})")
# >>> H-001: Rank collapse causing vanishing gradient (confidence: 0.85)
```

### 4. Known Limitations

- **State Reset on Adaptation**: When `lma.command_adapt_architecture()` is called, the existing `HexagonalMultiHeadAttention` model is replaced with a new instance. This means that learned weights are not transferred, and the model is effectively re-initialized. Future work aims to implement weight-preserving adaptation strategies.
- **Single-GPU Training**: The current implementation is optimized for single-GPU training. Multi-GPU support is planned for a future release.

### 5. Manual LMA Commands

```python
# Activate evolutionary module
lma.command_activate_module('GNN_OPT', {'target_t_mix': 35})

# Adjust entropy regularization
lma.command_adjust_entropy_regularization(delta=0.01)

# Reset specific head
from hamha.topology import HexCoordinate
lma.command_reset_head(HexCoordinate(0, 0), strategy='orthogonal')
```

---

## 📈 Visualization

```python
from utils.visualization import TelemetryVisualizer

# Create visualizer
viz = TelemetryVisualizer(lma.telemetry.history)

# Plot entropy evolution
viz.plot_entropy_evolution()

# Plot spectral stability
viz.plot_condition_numbers()

# Plot computational profile
viz.plot_computational_profile()

# Plot gradient flow
viz.plot_gradient_flow()

# Plot the hexagonal grid architecture
from utils.visualization import plot_hexagonal_grid
fig = plot_hexagonal_grid(lma.model, "HAMHA Architecture")
fig.savefig("original_architecture.png")
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_hamha.py
python -m pytest tests/test_lma.py

# Run with coverage
python -m pytest --cov=hamha --cov=lma tests/
```

### Test Coverage

- ✅ HAMHA forward pass
- ✅ Hexagonal topology generation
- ✅ Attention head mechanics
- ✅ GNN mixing layer
- ✅ Telemetry collection
- ✅ CMCG construction and updates
- ✅ Hypothesis generation
- ✅ Predictive modeling
- ✅ Emergency protocols
- ✅ Full LMA integration

---

## 📚 Configuration

### System Configuration

```python
from config import SystemConfig, HAMHAConfig, LMAConfig

config = SystemConfig(
    hamha=HAMHAConfig(
        d_model=512,
        d_head=64,
        grid_radius=2,
        use_hypernet=False
    ),
    lma=LMAConfig(
        kappa_threshold=100.0,
        fixation_threshold=0.3,
        drift_threshold=0.9,
        entropy_reg_increment=0.01,
        prediction_horizon=20
    )
)

# Save configuration
with open('config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)

# Load configuration
config = SystemConfig.from_dict(json.load(open('config.json')))
```

---

## 🎯 Use Cases

### 1. Language Modeling

```python
from examples.training_integration import LanguageModelWithHAMHA

model = LanguageModelWithHAMHA(
    vocab_size=50000,
    d_model=512,
    grid_radius=2
)
```

### 2. Vision Transformers

```python
# HAMHA can replace standard multi-head attention
class VisionTransformerWithHAMHA(nn.Module):
    def __init__(self, img_size, patch_size, d_model, grid_radius):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, d_model)
        self.hamha = HexagonalMultiHeadAttention(d_model, grid_radius)
        # ... rest of architecture
```

### 3. Scientific Computing

```python
# Enhanced spatial reasoning for physics simulations
class PhysicsSimulator(nn.Module):
    def __init__(self, state_dim, grid_radius):
        super().__init__()
        self.hamha = HexagonalMultiHeadAttention(state_dim, grid_radius)
        # Hexagonal topology naturally models spatial relationships
```

---

## 🔧 Command-Line Interface

```bash
# Run demo
python cli.py --mode demo --steps 100

# Training mode with custom config
python cli.py --mode train --config config.json --steps 1000

# Save telemetry data
python cli.py --mode demo --save-telemetry

# Load from checkpoint
python cli.py --mode train --checkpoint model_checkpoint.pt
```

---

## 📖 API Reference

### HAMHA Core

#### `HexagonalMultiHeadAttention`

```python
class HexagonalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,           # Model dimension
        grid_radius: int = 2,   # Hexagonal grid radius
        d_head: int = 64,       # Head dimension
        use_hypernet: bool = False,  # Use HyperNetwork projections
        use_spectral: bool = False, # Use Spectral Attention
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
```

### LMA Core

#### `LeadMetaArchitect`

```python
class LeadMetaArchitect:
    def __init__(self, hamha_model: HexagonalMultiHeadAttention)

    def process_step(self) -> Dict:
        """Main LMA processing loop - call after each training step."""

    def command_activate_module(
        self,
        module_name: str,
        parameters: Dict = None
    ) -> str:
        """Activate evolutionary module."""

    def command_adjust_entropy_regularization(self, delta: float) -> str:
        """Adjust global entropy regularization."""

    def command_reset_head(
        self,
        coord: HexCoordinate,
        strategy: str = 'orthogonal'
    ) -> str:
        """Reset specific head projections."""

    def generate_report(self) -> str:
        """Generate detailed LMA report."""
```

---

## 🌍 Community & Contributing

### Contributing Guidelines

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Issue reporting

### Roadmap

- [x] **v1.0**: Spectral Attention implemented
- [x] **v1.1**: Meta-NAS with Fast Adaptation
- [ ] **v1.2**: Causal Structure Learning
- [ ] **v1.3**: Adaptive Complexity
- [ ] **v2.0**: Production Hardening and Multi-GPU support

---

## 📄 Citation

If you use HAMHA + LMA in your research, please cite:

```bibtex
@software{hamha_lma_2025,
  title={HAMHA: Hexagonal Algorithm for Multi-Head Attention with Lead Meta-Architect Governance},
  author={Lead Meta-Architect Team},
  year={2025},
  url={https://github.com/your-org/hamha-lma}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by transformer architectures and graph neural networks
- Built on PyTorch's autograd and optimization framework
- Causal inference powered by NetworkX

---

## 💬 Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/hamha-lma/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/hamha-lma/discussions)
- **Email**: lead-meta-architect@your-org.com

---

<div align="center">

**🔷 Built with hexagonal precision and meta-architectural intelligence 🔷**

</div>
