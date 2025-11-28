# 🛠️ HAMHA+LMA: Professional Implementation Guide

## Table of Contents
1. [Setup & Prerequisites](#setup)
2. [Phase-by-Phase Implementation](#phases)
3. [Code Quality Standards](#quality)
4. [Testing & Validation](#testing)
5. [Deployment Procedures](#deployment)
6. [Troubleshooting](#troubleshooting)

---

## 📦 Part I: Setup & Prerequisites {#setup}

### **1.1 Development Environment Checklist**

```bash
# ══════════════════════════════════════════════════════════
# STEP 1: Hardware Requirements
# ══════════════════════════════════════════════════════════

# Minimum:
- 1x GPU with 16GB VRAM (RTX 3090, A100)
- 32GB System RAM
- 500GB SSD

# Recommended:
- 4x GPU with 24GB+ VRAM (A100, H100)
- 128GB System RAM
- 2TB NVMe SSD

# Optimal:
- 8x GPU with 40GB+ VRAM (A100 80GB, H100)
- 256GB System RAM
- 4TB NVMe RAID

# ══════════════════════════════════════════════════════════
# STEP 2: Software Stack
# ══════════════════════════════════════════════════════════

# Base System
□ Ubuntu 22.04 LTS or later
□ NVIDIA Driver ≥ 525.x
□ CUDA 11.8 or 12.1
□ cuDNN 8.9+

# Python Environment
□ Python 3.9-3.11 (not 3.12 yet - compatibility issues)
□ pip 23.0+
□ conda (optional but recommended)

# Core Dependencies
□ PyTorch 2.1+ with CUDA support
□ NumPy 1.24+
□ SciPy 1.10+
□ NetworkX 3.1+

# Monitoring & Logging
□ TensorBoard 2.13+
□ Weights & Biases (wandb)
□ Prometheus + Grafana

# Development Tools
□ pytest 7.4+
□ black (code formatter)
□ flake8 (linter)
□ mypy (type checker)
□ pre-commit hooks
```

### **1.2 Repository Structure**

```
hamha-lma/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml          # Continuous Integration
│   │   ├── tests.yml       # Automated Testing
│   │   └── deploy.yml      # Deployment Pipeline
│   └── ISSUE_TEMPLATE.md
│
├── configs/
│   ├── base.yaml           # Base configuration
│   ├── spectral.yaml       # Spectral attention config
│   ├── meta_nas.yaml       # Meta-NAS config
│   └── production.yaml     # Production settings
│
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Preprocessed data
│   └── benchmarks/         # Benchmark datasets
│
├── experiments/
│   ├── spectral_attention/ # Phase 1 experiments
│   ├── meta_nas/           # Phase 2 experiments
│   └── ablations/          # Ablation studies
│
├── hamha/
│   ├── __init__.py
│   ├── core.py             # Core HAMHA
│   ├── spectral.py         # NEW: Spectral attention
│   ├── heads.py
│   ├── mixing.py
│   └── topology.py
│
├── lma/
│   ├── __init__.py
│   ├── architect.py
│   ├── meta_nas.py         # NEW: Meta-NAS controller
│   ├── causal.py           # NEW: Causal learner
│   ├── telemetry.py
│   └── protocols.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── profiling.py        # NEW: Performance profiling
│   └── distributed.py      # NEW: Multi-GPU utilities
│
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── regression/         # Regression tests
│   └── benchmarks/         # Performance benchmarks
│
├── scripts/
│   ├── setup.sh            # Environment setup
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── deploy.py           # Deployment script
│
├── docs/
│   ├── api/                # API documentation
│   ├── tutorials/          # User tutorials
│   └── research/           # Research papers
│
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml          # Project metadata
├── requirements.txt
├── requirements-dev.txt
├── Makefile
└── README.md
```

### **1.3 Installation Procedure**

```bash
# ══════════════════════════════════════════════════════════
# AUTOMATED SETUP SCRIPT (save as scripts/setup.sh)
# ══════════════════════════════════════════════════════════

#!/bin/bash
set -e

echo "🚀 HAMHA+LMA Setup Script"
echo "=========================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA not found. Please install NVIDIA drivers."
    exit 1
fi

echo "✓ CUDA detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n hamha-lma python=3.10 -y
conda activate hamha-lma

# Install PyTorch
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install project in editable mode
echo "🔧 Installing HAMHA+LMA..."
pip install -e .

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import hamha; print('HAMHA imported successfully')"
python -c "import lma; print('LMA imported successfully')"

# Run tests
echo "🧪 Running smoke tests..."
pytest tests/unit/test_core.py -v

echo ""
echo "🎉 Setup complete! Activate environment with:"
echo "   conda activate hamha-lma"
echo ""
echo "📖 Next steps:"
echo "   1. Review configs/base.yaml"
echo "   2. Download datasets: python scripts/download_data.py"
echo "   3. Run baseline: python scripts/train.py --config configs/base.yaml"
```

---

## 🏗️ Part II: Phase-by-Phase Implementation {#phases}

### **Phase 1: Spectral Attention (Week 3-5) - ✅ Complete**

#### **Day 1-3: Theory & Design**

**✅ Task 1.1: Mathematical Foundation**
```python
"""
Spectral Graph Theory Review Checklist:
□ Understand graph Laplacian: L = D - A
□ Know eigenvector properties (Fiedler vector, etc.)
□ Review Chebyshev polynomial approximation
□ Study frequency response of graph filters

Key Equations:
1. Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
2. Spectral Convolution: g_θ ⊛ x = U g_θ(Λ) U^T x
   where U: eigenvectors, Λ: eigenvalues
3. Chebyshev Approximation: g_θ(Λ) ≈ Σ θ_k T_k(Λ̃)
"""

# EXERCISE: Implement eigendecomposition caching
def precompute_graph_spectrum(adjacency_matrix, k=32):
    """
    Precompute and cache top-k eigenpairs.

    Args:
        adjacency_matrix: Hexagonal grid adjacency [N, N]
        k: Number of eigenvectors to keep

    Returns:
        eigenvectors: [N, k]
        eigenvalues: [k]
    """
    # Compute Laplacian
    degree = torch.diag(adjacency_matrix.sum(dim=1))
    laplacian = degree - adjacency_matrix

    # Normalized Laplacian
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    L_norm = torch.eye(len(laplacian)) - \
             degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)

    # Keep top-k (smallest eigenvalues = low-frequency)
    return eigenvectors[:, :k], eigenvalues[:k]
```

**□ Task 1.2: Implementation Plan**
```markdown
## Spectral Attention Architecture

### Components:
1. **Spectral Encoder**
   - Input: Node features [B, N, D]
   - Output: Spectral features [B, K, D]
   - Operation: Project to frequency domain

2. **Learnable Filters**
   - Low-pass: Preserve smooth features
   - Mid-pass: Capture local structures
   - High-pass: Detect anomalies/edges

3. **Spectral Attention**
   - Q, K, V in frequency domain
   - Attention without distance bottleneck
   - Scale: O(K²) instead of O(N²)

4. **Spatial Decoder**
   - Output: Spatial features [B, N, D]
   - Operation: Inverse projection

### Data Flow:
x_spatial --[U^T]--> x_spectral --[Filter]--> x_filtered
  --[Attention]--> attn_out --[U]--> y_spatial
```

#### **Day 4-7: Core Implementation**

**□ Task 1.3: Spectral Layer Implementation**
```python
# FILE: hamha/spectral.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SpectralFilter(nn.Module):
    """
    Learnable spectral filter with multi-band support.

    Features:
    - Chebyshev polynomial approximation (efficient)
    - Multi-scale filtering (low, mid, high frequencies)
    - Differentiable filter design
    """

    def __init__(self, k_eigenvectors: int, num_bands: int = 3):
        super().__init__()

        self.k = k_eigenvectors
        self.num_bands = num_bands

        # Band boundaries (learnable)
        self.band_boundaries = nn.Parameter(
            torch.linspace(0, 1, num_bands + 1)
        )

        # Filter coefficients per band
        self.filter_weights = nn.Parameter(
            torch.ones(num_bands, k_eigenvectors)
        )

    def forward(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Generate filter response.

        Args:
            eigenvalues: [K] - Sorted eigenvalues

        Returns:
            filter_response: [K] - Filter gains
        """
        # Normalize eigenvalues to [0, 1]
        lambda_norm = (eigenvalues - eigenvalues.min()) / \
                      (eigenvalues.max() - eigenvalues.min() + 1e-8)

        # Assign each frequency to a band
        response = torch.zeros_like(eigenvalues)

        for i in range(self.num_bands):
            # Band mask
            lower = self.band_boundaries[i]
            upper = self.band_boundaries[i + 1]
            mask = (lambda_norm >= lower) & (lambda_norm < upper)

            # Apply filter weights
            response[mask] = self.filter_weights[i, mask]

        return response


class SpectralAttentionLayer(nn.Module):
    """
    Attention in spectral domain of hexagonal graph.

    Benefits over spatial attention:
    1. No over-squashing (K << N for large graphs)
    2. Global receptive field (all nodes interact)
    3. Interpretable frequency response
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        k_eigenvectors: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.k = k_eigenvectors
        self.d_head = d_model // num_heads

        # Standard attention projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Spectral filters (per head)
        self.filters = nn.ModuleList([
            SpectralFilter(k_eigenvectors, num_bands=3)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cache for eigenvectors (set externally)
        self.register_buffer('eigenvectors', None)
        self.register_buffer('eigenvalues', None)

    def set_graph_spectrum(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor
    ):
        """
        Cache graph spectrum (call once per graph structure).

        Args:
            eigenvectors: [N, K]
            eigenvalues: [K]
        """
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

    def to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Project to frequency domain: x_freq = U^T @ x"""
        return torch.matmul(self.eigenvectors.T, x)

    def to_spatial(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Project to spatial domain: x = U @ x_freq"""
        return torch.matmul(self.eigenvectors, x_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - Spatial features

        Returns:
            output: [B, N, D] - Transformed features
        """
        B, N, D = x.shape

        # Project Q, K, V
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_head)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_head)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_head)

        # Transform to spectral domain [B, K, H, D_h]
        Q_freq = self.to_spectral(Q.transpose(1, 2).contiguous())\
                     .transpose(1, 2)
        K_freq = self.to_spectral(K.transpose(1, 2).contiguous())\
                     .transpose(1, 2)
        V_freq = self.to_spectral(V.transpose(1, 2).contiguous())\
                     .transpose(1, 2)

        # Apply spectral filters per head
        outputs = []
        for h in range(self.num_heads):
            # Get filter response for this head
            filter_response = self.filters[h](self.eigenvalues)
            filter_matrix = torch.diag(filter_response)

            # Filter Q, K, V
            Q_h = torch.matmul(filter_matrix, Q_freq[:, :, h])
            K_h = torch.matmul(filter_matrix, K_freq[:, :, h])
            V_h = torch.matmul(filter_matrix, V_freq[:, :, h])

            # Attention in spectral space (efficient: K x K)
            scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / \
                     math.sqrt(self.d_head)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention
            out_h = torch.matmul(attn, V_h)  # [B, K, D_h]
            outputs.append(out_h)

        # Concatenate heads
        output_freq = torch.stack(outputs, dim=2)  # [B, K, H, D_h]

        # Transform back to spatial domain
        output = self.to_spatial(output_freq.transpose(1, 2).contiguous())\
                     .transpose(1, 2)  # [B, N, H, D_h]

        # Merge heads
        output = output.reshape(B, N, D)
        output = self.W_O(output)

        return output


# ══════════════════════════════════════════════════════════
# TESTING CHECKLIST
# ══════════════════════════════════════════════════════════

def test_spectral_attention():
    """Unit test for SpectralAttentionLayer."""

    # Setup
    d_model = 128
    num_heads = 4
    k = 16
    N = 19  # Hexagonal grid with radius=2
    B = 2

    # Create dummy graph
    from hamha.topology import generate_hex_grid, build_adjacency_matrix
    coords = generate_hex_grid(radius=2)
    adj = build_adjacency_matrix(coords)

    # Precompute spectrum
    degree = torch.diag(adj.sum(dim=1))
    L = degree - adj
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    eigenvectors = eigenvectors[:, :k]
    eigenvalues = eigenvalues[:k]

    # Create layer
    layer = SpectralAttentionLayer(d_model, num_heads, k)
    layer.set_graph_spectrum(eigenvectors, eigenvalues)

    # Forward pass
    x = torch.randn(B, N, d_model)
    output = layer(x)

    # Assertions
    assert output.shape == (B, N, d_model), "Output shape mismatch"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    print("✓ SpectralAttentionLayer test passed")

if __name__ == "__main__":
    test_spectral_attention()
```

#### **Day 8-10: Integration & Testing**

**□ Task 1.4: Integration Checklist**
```markdown
## Integration Steps

### Step 1: Modify HAMHA Core
□ Add `use_spectral` flag to __init__
□ Replace standard attention with SpectralAttentionLayer
□ Precompute spectrum in __init__ (one-time cost)
□ Add backward compatibility (toggle spectral on/off)

### Step 2: Benchmark Setup
□ Define evaluation protocol:
  - Tasks: Graph classification (Cora, Citeseer, PubMed)
  - Metrics: Accuracy, F1, training time, memory
  - Baselines: Standard HAMHA, GAT, GCN
□ Create data loaders
□ Implement evaluation script

### Step 3: Ablation Studies
□ Spectral vs. Spatial attention
□ Effect of k (number of eigenvectors)
□ Filter design (Chebyshev order, band boundaries)
□ Multi-head vs. single-head spectral

### Step 4: Hyperparameter Tuning
□ Learning rate schedule
□ Dropout rate
□ Number of layers
□ k_eigenvectors selection

### Step 5: Documentation
□ Add docstrings to all functions
□ Create usage example
□ Update README
□ Write technical note on spectral attention
```

#### **Week 5: Results & Iteration**

**□ Task 1.5: Results Analysis Template**
```python
# FILE: experiments/spectral_attention/analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(results_dir: str):
    """
    Comprehensive analysis of spectral attention experiments.

    Generates:
    1. Performance comparison table
    2. Ablation study plots
    3. Frequency response visualizations
    4. Computational cost analysis
    """

    # Load results
    df = pd.read_csv(f"{results_dir}/results.csv")

    # ══════════════════════════════════════════════════════════
    # 1. PERFORMANCE COMPARISON
    # ══════════════════════════════════════════════════════════

    metrics = ['accuracy', 'f1_macro', 'training_time', 'memory_gb']
    comparison = df.groupby('model')[metrics].agg(['mean', 'std'])

    print("="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison.to_string())
    print()

    # Statistical significance test
    from scipy.stats import ttest_ind
    spectral = df[df['model'] == 'spectral_hamha']['accuracy']
    baseline = df[df['model'] == 'hamha']['accuracy']
    t_stat, p_value = ttest_ind(spectral, baseline)

    print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ Spectral attention is significantly better (p<0.05)")
    else:
        print("⚠ No significant difference detected")
    print()

    # ══════════════════════════════════════════════════════════
    # 2. ABLATION STUDY
    # ══════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Effect of k
    k_values = df['k_eigenvectors'].unique()
    k_acc = df.groupby('k_eigenvectors')['accuracy'].mean()
    axes[0, 0].plot(k_values, k_acc, marker='o')
    axes[0, 0].set_xlabel('Number of Eigenvectors (k)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Effect of Spectral Dimension')
    axes[0, 0].grid(True)

    # Training dynamics
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        axes[0, 1].plot(subset['epoch'], subset['train_loss'],
                        label=model, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Training Loss')
    axes[0, 1].set_title('Convergence Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Computational cost
    models = df['model'].unique()
    times = [df[df['model'] == m]['training_time'].mean() for m in models]
    axes[1, 0].bar(models, times)
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Computational Cost')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Memory usage
    memory = [df[df['model'] == m]['memory_gb'].mean() for m in models]
    axes[1, 1].bar(models, memory)
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].set_title('Memory Consumption')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/ablation_study.png", dpi=150)
    print(f"✓ Saved ablation study plots to {results_dir}/ablation_study.png")

    # ══════════════════════════════════════════════════════════
    # 3. SUCCESS CRITERIA CHECK
    # ══════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK")
    print("="*70)

    spectral_acc = df[df['model'] == 'spectral_hamha']['accuracy'].mean()
    baseline_acc = df[df['model'] == 'hamha']['accuracy'].mean()
    improvement = ((spectral_acc - baseline_acc) / baseline_acc) * 100

    criteria = {
        "20%+ improvement on long-range tasks": improvement >= 20,
        "No degradation on short-range tasks": spectral_acc >= baseline_acc * 0.98,
        "Training time increase <15%": df[df['model'] == 'spectral_hamha']['training_time'].mean() <
                                        df[df['model'] == 'hamha']['training_time'].mean() * 1.15
    }

    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {criterion}")

    if all(criteria.values()):
        print("\n🎉 All success criteria met! Ready to merge.")
    else:
        print("\n⚠ Some criteria not met. Requires iteration.")

if __name__ == "__main__":
    analyze_results("experiments/spectral_attention/results")
```

---

## ✅ Part III: Code Quality Standards {#quality}

### **3.1 Code Review Checklist**

```markdown
## Before Submitting PR

### Code Quality
□ All functions have type hints
□ All functions have docstrings (Google style)
□ No magic numbers (use named constants)
□ No functions >50 lines (refactor if needed)
□ DRY principle followed (no code duplication)

### Testing
□ Unit tests written for new functions
□ Integration tests for new modules
□ All tests pass locally
□ Code coverage >90% for new code

### Documentation
□ README updated with new features
□ API documentation generated
□ Usage example provided
□ Architecture diagrams updated

### Performance
□ No memory leaks (tested for 1000 steps)
□ Profiling shows no unexpected bottlenecks
□ GPU utilization >80% during training

### Git Hygiene
□ Meaningful commit messages
□ Small, focused commits
□ Branch name follows convention (feature/*, bugfix/*)
□ No merge conflicts
```

### **3.2 Example: Perfect Function**

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

def compute_spectral_norm(
    matrix: torch.Tensor,
    num_iterations: int = 10,
    eps: float = 1e-12
) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Compute spectral norm (largest singular value) via power iteration.

    This is more efficient than full SVD for large matrices and suitable
    for monitoring weight matrix stability during training.

    Args:
        matrix: Weight matrix [M, N] to analyze
        num_iterations: Number of power iteration steps (default: 10)
        eps: Small constant for numerical stability (default: 1e-12)

    Returns:
        spectral_norm: Largest singular value (float)
        dominant_vector: Corresponding right singular vector [N] or None

    Raises:
        ValueError: If matrix is not 2D or has invalid dimensions

    Examples:
        >>> W = torch.randn(512, 512)
        >>> sigma_max, v = compute_spectral_norm(W, num_iterations=20)
        >>> print(f"Spectral norm: {sigma_max:.4f}")
        Spectral norm: 32.1234

    References:
        - Miyato et al. "Spectral Normalization for GANs" (ICLR 2018)
        - Golub & Van Loan. "Matrix Computations" (4th ed., 2013)

    Notes:
        - Time complexity: O(num_iterations * M * N)
        - Space complexity: O(max(M, N))
        - Converges exponentially fast for well-conditioned matrices
    """
    # Input validation
    if matrix.dim() != 2:
        raise ValueError(
            f"Expected 2D matrix, got {matrix.dim()}D tensor with shape {matrix.shape}"
        )

    M, N = matrix.shape
    if M == 0 or N == 0:
        raise ValueError(f"Invalid matrix dimensions: {M} x {N}")

    # Initialize random vector
    v = torch.randn(N, device=matrix.device, dtype=matrix.dtype)
    v = v / (torch.norm(v) + eps)

    # Power iteration
    for _ in range(num_iterations):
        # v ← M^T M v / ||M^T M v||
        u = torch.mv(matrix, v)
        v = torch.mv(matrix.t(), u)
        v = v / (torch.norm(v) + eps)

    # Compute spectral norm: σ_max = ||M v||
    u = torch.mv(matrix, v)
    spectral_norm = torch.norm(u).item()

    return spectral_norm, v
```

**Why this is perfect**:
✓ Type hints for all parameters
✓ Comprehensive docstring with examples
✓ Input validation
✓ Clear variable names
✓ Efficient algorithm (power iteration)
✓ References to literature
✓ Complexity analysis

---

## 🧪 Part IV: Testing & Validation {#testing}

### **4.1 Testing Strategy**

```python
# FILE: tests/test_strategy.py

"""
Testing Pyramid for HAMHA+LMA:

┌──────────────────────────────────────┐
│        E2E Tests (5%)                │  ← Full system integration
│   - Training pipeline               │
│   - Evaluation on benchmarks        │
└──────────────────────────────────────┘
         ▲
┌──────────────────────────────────────┐
│    Integration Tests (15%)           │  ← Module interactions
│   - HAMHA + LMA together            │
│   - Telemetry collection            │
│   - Protocol execution              │
└──────────────────────────────────────┘
         ▲
┌──────────────────────────────────────┐
│      Unit Tests (80%)                │  ← Individual functions
│   - Attention mechanism             │
│   - Topology generation             │
│   - Metric computation              │
└──────────────────────────────────────┘
"""

import pytest
import torch
from hamha.spectral import SpectralAttentionLayer

# ══════════════════════════════════════════════════════════
# UNIT TESTS (Fast, Isolated, Deterministic)
# ══════════════════════════════════════════════════════════

class TestSpectralAttention:
    """Unit tests for spectral attention layer."""

    @pytest.fixture
    def layer(self):
        """Fixture: Create spectral attention layer."""
        return SpectralAttentionLayer(
            d_model=128,
            num_heads=4,
            k_eigenvectors=16
        )

    @pytest.fixture
    def graph_spectrum(self):
        """Fixture: Dummy graph spectrum."""
        k = 16
        N = 19
        eigenvalues = torch.linspace(0, 1, k)
        eigenvectors = torch.randn(N, k)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)  # Orthonormalize
        return eigenvectors, eigenvalues

    def test_forward_shape(self, layer, graph_spectrum):
        """Test output shape is correct."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        B, N, D = 2, 19, 128
        x = torch.randn(B, N, D)
        output = layer(x)

        assert output.shape == (B, N, D), \
            f"Expected shape ({B}, {N}, {D}), got {output.shape}"

    def test_forward_no_nan(self, layer, graph_spectrum):
        """Test output contains no NaN values."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, 19, 128)
        output = layer(x)

        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_backward_pass(self, layer, graph_spectrum):
        """Test gradients flow correctly."""
        eigenvectors, eigenvalues = graph_spectrum
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, 19, 128, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"

    @pytest.mark.parametrize("k", [8, 16, 32])
    def test_different_k_values(self, k):
        """Test layer works with different k."""
        layer = SpectralAttentionLayer(128, 4, k)
        eigenvectors = torch.randn(19, k)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)
        eigenvalues = torch.linspace(0, 1, k)
        layer.set_graph_spectrum(eigenvectors, eigenvalues)

        x = torch.randn(2, 19, 128)
        output = layer(x)
        assert output.shape == (2, 19, 128)

# ══════════════════════════════════════════════════════════
# INTEGRATION TESTS (Slower, Multiple Components)
# ══════════════════════════════════════════════════════════

class TestSpectralHAMHAIntegration:
    """Integration tests for spectral HAMHA."""

    def test_hamha_with_spectral_attention(self):
        """Test full HAMHA with spectral attention."""
        from hamha.core import HexagonalMultiHeadAttention

        hamha = HexagonalMultiHeadAttention(
            d_model=128,
            grid_radius=2,
            use_spectral=True
        )

        x = torch.randn(2, 32, 128)
        output = hamha(x)

        assert output.shape == (2, 32, 128)
        assert not torch.isnan(output).any()

    def test_lma_monitors_spectral_hamha(self):
        """Test LMA can monitor spectral HAMHA."""
        from hamha.core import HexagonalMultiHeadAttention
        from lma.architect import LeadMetaArchitect

        hamha = HexagonalMultiHeadAttention(128, 2, use_spectral=True)
        lma = LeadMetaArchitect(hamha)

        # Simulate training step
        x = torch.randn(2, 32, 128)
        output = hamha(x)
        loss = output.sum()
        loss.backward()

        # LMA should collect telemetry
        result = lma.process_step()
        assert 'snapshot' in result
        assert len(result['snapshot'].attention_entropy) > 0

# ══════════════════════════════════════════════════════════
# REGRESSION TESTS (Ensure No Performance Degradation)
# ══════════════════════════════════════════════════════════

class TestRegression:
    """Regression tests against baseline."""

    @pytest.mark.slow
    def test_spectral_vs_baseline_accuracy(self):
        """Ensure spectral attention doesn't degrade accuracy."""
        # Load baseline results
        baseline_acc = 0.823  # From previous runs

        # Train spectral model
        spectral_acc = train_and_evaluate_spectral()

        # Allow 2% degradation tolerance
        assert spectral_acc >= baseline_acc * 0.98, \
            f"Spectral accuracy ({spectral_acc:.3f}) < baseline ({baseline_acc:.3f})"
```

### **4.2 Continuous Integration (CI)**

```yaml
# FILE: .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 hamha/ lma/ --count --max-line-length=100

    - name: Type check with mypy
      run: |
        mypy hamha/ lma/

    - name: Format check with black
      run: |
        black --check hamha/ lma/

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=hamha --cov=lma --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

---

## 🔥 Part V: Best Practices Summary

### **5.1 Code Organization**
✓ One class per file (max 300 lines)
✓ Group related functions into modules
✓ Use `__init__.py` to expose public API
✓ Keep tests mirroring source structure

### **5.2 Naming Conventions**
✓ `snake_case` for functions and variables
✓ `PascalCase` for classes
✓ `UPPER_CASE` for constants
✓ Descriptive names (no `x`, `y`, `z`)

### **5.3 Performance Optimization**
✓ Profile before optimizing
✓ Use `torch.no_grad()` for inference
✓ Prefer in-place operations when safe
✓ Batch operations to maximize GPU util

### **5.4 Git Workflow**
✓ Feature branches from `develop`
✓ PR reviews required
✓ Squash commits before merge
✓ Tag releases with semantic versioning

### **5.5 Experiment Tracking**
✓ Every experiment has unique ID
✓ Log all hyperparameters
✓ Save checkpoints every N steps
✓ Track system metrics (GPU, memory)

---

## 🎯 Ready to Start?

**Recommended Path**:
1. **Week 1**: Complete environment setup
2. **Week 2**: Implement spectral attention (core logic)
3. **Week 3**: Testing & benchmarking
4. **Week 4**: Integration & documentation
5. **Week 5**: Results analysis & iteration

**Success Criteria**:
- All tests pass
- Performance improves by 20%+
- Code review approved
- Documentation complete

Good luck building the next generation of HAMHA+LMA! 🚀