# 🚀 HAMHA + LMA Deployment Guide & Quick Reference

## Installation from Scratch

### 1. Clone and Setup

```bash
# Create project directory
mkdir hamha-lma && cd hamha-lma

# Initialize git
git init

# Copy all code files from artifacts into the following structure:
# hamha/
#   - core.py, heads.py, mixing.py, topology.py
# lma/
#   - architect.py, telemetry.py, hge.py, adp.py, cmcg.py, protocols.py, evolutionary.py
# utils/
#   - metrics.py, visualization.py
# tests/
#   - test_hamha.py, test_lma.py
# examples/
#   - training_integration.py
# Plus: main.py, config.py, cli.py, setup.py, requirements.txt, etc.

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run demo
python main.py
```

---

## Quick Start Examples

### Basic Usage

```python
import torch
from hamha.core import HexagonalMultiHeadAttention
from lma.architect import LeadMetaArchitect

# Initialize HAMHA
hamha = HexagonalMultiHeadAttention(
    d_model=512,      # Model dimension
    grid_radius=2,    # Creates 19 attention heads
    d_head=64,        # Head dimension
    use_hypernet=False
)

# Initialize LMA
lma = LeadMetaArchitect(hamha)

# Activate optimization modules
lma.command_activate_module('GNN_OPT', {'target_t_mix': 35})
lma.command_activate_module('ADAPT_BIAS', {'mode': 'exploration'})

# Forward pass
batch_size, seq_len = 4, 128
x = torch.randn(batch_size, seq_len, 512)
output = hamha(x)

# Simulate training step
loss = output.sum()
loss.backward()

# LMA monitoring
result = lma.process_step()
print(lma.generate_report())
```

---

## Integration Patterns

### Pattern 1: Training Loop Integration

```python
from torch.optim import AdamW
import torch.nn as nn

# Your model using HAMHA
class TransformerBlock(nn.Module):
    def __init__(self, d_model, grid_radius):
        super().__init__()
        self.attention = HexagonalMultiHeadAttention(d_model, grid_radius)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Initialize model and LMA
model = TransformerBlock(d_model=512, grid_radius=2)
lma = LeadMetaArchitect(model.attention)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # LMA monitoring (every N steps)
        if step % monitoring_frequency == 0:
            result = lma.process_step()
            
            # Log metrics
            wandb.log({
                'loss': loss.item(),
                'health': result['status']['health'],
                'avg_entropy': result['status']['avg_entropy'],
                'max_kappa': result['status']['max_kappa']
            })
            
            # Handle interventions
            if result['interventions']:
                print(f"LMA Interventions: {result['interventions']}")
```

### Pattern 2: Manual Control

```python
# Get current telemetry
snapshot = lma.telemetry.history[-1]

# Check specific head
coord_str = "H(0,0)"
entropy = snapshot.attention_entropy.get(coord_str, 1.0)
kappa = snapshot.condition_numbers.get(f"{coord_str}_Q", 1.0)

# Manual intervention if needed
if entropy < 0.7:
    lma.command_adjust_entropy_regularization(delta=0.02)
    print("Manually increased entropy regularization")

# Reset problematic head
if kappa > 200:
    from hamha.topology import HexCoordinate
    lma.command_reset_head(HexCoordinate(0, 0), strategy='orthogonal')
    print("Reset head H(0,0) due to rank collapse")

# Query predictions
predictions = lma.adp.predict_entropy_trajectory(
    lma.telemetry.history,
    coord_str="H(0,0)",
    steps_ahead=50
)
print(f"Fixation risk: {predictions['fixation_risk']}")
```

### Pattern 3: Custom Emergency Protocols

```python
class CustomProtocols(lma.protocols.__class__):
    """Extend emergency protocols with custom logic."""
    
    def trigger_custom_intervention(self, target_head_idx: int):
        """Custom intervention strategy."""
        head = self.model.heads[target_head_idx]
        
        # Custom logic: Gradually increase projection diversity
        with torch.no_grad():
            # Add orthogonal noise
            U, S, Vh = torch.linalg.svd(head.W_Q_base)
            noise = torch.randn_like(head.W_Q_base)
            noise_orth = noise - (noise @ U) @ U.T
            head.W_Q_base.data += noise_orth * 0.01
        
        self.protocol_history.append({
            'protocol': 'CUSTOM_INTERVENTION',
            'target_head': target_head_idx,
            'timestamp': time.time()
        })

# Replace protocols
lma.protocols = CustomProtocols(lma.model)
```

---

## Configuration Management

### Using Config Files

```python
from config import SystemConfig, HAMHAConfig, LMAConfig
import json

# Create configuration
config = SystemConfig(
    hamha=HAMHAConfig(
        d_model=768,
        d_head=96,
        grid_radius=3,  # 37 heads
        use_hypernet=True
    ),
    lma=LMAConfig(
        kappa_threshold=75.0,  # More strict
        fixation_threshold=0.4,  # More lenient
        entropy_reg_increment=0.005,  # Smaller steps
        prediction_horizon=50  # Longer predictions
    )
)

# Save configuration
with open('configs/production.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)

# Load and use
with open('configs/production.json', 'r') as f:
    config = SystemConfig.from_dict(json.load(f))

hamha = HexagonalMultiHeadAttention(
    d_model=config.hamha.d_model,
    grid_radius=config.hamha.grid_radius,
    d_head=config.hamha.d_head,
    use_hypernet=config.hamha.use_hypernet
)
```

---

## Monitoring & Visualization

### Real-time Monitoring Dashboard

```python
from utils.visualization import TelemetryVisualizer
import matplotlib.pyplot as plt

# After some training steps
viz = TelemetryVisualizer(lma.telemetry.history)

# Create monitoring dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Entropy evolution
plt.sca(axes[0, 0])
viz.plot_entropy_evolution()

# Plot 2: Spectral stability
plt.sca(axes[0, 1])
viz.plot_condition_numbers()

# Plot 3: Computational profile
plt.sca(axes[1, 0])
viz.plot_computational_profile()

# Plot 4: Gradient flow
plt.sca(axes[1, 1])
viz.plot_gradient_flow()

plt.tight_layout()
plt.savefig('monitoring_dashboard.png', dpi=150)
```

### Export Telemetry Data

```python
import pandas as pd

# Convert history to DataFrame
telemetry_data = []
for snapshot in lma.telemetry.history:
    row = {
        'step': snapshot.step,
        'timestamp': snapshot.timestamp,
        't_mix': snapshot.t_mix,
        't_total': snapshot.t_total,
        'throughput': snapshot.throughput_tps,
        'num_alerts': len(snapshot.alerts)
    }
    
    # Add per-head metrics
    for coord, entropy in snapshot.attention_entropy.items():
        row[f'entropy_{coord}'] = entropy
    
    telemetry_data.append(row)

df = pd.DataFrame(telemetry_data)
df.to_csv('telemetry_export.csv', index=False)
print(f"Exported {len(df)} telemetry snapshots")
```

---

## Advanced Features

### 1. HyperNetwork Mode

```python
# Enable dynamic projection generation
hamha = HexagonalMultiHeadAttention(
    d_model=512,
    grid_radius=2,
    use_hypernet=True  # Projects based on (q,r) and global context
)

# HyperNetwork adapts to input characteristics
x_simple = torch.randn(4, 64, 512)   # Simple input
x_complex = torch.randn(4, 256, 512)  # Complex input

output_simple = hamha(x_simple)
output_complex = hamha(x_complex)
# Different internal projections for different inputs!
```

### 2. Hypothesis Testing

```python
# Generate hypotheses from current state
snapshot = lma.telemetry.history[-1]
hypotheses = lma.hge.generate_from_snapshot(snapshot)

for hyp in hypotheses:
    print(f"\n{hyp.id}: {hyp.description}")
    print(f"Antecedent: {hyp.antecedent}")
    print(f"Consequent: {hyp.consequent}")
    print(f"Confidence: {hyp.confidence:.2f}")
    print(f"Prediction: {hyp.testable_prediction}")
    
    # Test hypothesis using EDAS
    # (Implementation depends on specific hypothesis)
```

### 3. Causal Graph Analysis

```python
import networkx as nx
import matplotlib.pyplot as plt

# Visualize causal relationships
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(lma.cmcg.graph)
nx.draw(lma.cmcg.graph, pos, with_labels=True, 
        node_color='lightblue', node_size=1500,
        font_size=10, font_weight='bold')

# Draw edge labels (confidence scores)
edge_labels = {(u, v): f"{d['confidence']:.2f}" 
               for u, v, d in lma.cmcg.graph.edges(data=True)}
nx.draw_networkx_edge_labels(lma.cmcg.graph, pos, edge_labels)

plt.title("Cross-Modal Causal Graph")
plt.tight_layout()
plt.savefig('causal_graph.png', dpi=150)

# Query causal relationships
effect = "fixation"
causes = lma.cmcg.get_likely_causes(effect, threshold=0.7)
print(f"Likely causes of {effect}: {causes}")
```

---

## Command-Line Interface

### Basic Commands

```bash
# Run demo with custom steps
python cli.py --mode demo --steps 200

# Training mode
python cli.py --mode train --d-model 768 --grid-radius 3

# Use custom config
python cli.py --mode train --config configs/production.json

# Save telemetry
python cli.py --mode demo --save-telemetry --steps 500
```

### Programmatic CLI Usage

```python
import subprocess

# Run HAMHA via CLI
result = subprocess.run([
    'python', 'cli.py',
    '--mode', 'demo',
    '--steps', '100',
    '--d-model', '512',
    '--grid-radius', '2'
], capture_output=True, text=True)

print(result.stdout)
```

---

## Performance Tuning

### Grid Size Selection

```python
# Grid size vs computational cost
radii_costs = {
    1: 7,    # 7 heads
    2: 19,   # 19 heads
    3: 37,   # 37 heads
    4: 61,   # 61 heads
}

# Formula: n_heads = 3 * r^2 + 3 * r + 1

# Choose based on:
# - Model capacity needs
# - Computational budget
# - Desired spatial reasoning granularity

# Small models: radius=1 or 2
# Large models: radius=3 or 4
```

### Memory Optimization

```python
# Use gradient checkpointing for large grids
from torch.utils.checkpoint import checkpoint

class CheckpointedHAMHA(HexagonalMultiHeadAttention):
    def forward(self, x):
        return checkpoint(super().forward, x)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = hamha(x)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Troubleshooting

### Issue: High T_mix Latency

```python
# Diagnosis
if lma.telemetry.history[-1].t_mix > 50:
    print("High T_mix detected")
    
    # Solution 1: Activate GNN optimization
    lma.command_activate_module('GNN_OPT', {'target_t_mix': 35})
    
    # Solution 2: Reduce grid size
    # hamha = HexagonalMultiHeadAttention(d_model=512, grid_radius=1)
```

### Issue: Entropy Fixation

```python
# Diagnosis
for coord, entropy in snapshot.attention_entropy.items():
    if entropy < 0.3:
        print(f"Fixation detected in {coord}")
        
        # Solution: Automatic (AAP_AD triggers)
        # Or manual:
        lma.command_adjust_entropy_regularization(delta=0.02)
```

### Issue: Rank Collapse

```python
# Diagnosis
for key, kappa in snapshot.condition_numbers.items():
    if kappa > 100:
        print(f"Rank collapse in {key}")
        
        # Solution: Reset head
        coord = HexCoordinate(0, 0)  # Extract from key
        lma.command_reset_head(coord, strategy='orthogonal')
```

---

## Production Deployment

### Docker Container

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hamha-lma
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: hamha
        image: your-org/hamha-lma:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "16Gi"
            cpu: "8"
```

---

## Support & Resources

- **Documentation**: See `docs/` folder
- **Examples**: See `examples/` folder
- **Issues**: GitHub Issues
- **Community**: GitHub Discussions

---

**🔷 Built with hexagonal precision and meta-architectural intelligence 🔷**
