# Project Structure

```
hamha-lma/
│
├── hamha/                      # Core HAMHA implementation
│   ├── __init__.py
│   ├── core.py                # Main HAMHA module
│   ├── heads.py               # Attention head implementations
│   ├── mixing.py              # GNN mixing layer
│   └── topology.py            # Hexagonal grid utilities
│
├── lma/                        # Lead Meta-Architect
│   ├── __init__.py
│   ├── architect.py           # Main LMA controller
│   ├── telemetry.py           # Telemetry collection
│   ├── hge.py                 # Hypothesis Generation Engine
│   ├── adp.py                 # Architectural Dynamics Predictor
│   ├── cmcg.py                # Cross-Modal Causal Graph
│   ├── edas.py                # Experiment Design & Analysis
│   ├── protocols.py           # Emergency response protocols
│   └── evolutionary.py        # Evolutionary horizon modules
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── metrics.py             # Metric computation
│   └── visualization.py       # Telemetry visualization
│
├── tests/                      # Testing suite
│   ├── __init__.py
│   ├── test_hamha.py          # HAMHA tests
│   ├── test_lma.py            # LMA tests
│   ├── test_telemetry.py      # Telemetry tests
│   └── test_integration.py    # Integration tests
│
├── examples/                   # Usage examples
│   ├── __init__.py
│   ├── training_integration.py # Training loop example
│   ├── custom_architecture.py  # Custom model example
│   └── notebooks/              # Jupyter notebooks
│       ├── demo.ipynb
│       └── analysis.ipynb
│
├── docs/                       # Documentation
│   ├── conf.py                # Sphinx config
│   ├── index.rst
│   ├── api/
│   ├── tutorials/
│   └── architecture/
│
├── config.py                   # System configuration
├── main.py                     # Main entry point
├── cli.py                      # Command-line interface
│
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Development dependencies
├── Makefile                    # Build commands
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── PROJECT_STRUCTURE.md        # This file
```

## Module Descriptions

### Core Modules

**hamha/core.py**
- `HexagonalMultiHeadAttention`: Main attention mechanism
- Grid management and head coordination

**hamha/heads.py**
- `AttentionHead`: Individual attention head
- `CoordinateBiasFunction`: Coordinate-dependent biases
- `HyperNetwork`: Dynamic projection generation

**hamha/mixing.py**
- `GNNMixingLayer`: Graph neural network aggregation
- Neighbor-based information propagation

**hamha/topology.py**
- `HexCoordinate`: Hexagonal coordinate system
- `generate_hex_grid()`: Grid generation
- `build_adjacency_matrix()`: Topology construction

### LMA Modules

**lma/architect.py**
- `LeadMetaArchitect`: Central control system
- Integration of all LMA subsystems
- Command interface

**lma/telemetry.py**
- `TelemetryCollector`: Real-time data collection
- `TelemetrySnapshot`: Data structure for snapshots
- Alert generation

**lma/hge.py**
- `HypothesisGenerationEngine`: Causal hypothesis creation
- `Hypothesis`: Hypothesis data structure
- Pattern matching algorithms

**lma/adp.py**
- `ArchitecturalDynamicsPredictor`: Forecasting system
- Time-series analysis
- Risk assessment

**lma/cmcg.py**
- `CrossModalCausalGraph`: Bayesian causal network
- Edge confidence updating
- Cause/effect inference

**lma/protocols.py**
- `EmergencyProtocols`: Intervention system
- AAP_AD Phase 1/2 implementations
- Head reset procedures

**lma/evolutionary.py**
- `EvolutionaryModules`: Module management
- Module activation/deactivation
- Progress tracking

### Utility Modules

**utils/metrics.py**
- `MetricsCalculator`: Metric computation utilities
- Condition number, entropy, gradient norms
- Spectral analysis

**utils/visualization.py**
- `TelemetryVisualizer`: Plotting utilities
- Entropy evolution plots
- Computational profile visualization

## Data Flow

```
Input → HAMHA → Output
  ↓       ↑
  ↓       │
  ↓    Commands
  ↓       │
  └→ LMA ─┘
     ├─ Telemetry Collector
     ├─ CMCG
     ├─ HGE
     ├─ ADP
     └─ Protocols
```

## Configuration Files

- `config.py`: System-wide configuration
- `setup.py`: Package metadata and dependencies
- `requirements.txt`: Runtime dependencies
- `requirements-dev.txt`: Development dependencies
- `.gitignore`: Git exclusions
- `Makefile`: Build automation
