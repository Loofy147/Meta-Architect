import torch
from hamha.core import HexagonalMultiHeadAttention
from hamha.topology import generate_hex_grid, HexCoordinate
from hamha.heads import AttentionHead
from lma.telemetry import TelemetryCollector
from lma.cmcg import CrossModalCausalGraph
from lma.hge import HypothesisGenerationEngine
from lma.telemetry import TelemetrySnapshot
from lma.architect import LeadMetaArchitect

def test_hamha_forward():
    """Test basic forward pass."""
    d_model = 128
    grid_radius = 1
    batch_size = 2
    seq_len = 32
    model = HexagonalMultiHeadAttention(d_model, grid_radius)
    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)
    assert output.shape == (batch_size, seq_len, d_model)

def test_hexagonal_topology():
    """Test hexagonal grid generation."""
    coords = generate_hex_grid(radius=2)
    assert len(coords) == 19  # 1 + 6 + 12 for radius 2

    # Check neighbors
    center = HexCoordinate(0, 0)
    neighbors = center.neighbors()
    assert len(neighbors) == 6

def test_attention_head():
    """Test single attention head."""
    d_model = 128
    batch_size = 2
    seq_len = 32
    coord = HexCoordinate(0, 0)
    head = AttentionHead(coord, d_model, 64)
    x = torch.randn(batch_size, seq_len, d_model)
    output = head(x)
    assert output.shape == (batch_size, seq_len, 64)

def test_telemetry_collection():
    """Test telemetry collector."""
    model = HexagonalMultiHeadAttention(128, 1)
    collector = TelemetryCollector(model)
    snapshot = collector.collect()
    assert snapshot is not None
    assert snapshot.step == 0

def test_cmcg_creation():
    """Test CMCG initialization."""
    cmcg = CrossModalCausalGraph()
    assert cmcg.graph.number_of_nodes() > 0
    assert cmcg.graph.number_of_edges() > 0

def test_hypothesis_generation():
    """Test HGE."""
    cmcg = CrossModalCausalGraph()
    hge = HypothesisGenerationEngine(cmcg)
    snapshot = TelemetrySnapshot(step=0, timestamp=0.0)
    snapshot.condition_numbers = {'H(0,0)_Q': 150.0}
    snapshot.gradient_norms = {'H(0,0)': 1e-7}

    hypotheses = hge.generate_from_snapshot(snapshot)
    assert len(hypotheses) > 0

def test_lma_integration():
    """Test full LMA integration."""
    model = HexagonalMultiHeadAttention(128, 1)
    lma = LeadMetaArchitect(model)
    # Create dummy input and forward/backward pass to generate grads and attention weights
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 128)
    output = model(x)
    loss = output.sum()
    loss.backward()

    result = lma.process_step()
    assert 'snapshot' in result
    assert 'status' in result
