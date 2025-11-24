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
