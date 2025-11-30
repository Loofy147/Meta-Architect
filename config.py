from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HAMHAConfig:
    """Configuration settings for the HAMHA model.

    Attributes:
        d_model (int): The main dimensionality of the model.
        d_head (int): The dimensionality of each attention head.
        grid_radius (int): The radius of the hexagonal grid.
        use_hypernet (bool): Whether to use a HyperNetwork for projections.
        dropout (float): The dropout rate.
    """

    d_model: int = 512
    d_head: int = 64
    grid_radius: int = 2
    use_hypernet: bool = False
    dropout: float = 0.1


@dataclass
class LMAConfig:
    """Configuration settings for the Lead Meta-Architect.

    This dataclass holds all the tunable parameters for the LMA's subsystems,
    including telemetry collection, alert thresholds, intervention parameters,
    and prediction settings.

    Attributes:
        telemetry_window_size (int): The number of historical steps to consider
            for trend analysis.
        telemetry_collection_frequency (int): How often (in steps) to collect
            telemetry.
        kappa_threshold (float): The condition number threshold for triggering
            a rank collapse alert.
        fixation_threshold (float): The attention entropy threshold for
            triggering a fixation alert.
        drift_threshold (float): The attention entropy threshold for
            triggering a drift alert.
        entropy_derivative_threshold (float): The rate of entropy change
            threshold for a drift alert.
        vanishing_gradient_threshold (float): The lower bound for gradient norms.
        exploding_gradient_threshold (float): The upper bound for gradient norms.
        entropy_reg_increment (float): The amount to increase entropy
            regularization during a soft intervention.
        self_mixing_decay_factor (float): The factor to decay a head's self-
            influence during a soft intervention.
        neighbor_boost_factor (float): The factor to boost a head's neighbor
            influence during a soft intervention.
        prediction_horizon (int): The number of steps into the future to forecast.
        prediction_confidence_threshold (float): The confidence threshold for
            acting on a prediction.
        cmcg_edge_confidence_threshold (float): The confidence threshold for
            causal graph queries.
        gnn_opt_target_t_mix (float): The target mixing time for the GNN_OPT
            evolutionary module.
        adapt_bias_mode (str): The operational mode for the ADAPT_BIAS
            evolutionary module.
    """

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
    """A top-level container for the complete system configuration.

    This dataclass aggregates the configurations for all major subsystems,
    providing a single object to manage the overall system setup.

    Attributes:
        hamha (HAMHAConfig): The configuration for the HAMHA model.
        lma (LMAConfig): The configuration for the Lead Meta-Architect.
    """

    hamha: HAMHAConfig = None
    lma: LMAConfig = None

    def __post_init__(self):
        if self.hamha is None:
            self.hamha = HAMHAConfig()
        if self.lma is None:
            self.lma = LMAConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration to a dictionary.

        Returns:
            Dict[str, Any]: The configuration as a nested dictionary.
        """
        return {"hamha": self.hamha.__dict__, "lma": self.lma.__dict__}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemConfig":
        """Creates a SystemConfig instance from a dictionary.

        Args:
            config_dict (Dict[str, Any]): The dictionary to deserialize.

        Returns:
            SystemConfig: A new instance of the SystemConfig.
        """
        hamha_config = HAMHAConfig(**config_dict.get("hamha", {}))
        lma_config = LMAConfig(**config_dict.get("lma", {}))
        return cls(hamha=hamha_config, lma=lma_config)
