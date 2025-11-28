from typing import List, Dict, Any

# Defines the search space for the Meta-NAS controller.
# This centralized definition ensures that generated architectures are always valid
# and provides a single point of control for experimentation.

SEARCH_SPACE: Dict[str, List[Any]] = {
    "d_head": [16, 32, 64, 128],
    "k_eigenvectors": [8, 16, 32, 64],
    "use_spectral": [True, False],
    "use_hypernet": [True, False],
    "grid_radius": [1, 2, 3],
}

def is_valid_architecture(arch_params: Dict[str, Any]) -> bool:
    """
    Validates a set of architecture parameters against the defined search space.

    Args:
        arch_params: A dictionary of hyperparameters.

    Returns:
        True if the architecture is valid, False otherwise.
    """
    for key, value in arch_params.items():
        if key not in SEARCH_SPACE:
            return False
        if value not in SEARCH_SPACE[key]:
            return False
    return True
