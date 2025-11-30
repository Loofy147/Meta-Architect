from typing import List, Dict, Any

"""Defines the search space for the Meta-NAS controller.

This module provides a centralized definition of the architectural choices
that the `MetaNASController` can make. This ensures that any generated
architecture is valid and provides a single point of control for
experimentation with the search space.
"""

SEARCH_SPACE: Dict[str, List[Any]] = {
    "d_head": [16, 32, 64, 128],
    "k_eigenvectors": [8, 16, 32, 64],
    "use_spectral": [True, False],
    "use_hypernet": [True, False],
    "grid_radius": [1, 2, 3],
}

def is_valid_architecture(arch_params: Dict[str, Any]) -> bool:
    """Validates a set of architecture parameters against the search space.

    This function ensures that a given architecture is valid by checking that:
    1. All hyperparameter names are defined in the `SEARCH_SPACE`.
    2. The value of each hyperparameter is one of the allowed values.

    Args:
        arch_params (Dict[str, Any]): A dictionary of hyperparameters
            representing the architecture to be validated.

    Returns:
        bool: True if the architecture is valid, False otherwise.
    """
    for key, value in arch_params.items():
        if key not in SEARCH_SPACE:
            return False
        if value not in SEARCH_SPACE[key]:
            return False
    return True
