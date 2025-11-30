from typing import List, Dict


class EvolutionaryModules:
    """Manages the lifecycle of long-term evolutionary modules.

    This class acts as a registry and controller for different "evolutionary"
    subsystems of the LMA. These modules represent experimental or long-term
    research directions that can be activated or deactivated as needed.

    Attributes:
        modules (Dict[str, Dict]): A dictionary tracking the status and
            parameters of each registered module.
    """

    def __init__(self):
        """Initializes the EvolutionaryModules manager."""
        self.modules = {
            "GNN_OPT": {"active": False, "progress": 0},
            "ADAPT_BIAS": {"active": False, "progress": 0},
            "ADV_GRID": {"active": False, "progress": 0},
            "SANDBOX": {"active": False, "progress": 0},
        }

    def activate_module(self, module_name: str, parameters: Dict = None):
        """Activates an evolutionary module.

        Args:
            module_name (str): The name of the module to activate.
            parameters (Dict, optional): A dictionary of parameters to configure
                the module. Defaults to None.

        Returns:
            str: A message indicating the result of the operation.
        """
        if module_name in self.modules:
            self.modules[module_name]["active"] = True
            self.modules[module_name]["parameters"] = parameters or {}
            return f"Module {module_name} activated"
        return f"Unknown module: {module_name}"

    def deactivate_module(self, module_name: str):
        """Deactivates an evolutionary module.

        Args:
            module_name (str): The name of the module to deactivate.

        Returns:
            str: A message indicating the result of the operation.
        """
        if module_name in self.modules:
            self.modules[module_name]["active"] = False
            return f"Module {module_name} deactivated"
        return f"Unknown module: {module_name}"

    def get_active_modules(self) -> List[str]:
        """Returns a list of currently active modules.

        Returns:
            List[str]: A list of the names of the active modules.
        """
        return [name for name, info in self.modules.items() if info["active"]]
