from typing import Dict, List

class EvolutionaryModules:
    """Evolutionary horizon modules for HAMHA development."""

    def __init__(self):
        self.modules = {
            'GNN_OPT': {'active': False, 'progress': 0},
            'ADAPT_BIAS': {'active': False, 'progress': 0},
            'ADV_GRID': {'active': False, 'progress': 0},
            'SANDBOX': {'active': False, 'progress': 0}
        }

    def activate_module(self, module_name: str, parameters: Dict = None):
        """Activate an evolutionary module."""
        if module_name in self.modules:
            self.modules[module_name]['active'] = True
            self.modules[module_name]['parameters'] = parameters or {}
            return f"Module {module_name} activated"
        return f"Unknown module: {module_name}"

    def deactivate_module(self, module_name: str):
        """Deactivate an evolutionary module."""
        if module_name in self.modules:
            self.modules[module_name]['active'] = False
            return f"Module {module_name} deactivated"
        return f"Unknown module: {module_name}"

    def get_active_modules(self) -> List[str]:
        """Return list of currently active modules."""
        return [name for name, info in self.modules.items() if info['active']]
