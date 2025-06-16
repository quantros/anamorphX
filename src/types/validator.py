from dataclasses import dataclass
from typing import Any, Dict, List
# from . import AnamorphType
class TypeValidator:
    def __init__(self): self.stats = {"total": 0, "passed": 0}
    def validate_neuron(self, value): return hasattr(value, "activate")
    def validate_tensor(self, value): return hasattr(value, "shape")
    def validate_layer(self, value): return hasattr(value, "forward")
validator = TypeValidator()
