# Neural Backend Extensions для AnamorphX

__version__ = "0.1.0"
__author__ = "AnamorphX Team"

# Проверка доступности PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None
    print("Warning: PyTorch not available")

class NeuralExtension:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.enabled = TORCH_AVAILABLE
