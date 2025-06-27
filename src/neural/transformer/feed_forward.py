"""
Feed-Forward Network для AnamorphX Neural Backend Extensions

Реализует двухслойную feed-forward сеть для Transformer архитектуры.
Поддерживает как PyTorch, так и fallback режимы.
"""

import numpy as np
from typing import Union
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch не найден. Используется fallback реализация Feed-Forward Network.")


class FeedForwardNetwork:
    """
    Feed-Forward Network для Transformer блока.
    
    Архитектура: Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)
    Следует стандартной архитектуре из "Attention Is All You Need".
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 use_torch: bool = True):
        """
        Инициализация Feed-Forward сети.
        
        Args:
            d_model: Размерность модели (входа и выхода)
            d_ff: Размерность скрытого слоя (обычно 4 * d_model)
            dropout: Вероятность dropout
            activation: Функция активации ('relu', 'gelu', 'swish')
            use_torch: Использовать PyTorch если доступен
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        if self.use_torch:
            self._init_torch_layers()
        else:
            self._init_numpy_params()
    
    def _init_torch_layers(self):
        """Инициализация PyTorch слоев."""
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Инициализация весов (Xavier/Glorot)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)
        
    def _init_numpy_params(self):
        """Инициализация параметров для NumPy реализации."""
        # Xavier/Glorot инициализация
        limit1 = np.sqrt(6.0 / (self.d_model + self.d_ff))
        limit2 = np.sqrt(6.0 / (self.d_ff + self.d_model))
        
        self.W1 = np.random.uniform(-limit1, limit1, (self.d_model, self.d_ff)).astype(np.float32)
        self.b1 = np.zeros(self.d_ff, dtype=np.float32)
        
        self.W2 = np.random.uniform(-limit2, limit2, (self.d_ff, self.d_model)).astype(np.float32)
        self.b2 = np.zeros(self.d_model, dtype=np.float32)
    
    def _apply_activation(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """Применяет функцию активации."""
        if self.use_torch and isinstance(x, torch.Tensor):
            if self.activation == 'relu':
                return F.relu(x)
            elif self.activation == 'gelu':
                return F.gelu(x)
            elif self.activation == 'swish':
                return x * torch.sigmoid(x)
            else:
                return F.relu(x)  # По умолчанию ReLU
        else:
            # NumPy реализации
            if self.activation == 'relu':
                return np.maximum(0, x)
            elif self.activation == 'gelu':
                # Приближенная GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
            elif self.activation == 'swish':
                return x * (1 / (1 + np.exp(-x)))  # x * sigmoid(x)
            else:
                return np.maximum(0, x)  # По умолчанию ReLU
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """Прямой проход Feed-Forward сети."""
        if self.use_torch and TORCH_AVAILABLE:
            return self._forward_torch(x)
        else:
            return self._forward_numpy(x)
    
    def _forward_torch(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """PyTorch реализация прямого прохода."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # x → linear1 → activation → dropout → linear2
        x = self.linear1(x)
        x = self._apply_activation(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        
        return x
    
    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy fallback реализация прямого прохода."""
        if hasattr(x, 'detach'):  # Если это torch.Tensor
            x = x.detach().cpu().numpy()
        
        x = np.asarray(x, dtype=np.float32)
        
        # Первый линейный слой
        x = np.dot(x, self.W1) + self.b1
        
        # Активация
        x = self._apply_activation(x)
        
        # Dropout (в режиме inference - просто масштабирование)
        if self.dropout > 0:
            x = x * (1 - self.dropout)
        
        # Второй линейный слой
        x = np.dot(x, self.W2) + self.b2
        
        return x
    
    def __call__(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """Делает объект вызываемым."""
        return self.forward(x)


def demo_feed_forward_network():
    """Демонстрация работы Feed-Forward Network."""
    print("🔧 Демонстрация Feed-Forward Network")
    print("=" * 50)
    
    # Параметры
    batch_size, seq_len, d_model = 2, 16, 512
    d_ff = 2048
    
    # Создаем тестовые данные
    if TORCH_AVAILABLE:
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"📊 Входные данные (PyTorch): {x.shape}")
    else:
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        print(f"📊 Входные данные (NumPy): {x.shape}")
    
    # Тест стандартной FFN
    print(f"\n🔧 Тест: Feed-Forward Network")
    print("-" * 40)
    
    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, activation='relu')
    print(f"✅ FFN создана (d_model={d_model}, d_ff={d_ff})")
    print(f"   Режим: {'PyTorch' if ffn.use_torch else 'NumPy fallback'}")
    print(f"   Активация: {ffn.activation}")
    
    import time
    start_time = time.time()
    output = ffn(x)
    end_time = time.time()
    
    print(f"✅ Forward pass: {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Форма: {x.shape} → {output.shape}")
    
    # Тест различных активаций
    activations = ['relu', 'gelu', 'swish']
    print(f"\n🎯 Тест активаций:")
    
    for activation in activations:
        ffn_act = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, activation=activation)
        
        start_time = time.time()
        output_act = ffn_act(x)
        end_time = time.time()
        
        print(f"   {activation.upper()}: {(end_time - start_time) * 1000:.2f}мс")
    
    return {
        'ffn': ffn,
        'input_shape': x.shape,
        'output_shape': output.shape,
        'processing_time_ms': (end_time - start_time) * 1000
    }


if __name__ == "__main__":
    demo_feed_forward_network() 