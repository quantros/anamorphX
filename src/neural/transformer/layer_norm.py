"""
Layer Normalization для AnamorphX Neural Backend Extensions

Реализует нормализацию по слоям для стабилизации обучения в Transformer архитектурах.
Поддерживает как PyTorch, так и fallback режимы.
"""

import numpy as np
from typing import Optional, Tuple, Union
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch не найден. Используется fallback реализация Layer Normalization.")


class LayerNormalization:
    """
    Layer Normalization для стабилизации обучения в нейронных сетях.
    
    Нормализует входные данные по последней размерности (features),
    применяя обучаемые параметры gamma (scale) и beta (shift).
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 eps: float = 1e-6,
                 use_torch: bool = True):
        """
        Инициализация Layer Normalization.
        
        Args:
            d_model: Размерность модели (количество признаков)
            eps: Малое значение для численной стабильности
            use_torch: Использовать PyTorch если доступен
        """
        self.d_model = d_model
        self.eps = eps
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        if self.use_torch:
            self._init_torch_layer()
        else:
            self._init_numpy_params()
    
    def _init_torch_layer(self):
        """Инициализация PyTorch слоя."""
        self.layer_norm = nn.LayerNorm(self.d_model, eps=self.eps)
        
    def _init_numpy_params(self):
        """Инициализация параметров для NumPy реализации."""
        # Обучаемые параметры
        self.gamma = np.ones(self.d_model, dtype=np.float32)  # scale
        self.beta = np.zeros(self.d_model, dtype=np.float32)  # shift
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Прямой проход Layer Normalization.
        
        Args:
            x: Входной тензор формы (..., d_model)
            
        Returns:
            Нормализованный тензор той же формы
        """
        if self.use_torch and TORCH_AVAILABLE:
            return self._forward_torch(x)
        else:
            return self._forward_numpy(x)
    
    def _forward_torch(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """PyTorch реализация прямого прохода."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        return self.layer_norm(x)
    
    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy fallback реализация прямого прохода."""
        if hasattr(x, 'detach'):  # Если это torch.Tensor
            x = x.detach().cpu().numpy()
        
        x = np.asarray(x, dtype=np.float32)
        
        # Вычисляем среднее и дисперсию по последней размерности
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Нормализация
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Применяем обучаемые параметры
        return self.gamma * x_norm + self.beta
    
    def __call__(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """Делает объект вызываемым."""
        return self.forward(x)


class PreNormResidualBlock:
    """
    Pre-Norm Residual Block с Layer Normalization.
    
    Применяет Layer Norm перед основной операцией и добавляет residual connection.
    Это современный подход, используемый в GPT и других Transformer моделях.
    """
    
    def __init__(self, d_model: int = 512, eps: float = 1e-6):
        """
        Инициализация Pre-Norm блока.
        
        Args:
            d_model: Размерность модели
            eps: Параметр для численной стабильности
        """
        self.layer_norm = LayerNormalization(d_model, eps)
        self.d_model = d_model
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor'], 
                sublayer_fn: callable) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Прямой проход Pre-Norm блока.
        
        Args:
            x: Входной тензор
            sublayer_fn: Функция подслоя (например, attention или feed-forward)
            
        Returns:
            Выход с residual connection: x + sublayer(LayerNorm(x))
        """
        # Pre-normalization
        normalized = self.layer_norm(x)
        
        # Применяем подслой
        sublayer_output = sublayer_fn(normalized)
        
        # Residual connection
        if hasattr(x, 'shape') and hasattr(sublayer_output, 'shape'):
            if x.shape != sublayer_output.shape:
                warnings.warn(f"Размерности не совпадают: {x.shape} vs {sublayer_output.shape}")
        
        return x + sublayer_output


def demo_layer_normalization():
    """Демонстрация работы Layer Normalization."""
    print("🧠 Демонстрация Layer Normalization")
    print("=" * 50)
    
    # Параметры
    batch_size, seq_len, d_model = 2, 10, 512
    
    # Создаем тестовые данные
    if TORCH_AVAILABLE:
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"📊 Входные данные (PyTorch): {x.shape}")
    else:
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        print(f"📊 Входные данные (NumPy): {x.shape}")
    
    # Создаем Layer Normalization
    layer_norm = LayerNormalization(d_model=d_model)
    
    print(f"⚙️  Режим: {'PyTorch' if layer_norm.use_torch else 'NumPy fallback'}")
    
    # Прямой проход
    import time
    start_time = time.time()
    
    normalized = layer_norm(x)
    
    end_time = time.time()
    
    print(f"✅ Нормализация завершена за {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Выходная форма: {normalized.shape}")
    
    # Проверяем статистики
    if hasattr(normalized, 'detach'):
        stats_data = normalized.detach().cpu().numpy()
    else:
        stats_data = normalized
    
    mean = np.mean(stats_data, axis=-1)
    std = np.std(stats_data, axis=-1)
    
    print(f"📊 Среднее значение: {np.mean(mean):.6f} (должно быть ~0)")
    print(f"📊 Стандартное отклонение: {np.mean(std):.6f} (должно быть ~1)")
    
    # Тест Pre-Norm блока
    print("\n🔄 Тест Pre-Norm Residual Block")
    print("-" * 30)
    
    pre_norm_block = PreNormResidualBlock(d_model)
    
    # Простая функция подслоя (identity)
    def identity_sublayer(x):
        return x * 0.1  # Небольшое изменение для демонстрации
    
    start_time = time.time()
    residual_output = pre_norm_block.forward(x, identity_sublayer)
    end_time = time.time()
    
    print(f"✅ Pre-Norm блок завершен за {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Residual выход: {residual_output.shape}")
    
    return {
        'layer_norm': layer_norm,
        'pre_norm_block': pre_norm_block,
        'input_shape': x.shape,
        'output_shape': normalized.shape,
        'processing_time_ms': (end_time - start_time) * 1000
    }


if __name__ == "__main__":
    demo_layer_normalization() 