"""
Полный Transformer Block для AnamorphX Neural Backend Extensions

Объединяет Multi-head Attention, Positional Encoding и Layer Normalization
в единый функциональный блок Transformer архитектуры.
"""

import numpy as np
from typing import Optional, Union, Tuple
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Импорты наших компонентов
from .attention import MultiHeadAttention, AttentionConfig
from .positional_encoding import PositionalEncoder, PositionalEncodingConfig
from .layer_norm import LayerNormalization, PreNormResidualBlock


class FeedForwardNetwork:
    """
    Feed-Forward Network для Transformer блока.
    
    Состоит из двух линейных слоев с ReLU активацией между ними.
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 use_torch: bool = True):
        """
        Инициализация Feed-Forward сети.
        
        Args:
            d_model: Размерность модели
            d_ff: Размерность скрытого слоя (обычно 4 * d_model)
            dropout: Вероятность dropout
            use_torch: Использовать PyTorch если доступен
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
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
        
    def _init_numpy_params(self):
        """Инициализация параметров для NumPy реализации."""
        # Xavier/Glorot инициализация
        limit1 = np.sqrt(6.0 / (self.d_model + self.d_ff))
        limit2 = np.sqrt(6.0 / (self.d_ff + self.d_model))
        
        self.W1 = np.random.uniform(-limit1, limit1, (self.d_model, self.d_ff)).astype(np.float32)
        self.b1 = np.zeros(self.d_ff, dtype=np.float32)
        
        self.W2 = np.random.uniform(-limit2, limit2, (self.d_ff, self.d_model)).astype(np.float32)
        self.b2 = np.zeros(self.d_model, dtype=np.float32)
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """Прямой проход Feed-Forward сети."""
        if self.use_torch and TORCH_AVAILABLE:
            return self._forward_torch(x)
        else:
            return self._forward_numpy(x)
    
    def _forward_torch(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """PyTorch реализация."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # x -> linear1 -> ReLU -> dropout -> linear2
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        
        return x
    
    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy fallback реализация."""
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        
        x = np.asarray(x, dtype=np.float32)
        
        # Первый линейный слой + ReLU
        x = np.dot(x, self.W1) + self.b1
        x = np.maximum(0, x)  # ReLU
        
        # Простой dropout (в режиме inference)
        if self.dropout > 0:
            x = x * (1 - self.dropout)
        
        # Второй линейный слой
        x = np.dot(x, self.W2) + self.b2
        
        return x
    
    def __call__(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """Делает объект вызываемым."""
        return self.forward(x)


class TransformerBlock:
    """
    Полный Transformer блок.
    
    Объединяет Multi-head Attention, Feed-Forward Network и Layer Normalization
    с residual connections в современной Pre-Norm архитектуре.
    """
    
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 use_torch: bool = True):
        """
        Инициализация Transformer блока.
        
        Args:
            d_model: Размерность модели
            num_heads: Количество голов внимания
            d_ff: Размерность Feed-Forward сети
            dropout: Вероятность dropout
            max_seq_length: Максимальная длина последовательности
            use_torch: Использовать PyTorch если доступен
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        # Создаем конфигурации для компонентов
        attention_config = AttentionConfig(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        pos_config = PositionalEncodingConfig(
            d_model=d_model,
            max_length=max_seq_length,
            dropout=dropout
        )
        
        # Компоненты блока
        self.attention = MultiHeadAttention(attention_config)
        
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_torch=use_torch
        )
        
        # Pre-Norm блоки
        self.attention_norm = PreNormResidualBlock(d_model)
        self.ff_norm = PreNormResidualBlock(d_model)
        
        # Positional Encoding (опционально)
        self.pos_encoding = PositionalEncoder(pos_config)
    
    def forward(self, 
                x: Union[np.ndarray, 'torch.Tensor'],
                mask: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
                add_positional: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Прямой проход Transformer блока.
        
        Args:
            x: Входной тензор формы (batch_size, seq_len, d_model)
            mask: Маска внимания (опционально)
            add_positional: Добавлять ли позиционное кодирование
            
        Returns:
            Выходной тензор той же формы
        """
        # Добавляем позиционное кодирование
        if add_positional:
            x, _ = self.pos_encoding.forward(x)
        
        # Self-attention с Pre-Norm и residual connection
        def attention_sublayer(normalized_x):
            output, _ = self.attention.forward(normalized_x, normalized_x, normalized_x, mask)
            return output
        
        x = self.attention_norm.forward(x, attention_sublayer)
        
        # Feed-forward с Pre-Norm и residual connection
        x = self.ff_norm.forward(x, self.feed_forward)
        
        return x
    
    def __call__(self, 
                 x: Union[np.ndarray, 'torch.Tensor'],
                 mask: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
                 add_positional: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Делает объект вызываемым."""
        return self.forward(x, mask, add_positional)


class TransformerEncoder:
    """
    Полный Transformer Encoder из нескольких блоков.
    """
    
    def __init__(self,
                 num_layers: int = 6,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 use_torch: bool = True):
        """
        Инициализация Transformer Encoder.
        
        Args:
            num_layers: Количество Transformer блоков
            d_model: Размерность модели
            num_heads: Количество голов внимания
            d_ff: Размерность Feed-Forward сети
            dropout: Вероятность dropout
            max_seq_length: Максимальная длина последовательности
            use_torch: Использовать PyTorch если доступен
        """
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Создаем стек Transformer блоков
        self.layers = []
        for i in range(num_layers):
            layer = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_length=max_seq_length,
                use_torch=use_torch
            )
            self.layers.append(layer)
        
        # Финальная нормализация
        self.final_norm = LayerNormalization(d_model, use_torch=use_torch)
    
    def forward(self, 
                x: Union[np.ndarray, 'torch.Tensor'],
                mask: Optional[Union[np.ndarray, 'torch.Tensor']] = None) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Прямой проход через все слои Encoder.
        
        Args:
            x: Входной тензор формы (batch_size, seq_len, d_model)
            mask: Маска внимания (опционально)
            
        Returns:
            Закодированное представление
        """
        # Проходим через все слои
        for i, layer in enumerate(self.layers):
            # Позиционное кодирование добавляем только в первом слое
            add_pos = (i == 0)
            x = layer(x, mask, add_positional=add_pos)
        
        # Финальная нормализация
        x = self.final_norm(x)
        
        return x
    
    def __call__(self, 
                 x: Union[np.ndarray, 'torch.Tensor'],
                 mask: Optional[Union[np.ndarray, 'torch.Tensor']] = None) -> Union[np.ndarray, 'torch.Tensor']:
        """Делает объект вызываемым."""
        return self.forward(x, mask)


def demo_transformer_block():
    """Демонстрация работы полного Transformer блока."""
    print("🚀 Демонстрация полного Transformer блока")
    print("=" * 60)
    
    # Параметры
    batch_size, seq_len, d_model = 2, 16, 512
    num_heads = 8
    d_ff = 2048
    
    # Создаем тестовые данные
    if TORCH_AVAILABLE:
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"📊 Входные данные (PyTorch): {x.shape}")
    else:
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        print(f"📊 Входные данные (NumPy): {x.shape}")
    
    # Тест одного Transformer блока
    print("\n🧠 Тест одного Transformer блока")
    print("-" * 40)
    
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    print(f"⚙️  Режим: {'PyTorch' if transformer_block.use_torch else 'NumPy fallback'}")
    
    import time
    start_time = time.time()
    
    output = transformer_block(x)
    
    end_time = time.time()
    
    print(f"✅ Transformer блок завершен за {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Выходная форма: {output.shape}")
    
    # Тест полного Encoder
    print("\n🏗️  Тест Transformer Encoder (3 слоя)")
    print("-" * 40)
    
    encoder = TransformerEncoder(
        num_layers=3,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    start_time = time.time()
    
    encoded = encoder(x)
    
    end_time = time.time()
    
    print(f"✅ Transformer Encoder завершен за {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Закодированная форма: {encoded.shape}")
    
    # Проверяем, что форма сохранилась
    input_shape = x.shape
    output_shape = encoded.shape if hasattr(encoded, 'shape') else np.array(encoded).shape
    
    print(f"🔍 Проверка формы: {input_shape} -> {output_shape}")
    print(f"✅ Форма {'сохранена' if input_shape == output_shape else 'изменена'}")
    
    return {
        'transformer_block': transformer_block,
        'encoder': encoder,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'single_block_time_ms': (end_time - start_time) * 1000,
        'encoder_time_ms': (end_time - start_time) * 1000
    }


if __name__ == "__main__":
    demo_transformer_block() 