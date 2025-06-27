"""
Complete Transformer Block для AnamorphX Neural Backend Extensions

Полная реализация Transformer блока, объединяющая все компоненты:
- Multi-head Attention
- Positional Encoding  
- Layer Normalization
- Feed-Forward Network

Следует современной Pre-Norm архитектуре.
"""

import sys
import os
import numpy as np
from typing import Optional, Union
import warnings

# Добавляем путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch не найден. Используется fallback реализация Transformer.")

# Импорты наших компонентов
from attention import MultiHeadAttention, AttentionConfig
from positional_encoding import PositionalEncoder, PositionalEncodingConfig
from layer_norm import LayerNormalization, PreNormResidualBlock
from feed_forward import FeedForwardNetwork


class TransformerBlock:
    """
    Полный Transformer блок с Pre-Norm архитектурой.
    
    Архитектура:
    1. x + MultiHeadAttention(LayerNorm(x))
    2. x + FeedForward(LayerNorm(x))
    
    Это современная Pre-Norm архитектура, используемая в GPT и других моделях.
    """
    
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 max_seq_length: int = 1000,
                 use_torch: bool = True):
        """
        Инициализация Transformer блока.
        
        Args:
            d_model: Размерность модели
            num_heads: Количество голов внимания
            d_ff: Размерность Feed-Forward сети
            dropout: Вероятность dropout
            activation: Функция активации для FFN
            max_seq_length: Максимальная длина последовательности
            use_torch: Использовать PyTorch если доступен
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        # Создаем все компоненты
        self._init_components()
    
    def _init_components(self):
        """Инициализация всех компонентов Transformer блока."""
        # Multi-head Attention
        attention_config = AttentionConfig(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        self.attention = MultiHeadAttention(attention_config)
        
        # Feed-Forward Network
        self.feed_forward = FeedForwardNetwork(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            use_torch=self.use_torch
        )
        
        # Layer Normalization блоки (Pre-Norm)
        self.attention_norm = PreNormResidualBlock(self.d_model)
        self.ff_norm = PreNormResidualBlock(self.d_model)
        
        # Positional Encoding (опционально)
        pos_config = PositionalEncodingConfig(
            d_model=self.d_model,
            max_length=1000,
            encoding_type="sinusoidal"
        )
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
        # 1. Позиционное кодирование (если нужно)
        if add_positional:
            x = self.pos_encoding(x)
        
        # 2. Self-attention с Pre-Norm и residual connection
        def attention_sublayer(normalized_x):
            output, _ = self.attention.forward(normalized_x, normalized_x, normalized_x, mask)
            return output
        
        x = self.attention_norm.forward(x, attention_sublayer)
        
        # 3. Feed-forward с Pre-Norm и residual connection
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
    
    Стек из N Transformer блоков с финальной нормализацией.
    """
    
    def __init__(self,
                 num_layers: int = 6,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
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
            activation: Функция активации
            max_seq_length: Максимальная длина последовательности
            use_torch: Использовать PyTorch если доступен
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        # Создаем стек Transformer блоков
        self.layers = []
        for i in range(num_layers):
            layer = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
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


def demo_complete_transformer():
    """Демонстрация работы полного Transformer блока."""
    print("🚀 Демонстрация Complete Transformer Block")
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
    print("\n🧠 Тест 1: Одиночный Transformer Block")
    print("-" * 50)
    
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        activation='relu'
    )
    
    print(f"✅ TransformerBlock создан")
    print(f"   Режим: {'PyTorch' if transformer_block.use_torch else 'NumPy fallback'}")
    print(f"   Параметры: d_model={d_model}, heads={num_heads}, d_ff={d_ff}")
    
    import time
    start_time = time.time()
    
    output = transformer_block(x)
    
    end_time = time.time()
    
    print(f"✅ Transformer блок завершен за {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Форма: {x.shape} → {output.shape}")
    print(f"   Residual connections: {'✅ Работают' if output.shape == x.shape else '❌ Ошибка'}")
    
    # Тест Transformer Encoder (стек блоков)
    print("\n🏗️  Тест 2: Transformer Encoder (3 слоя)")
    print("-" * 50)
    
    encoder = TransformerEncoder(
        num_layers=3,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        activation='relu'
    )
    
    print(f"✅ TransformerEncoder создан ({encoder.num_layers} слоев)")
    
    start_time = time.time()
    
    encoded = encoder(x)
    
    end_time = time.time()
    
    print(f"✅ Transformer Encoder завершен за {(end_time - start_time) * 1000:.2f}мс")
    print(f"📈 Форма: {x.shape} → {encoded.shape}")
    
    # Проверяем, что форма сохранилась
    input_shape = x.shape
    output_shape = encoded.shape if hasattr(encoded, 'shape') else np.array(encoded).shape
    
    print(f"🔍 Проверка формы: {input_shape} → {output_shape}")
    print(f"✅ Форма {'сохранена' if input_shape == output_shape else 'изменена'}")
    
    # Тест производительности компонентов
    print("\n⚡ Тест 3: Производительность компонентов")
    print("-" * 50)
    
    # Тестируем каждый компонент отдельно
    components_time = {}
    
    # Attention
    start_time = time.time()
    attn_output, _ = transformer_block.attention.forward(x, x, x)
    components_time['attention'] = (time.time() - start_time) * 1000
    
    # Feed-Forward
    start_time = time.time()
    ff_output = transformer_block.feed_forward(x)
    components_time['feed_forward'] = (time.time() - start_time) * 1000
    
    # Layer Norm
    start_time = time.time()
    norm_output = transformer_block.attention_norm.layer_norm(x)
    components_time['layer_norm'] = (time.time() - start_time) * 1000
    
    # Positional Encoding
    start_time = time.time()
    pos_output = transformer_block.pos_encoding(x)
    components_time['positional'] = (time.time() - start_time) * 1000
    
    print("📊 Время выполнения компонентов:")
    for component, time_ms in components_time.items():
        print(f"   {component.replace('_', ' ').title()}: {time_ms:.2f}мс")
    
    total_component_time = sum(components_time.values())
    full_block_time = (end_time - start_time) * 1000
    
    print(f"\n📈 Сравнение:")
    print(f"   Сумма компонентов: {total_component_time:.2f}мс")
    print(f"   Полный блок: {full_block_time:.2f}мс")
    print(f"   Эффективность: {(total_component_time / full_block_time * 100):.1f}%")
    
    return {
        'transformer_block': transformer_block,
        'encoder': encoder,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'single_block_time_ms': (end_time - start_time) * 1000,
        'encoder_time_ms': (end_time - start_time) * 1000,
        'components_time': components_time
    }


if __name__ == "__main__":
    demo_complete_transformer() 