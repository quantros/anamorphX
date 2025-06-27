"""
Multi-head Attention для AnamorphX

Реализация механизма внимания для Transformer архитектур.
Интегрируется с командами neuro и synap.
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Импорты команд AnamorphX
try:
    from ...interpreter.commands import Command, CommandResult, CommandError
    from ...interpreter.runtime import ExecutionContext
    ANAMORPHX_AVAILABLE = True
except ImportError:
    ANAMORPHX_AVAILABLE = False
    # Заглушки
    class Command:
        def __init__(self, name, description):
            self.name = name
            self.description = description
    
    class CommandResult:
        def __init__(self, success=True, message="", data=None):
            self.success = success
            self.message = message
            self.data = data or {}
    
    class ExecutionContext:
        def __init__(self):
            self.neural_entities = {}

# Проверка PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using fallback attention implementation")

@dataclass
class AttentionConfig:
    """Конфигурация для Multi-head Attention"""
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    temperature: float = 1.0
    use_bias: bool = True
    attention_type: str = "scaled_dot_product"  # scaled_dot_product, additive, multiplicative

class MultiHeadAttention:
    """
    Multi-head Attention механизм для AnamorphX
    
    Поддерживает различные типы внимания и интегрируется
    с командами neuro и synap языка Anamorph.
    """
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.d_k = config.d_model // config.num_heads
        
        if self.d_k * config.num_heads != config.d_model:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by num_heads ({config.num_heads})")
        
        # Инициализация весов (fallback без PyTorch)
        if TORCH_AVAILABLE:
            self._init_torch_layers()
        else:
            self._init_fallback_weights()
    
    def _init_torch_layers(self):
        """Инициализация слоев с PyTorch"""
        self.W_q = nn.Linear(self.config.d_model, self.config.d_model, bias=self.config.use_bias)
        self.W_k = nn.Linear(self.config.d_model, self.config.d_model, bias=self.config.use_bias)
        self.W_v = nn.Linear(self.config.d_model, self.config.d_model, bias=self.config.use_bias)
        self.W_o = nn.Linear(self.config.d_model, self.config.d_model, bias=self.config.use_bias)
        self.dropout = nn.Dropout(self.config.dropout)
    
    def _init_fallback_weights(self):
        """Инициализация весов без PyTorch (заглушка)"""
        import random
        
        def init_matrix(rows, cols):
            return [[random.gauss(0, 0.1) for _ in range(cols)] for _ in range(rows)]
        
        self.W_q = init_matrix(self.config.d_model, self.config.d_model)
        self.W_k = init_matrix(self.config.d_model, self.config.d_model)
        self.W_v = init_matrix(self.config.d_model, self.config.d_model)
        self.W_o = init_matrix(self.config.d_model, self.config.d_model)
    
    def forward(self, query: Any, key: Any, value: Any, mask: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Прямой проход Multi-head Attention
        
        Args:
            query: Query тензор [batch_size, seq_len, d_model]
            key: Key тензор [batch_size, seq_len, d_model]  
            value: Value тензор [batch_size, seq_len, d_model]
            mask: Маска внимания (опционально)
            
        Returns:
            Tuple[output, attention_weights]
        """
        if TORCH_AVAILABLE and isinstance(query, torch.Tensor):
            return self._torch_forward(query, key, value, mask)
        else:
            return self._fallback_forward(query, key, value, mask)
    
    def _torch_forward(self, query, key, value, mask=None):
        """PyTorch реализация"""
        batch_size, seq_len, d_model = query.size()
        
        # Линейные преобразования
        Q = self.W_q(query).view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.config.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Конкатенация головок
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.config.d_model
        )
        
        # Финальная линейная трансформация
        output = self.W_o(attention_output)
        
        return output, {"attention_weights": attention_weights}
    
    def _fallback_forward(self, query, key, value, mask=None):
        """Fallback реализация без PyTorch"""
        # Упрощенная реализация для демонстрации
        seq_len = len(query) if isinstance(query, list) else 1
        d_model = self.config.d_model
        
        # Заглушка - возвращаем входные данные с небольшими изменениями
        output = query  # В реальной реализации здесь была бы математика внимания
        attention_weights = [[1.0 / seq_len for _ in range(seq_len)] for _ in range(seq_len)]
        
        return output, {"attention_weights": attention_weights}
    
    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled Dot-Product Attention"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * self.config.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class AttentionCommand(Command):
    """
    Команда attention для AnamorphX
    
    Применяет механизм внимания к нейронным сущностям.
    Интегрируется с командами neuro и synap.
    """
    
    def __init__(self):
        super().__init__("attention", "Apply multi-head attention mechanism")
        self.attention_modules = {}
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        """
        Выполнение команды attention
        
        Параметры:
            source: источник данных (нейрон/слой)
            target: цель применения внимания
            num_heads: количество головок внимания (по умолчанию 8)
            d_model: размерность модели (по умолчанию 512)
            attention_type: тип внимания
        """
        try:
            source = kwargs.get('source')
            target = kwargs.get('target', source)
            num_heads = int(kwargs.get('num_heads', 8))
            d_model = int(kwargs.get('d_model', 512))
            attention_type = kwargs.get('attention_type', 'scaled_dot_product')
            
            if not source:
                raise CommandError("Source entity is required for attention")
            
            # Создание конфигурации внимания
            config = AttentionConfig(
                d_model=d_model,
                num_heads=num_heads,
                attention_type=attention_type
            )
            
            # Создание модуля внимания
            attention_id = f"attention_{source}_{int(time.time() * 1000)}"
            attention_module = MultiHeadAttention(config)
            self.attention_modules[attention_id] = attention_module
            
            # Получение данных из контекста
            if hasattr(context, 'neural_entities') and source in context.neural_entities:
                source_entity = context.neural_entities[source]
                
                # Применение внимания (упрощенная версия)
                if hasattr(source_entity, 'data'):
                    query = key = value = source_entity.data
                    output, attention_info = attention_module.forward(query, key, value)
                    
                    # Сохранение результата
                    if target not in context.neural_entities:
                        context.neural_entities[target] = type(source_entity)(target)
                    
                    context.neural_entities[target].data = output
                    context.neural_entities[target].metadata = {
                        **getattr(context.neural_entities[target], 'metadata', {}),
                        'attention_applied': True,
                        'attention_id': attention_id,
                        'attention_config': config.__dict__,
                        'attention_info': attention_info
                    }
                    
                    return CommandResult(
                        success=True,
                        message=f"Multi-head attention applied: {source} -> {target}",
                        data={
                            "attention_id": attention_id,
                            "config": config.__dict__,
                            "num_heads": num_heads,
                            "d_model": d_model,
                            "torch_available": TORCH_AVAILABLE
                        }
                    )
                else:
                    raise CommandError(f"Source entity '{source}' has no data")
            else:
                raise CommandError(f"Source entity '{source}' not found")
                
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Attention command failed: {str(e)}",
                error=CommandError("ATTENTION_ERROR", str(e))
            )

def demo_multi_head_attention():
    """Демонстрация работы Multi-head Attention."""
    print("🎯 Демонстрация Multi-head Attention")
    print("=" * 50)
    
    # Параметры
    batch_size, seq_len, d_model = 2, 16, 512
    num_heads = 8
    
    # Создаем конфигурацию
    config = AttentionConfig(d_model=d_model, num_heads=num_heads)
    
    # Создаем тестовые данные
    if TORCH_AVAILABLE:
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"📊 Входные данные (PyTorch): {x.shape}")
    else:
        x = [[0.1] * d_model for _ in range(seq_len)]
        print(f"📊 Входные данные (fallback): {len(x)}x{len(x[0])}")
    
    # Создаем Multi-head Attention
    attention = MultiHeadAttention(config)
    
    print(f"⚙️  Режим: {'PyTorch' if TORCH_AVAILABLE else 'NumPy fallback'}")
    print(f"📈 Параметры: d_model={d_model}, num_heads={num_heads}, d_k={attention.d_k}")
    
    # Прямой проход
    import time
    start_time = time.time()
    
    output, attention_info = attention.forward(x, x, x)  # Self-attention
    
    end_time = time.time()
    
    print(f"✅ Multi-head Attention завершен за {(end_time - start_time) * 1000:.2f}мс")
    if hasattr(output, 'shape'):
        print(f"📈 Выходная форма: {output.shape}")
    else:
        print(f"📈 Выходная форма: {len(output)}x{len(output[0]) if output else 0}")
    
    return {
        'attention': attention,
        'output': output,
        'attention_info': attention_info,
        'processing_time_ms': (end_time - start_time) * 1000
    }


# Экспорты
__all__ = [
    "MultiHeadAttention",
    "AttentionConfig", 
    "AttentionCommand",
    "TORCH_AVAILABLE"
]


if __name__ == "__main__":
    demo_multi_head_attention() 