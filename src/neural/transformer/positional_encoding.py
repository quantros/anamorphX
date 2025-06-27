"""
Positional Encoding для AnamorphX

Реализация позиционного кодирования для Transformer архитектур.
Поддерживает синусоидальное и обучаемое кодирование.
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Проверка PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using fallback positional encoding")

@dataclass
class PositionalEncodingConfig:
    """Конфигурация для Positional Encoding"""
    d_model: int = 512
    max_length: int = 5000
    dropout: float = 0.1
    encoding_type: str = "sinusoidal"  # sinusoidal, learned
    temperature: float = 10000.0
    normalize: bool = True

class PositionalEncoder:
    """
    Positional Encoding для AnamorphX
    
    Поддерживает синусоидальное и обучаемое позиционное кодирование
    для Transformer архитектур.
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        self.config = config
        
        if TORCH_AVAILABLE:
            self._init_torch_encoding()
        else:
            self._init_fallback_encoding()
    
    def _init_torch_encoding(self):
        """Инициализация с PyTorch"""
        if self.config.encoding_type == "sinusoidal":
            self._init_sinusoidal_torch()
        elif self.config.encoding_type == "learned":
            self._init_learned_torch()
        else:
            raise ValueError(f"Unknown encoding type: {self.config.encoding_type}")
        
        self.dropout = nn.Dropout(self.config.dropout)
    
    def _init_sinusoidal_torch(self):
        """Инициализация синусоидального кодирования с PyTorch"""
        pe = torch.zeros(self.config.max_length, self.config.d_model)
        position = torch.arange(0, self.config.max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2).float() * 
                           (-math.log(self.config.temperature) / self.config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if self.config.normalize:
            pe = pe / math.sqrt(self.config.d_model)
        
        self.pe = pe.unsqueeze(0)
    
    def _init_learned_torch(self):
        """Инициализация обучаемого кодирования с PyTorch"""
        self.pe = nn.Embedding(self.config.max_length, self.config.d_model)
        nn.init.normal_(self.pe.weight, std=0.02)
    
    def _init_fallback_encoding(self):
        """Инициализация без PyTorch (fallback)"""
        if self.config.encoding_type == "sinusoidal":
            self._init_sinusoidal_fallback()
        elif self.config.encoding_type == "learned":
            self._init_learned_fallback()
    
    def _init_sinusoidal_fallback(self):
        """Fallback синусоидальное кодирование"""
        import random
        
        # Упрощенная реализация для демонстрации
        self.pe = []
        for pos in range(self.config.max_length):
            encoding = []
            for i in range(self.config.d_model):
                if i % 2 == 0:
                    # sin для четных позиций
                    val = math.sin(pos / (self.config.temperature ** (2 * i / self.config.d_model)))
                else:
                    # cos для нечетных позиций
                    val = math.cos(pos / (self.config.temperature ** (2 * (i-1) / self.config.d_model)))
                
                if self.config.normalize:
                    val = val / math.sqrt(self.config.d_model)
                
                encoding.append(val)
            self.pe.append(encoding)
    
    def _init_learned_fallback(self):
        """Fallback обучаемое кодирование"""
        import random
        
        # Случайная инициализация для демонстрации
        self.pe = []
        for pos in range(self.config.max_length):
            encoding = [random.gauss(0, 0.02) for _ in range(self.config.d_model)]
            self.pe.append(encoding)
    
    def forward(self, x: Any, positions: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Применение позиционного кодирования
        
        Args:
            x: Входной тензор [batch_size, seq_len, d_model]
            positions: Позиции для кодирования (опционально)
            
        Returns:
            Tuple[encoded_output, encoding_info]
        """
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return self._torch_forward(x, positions)
        else:
            return self._fallback_forward(x, positions)
    
    def _torch_forward(self, x, positions=None):
        """PyTorch реализация forward pass"""
        batch_size, seq_len, d_model = x.size()
        
        if self.config.encoding_type == "sinusoidal":
            # Синусоидальное кодирование
            if positions is None:
                pe = self.pe[:, :seq_len, :]
            else:
                pe = self.pe[:, positions, :]
            
            output = x + pe
        
        elif self.config.encoding_type == "learned":
            # Обучаемое кодирование
            if positions is None:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            
            pe = self.pe(positions)
            output = x + pe
        
        output = self.dropout(output)
        
        encoding_info = {
            "encoding_type": self.config.encoding_type,
            "seq_len": seq_len,
            "d_model": d_model,
            "max_length": self.config.max_length
        }
        
        return output, encoding_info
    
    def _fallback_forward(self, x, positions=None):
        """Fallback реализация без PyTorch"""
        if isinstance(x, list):
            seq_len = len(x)
            if seq_len == 0:
                return x, {"encoding_type": self.config.encoding_type}
            
            # Применяем позиционное кодирование
            output = []
            for i, token_embedding in enumerate(x):
                if i < len(self.pe):
                    pos_encoding = self.pe[i]
                    if isinstance(token_embedding, list) and len(token_embedding) == len(pos_encoding):
                        # Складываем эмбеддинги токенов и позиционное кодирование
                        encoded_token = [t + p for t, p in zip(token_embedding, pos_encoding)]
                    else:
                        encoded_token = token_embedding
                else:
                    encoded_token = token_embedding
                
                output.append(encoded_token)
        else:
            output = x
        
        encoding_info = {
            "encoding_type": self.config.encoding_type,
            "seq_len": len(output) if isinstance(output, list) else 1,
            "fallback_mode": True
        }
        
        return output, encoding_info
    
    def get_encoding(self, positions: Any) -> Any:
        """
        Получить позиционное кодирование для конкретных позиций
        
        Args:
            positions: Позиции для кодирования
            
        Returns:
            Позиционные кодировки
        """
        if TORCH_AVAILABLE and hasattr(positions, 'device'):
            if self.config.encoding_type == "sinusoidal":
                return self.pe[:, positions, :]
            elif self.config.encoding_type == "learned":
                return self.pe(positions)
        else:
            # Fallback
            if isinstance(positions, list):
                return [self.pe[pos] for pos in positions if pos < len(self.pe)]
            else:
                return self.pe[positions] if positions < len(self.pe) else None

class PositionalEncodingCommand:
    """
    Команда positional_encoding для AnamorphX
    
    Применяет позиционное кодирование к последовательностям.
    """
    
    def __init__(self):
        self.name = "positional_encoding"
        self.description = "Apply positional encoding to sequences"
        self.encoders = {}
    
    def execute(self, context, **kwargs):
        """
        Выполнение команды positional_encoding
        
        Параметры:
            source: источник данных (последовательность)
            target: цель применения кодирования
            d_model: размерность модели (по умолчанию 512)
            max_length: максимальная длина последовательности
            encoding_type: тип кодирования (sinusoidal/learned)
        """
        try:
            source = kwargs.get('source')
            target = kwargs.get('target', source)
            d_model = int(kwargs.get('d_model', 512))
            max_length = int(kwargs.get('max_length', 5000))
            encoding_type = kwargs.get('encoding_type', 'sinusoidal')
            
            if not source:
                raise ValueError("Source is required for positional encoding")
            
            # Создание конфигурации
            config = PositionalEncodingConfig(
                d_model=d_model,
                max_length=max_length,
                encoding_type=encoding_type
            )
            
            # Создание энкодера
            encoder_id = f"pos_enc_{source}_{int(time.time() * 1000)}"
            encoder = PositionalEncoder(config)
            self.encoders[encoder_id] = encoder
            
            # Получение данных из контекста
            if hasattr(context, 'neural_entities') and source in context.neural_entities:
                source_entity = context.neural_entities[source]
                
                if hasattr(source_entity, 'data'):
                    input_data = source_entity.data
                    output, encoding_info = encoder.forward(input_data)
                    
                    # Сохранение результата
                    if target not in context.neural_entities:
                        context.neural_entities[target] = type(source_entity)(target)
                    
                    context.neural_entities[target].data = output
                    context.neural_entities[target].metadata = {
                        **getattr(context.neural_entities[target], 'metadata', {}),
                        'positional_encoding_applied': True,
                        'encoder_id': encoder_id,
                        'encoding_config': config.__dict__,
                        'encoding_info': encoding_info
                    }
                    
                    return {
                        "success": True,
                        "message": f"Positional encoding applied: {source} -> {target}",
                        "data": {
                            "encoder_id": encoder_id,
                            "config": config.__dict__,
                            "encoding_type": encoding_type,
                            "d_model": d_model,
                            "max_length": max_length,
                            "torch_available": TORCH_AVAILABLE
                        }
                    }
                else:
                    raise ValueError(f"Source entity '{source}' has no data")
            else:
                raise ValueError(f"Source entity '{source}' not found")
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Positional encoding command failed: {str(e)}",
                "error": str(e)
            }

# Экспорты
__all__ = [
    "PositionalEncoder",
    "PositionalEncodingConfig",
    "PositionalEncodingCommand",
    "TORCH_AVAILABLE"
] 