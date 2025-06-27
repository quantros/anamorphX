"""
🧠 Advanced Neural Engine для AnamorphX Enterprise
Продвинутые нейронные возможности для enterprise уровня
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import asyncio
import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Типы нейронных моделей"""
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    RESNET = "resnet"
    BERT = "bert"
    GPT = "gpt"

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.2

class TransformerModel(nn.Module):
    """Transformer модель для enterprise задач"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LSTMModel(nn.Module):
    """LSTM модель с вниманием"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 3, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attended = self.attention(lstm_out)
        output = self.classifier(self.dropout(attended))
        return output

class AttentionLayer(nn.Module):
    """Слой внимания"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        return attended

class CNNModel(nn.Module):
    """CNN модель для классификации"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 num_filters: int = 100, filter_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=size)
            for size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embedding_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.classifier(self.dropout(concatenated))
        return output

class AdvancedNeuralEngine:
    """Продвинутый нейронный движок"""
    
    def __init__(self, device: str = "auto", max_workers: int = 4):
        self.device = self._setup_device(device)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Модели
        self.models: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, optim.Optimizer] = {}
        self.schedulers: Dict[str, Any] = {}
        
        # Метрики
        self.training_metrics: Dict[str, List[float]] = {}
        self.inference_stats: Dict[str, Any] = {}
        
        # Конфигурации
        self.model_configs: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(__name__)
        
        print(f"🧠 Advanced Neural Engine инициализирован")
        print(f"   ⚡ Device: {self.device}")
        print(f"   🔧 Max Workers: {max_workers}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Настройка устройства"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        torch_device = torch.device(device)
        print(f"🖥️ Использование устройства: {torch_device}")
        return torch_device
    
    async def create_model(self, model_name: str, model_type: ModelType, 
                          config: Dict[str, Any]) -> bool:
        """Создание новой модели"""
        try:
            if model_type == ModelType.TRANSFORMER:
                model = TransformerModel(
                    vocab_size=config.get('vocab_size', 10000),
                    d_model=config.get('d_model', 512),
                    nhead=config.get('nhead', 8),
                    num_layers=config.get('num_layers', 6),
                    dim_feedforward=config.get('dim_feedforward', 2048),
                    dropout=config.get('dropout', 0.1)
                )
            else:
                # Простая LSTM модель как fallback
                class SimpleLSTM(nn.Module):
                    def __init__(self, vocab_size, hidden_dim=256):
                        super().__init__()
                        self.embedding = nn.Embedding(vocab_size, hidden_dim)
                        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                        self.classifier = nn.Linear(hidden_dim, vocab_size)
                    
                    def forward(self, x):
                        x = self.embedding(x)
                        x, _ = self.lstm(x)
                        x = self.classifier(x[:, -1, :])
                        return x
                
                model = SimpleLSTM(config.get('vocab_size', 10000))
            
            model = model.to(self.device)
            self.models[model_name] = model
            self.model_configs[model_name] = config
            
            # Создание оптимизатора
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', 0.001),
                weight_decay=config.get('weight_decay', 1e-5)
            )
            self.optimizers[model_name] = optimizer
            
            print(f"✅ Модель {model_name} создана")
            print(f"   📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка создания модели {model_name}: {e}")
            return False
    
    async def train_model(self, model_name: str, train_data: List[Any], 
                         training_config: TrainingConfig) -> Dict[str, Any]:
        """Асинхронное обучение модели"""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена")
        
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        scheduler = self.schedulers[model_name]
        
        # Запуск обучения в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._train_model_sync, 
            model, optimizer, scheduler, train_data, training_config
        )
        
        return result
    
    def _train_model_sync(self, model: nn.Module, optimizer: optim.Optimizer,
                         scheduler: Any, train_data: List[Any], 
                         config: TrainingConfig) -> Dict[str, Any]:
        """Синхронное обучение модели"""
        model.train()
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"🚀 Начало обучения (epochs: {config.epochs})")
        
        for epoch in range(config.epochs):
            epoch_losses = []
            
            # Обучение по батчам
            for batch_idx in range(0, len(train_data), config.batch_size):
                batch = train_data[batch_idx:batch_idx + config.batch_size]
                
                # Преобразование данных в тензоры
                inputs, targets = self._prepare_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            scheduler.step()
            
            # Early stopping
            if config.early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"⏹️ Early stopping на epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                print(f"📈 Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return {
            'final_loss': losses[-1] if losses else 0,
            'best_loss': best_loss,
            'total_epochs': len(losses),
            'losses': losses
        }
    
    def _prepare_batch(self, batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Подготовка батча данных"""
        # Простая обработка для демонстрации
        if isinstance(batch[0], str):
            # Простая токенизация
            inputs = []
            targets = []
            
            for item in batch:
                # Токенизация строки
                tokens = [hash(word) % 10000 for word in item.split()]
                if len(tokens) < 50:
                    tokens.extend([0] * (50 - len(tokens)))
                else:
                    tokens = tokens[:50]
                
                inputs.append(tokens)
                targets.append(len(tokens) % 10)  # Простая цель
            
            return torch.LongTensor(inputs), torch.LongTensor(targets)
        else:
            # Обработка других типов данных
            inputs = torch.FloatTensor(batch)
            targets = torch.LongTensor([0] * len(batch))
            return inputs, targets
    
    async def predict(self, model_name: str, input_data: Any) -> Dict[str, Any]:
        """Асинхронное предсказание"""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена")
        
        model = self.models[model_name]
        
        # Запуск предсказания в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._predict_sync, model, input_data
        )
        
        return result
    
    def _predict_sync(self, model: nn.Module, input_data: Any) -> Dict[str, Any]:
        """Синхронное предсказание"""
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            # Простая обработка входных данных
            if isinstance(input_data, str):
                # Токенизация строки
                tokens = [hash(word) % 10000 for word in input_data.split()]
                if len(tokens) < 50:
                    tokens.extend([0] * (50 - len(tokens)))
                else:
                    tokens = tokens[:50]
                inputs = torch.LongTensor([tokens]).to(self.device)
            else:
                inputs = torch.FloatTensor([[input_data]]).to(self.device)
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities).item()
        
        processing_time = time.time() - start_time
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist(),
            'processing_time': processing_time
        }
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        if model_name not in self.models:
            return {'error': f'Модель {model_name} не найдена'}
        
        model = self.models[model_name]
        config = self.model_configs[model_name]
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'name': model_name,
            'type': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(next(model.parameters()).device),
            'config': config,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Примерный размер
        }
    
    async def save_model(self, model_name: str, path: str) -> bool:
        """Сохранение модели"""
        if model_name not in self.models:
            return False
        
        try:
            model = self.models[model_name]
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.model_configs[model_name],
                'model_type': model.__class__.__name__
            }, path)
            
            print(f"💾 Модель {model_name} сохранена в {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")
            return False
    
    async def load_model(self, model_name: str, path: str) -> bool:
        """Загрузка модели"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint['config']
            model_type = checkpoint['model_type']
            
            # Создание модели соответствующего типа
            if model_type == 'TransformerModel':
                model = TransformerModel(**config)
            elif model_type == 'LSTMModel':
                model = LSTMModel(**config)
            elif model_type == 'CNNModel':
                model = CNNModel(**config)
            else:
                raise ValueError(f"Неизвестный тип модели: {model_type}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            
            self.models[model_name] = model
            self.model_configs[model_name] = config
            
            print(f"📁 Модель {model_name} загружена из {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Получение списка доступных моделей"""
        models_info = []
        
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            models_info.append({
                'name': name,
                'type': model.__class__.__name__,
                'parameters': total_params,
                'device': str(next(model.parameters()).device)
            })
        
        return models_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return {
            'total_models': len(self.models),
            'device': str(self.device),
            'max_workers': self.max_workers,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'memory_cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
    
    async def optimize_model(self, model_name: str) -> Dict[str, Any]:
        """Оптимизация модели"""
        if model_name not in self.models:
            return {'error': f'Модель {model_name} не найдена'}
        
        model = self.models[model_name]
        
        # Квантизация модели (если поддерживается)
        try:
            if hasattr(torch.quantization, 'quantize_dynamic'):
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                
                original_size = sum(p.numel() for p in model.parameters()) * 4
                quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1  # Примерно
                
                self.models[f"{model_name}_quantized"] = quantized_model
                
                return {
                    'original_size_mb': original_size / (1024 * 1024),
                    'quantized_size_mb': quantized_size / (1024 * 1024),
                    'compression_ratio': original_size / quantized_size,
                    'quantized_model_name': f"{model_name}_quantized"
                }
            else:
                return {'info': 'Квантизация не поддерживается'}
                
        except Exception as e:
            return {'error': f'Ошибка оптимизации: {e}'}
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.executor.shutdown(wait=True)
        
        # Очистка CUDA кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 Ресурсы Neural Engine очищены") 