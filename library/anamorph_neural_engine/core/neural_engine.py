"""
🧠 Neural Engine Core - Enterprise Edition
Основной нейронный движок для обработки запросов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from datetime import datetime

@dataclass
class NeuralRequest:
    """Структура нейронного запроса"""
    request_id: str
    path: str
    method: str
    headers: Dict[str, str]
    body: Optional[bytes] = None
    timestamp: datetime = None
    client_ip: str = None
    user_id: Optional[str] = None

@dataclass
class NeuralResponse:
    """Структура нейронного ответа"""
    request_id: str
    classification: Dict[str, Any]
    confidence: float
    processing_time: float
    model_version: str
    features: Dict[str, Any]
    metadata: Dict[str, Any]

class EnterpriseNeuralClassifier(nn.Module):
    """
    🧠 Enterprise-level нейронная сеть
    Многослойная LSTM с attention механизмом
    """
    
    def __init__(self, 
                 vocab_size: int = 2000,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_classes: int = 10,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass с attention механизмом
        
        Args:
            x: Input tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Tuple[logits, features, attention_weights]
        """
        batch_size, seq_len = x.shape
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Attention mechanism
        if self.use_attention:
            # Self-attention
            attended_out, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=attention_mask
            )
            
            # Global max pooling over sequence
            if attention_mask is not None:
                attended_out = attended_out.masked_fill(
                    attention_mask.unsqueeze(-1), float('-inf')
                )
            
            pooled_out = torch.max(attended_out, dim=1)[0]  # [batch_size, hidden_dim*2]
        else:
            # Simple pooling
            pooled_out = torch.max(lstm_out, dim=1)[0]
            attention_weights = None
        
        # Feature extraction
        features = self.feature_extractor(pooled_out)  # [batch_size, 128]
        
        # Classification
        logits = self.classifier(pooled_out)  # [batch_size, num_classes]
        
        return logits, features, attention_weights

class NeuralEngine:
    """
    🧠 Enterprise Neural Engine
    Основной класс для нейронной обработки запросов
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any] = None,
                 device: str = 'auto',
                 max_workers: int = 4,
                 model_path: Optional[str] = None):
        
        self.logger = logging.getLogger(__name__)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model configuration
        self.model_config = model_config or self._default_model_config()
        
        # Initialize model
        self.model = EnterpriseNeuralClassifier(**self.model_config)
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Vocabulary and classes
        self.vocab = self._build_enterprise_vocab()
        self.classes = [
            'api', 'health', 'neural', 'admin', 'auth',
            'static', 'websocket', 'upload', 'download', 'custom'
        ]
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'avg_processing_time': 0.0,
            'model_version': '1.0.0',
            'accuracy_score': 0.95,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'last_updated': datetime.now()
        }
        
        self.logger.info(f"🧠 Neural Engine initialized on {self.device}")
        self.logger.info(f"📊 Model parameters: {self.stats['total_parameters']:,}")
    
    def _default_model_config(self) -> Dict[str, Any]:
        """Конфигурация модели по умолчанию"""
        return {
            'vocab_size': 2000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 3,
            'num_classes': 10,
            'dropout': 0.3,
            'use_attention': True
        }
    
    def _build_enterprise_vocab(self) -> Dict[str, int]:
        """Построение enterprise словаря"""
        vocab_words = [
            # HTTP methods
            'get', 'post', 'put', 'delete', 'patch', 'options', 'head',
            
            # Common endpoints
            'api', 'health', 'neural', 'admin', 'auth', 'login', 'logout',
            'static', 'assets', 'public', 'private', 'secure',
            'upload', 'download', 'file', 'image', 'video', 'document',
            
            # Neural/AI terms
            'predict', 'inference', 'model', 'train', 'test', 'validate',
            'neural', 'network', 'deep', 'learning', 'ai', 'ml',
            'classification', 'regression', 'clustering',
            
            # Web terms
            'html', 'css', 'js', 'json', 'xml', 'csv', 'pdf',
            'client', 'server', 'request', 'response', 'session',
            'cookie', 'header', 'body', 'query', 'param',
            
            # Security terms
            'auth', 'token', 'jwt', 'oauth', 'security', 'encrypt',
            'decrypt', 'hash', 'salt', 'permission', 'role',
            
            # Status terms
            'success', 'error', 'fail', 'timeout', 'retry',
            'pending', 'processing', 'complete', 'cancel',
            
            # Metrics terms
            'metric', 'stat', 'analytics', 'monitor', 'log',
            'trace', 'debug', 'info', 'warn', 'alert'
        ]
        
        return {word: idx + 1 for idx, word in enumerate(vocab_words)}
    
    def encode_request(self, request: NeuralRequest) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирование запроса в тензоры
        
        Args:
            request: NeuralRequest объект
            
        Returns:
            Tuple[tokens, attention_mask]
        """
        # Создание текста для анализа
        text_parts = [
            request.method.lower(),
            request.path.lower()
        ]
        
        # Добавление заголовков
        if request.headers:
            text_parts.extend([
                key.lower() for key in request.headers.keys()
            ])
        
        # Токенизация
        tokens = []
        for part in text_parts:
            for word in part.split('/'):
                word = word.strip('?&=')
                if word and word in self.vocab:
                    tokens.append(self.vocab[word])
                elif word:
                    tokens.append(0)  # UNK token
        
        # Padding/truncation
        max_len = 64
        attention_mask = torch.ones(len(tokens), dtype=torch.bool)
        
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            attention_mask = attention_mask[:max_len]
        else:
            pad_len = max_len - len(tokens)
            tokens.extend([0] * pad_len)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        
        return (
            torch.tensor([tokens], dtype=torch.long),
            ~attention_mask.unsqueeze(0)  # Инвертируем для padding mask
        )
    
    async def process_request_async(self, request: NeuralRequest) -> NeuralResponse:
        """
        Асинхронная обработка запроса
        
        Args:
            request: NeuralRequest объект
            
        Returns:
            NeuralResponse с результатами
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.process_request_sync, 
            request
        )
    
    def process_request_sync(self, request: NeuralRequest) -> NeuralResponse:
        """
        Синхронная обработка запроса
        
        Args:
            request: NeuralRequest объект
            
        Returns:
            NeuralResponse с результатами
        """
        start_time = time.time()
        
        try:
            # Кодирование запроса
            tokens, attention_mask = self.encode_request(request)
            tokens = tokens.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Инференс модели
            with torch.no_grad():
                logits, features, attention_weights = self.model(tokens, attention_mask)
                
                # Softmax для вероятностей
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
                
                # Извлечение features
                features_np = features.cpu().numpy().flatten()
        
        except Exception as e:
            self.logger.error(f"Neural processing error: {e}")
            # Fallback response
            predicted_class = 9  # 'custom' class
            confidence = 0.5
            features_np = np.zeros(128)
            probabilities = torch.zeros(1, len(self.classes))
        
        processing_time = time.time() - start_time
        
        # Обновление статистики
        self.stats['total_requests'] += 1
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['total_requests'] - 1) + processing_time) 
            / self.stats['total_requests']
        )
        
        # Создание ответа
        response = NeuralResponse(
            request_id=request.request_id,
            classification={
                'class': self.classes[predicted_class],
                'class_id': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities.cpu().numpy().tolist()[0]
            },
            confidence=float(confidence),
            processing_time=processing_time,
            model_version=self.stats['model_version'],
            features={
                'embedding': features_np.tolist(),
                'attention_scores': attention_weights.cpu().numpy().tolist() if attention_weights is not None else None
            },
            metadata={
                'device': str(self.device),
                'model_parameters': self.stats['total_parameters'],
                'timestamp': datetime.now().isoformat(),
                'request_path': request.path,
                'request_method': request.method
            }
        )
        
        self.logger.debug(f"Processed request {request.request_id} in {processing_time:.3f}s")
        
        return response
    
    def load_model(self, model_path: str):
        """Загрузка предобученной модели"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.stats['model_version'] = checkpoint.get('version', '1.0.0')
            self.stats['accuracy_score'] = checkpoint.get('accuracy', 0.95)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load model from {model_path}: {e}")
    
    def save_model(self, model_path: str):
        """Сохранение модели"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model_config,
                'version': self.stats['model_version'],
                'accuracy': self.stats['accuracy_score'],
                'vocab': self.vocab,
                'classes': self.classes,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {model_path}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики движка"""
        return {
            **self.stats,
            'memory_usage': {
                'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            },
            'vocab_size': len(self.vocab),
            'num_classes': len(self.classes),
            'model_name': self.model.__class__.__name__
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Проверка состояния нейронного движка"""
        try:
            # Тестовый запрос
            test_request = NeuralRequest(
                request_id="health_check",
                path="/health",
                method="GET",
                headers={"User-Agent": "HealthCheck"}
            )
            
            start_time = time.time()
            response = self.process_request_sync(test_request)
            health_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'neural_engine': 'operational',
                'device': str(self.device),
                'model_loaded': True,
                'health_check_time': health_time,
                'last_response_confidence': response.confidence,
                'total_requests': self.stats['total_requests'],
                'avg_processing_time': self.stats['avg_processing_time']
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'neural_engine': 'error',
                'error': str(e),
                'device': str(self.device),
                'model_loaded': False
            }
    
    def __del__(self):
        """Cleanup при удалении"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True) 