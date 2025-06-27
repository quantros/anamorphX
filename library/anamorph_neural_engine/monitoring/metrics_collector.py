"""
📊 Metrics Collector - Enterprise Edition
Сбор и агрегация метрик для мониторинга
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import psutil
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

@dataclass
class SystemMetrics:
    """Системные метрики"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    load_average: List[float]
    uptime_seconds: float
    
@dataclass
class ApplicationMetrics:
    """Метрики приложения"""
    requests_total: int
    requests_per_second: float
    neural_inferences_total: int
    neural_avg_time: float
    websocket_connections: int
    active_users: int
    error_rate: float
    cache_hit_rate: float

@dataclass
class NeuralMetrics:
    """Метрики нейронной сети"""
    model_parameters: int
    inference_count: int
    avg_inference_time: float
    avg_confidence: float
    classification_distribution: Dict[str, int]
    device_usage: Dict[str, Any]
    memory_usage: Dict[str, float]

class MetricsCollector:
    """
    📊 Enterprise Metrics Collector
    Сбор, агрегация и хранение метрик
    """
    
    def __init__(self, 
                 redis_url: str = 'redis://localhost:6379',
                 collection_interval: int = 10,
                 retention_days: int = 7):
        
        self.redis_url = redis_url
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)
        
        # Redis client
        self.redis_client = None
        
        # Metrics storage
        self.current_metrics = {}
        self.historical_metrics = []
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Collection state
        self.is_collecting = False
        self.start_time = time.time()
        self.last_collection = None
        
        # Application counters
        self.request_counter = 0
        self.neural_counter = 0
        self.error_counter = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Neural tracking
        self.neural_times = []
        self.neural_confidences = []
        self.neural_classifications = {}
        
    def _setup_prometheus_metrics(self):
        """Настройка Prometheus метрик"""
        
        # System metrics
        self.prom_cpu_usage = Gauge(
            'system_cpu_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.prom_memory_usage = Gauge(
            'system_memory_percent', 
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.prom_disk_usage = Gauge(
            'system_disk_percent',
            'Disk usage percentage', 
            registry=self.registry
        )
        
        # Application metrics
        self.prom_requests_total = Counter(
            'app_requests_total',
            'Total requests processed',
            registry=self.registry
        )
        
        self.prom_neural_inferences = Counter(
            'neural_inferences_total',
            'Total neural inferences',
            registry=self.registry
        )
        
        self.prom_neural_time = Histogram(
            'neural_inference_seconds',
            'Neural inference time',
            registry=self.registry
        )
        
        self.prom_websocket_connections = Gauge(
            'websocket_connections_active',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        self.prom_error_rate = Gauge(
            'app_error_rate',
            'Application error rate',
            registry=self.registry
        )
        
        # Neural metrics
        self.prom_neural_confidence = Gauge(
            'neural_avg_confidence',
            'Average neural confidence',
            registry=self.registry
        )
        
        self.prom_classification_count = Counter(
            'neural_classifications_total',
            'Neural classifications by class',
            ['class_name'],
            registry=self.registry
        )
    
    async def initialize(self):
        """Инициализация сборщика метрик"""
        
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.logger.info("✅ Redis connected for metrics")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize metrics storage
        await self._initialize_storage()
        
        self.logger.info("📊 Metrics Collector initialized")
    
    async def _initialize_storage(self):
        """Инициализация хранилища метрик"""
        
        # Load historical metrics from Redis
        if self.redis_client:
            try:
                # Get last 24 hours of metrics
                end_time = time.time()
                start_time = end_time - (24 * 3600)  # 24 hours ago
                
                metrics_keys = await self.redis_client.zrangebyscore(
                    'metrics:timeline',
                    start_time,
                    end_time
                )
                
                if metrics_keys:
                    self.logger.info(f"Loaded {len(metrics_keys)} historical metrics")
                
            except Exception as e:
                self.logger.warning(f"Failed to load historical metrics: {e}")
    
    async def start(self):
        """Запуск сбора метрик"""
        
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.logger.info("📊 Starting metrics collection")
        
        # Start collection loop
        asyncio.create_task(self._collection_loop())
    
    async def stop(self):
        """Остановка сбора метрик"""
        
        self.is_collecting = False
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("📊 Metrics collection stopped")
    
    async def _collection_loop(self):
        """Основной цикл сбора метрик"""
        
        while self.is_collecting:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _collect_metrics(self):
        """Сбор всех метрик"""
        
        collection_time = time.time()
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Collect application metrics
        app_metrics = await self._collect_app_metrics()
        
        # Collect neural metrics
        neural_metrics = await self._collect_neural_metrics()
        
        # Combine all metrics
        all_metrics = {
            'timestamp': collection_time,
            'datetime': datetime.fromtimestamp(collection_time).isoformat(),
            'system': asdict(system_metrics),
            'application': asdict(app_metrics),
            'neural': asdict(neural_metrics)
        }
        
        # Update current metrics
        self.current_metrics = all_metrics
        self.last_collection = collection_time
        
        # Store in historical data
        self.historical_metrics.append(all_metrics)
        
        # Keep only recent history in memory
        if len(self.historical_metrics) > 1440:  # 24 hours at 1-minute intervals
            self.historical_metrics = self.historical_metrics[-1440:]
        
        # Update Prometheus metrics
        await self._update_prometheus_metrics(system_metrics, app_metrics, neural_metrics)
        
        # Store in Redis
        await self._store_metrics_redis(all_metrics)
        
        self.logger.debug(f"Metrics collected: {collection_time}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Сбор системных метрик"""
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            memory_total_mb = memory.total / 1024 / 1024
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / 1024 / 1024 / 1024
            disk_total_gb = disk.total / 1024 / 1024 / 1024
            
            # Load average
            load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                load_average=load_avg,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, [0, 0, 0], 0)
    
    async def _collect_app_metrics(self) -> ApplicationMetrics:
        """Сбор метрик приложения"""
        
        try:
            # Calculate rates
            time_diff = self.collection_interval
            requests_per_second = self.request_counter / time_diff if time_diff > 0 else 0
            
            # Calculate error rate
            total_requests = max(self.request_counter, 1)
            error_rate = (self.error_counter / total_requests) * 100
            
            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            metrics = ApplicationMetrics(
                requests_total=self.request_counter,
                requests_per_second=requests_per_second,
                neural_inferences_total=self.neural_counter,
                neural_avg_time=sum(self.neural_times) / len(self.neural_times) if self.neural_times else 0,
                websocket_connections=0,  # Will be updated externally
                active_users=0,  # Will be updated externally
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate
            )
            
            # Reset counters for next interval
            self.request_counter = 0
            self.neural_counter = 0
            self.error_counter = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.neural_times = []
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"App metrics collection error: {e}")
            return ApplicationMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    async def _collect_neural_metrics(self) -> NeuralMetrics:
        """Сбор метрик нейронной сети"""
        
        try:
            # Average confidence
            avg_confidence = sum(self.neural_confidences) / len(self.neural_confidences) if self.neural_confidences else 0
            
            # Average inference time
            avg_inference_time = sum(self.neural_times) / len(self.neural_times) if self.neural_times else 0
            
            # Device usage (if CUDA available)
            device_usage = {}
            memory_usage = {}
            
            try:
                import torch
                if torch.cuda.is_available():
                    device_usage['cuda_available'] = True
                    device_usage['cuda_device_count'] = torch.cuda.device_count()
                    memory_usage['cuda_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    memory_usage['cuda_cached'] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                else:
                    device_usage['cuda_available'] = False
            except ImportError:
                device_usage['torch_available'] = False
            
            metrics = NeuralMetrics(
                model_parameters=0,  # Will be updated externally
                inference_count=len(self.neural_times),
                avg_inference_time=avg_inference_time,
                avg_confidence=avg_confidence,
                classification_distribution=self.neural_classifications.copy(),
                device_usage=device_usage,
                memory_usage=memory_usage
            )
            
            # Reset neural tracking for next interval
            self.neural_confidences = []
            self.neural_classifications = {}
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Neural metrics collection error: {e}")
            return NeuralMetrics(0, 0, 0, 0, {}, {}, {})
    
    async def _update_prometheus_metrics(self, 
                                       system: SystemMetrics, 
                                       app: ApplicationMetrics, 
                                       neural: NeuralMetrics):
        """Обновление Prometheus метрик"""
        
        try:
            # System metrics
            self.prom_cpu_usage.set(system.cpu_percent)
            self.prom_memory_usage.set(system.memory_percent)
            self.prom_disk_usage.set(system.disk_percent)
            
            # Application metrics
            self.prom_requests_total.inc(app.requests_total)
            self.prom_neural_inferences.inc(app.neural_inferences_total)
            self.prom_websocket_connections.set(app.websocket_connections)
            self.prom_error_rate.set(app.error_rate)
            
            # Neural metrics
            self.prom_neural_confidence.set(neural.avg_confidence)
            
            # Update classification counters
            for class_name, count in neural.classification_distribution.items():
                self.prom_classification_count.labels(class_name=class_name).inc(count)
                
        except Exception as e:
            self.logger.error(f"Prometheus metrics update error: {e}")
    
    async def _store_metrics_redis(self, metrics: Dict[str, Any]):
        """Хранение метрик в Redis"""
        
        if not self.redis_client:
            return
        
        try:
            timestamp = metrics['timestamp']
            metrics_key = f"metrics:{int(timestamp)}"
            
            # Store metrics data
            await self.redis_client.set(
                metrics_key,
                json.dumps(metrics),
                ex=self.retention_days * 24 * 3600  # TTL in seconds
            )
            
            # Add to timeline for easy querying
            await self.redis_client.zadd(
                'metrics:timeline',
                {metrics_key: timestamp}
            )
            
            # Cleanup old timeline entries
            cutoff_time = timestamp - (self.retention_days * 24 * 3600)
            await self.redis_client.zremrangebyscore(
                'metrics:timeline',
                0,
                cutoff_time
            )
            
        except Exception as e:
            self.logger.error(f"Redis metrics storage error: {e}")
    
    # Public methods for external components to record metrics
    
    def record_request(self):
        """Записать обработанный запрос"""
        self.request_counter += 1
    
    def record_neural_inference(self, inference_time: float, confidence: float, classification: str):
        """Записать нейронную инференцию"""
        self.neural_counter += 1
        self.neural_times.append(inference_time)
        self.neural_confidences.append(confidence)
        
        if classification in self.neural_classifications:
            self.neural_classifications[classification] += 1
        else:
            self.neural_classifications[classification] = 1
    
    def record_error(self):
        """Записать ошибку"""
        self.error_counter += 1
    
    def record_cache_hit(self):
        """Записать попадание в кэш"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Записать промах кэша"""
        self.cache_misses += 1
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Получить текущие метрики"""
        return self.current_metrics.copy() if self.current_metrics else {}
    
    async def get_historical_metrics(self, 
                                   start_time: Optional[float] = None,
                                   end_time: Optional[float] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Получить исторические метрики"""
        
        if not start_time:
            start_time = time.time() - 3600  # Last hour
        if not end_time:
            end_time = time.time()
        
        # Try Redis first
        if self.redis_client:
            try:
                metrics_keys = await self.redis_client.zrangebyscore(
                    'metrics:timeline',
                    start_time,
                    end_time,
                    start=0,
                    num=limit
                )
                
                if metrics_keys:
                    metrics_data = []
                    for key in metrics_keys:
                        data = await self.redis_client.get(key)
                        if data:
                            metrics_data.append(json.loads(data))
                    return metrics_data
                    
            except Exception as e:
                self.logger.error(f"Redis historical metrics error: {e}")
        
        # Fallback to in-memory data
        filtered_metrics = [
            m for m in self.historical_metrics
            if start_time <= m['timestamp'] <= end_time
        ]
        
        return filtered_metrics[-limit:] if len(filtered_metrics) > limit else filtered_metrics
    
    async def get_prometheus_metrics(self) -> str:
        """Получить метрики в формате Prometheus"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья сборщика метрик"""
        
        health = {
            'status': 'healthy',
            'collecting': self.is_collecting,
            'last_collection': self.last_collection,
            'redis_connected': self.redis_client is not None,
            'metrics_count': len(self.historical_metrics),
            'collection_interval': self.collection_interval
        }
        
        # Check if collection is working
        if self.last_collection:
            time_since_last = time.time() - self.last_collection
            if time_since_last > self.collection_interval * 2:
                health['status'] = 'degraded'
                health['warning'] = 'Collection may be lagging'
        
        return health
    
    async def get_system_summary(self) -> Dict[str, Any]:
        """Получить сводку системных метрик"""
        
        if not self.current_metrics:
            return {}
        
        system_metrics = self.current_metrics.get('system', {})
        app_metrics = self.current_metrics.get('application', {})
        neural_metrics = self.current_metrics.get('neural', {})
        
        return {
            'system_health': 'good' if system_metrics.get('cpu_percent', 0) < 80 else 'warning',
            'memory_usage': f"{system_metrics.get('memory_percent', 0):.1f}%",
            'cpu_usage': f"{system_metrics.get('cpu_percent', 0):.1f}%",
            'requests_per_second': f"{app_metrics.get('requests_per_second', 0):.1f}",
            'neural_avg_time': f"{neural_metrics.get('avg_inference_time', 0)*1000:.1f}ms",
            'neural_confidence': f"{neural_metrics.get('avg_confidence', 0)*100:.1f}%",
            'uptime': str(timedelta(seconds=int(system_metrics.get('uptime_seconds', 0))))
        } 