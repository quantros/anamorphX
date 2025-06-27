"""
📊 Real-time Analytics для AnamorphX Enterprise
Система аналитики в реальном времени с ML прогнозированием
"""

import asyncio
import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque, defaultdict
import statistics
import numpy as np
from datetime import datetime, timedelta
import psutil
import socket
import uuid

class MetricType(Enum):
    """Типы метрик"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    """Уровни серьезности алертов"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Метрика"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class Alert:
    """Алерт"""
    alert_id: str
    metric_name: str
    condition: str
    severity: AlertSeverity
    message: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MetricAggregator:
    """Агрегатор метрик"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()
    
    def add_metric(self, metric: Metric):
        """Добавление метрики"""
        with self.lock:
            self.metrics_buffer[metric.name].append(metric)
    
    def get_aggregated_metrics(self, metric_name: str, 
                             aggregation: str = "avg") -> Optional[float]:
        """Получение агрегированных метрик"""
        with self.lock:
            if metric_name not in self.metrics_buffer:
                return None
            
            values = [m.value for m in self.metrics_buffer[metric_name]]
            
            if not values:
                return None
            
            if aggregation == "avg":
                return statistics.mean(values)
            elif aggregation == "sum":
                return sum(values)
            elif aggregation == "min":
                return min(values)
            elif aggregation == "max":
                return max(values)
            elif aggregation == "median":
                return statistics.median(values)
            elif aggregation == "std":
                return statistics.stdev(values) if len(values) > 1 else 0
            else:
                return values[-1]  # latest
    
    def get_time_series(self, metric_name: str, 
                       duration_seconds: int = 3600) -> List[Dict[str, Any]]:
        """Получение временного ряда"""
        with self.lock:
            if metric_name not in self.metrics_buffer:
                return []
            
            current_time = time.time()
            cutoff_time = current_time - duration_seconds
            
            filtered_metrics = [
                m for m in self.metrics_buffer[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            return [
                {
                    'timestamp': m.timestamp,
                    'value': m.value,
                    'tags': m.tags
                }
                for m in filtered_metrics
            ]

class SystemMetricsCollector:
    """Сборщик системных метрик"""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.is_running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.metrics_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
    
    def start_collection(self):
        """Запуск сбора метрик"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info(f"📊 Сбор системных метрик запущен (интервал: {self.collection_interval}s)")
    
    def stop_collection(self):
        """Остановка сбора метрик"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _collection_loop(self):
        """Цикл сбора метрик"""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                for metric in metrics:
                    self.metrics_queue.put(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка сбора метрик: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> List[Metric]:
        """Сбор системных метрик"""
        timestamp = time.time()
        metrics = []
        
        # CPU метрики
        cpu_percent = psutil.cpu_percent(interval=None)
        metrics.append(Metric(
            name="system.cpu.usage_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="percent"
        ))
        
        # Memory метрики
        memory = psutil.virtual_memory()
        metrics.extend([
            Metric(
                name="system.memory.usage_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                unit="percent"
            ),
            Metric(
                name="system.memory.available_gb",
                value=memory.available / (1024**3),
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                unit="GB"
            ),
            Metric(
                name="system.memory.used_gb",
                value=memory.used / (1024**3),
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                unit="GB"
            )
        ])
        
        # Disk метрики
        disk = psutil.disk_usage('/')
        metrics.extend([
            Metric(
                name="system.disk.usage_percent",
                value=(disk.used / disk.total) * 100,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                unit="percent"
            ),
            Metric(
                name="system.disk.free_gb",
                value=disk.free / (1024**3),
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                unit="GB"
            )
        ])
        
        # Network метрики
        try:
            network = psutil.net_io_counters()
            metrics.extend([
                Metric(
                    name="system.network.bytes_sent",
                    value=network.bytes_sent,
                    metric_type=MetricType.COUNTER,
                    timestamp=timestamp,
                    unit="bytes"
                ),
                Metric(
                    name="system.network.bytes_recv",
                    value=network.bytes_recv,
                    metric_type=MetricType.COUNTER,
                    timestamp=timestamp,
                    unit="bytes"
                )
            ])
        except Exception:
            pass  # Network stats may not be available
        
        # Process метрики
        try:
            process = psutil.Process()
            metrics.extend([
                Metric(
                    name="process.memory.rss_mb",
                    value=process.memory_info().rss / (1024**2),
                    metric_type=MetricType.GAUGE,
                    timestamp=timestamp,
                    unit="MB"
                ),
                Metric(
                    name="process.cpu.percent",
                    value=process.cpu_percent(),
                    metric_type=MetricType.GAUGE,
                    timestamp=timestamp,
                    unit="percent"
                ),
                Metric(
                    name="process.threads.count",
                    value=process.num_threads(),
                    metric_type=MetricType.GAUGE,
                    timestamp=timestamp,
                    unit="count"
                )
            ])
        except Exception:
            pass
        
        return metrics
    
    def get_metrics(self) -> List[Metric]:
        """Получение собранных метрик"""
        metrics = []
        try:
            while not self.metrics_queue.empty():
                metrics.append(self.metrics_queue.get_nowait())
        except queue.Empty:
            pass
        return metrics

class AlertManager:
    """Менеджер алертов"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def add_alert_rule(self, rule_name: str, metric_name: str, 
                      condition: str, threshold: float, 
                      severity: AlertSeverity = AlertSeverity.WARNING,
                      message_template: str = None):
        """Добавление правила алерта"""
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq', 'gte', 'lte'
            'threshold': threshold,
            'severity': severity,
            'message_template': message_template or f"{metric_name} {condition} {threshold}"
        }
        
        self.logger.info(f"🚨 Добавлено правило алерта: {rule_name}")
    
    def check_alerts(self, metrics: List[Metric]):
        """Проверка алертов"""
        current_metric_values = {m.name: m.value for m in metrics}
        
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric_name']
            
            if metric_name not in current_metric_values:
                continue
            
            current_value = current_metric_values[metric_name]
            threshold = rule['threshold']
            condition = rule['condition']
            
            should_alert = False
            
            if condition == 'gt' and current_value > threshold:
                should_alert = True
            elif condition == 'lt' and current_value < threshold:
                should_alert = True
            elif condition == 'gte' and current_value >= threshold:
                should_alert = True
            elif condition == 'lte' and current_value <= threshold:
                should_alert = True
            elif condition == 'eq' and abs(current_value - threshold) < 0.001:
                should_alert = True
            
            if should_alert:
                self._trigger_alert(rule_name, rule, current_value)
            else:
                self._resolve_alert(rule_name)
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], current_value: float):
        """Срабатывание алерта"""
        if rule_name in self.active_alerts:
            return  # Алерт уже активен
        
        alert_id = f"{rule_name}_{int(time.time())}"
        message = rule['message_template'].format(
            metric_name=rule['metric_name'],
            value=current_value,
            threshold=rule['threshold']
        )
        
        alert = Alert(
            alert_id=alert_id,
            metric_name=rule['metric_name'],
            condition=f"{rule['condition']} {rule['threshold']}",
            severity=rule['severity'],
            message=message,
            timestamp=time.time(),
            metadata={
                'rule_name': rule_name,
                'current_value': current_value,
                'threshold': rule['threshold']
            }
        )
        
        self.active_alerts[rule_name] = alert
        self.alert_history.append(alert)
        
        # Уведомление callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Ошибка в alert callback: {e}")
        
        self.logger.warning(f"🚨 ALERT: {message}")
    
    def _resolve_alert(self, rule_name: str):
        """Разрешение алерта"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            del self.active_alerts[rule_name]
            
            self.logger.info(f"✅ RESOLVED: {alert.message}")
    
    def add_alert_callback(self, callback: Callable):
        """Добавление callback для алертов"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """Получение активных алертов"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Получение истории алертов"""
        return self.alert_history[-limit:]

class PerformanceAnalyzer:
    """Анализатор производительности"""
    
    def __init__(self, aggregator: MetricAggregator):
        self.aggregator = aggregator
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance_trends(self, duration_hours: int = 24) -> Dict[str, Any]:
        """Анализ трендов производительности"""
        duration_seconds = duration_hours * 3600
        analysis = {}
        
        # Ключевые метрики для анализа
        key_metrics = [
            "system.cpu.usage_percent",
            "system.memory.usage_percent",
            "system.disk.usage_percent",
            "process.memory.rss_mb"
        ]
        
        for metric_name in key_metrics:
            time_series = self.aggregator.get_time_series(metric_name, duration_seconds)
            
            if not time_series:
                continue
            
            values = [point['value'] for point in time_series]
            
            if len(values) < 2:
                continue
            
            # Статистический анализ
            trend_analysis = self._analyze_trend(values)
            anomalies = self._detect_anomalies(values)
            
            analysis[metric_name] = {
                'trend': trend_analysis,
                'anomalies': anomalies,
                'statistics': {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values)
                },
                'data_points': len(values)
            }
        
        # Общая оценка здоровья системы
        health_score = self._calculate_health_score(analysis)
        analysis['overall_health'] = health_score
        
        return analysis
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Анализ тренда"""
        if len(values) < 3:
            return {'direction': 'unknown', 'strength': 0}
        
        # Простая линейная регрессия для определения тренда
        x = list(range(len(values)))
        
        # Расчет коэффициента корреляции
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)
        
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(len(values)))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(len(values)))
        denominator_y = sum((values[i] - mean_y) ** 2 for i in range(len(values)))
        
        if denominator_x == 0 or denominator_y == 0:
            correlation = 0
        else:
            correlation = numerator / (denominator_x * denominator_y) ** 0.5
        
        # Определение направления тренда
        if correlation > 0.3:
            direction = 'increasing'
        elif correlation < -0.3:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'strength': abs(correlation),
            'correlation': correlation
        }
    
    def _detect_anomalies(self, values: List[float], threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Обнаружение аномалий"""
        if len(values) < 10:
            return []
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        anomalies = []
        
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'index': i,
                    'value': value,
                    'z_score': z_score,
                    'type': 'outlier'
                })
        
        return anomalies
    
    def _calculate_health_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет общего показателя здоровья системы"""
        scores = []
        
        # Оценка на основе средних значений метрик
        for metric_name, data in analysis.items():
            if metric_name == 'overall_health':
                continue
            
            stats = data.get('statistics', {})
            mean_val = stats.get('mean', 0)
            
            # Простая оценка здоровья (обратная к загруженности)
            if 'usage_percent' in metric_name:
                score = max(0, (100 - mean_val) / 100)
            else:
                score = 0.8  # Нейтральная оценка для других метрик
            
            scores.append(score)
        
        overall_score = statistics.mean(scores) if scores else 0.5
        
        # Определение статуса
        if overall_score >= 0.8:
            status = 'excellent'
        elif overall_score >= 0.6:
            status = 'good'
        elif overall_score >= 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'score': overall_score,
            'status': status,
            'component_scores': len(scores)
        }

class RealTimeAnalytics:
    """Основной класс real-time аналитики"""
    
    def __init__(self, collection_interval: float = 5.0):
        self.aggregator = MetricAggregator()
        self.system_collector = SystemMetricsCollector(collection_interval)
        self.alert_manager = AlertManager()
        self.performance_analyzer = PerformanceAnalyzer(self.aggregator)
        
        # WebSocket клиенты для real-time уведомлений
        self.websocket_clients: List[Any] = []
        
        # Задачи
        self.analytics_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        
        # Настройка базовых алертов
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Настройка базовых алертов"""
        self.alert_manager.add_alert_rule(
            "high_cpu",
            "system.cpu.usage_percent",
            "gt", 80.0,
            AlertSeverity.WARNING,
            "High CPU usage: {value:.1f}% > {threshold}%"
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory",
            "system.memory.usage_percent", 
            "gt", 85.0,
            AlertSeverity.ERROR,
            "High memory usage: {value:.1f}% > {threshold}%"
        )
        
        self.alert_manager.add_alert_rule(
            "disk_space_low",
            "system.disk.usage_percent",
            "gt", 90.0,
            AlertSeverity.CRITICAL,
            "Low disk space: {value:.1f}% used > {threshold}%"
        )
    
    async def start(self):
        """Запуск системы аналитики"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Запуск сбора системных метрик
        self.system_collector.start_collection()
        
        # Запуск основного цикла аналитики
        self.analytics_task = asyncio.create_task(self._analytics_loop())
        
        # Добавление callback для алертов
        self.alert_manager.add_alert_callback(self._on_alert)
        
        self.logger.info("📊 Real-time Analytics система запущена")
    
    async def stop(self):
        """Остановка системы аналитики"""
        self.is_running = False
        
        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass
        
        self.system_collector.stop_collection()
        
        self.logger.info("📊 Real-time Analytics система остановлена")
    
    async def _analytics_loop(self):
        """Основной цикл аналитики"""
        while self.is_running:
            try:
                # Получение новых метрик
                system_metrics = self.system_collector.get_metrics()
                
                # Добавление в агрегатор
                for metric in system_metrics:
                    self.aggregator.add_metric(metric)
                
                # Проверка алертов
                if system_metrics:
                    self.alert_manager.check_alerts(system_metrics)
                
                # Отправка real-time обновлений
                await self._broadcast_real_time_update()
                
                await asyncio.sleep(5)  # Обновление каждые 5 секунд
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле аналитики: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_real_time_update(self):
        """Broadcast real-time обновлений"""
        if not self.websocket_clients:
            return
        
        # Получение текущих метрик
        current_metrics = self.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        update_data = {
            'type': 'metrics_update',
            'timestamp': time.time(),
            'metrics': current_metrics,
            'alerts': [asdict(alert) for alert in active_alerts]
        }
        
        # Отправка всем подключенным клиентам
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_str(json.dumps(update_data))
            except Exception:
                disconnected_clients.append(client)
        
        # Удаление отключенных клиентов
        for client in disconnected_clients:
            self.websocket_clients.remove(client)
    
    def _on_alert(self, alert: Alert):
        """Callback для алертов"""
        self.logger.warning(f"🚨 Alert triggered: {alert.message}")
        
        # Отправка алерта в real-time
        asyncio.create_task(self._broadcast_alert(alert))
    
    async def _broadcast_alert(self, alert: Alert):
        """Broadcast алерта"""
        if not self.websocket_clients:
            return
        
        alert_data = {
            'type': 'alert',
            'alert': asdict(alert)
        }
        
        for client in self.websocket_clients:
            try:
                await client.send_str(json.dumps(alert_data))
            except Exception:
                pass
    
    def add_websocket_client(self, websocket):
        """Добавление WebSocket клиента"""
        self.websocket_clients.append(websocket)
        self.logger.info(f"📡 WebSocket клиент подключен (всего: {len(self.websocket_clients)})")
    
    def remove_websocket_client(self, websocket):
        """Удаление WebSocket клиента"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            self.logger.info(f"📡 WebSocket клиент отключен (всего: {len(self.websocket_clients)})")
    
    def add_custom_metric(self, metric: Metric):
        """Добавление пользовательской метрики"""
        self.aggregator.add_metric(metric)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Получение текущих метрик"""
        metrics = {}
        
        # Системные метрики
        system_metrics = [
            "system.cpu.usage_percent",
            "system.memory.usage_percent", 
            "system.disk.usage_percent",
            "process.memory.rss_mb"
        ]
        
        for metric_name in system_metrics:
            value = self.aggregator.get_aggregated_metrics(metric_name, "avg")
            if value is not None:
                metrics[metric_name] = value
        
        return metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Получение данных для dashboard"""
        return {
            'current_metrics': self.get_current_metrics(),
            'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            'alert_history': [asdict(alert) for alert in self.alert_manager.get_alert_history(10)],
            'performance_analysis': self.performance_analyzer.analyze_performance_trends(1),  # 1 час
            'system_info': {
                'hostname': socket.gethostname(),
                'uptime': time.time() - psutil.boot_time(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        } 