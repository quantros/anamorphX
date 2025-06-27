"""
🌐 Distributed Computing для AnamorphX Enterprise
Система распределенных вычислений и кластерного управления
"""

import asyncio
import aiohttp
import json
import time
import uuid
import hashlib
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import socket
import psutil

class NodeType(Enum):
    """Типы узлов в кластере"""
    MASTER = "master"
    WORKER = "worker"
    BACKUP = "backup"

class TaskStatus(Enum):
    """Статусы задач"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ClusterNode:
    """Узел кластера"""
    node_id: str
    host: str
    port: int
    node_type: NodeType
    status: str = "active"
    last_heartbeat: float = 0
    cpu_count: int = 0
    memory_gb: float = 0
    gpu_count: int = 0
    current_load: float = 0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

@dataclass
class DistributedTask:
    """Распределенная задача"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    requirements: Dict[str, Any] = None
    created_at: float = 0
    started_at: float = 0
    completed_at: float = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = {}
        if self.created_at == 0:
            self.created_at = time.time()

class LoadBalancer:
    """Балансировщик нагрузки"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index = 0
        
    def select_node(self, nodes: List[ClusterNode], 
                   requirements: Dict[str, Any] = None) -> Optional[ClusterNode]:
        """Выбор оптимального узла"""
        if not nodes:
            return None
        
        # Фильтрация узлов по требованиям
        suitable_nodes = self._filter_nodes(nodes, requirements or {})
        
        if not suitable_nodes:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(suitable_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(suitable_nodes)
        elif self.strategy == "random":
            import random
            return random.choice(suitable_nodes)
        else:
            return suitable_nodes[0]
    
    def _filter_nodes(self, nodes: List[ClusterNode], 
                     requirements: Dict[str, Any]) -> List[ClusterNode]:
        """Фильтрация узлов по требованиям"""
        filtered = []
        
        for node in nodes:
            if node.status != "active":
                continue
            
            # Проверка требований
            if requirements.get('min_memory', 0) > node.memory_gb:
                continue
            if requirements.get('min_cpu', 0) > node.cpu_count:
                continue
            if requirements.get('gpu_required', False) and node.gpu_count == 0:
                continue
            if requirements.get('max_load', 1.0) < node.current_load:
                continue
            
            required_capabilities = requirements.get('capabilities', [])
            if not all(cap in node.capabilities for cap in required_capabilities):
                continue
            
            filtered.append(node)
        
        return filtered
    
    def _round_robin_selection(self, nodes: List[ClusterNode]) -> ClusterNode:
        """Round-robin выбор"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return node
    
    def _least_loaded_selection(self, nodes: List[ClusterNode]) -> ClusterNode:
        """Выбор наименее загруженного узла"""
        return min(nodes, key=lambda n: n.current_load)

class DistributedTaskManager:
    """Менеджер распределенных задач"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue = asyncio.Queue()
        self.result_callbacks: Dict[str, Callable] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
    async def submit_task(self, task_type: str, payload: Dict[str, Any],
                         priority: int = 1, requirements: Dict[str, Any] = None) -> str:
        """Отправка задачи в очередь"""
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            requirements=requirements or {}
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task)
        
        self.logger.info(f"📋 Задача {task_id} добавлена в очередь")
        return task_id
    
    async def execute_task(self, task: DistributedTask) -> Any:
        """Выполнение задачи"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            self.logger.info(f"🚀 Выполнение задачи {task.task_id}")
            
            # Выполнение в зависимости от типа задачи
            if task.task_type == "neural_training":
                result = await self._execute_neural_training(task)
            elif task.task_type == "neural_inference":
                result = await self._execute_neural_inference(task)
            elif task.task_type == "data_processing":
                result = await self._execute_data_processing(task)
            elif task.task_type == "model_optimization":
                result = await self._execute_model_optimization(task)
            else:
                raise ValueError(f"Неизвестный тип задачи: {task.task_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            self.logger.info(f"✅ Задача {task.task_id} выполнена")
            return result
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            self.logger.error(f"❌ Ошибка выполнения задачи {task.task_id}: {e}")
            raise
    
    async def _execute_neural_training(self, task: DistributedTask) -> Dict[str, Any]:
        """Выполнение обучения нейронной сети"""
        payload = task.payload
        model_config = payload.get('model_config', {})
        training_data = payload.get('training_data', [])
        epochs = payload.get('epochs', 10)
        
        # Симуляция обучения
        await asyncio.sleep(2)  # Имитация времени обучения
        
        return {
            'model_id': f"model_{task.task_id[:8]}",
            'epochs_completed': epochs,
            'final_loss': 0.001,
            'accuracy': 0.95,
            'training_time': 2.0
        }
    
    async def _execute_neural_inference(self, task: DistributedTask) -> Dict[str, Any]:
        """Выполнение инференса"""
        payload = task.payload
        model_id = payload.get('model_id')
        input_data = payload.get('input_data')
        
        # Симуляция инференса
        await asyncio.sleep(0.1)
        
        return {
            'predictions': [0.8, 0.2],
            'confidence': 0.8,
            'processing_time': 0.1
        }
    
    async def _execute_data_processing(self, task: DistributedTask) -> Dict[str, Any]:
        """Выполнение обработки данных"""
        payload = task.payload
        data = payload.get('data', [])
        operation = payload.get('operation', 'transform')
        
        # Симуляция обработки данных
        await asyncio.sleep(1)
        
        return {
            'processed_items': len(data),
            'operation': operation,
            'output_size': len(data) * 2
        }
    
    async def _execute_model_optimization(self, task: DistributedTask) -> Dict[str, Any]:
        """Выполнение оптимизации модели"""
        payload = task.payload
        model_id = payload.get('model_id')
        optimization_type = payload.get('optimization_type', 'quantization')
        
        # Симуляция оптимизации
        await asyncio.sleep(3)
        
        return {
            'original_size_mb': 100,
            'optimized_size_mb': 25,
            'compression_ratio': 4.0,
            'optimization_type': optimization_type
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Получение статуса задачи"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'task_id': task.task_id,
            'status': task.status.value,
            'progress': self._calculate_progress(task),
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'result': task.result,
            'error': task.error
        }
    
    def _calculate_progress(self, task: DistributedTask) -> float:
        """Расчет прогресса выполнения задачи"""
        if task.status == TaskStatus.PENDING:
            return 0.0
        elif task.status == TaskStatus.RUNNING:
            # Простая оценка прогресса
            if task.started_at > 0:
                elapsed = time.time() - task.started_at
                # Предполагаем, что задача займет от 1 до 10 секунд
                estimated_duration = 5.0
                return min(elapsed / estimated_duration, 0.99)
            return 0.1
        elif task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return 1.0
        return 0.0

class ClusterManager:
    """Менеджер кластера"""
    
    def __init__(self, node_id: str, node_type: NodeType = NodeType.WORKER,
                 host: str = "localhost", port: int = 8080):
        self.node_id = node_id
        self.node_type = node_type
        self.host = host
        self.port = port
        
        # Состояние кластера
        self.nodes: Dict[str, ClusterNode] = {}
        self.master_node: Optional[str] = None
        
        # Компоненты
        self.load_balancer = LoadBalancer("least_loaded")
        self.task_manager = DistributedTaskManager(node_id)
        
        # Сетевые компоненты
        self.session: Optional[aiohttp.ClientSession] = None
        self.heartbeat_interval = 30  # секунд
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Статистика
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'uptime_start': time.time()
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Создание информации о текущем узле
        self._create_current_node()
    
    def _create_current_node(self):
        """Создание информации о текущем узле"""
        self.current_node = ClusterNode(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            node_type=self.node_type,
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_count=0,  # Упрощено
            capabilities=["neural_training", "neural_inference", "data_processing"]
        )
    
    async def start(self):
        """Запуск кластерного узла"""
        self.session = aiohttp.ClientSession()
        
        # Регистрация в кластере
        if self.node_type == NodeType.WORKER:
            await self._register_with_master()
        
        # Запуск heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Запуск обработки задач
        asyncio.create_task(self._task_processing_loop())
        
        print(f"🌐 Кластерный узел {self.node_id} запущен")
        print(f"   🔧 Тип: {self.node_type.value}")
        print(f"   🖥️ CPU: {self.current_node.cpu_count} cores")
        print(f"   💾 Memory: {self.current_node.memory_gb:.1f} GB")
    
    async def _register_with_master(self):
        """Регистрация worker узла в master"""
        # В реальной реализации здесь был бы поиск master узла
        # и отправка запроса на регистрацию
        pass
    
    async def _heartbeat_loop(self):
        """Цикл отправки heartbeat"""
        while True:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Ошибка heartbeat: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat(self):
        """Отправка heartbeat"""
        self.current_node.last_heartbeat = time.time()
        self.current_node.current_load = self._calculate_current_load()
        
        # Обновление статуса в локальном реестре
        self.nodes[self.node_id] = self.current_node
        
        # В реальной реализации здесь была бы отправка в master
        self.logger.debug(f"💓 Heartbeat: load={self.current_node.current_load:.2f}")
    
    def _calculate_current_load(self) -> float:
        """Расчет текущей нагрузки узла"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Простая формула комбинированной нагрузки
        return (cpu_percent + memory_percent) / 200.0
    
    async def _task_processing_loop(self):
        """Цикл обработки задач"""
        while True:
            try:
                # Получение задачи из очереди
                task = await self.task_manager.task_queue.get()
                
                # Выполнение задачи
                start_time = time.time()
                try:
                    result = await self.task_manager.execute_task(task)
                    self.stats['tasks_completed'] += 1
                except Exception as e:
                    self.stats['tasks_failed'] += 1
                    self.logger.error(f"Ошибка выполнения задачи: {e}")
                
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                
                # Уведомление о завершении
                self.task_manager.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле обработки задач: {e}")
                await asyncio.sleep(1)
    
    async def submit_distributed_task(self, task_type: str, payload: Dict[str, Any],
                                    requirements: Dict[str, Any] = None) -> str:
        """Отправка распределенной задачи"""
        # Выбор оптимального узла
        available_nodes = [node for node in self.nodes.values() 
                          if node.status == "active" and node.node_type == NodeType.WORKER]
        
        target_node = self.load_balancer.select_node(available_nodes, requirements)
        
        if not target_node:
            # Выполнение на текущем узле
            return await self.task_manager.submit_task(task_type, payload, 
                                                     requirements=requirements)
        
        # Отправка задачи на другой узел
        task_id = await self._send_task_to_node(target_node, task_type, payload, requirements)
        return task_id
    
    async def _send_task_to_node(self, node: ClusterNode, task_type: str,
                               payload: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """Отправка задачи на конкретный узел"""
        if node.node_id == self.node_id:
            # Локальное выполнение
            return await self.task_manager.submit_task(task_type, payload, 
                                                     requirements=requirements)
        
        # Удаленное выполнение (через HTTP API)
        try:
            task_data = {
                'task_type': task_type,
                'payload': payload,
                'requirements': requirements
            }
            
            url = f"http://{node.host}:{node.port}/api/tasks/submit"
            async with self.session.post(url, json=task_data) as response:
                result = await response.json()
                return result.get('task_id')
                
        except Exception as e:
            self.logger.error(f"Ошибка отправки задачи на узел {node.node_id}: {e}")
            # Fallback на локальное выполнение
            return await self.task_manager.submit_task(task_type, payload,
                                                     requirements=requirements)
    
    def add_node(self, node: ClusterNode):
        """Добавление узла в кластер"""
        self.nodes[node.node_id] = node
        print(f"➕ Узел {node.node_id} добавлен в кластер")
    
    def remove_node(self, node_id: str):
        """Удаление узла из кластера"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            print(f"➖ Узел {node_id} удален из кластера")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Получение статуса кластера"""
        active_nodes = [n for n in self.nodes.values() if n.status == "active"]
        
        total_cpu = sum(node.cpu_count for node in active_nodes)
        total_memory = sum(node.memory_gb for node in active_nodes)
        avg_load = sum(node.current_load for node in active_nodes) / len(active_nodes) if active_nodes else 0
        
        return {
            'cluster_size': len(self.nodes),
            'active_nodes': len(active_nodes),
            'total_cpu_cores': total_cpu,
            'total_memory_gb': total_memory,
            'average_load': avg_load,
            'master_node': self.master_node,
            'current_node': self.node_id,
            'stats': self.stats
        }
    
    def get_node_info(self, node_id: str = None) -> Optional[Dict[str, Any]]:
        """Получение информации об узле"""
        if node_id is None:
            node_id = self.node_id
        
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        return asdict(node)
    
    async def stop(self):
        """Остановка кластерного узла"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        if self.session:
            await self.session.close()
        
        self.task_manager.executor.shutdown(wait=True)
        
        print(f"🛑 Кластерный узел {self.node_id} остановлен")

class DistributedNeuralNetwork:
    """Распределенная нейронная сеть"""
    
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager = cluster_manager
        self.model_shards: Dict[str, List[str]] = {}  # model_id -> list of node_ids
        self.logger = logging.getLogger(__name__)
    
    async def train_distributed_model(self, model_config: Dict[str, Any],
                                    training_data: List[Any]) -> str:
        """Распределенное обучение модели"""
        model_id = f"distributed_model_{uuid.uuid4().hex[:8]}"
        
        # Разделение данных между узлами
        data_shards = self._shard_data(training_data)
        
        # Отправка задач обучения на разные узлы
        training_tasks = []
        for shard_data in data_shards:
            task_id = await self.cluster_manager.submit_distributed_task(
                "neural_training",
                {
                    'model_config': model_config,
                    'training_data': shard_data,
                    'model_id': model_id
                },
                requirements={'min_memory': 2.0, 'capabilities': ['neural_training']}
            )
            training_tasks.append(task_id)
        
        # Ожидание завершения всех задач
        await self._wait_for_tasks(training_tasks)
        
        print(f"🧠 Распределенная модель {model_id} обучена на {len(data_shards)} узлах")
        return model_id
    
    def _shard_data(self, data: List[Any], max_shards: int = 4) -> List[List[Any]]:
        """Разделение данных на части"""
        if len(data) <= max_shards:
            return [[item] for item in data]
        
        shard_size = len(data) // max_shards
        shards = []
        
        for i in range(max_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < max_shards - 1 else len(data)
            shards.append(data[start_idx:end_idx])
        
        return shards
    
    async def _wait_for_tasks(self, task_ids: List[str], timeout: float = 300):
        """Ожидание завершения задач"""
        start_time = time.time()
        
        while True:
            completed_tasks = 0
            
            for task_id in task_ids:
                status = self.cluster_manager.task_manager.get_task_status(task_id)
                if status and status['status'] in ['completed', 'failed']:
                    completed_tasks += 1
            
            if completed_tasks == len(task_ids):
                break
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Задачи не завершились за {timeout} секунд")
            
            await asyncio.sleep(1)
    
    async def parallel_inference(self, model_id: str, input_batch: List[Any]) -> List[Dict[str, Any]]:
        """Параллельный инференс"""
        # Разделение батча между узлами
        batch_shards = self._shard_data(input_batch)
        
        # Отправка задач инференса
        inference_tasks = []
        for shard in batch_shards:
            task_id = await self.cluster_manager.submit_distributed_task(
                "neural_inference",
                {
                    'model_id': model_id,
                    'input_data': shard
                },
                requirements={'capabilities': ['neural_inference']}
            )
            inference_tasks.append(task_id)
        
        # Ожидание результатов
        await self._wait_for_tasks(inference_tasks)
        
        # Сбор результатов
        results = []
        for task_id in inference_tasks:
            status = self.cluster_manager.task_manager.get_task_status(task_id)
            if status and status['result']:
                results.extend(status['result'].get('predictions', []))
        
        return results 