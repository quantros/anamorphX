"""
Команды управления потоком AnamorphX

Команды для управления потоками данных, сигналов и выполнения в нейронной сети.
"""

import time
import uuid
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Импорты команд с обработкой ошибок
try:
    from .commands import FlowControlCommand, CommandResult, CommandError
    from .runtime import ExecutionContext
except ImportError as e:
    print(f"Warning: Could not import commands: {e}")
    # Создаем заглушки
    class CommandResult:
        def __init__(self, success=True, message="", data=None, error=None):
            self.success = success
            self.message = message
            self.data = data
            self.error = error
    
    class CommandError(Exception):
        def __init__(self, code="", message=""):
            self.code = code
            self.message = message
    
    class FlowControlCommand:
        def __init__(self, name="", description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
    
    class ExecutionContext:
        def __init__(self):
            self.neural_entities = {}
            self.synapses = {}
            self.variables = {}
            self.signals = {}
            self.flow_states = {}


# =============================================================================
# ENUMS И DATACLASSES
# =============================================================================

class FlowState(Enum):
    """Состояния потока выполнения"""
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    HALTED = "halted"
    YIELDED = "yielded"


class SignalType(Enum):
    """Типы сигналов"""
    DATA = "data"
    CONTROL = "control"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class LoopType(Enum):
    """Типы циклов"""
    WHILE = "while"
    FOR = "for"
    INFINITE = "infinite"
    CONDITIONAL = "conditional"


@dataclass
class Signal:
    """Сигнал для передачи между узлами"""
    id: str
    source_id: str
    target_id: str
    signal_type: SignalType
    data: Any
    priority: int
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowContext:
    """Контекст выполнения потока"""
    id: str
    state: FlowState
    current_node: Optional[str]
    stack: List[Any] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# =============================================================================
# КОМАНДЫ УПРАВЛЕНИЯ ПОТОКОМ (10 команд)
# =============================================================================

class PulseCommand(FlowControlCommand):
    """Отправка сигнала между узлами"""
    
    def __init__(self):
        super().__init__("pulse", "Send signal between neural nodes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get('from')
            target = kwargs.get('to')
            data = kwargs.get('data')
            priority = kwargs.get('priority', 5)
            signal_type = kwargs.get('type', 'data')
            
            if not source or not target:
                raise CommandError("Source and target nodes are required")
            
            # Проверка существования узлов
            if source not in context.neural_network.nodes:
                raise CommandError(f"Source node '{source}' not found")
            if target not in context.neural_network.nodes:
                raise CommandError(f"Target node '{target}' not found")
            
            source_node = context.neural_network.nodes[source]
            target_node = context.neural_network.nodes[target]
            
            # Создание сигнала
            signal_id = str(uuid.uuid4())
            signal = Signal(
                id=signal_id,
                source_id=source_node.id,
                target_id=target_node.id,
                signal_type=SignalType(signal_type),
                data=data,
                priority=int(priority),
                created_at=time.time(),
                metadata={'created_by': 'pulse_command'}
            )
            
            # Сохранение сигнала
            if not hasattr(context.neural_network, 'signals'):
                context.neural_network.signals = {}
            context.neural_network.signals[signal_id] = signal
            
            # Обновление активности узлов
            source_node.metadata['last_activity'] = time.time()
            target_node.metadata['last_activity'] = time.time()
            
            return CommandResult(
                success=True,
                data={
                    'signal_id': signal_id,
                    'source': source,
                    'target': target,
                    'type': signal_type,
                    'priority': priority,
                    'data_size': len(str(data)) if data else 0
                },
                message=f"Signal sent from '{source}' to '{target}'"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to send pulse: {e}"
            )


class DriftCommand(FlowControlCommand):
    """Перенос данных между узлами"""
    
    def __init__(self):
        super().__init__("drift", "Transfer data between neural nodes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get('from')
            target = kwargs.get('to')
            data_key = kwargs.get('key')
            mode = kwargs.get('mode', 'copy')  # 'copy', 'move', 'reference'
            
            if not source or not target:
                raise CommandError("Source and target nodes are required")
            
            # Проверка существования узлов
            if source not in context.neural_network.nodes:
                raise CommandError(f"Source node '{source}' not found")
            if target not in context.neural_network.nodes:
                raise CommandError(f"Target node '{target}' not found")
            
            source_node = context.neural_network.nodes[source]
            target_node = context.neural_network.nodes[target]
            
            # Определение данных для переноса
            if data_key:
                if data_key not in source_node.data:
                    raise CommandError(f"Data key '{data_key}' not found in source node")
                data_to_transfer = {data_key: source_node.data[data_key]}
            else:
                data_to_transfer = source_node.data.copy()
            
            # Выполнение переноса
            transferred_keys = []
            for key, value in data_to_transfer.items():
                if mode == 'copy':
                    target_node.data[key] = value
                elif mode == 'move':
                    target_node.data[key] = value
                    if key in source_node.data:
                        del source_node.data[key]
                elif mode == 'reference':
                    # В реальной реализации здесь была бы ссылка
                    target_node.data[key] = value
                
                transferred_keys.append(key)
            
            return CommandResult(
                success=True,
                data={
                    'source': source,
                    'target': target,
                    'mode': mode,
                    'transferred_keys': transferred_keys,
                    'data_count': len(transferred_keys)
                },
                message=f"Data drifted from '{source}' to '{target}' ({mode} mode)"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to drift data: {e}"
            )


class EchoCommand(FlowControlCommand):
    """Отражение сигналов"""
    
    def __init__(self):
        super().__init__("echo", "Echo signals back to source")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_name = kwargs.get('node')
            signal_filter = kwargs.get('filter', {})
            delay = kwargs.get('delay', 0.0)
            amplification = kwargs.get('amplification', 1.0)
            
            if not node_name:
                raise CommandError("Node name is required")
            
            # Проверка существования узла
            if node_name not in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' not found")
            
            node = context.neural_network.nodes[node_name]
            echoed_signals = []
            
            # Поиск сигналов для отражения
            if hasattr(context.neural_network, 'signals'):
                for signal_id, signal in context.neural_network.signals.items():
                    if signal.target_id == node.id:
                        # Применение фильтра
                        should_echo = True
                        for filter_key, filter_value in signal_filter.items():
                            if hasattr(signal, filter_key):
                                if getattr(signal, filter_key) != filter_value:
                                    should_echo = False
                                    break
                        
                        if should_echo:
                            # Создание эхо-сигнала
                            echo_signal_id = str(uuid.uuid4())
                            echo_data = signal.data
                            
                            # Применение усиления
                            if isinstance(echo_data, (int, float)):
                                echo_data *= amplification
                            
                            echo_signal = Signal(
                                id=echo_signal_id,
                                source_id=signal.target_id,
                                target_id=signal.source_id,
                                signal_type=signal.signal_type,
                                data=echo_data,
                                priority=signal.priority,
                                created_at=time.time() + delay,
                                metadata={
                                    'echo_of': signal_id,
                                    'amplification': amplification,
                                    'delay': delay
                                }
                            )
                            
                            context.neural_network.signals[echo_signal_id] = echo_signal
                            echoed_signals.append(echo_signal_id)
            
            return CommandResult(
                success=True,
                data={
                    'node': node_name,
                    'echoed_signals': echoed_signals,
                    'echo_count': len(echoed_signals),
                    'amplification': amplification,
                    'delay': delay
                },
                message=f"Echoed {len(echoed_signals)} signals from node '{node_name}'"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to echo signals: {e}"
            )


class ReflectCommand(FlowControlCommand):
    """Отражение состояния узлов"""
    
    def __init__(self):
        super().__init__("reflect", "Reflect current state of neural nodes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('target', 'all')  # 'all', 'active', specific node
            include_data = kwargs.get('include_data', False)
            include_connections = kwargs.get('include_connections', True)
            
            reflection_data = {}
            
            if target == 'all':
                nodes_to_reflect = context.neural_network.nodes.values()
            elif target == 'active':
                current_time = time.time()
                nodes_to_reflect = [
                    node for node in context.neural_network.nodes.values()
                    if current_time - node.metadata.get('last_activity', node.created_at) < 3600
                ]
            else:
                if target not in context.neural_network.nodes:
                    raise CommandError(f"Node '{target}' not found")
                nodes_to_reflect = [context.neural_network.nodes[target]]
            
            # Сбор информации о состоянии
            for node in nodes_to_reflect:
                node_reflection = {
                    'id': node.id,
                    'name': node.name,
                    'type': node.node_type.value,
                    'state': node.state,
                    'created_at': node.created_at,
                    'last_activity': node.metadata.get('last_activity', node.created_at),
                    'metadata_keys': list(node.metadata.keys())
                }
                
                if include_connections:
                    node_reflection['connections'] = list(node.connections)
                    node_reflection['connection_count'] = len(node.connections)
                
                if include_data:
                    node_reflection['data'] = node.data
                    node_reflection['data_keys'] = list(node.data.keys())
                else:
                    node_reflection['data_keys'] = list(node.data.keys())
                    node_reflection['data_count'] = len(node.data)
                
                reflection_data[node.name] = node_reflection
            
            return CommandResult(
                success=True,
                data={
                    'target': target,
                    'reflection_data': reflection_data,
                    'node_count': len(reflection_data),
                    'include_data': include_data,
                    'include_connections': include_connections,
                    'timestamp': time.time()
                },
                message=f"Reflected state of {len(reflection_data)} nodes"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to reflect state: {e}"
            )


class AbsorbCommand(FlowControlCommand):
    """Поглощение сигналов"""
    
    def __init__(self):
        super().__init__("absorb", "Absorb incoming signals")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_name = kwargs.get('node')
            signal_types = kwargs.get('types', ['data'])
            max_signals = kwargs.get('max_signals', 10)
            store_absorbed = kwargs.get('store', True)
            
            if not node_name:
                raise CommandError("Node name is required")
            
            # Проверка существования узла
            if node_name not in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' not found")
            
            node = context.neural_network.nodes[node_name]
            absorbed_signals = []
            
            # Поиск и поглощение сигналов
            if hasattr(context.neural_network, 'signals'):
                signals_to_remove = []
                for signal_id, signal in context.neural_network.signals.items():
                    if (signal.target_id == node.id and 
                        signal.signal_type.value in signal_types and
                        len(absorbed_signals) < max_signals):
                        
                        absorbed_signal_data = {
                            'id': signal_id,
                            'source_id': signal.source_id,
                            'type': signal.signal_type.value,
                            'data': signal.data,
                            'priority': signal.priority,
                            'absorbed_at': time.time()
                        }
                        
                        absorbed_signals.append(absorbed_signal_data)
                        signals_to_remove.append(signal_id)
                        
                        # Сохранение поглощенного сигнала в узле
                        if store_absorbed:
                            if 'absorbed_signals' not in node.data:
                                node.data['absorbed_signals'] = []
                            node.data['absorbed_signals'].append(absorbed_signal_data)
                
                # Удаление поглощенных сигналов
                for signal_id in signals_to_remove:
                    del context.neural_network.signals[signal_id]
            
            # Обновление метаданных узла
            node.metadata['last_absorption'] = time.time()
            node.metadata['total_absorbed'] = node.metadata.get('total_absorbed', 0) + len(absorbed_signals)
            
            return CommandResult(
                success=True,
                data={
                    'node': node_name,
                    'absorbed_signals': absorbed_signals,
                    'absorption_count': len(absorbed_signals),
                    'signal_types': signal_types,
                    'stored': store_absorbed
                },
                message=f"Absorbed {len(absorbed_signals)} signals into node '{node_name}'"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to absorb signals: {e}"
            )


class DiffuseCommand(FlowControlCommand):
    """Распространение сигналов"""
    
    def __init__(self):
        super().__init__("diffuse", "Diffuse signals across network")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source_node = kwargs.get('from')
            data = kwargs.get('data')
            radius = kwargs.get('radius', 1)
            decay_factor = kwargs.get('decay', 0.9)
            signal_type = kwargs.get('type', 'data')
            
            if not source_node:
                raise CommandError("Source node is required")
            
            # Проверка существования узла
            if source_node not in context.neural_network.nodes:
                raise CommandError(f"Source node '{source_node}' not found")
            
            source = context.neural_network.nodes[source_node]
            diffused_signals = []
            
            # Алгоритм распространения по радиусу
            visited_nodes = set()
            current_level = {source.id}
            current_data = data
            
            for level in range(radius):
                next_level = set()
                
                for node_id in current_level:
                    if node_id in visited_nodes:
                        continue
                    
                    visited_nodes.add(node_id)
                    
                    # Поиск соседних узлов
                    if hasattr(context.neural_network, 'synapses'):
                        for synapse in context.neural_network.synapses.values():
                            if synapse.source_id == node_id:
                                target_id = synapse.target_id
                                if target_id not in visited_nodes:
                                    next_level.add(target_id)
                                    
                                    # Создание диффузного сигнала
                                    signal_id = str(uuid.uuid4())
                                    diffused_data = current_data
                                    
                                    # Применение затухания
                                    if isinstance(diffused_data, (int, float)):
                                        diffused_data *= (decay_factor ** (level + 1))
                                    
                                    signal = Signal(
                                        id=signal_id,
                                        source_id=node_id,
                                        target_id=target_id,
                                        signal_type=SignalType(signal_type),
                                        data=diffused_data,
                                        priority=5,
                                        created_at=time.time(),
                                        metadata={
                                            'diffusion_level': level + 1,
                                            'decay_factor': decay_factor,
                                            'original_source': source.id
                                        }
                                    )
                                    
                                    if not hasattr(context.neural_network, 'signals'):
                                        context.neural_network.signals = {}
                                    context.neural_network.signals[signal_id] = signal
                                    diffused_signals.append(signal_id)
                
                current_level = next_level
                if not current_level:
                    break
            
            return CommandResult(
                success=True,
                data={
                    'source': source_node,
                    'radius': radius,
                    'decay_factor': decay_factor,
                    'diffused_signals': diffused_signals,
                    'signal_count': len(diffused_signals),
                    'affected_nodes': len(visited_nodes)
                },
                message=f"Diffused {len(diffused_signals)} signals from '{source_node}' with radius {radius}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to diffuse signals: {e}"
            )


class MergeCommand(FlowControlCommand):
    """Слияние потоков данных"""
    
    def __init__(self):
        super().__init__("merge", "Merge multiple data flows")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source_nodes = kwargs.get('sources', [])
            target_node = kwargs.get('target')
            merge_strategy = kwargs.get('strategy', 'combine')  # 'combine', 'average', 'max', 'min'
            data_key = kwargs.get('key')
            
            if not source_nodes or not target_node:
                raise CommandError("Source nodes and target node are required")
            
            # Проверка существования узлов
            sources = []
            for node_name in source_nodes:
                if node_name not in context.neural_network.nodes:
                    raise CommandError(f"Source node '{node_name}' not found")
                sources.append(context.neural_network.nodes[node_name])
            
            if target_node not in context.neural_network.nodes:
                raise CommandError(f"Target node '{target_node}' not found")
            
            target = context.neural_network.nodes[target_node]
            
            # Сбор данных для слияния
            merge_data = []
            for source in sources:
                if data_key:
                    if data_key in source.data:
                        merge_data.append(source.data[data_key])
                else:
                    merge_data.append(source.data)
            
            # Выполнение слияния
            merged_result = None
            if merge_strategy == 'combine':
                if data_key:
                    merged_result = merge_data
                else:
                    merged_result = {}
                    for data_dict in merge_data:
                        if isinstance(data_dict, dict):
                            merged_result.update(data_dict)
            
            elif merge_strategy == 'average':
                numeric_data = [d for d in merge_data if isinstance(d, (int, float))]
                if numeric_data:
                    merged_result = sum(numeric_data) / len(numeric_data)
            
            elif merge_strategy == 'max':
                numeric_data = [d for d in merge_data if isinstance(d, (int, float))]
                if numeric_data:
                    merged_result = max(numeric_data)
            
            elif merge_strategy == 'min':
                numeric_data = [d for d in merge_data if isinstance(d, (int, float))]
                if numeric_data:
                    merged_result = min(numeric_data)
            
            # Сохранение результата
            result_key = data_key or 'merged_data'
            target.data[result_key] = merged_result
            
            # Обновление метаданных
            target.metadata['last_merge'] = time.time()
            target.metadata['merge_sources'] = source_nodes
            target.metadata['merge_strategy'] = merge_strategy
            
            return CommandResult(
                success=True,
                data={
                    'sources': source_nodes,
                    'target': target_node,
                    'strategy': merge_strategy,
                    'data_key': data_key,
                    'merged_result': merged_result,
                    'source_count': len(source_nodes)
                },
                message=f"Merged data from {len(source_nodes)} sources into '{target_node}'"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to merge flows: {e}"
            )


class SplitCommand(FlowControlCommand):
    """Разделение потоков данных"""
    
    def __init__(self):
        super().__init__("split", "Split data flow into multiple streams")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source_node = kwargs.get('source')
            target_nodes = kwargs.get('targets', [])
            split_strategy = kwargs.get('strategy', 'duplicate')  # 'duplicate', 'distribute', 'partition'
            data_key = kwargs.get('key')
            
            if not source_node or not target_nodes:
                raise CommandError("Source node and target nodes are required")
            
            # Проверка существования узлов
            if source_node not in context.neural_network.nodes:
                raise CommandError(f"Source node '{source_node}' not found")
            
            source = context.neural_network.nodes[source_node]
            targets = []
            for node_name in target_nodes:
                if node_name not in context.neural_network.nodes:
                    raise CommandError(f"Target node '{node_name}' not found")
                targets.append(context.neural_network.nodes[node_name])
            
            # Получение данных для разделения
            if data_key:
                if data_key not in source.data:
                    raise CommandError(f"Data key '{data_key}' not found in source node")
                source_data = source.data[data_key]
            else:
                source_data = source.data
            
            # Выполнение разделения
            split_results = []
            
            if split_strategy == 'duplicate':
                # Дублирование данных во все целевые узлы
                for i, target in enumerate(targets):
                    result_key = data_key or f'split_data_{i}'
                    target.data[result_key] = source_data
                    split_results.append({
                        'target': target.name,
                        'key': result_key,
                        'data_size': len(str(source_data))
                    })
            
            elif split_strategy == 'distribute':
                # Распределение элементов данных
                if isinstance(source_data, (list, tuple)):
                    for i, target in enumerate(targets):
                        target_data = []
                        for j in range(i, len(source_data), len(targets)):
                            target_data.append(source_data[j])
                        
                        result_key = data_key or f'distributed_data_{i}'
                        target.data[result_key] = target_data
                        split_results.append({
                            'target': target.name,
                            'key': result_key,
                            'data_size': len(target_data)
                        })
            
            elif split_strategy == 'partition':
                # Разделение данных на части
                if isinstance(source_data, (list, tuple)):
                    chunk_size = len(source_data) // len(targets)
                    remainder = len(source_data) % len(targets)
                    
                    start_idx = 0
                    for i, target in enumerate(targets):
                        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                        target_data = source_data[start_idx:end_idx]
                        
                        result_key = data_key or f'partition_data_{i}'
                        target.data[result_key] = target_data
                        split_results.append({
                            'target': target.name,
                            'key': result_key,
                            'data_size': len(target_data)
                        })
                        
                        start_idx = end_idx
            
            # Обновление метаданных
            for target in targets:
                target.metadata['last_split_receive'] = time.time()
                target.metadata['split_source'] = source_node
                target.metadata['split_strategy'] = split_strategy
            
            return CommandResult(
                success=True,
                data={
                    'source': source_node,
                    'targets': target_nodes,
                    'strategy': split_strategy,
                    'data_key': data_key,
                    'split_results': split_results,
                    'target_count': len(targets)
                },
                message=f"Split data from '{source_node}' to {len(targets)} targets using {split_strategy} strategy"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to split flow: {e}"
            )


class LoopCommand(FlowControlCommand):
    """Циклические операции"""
    
    def __init__(self):
        super().__init__("loop", "Execute cyclic operations")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            loop_type = kwargs.get('type', 'while')
            condition = kwargs.get('condition')
            iterations = kwargs.get('iterations', 10)
            commands = kwargs.get('commands', [])
            node_name = kwargs.get('node')
            
            if not node_name:
                raise CommandError("Node name is required")
            
            # Проверка существования узла
            if node_name not in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' not found")
            
            node = context.neural_network.nodes[node_name]
            loop_id = str(uuid.uuid4())
            
            # Создание контекста цикла
            loop_context = {
                'id': loop_id,
                'type': loop_type,
                'node': node_name,
                'start_time': time.time(),
                'iterations_completed': 0,
                'max_iterations': iterations,
                'condition': condition,
                'commands': commands,
                'state': 'running'
            }
            
            # Выполнение цикла
            execution_log = []
            
            if loop_type == 'for':
                for i in range(iterations):
                    iteration_start = time.time()
                    
                    # Симуляция выполнения команд
                    iteration_result = {
                        'iteration': i,
                        'timestamp': iteration_start,
                        'commands_executed': len(commands),
                        'status': 'completed'
                    }
                    
                    execution_log.append(iteration_result)
                    loop_context['iterations_completed'] = i + 1
                    
                    # Обновление данных узла
                    node.data[f'loop_iteration_{i}'] = iteration_result
            
            elif loop_type == 'while':
                iteration = 0
                while iteration < iterations:  # Защита от бесконечного цикла
                    iteration_start = time.time()
                    
                    # Простая проверка условия (в реальной реализации была бы более сложная)
                    if condition and 'false' in str(condition).lower():
                        break
                    
                    iteration_result = {
                        'iteration': iteration,
                        'timestamp': iteration_start,
                        'condition_check': True,
                        'status': 'completed'
                    }
                    
                    execution_log.append(iteration_result)
                    loop_context['iterations_completed'] = iteration + 1
                    iteration += 1
            
            elif loop_type == 'infinite':
                # Создание бесконечного цикла (с ограничением для безопасности)
                loop_context['state'] = 'infinite'
                loop_context['max_iterations'] = float('inf')
                
                # В реальной реализации здесь был бы механизм остановки
                execution_log.append({
                    'iteration': 0,
                    'timestamp': time.time(),
                    'status': 'infinite_started',
                    'note': 'Infinite loop started - use halt command to stop'
                })
            
            # Завершение цикла
            loop_context['end_time'] = time.time()
            loop_context['total_time'] = loop_context['end_time'] - loop_context['start_time']
            loop_context['state'] = 'completed'
            
            # Сохранение контекста цикла
            if 'active_loops' not in node.data:
                node.data['active_loops'] = {}
            node.data['active_loops'][loop_id] = loop_context
            
            return CommandResult(
                success=True,
                data={
                    'loop_id': loop_id,
                    'node': node_name,
                    'type': loop_type,
                    'iterations_completed': loop_context['iterations_completed'],
                    'total_time': loop_context['total_time'],
                    'execution_log': execution_log,
                    'state': loop_context['state']
                },
                message=f"Loop executed on node '{node_name}': {loop_context['iterations_completed']} iterations"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to execute loop: {e}"
            )


class HaltCommand(FlowControlCommand):
    """Остановка выполнения"""
    
    def __init__(self):
        super().__init__("halt", "Halt execution of nodes or processes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('target', 'current')  # 'current', 'all', 'node', 'loop'
            target_id = kwargs.get('id')
            reason = kwargs.get('reason', 'manual_halt')
            graceful = kwargs.get('graceful', True)
            
            halted_items = []
            
            if target == 'all':
                # Остановка всех узлов
                for node_name, node in context.neural_network.nodes.items():
                    node.state = 'halted'
                    node.metadata['halt_reason'] = reason
                    node.metadata['halt_time'] = time.time()
                    halted_items.append(f"node:{node_name}")
                
                # Остановка всех активных циклов
                for node in context.neural_network.nodes.values():
                    if 'active_loops' in node.data:
                        for loop_id, loop_context in node.data['active_loops'].items():
                            if loop_context['state'] in ['running', 'infinite']:
                                loop_context['state'] = 'halted'
                                loop_context['halt_reason'] = reason
                                loop_context['halt_time'] = time.time()
                                halted_items.append(f"loop:{loop_id}")
            
            elif target == 'node' and target_id:
                # Остановка конкретного узла
                if target_id in context.neural_network.nodes:
                    node = context.neural_network.nodes[target_id]
                    node.state = 'halted'
                    node.metadata['halt_reason'] = reason
                    node.metadata['halt_time'] = time.time()
                    halted_items.append(f"node:{target_id}")
                    
                    # Остановка циклов узла
                    if 'active_loops' in node.data:
                        for loop_id, loop_context in node.data['active_loops'].items():
                            if loop_context['state'] in ['running', 'infinite']:
                                loop_context['state'] = 'halted'
                                loop_context['halt_reason'] = reason
                                loop_context['halt_time'] = time.time()
                                halted_items.append(f"loop:{loop_id}")
                else:
                    raise CommandError(f"Node '{target_id}' not found")
            
            elif target == 'loop' and target_id:
                # Остановка конкретного цикла
                loop_found = False
                for node in context.neural_network.nodes.values():
                    if 'active_loops' in node.data and target_id in node.data['active_loops']:
                        loop_context = node.data['active_loops'][target_id]
                        if loop_context['state'] in ['running', 'infinite']:
                            loop_context['state'] = 'halted'
                            loop_context['halt_reason'] = reason
                            loop_context['halt_time'] = time.time()
                            halted_items.append(f"loop:{target_id}")
                            loop_found = True
                            break
                
                if not loop_found:
                    raise CommandError(f"Loop '{target_id}' not found or not running")
            
            # Очистка сигналов если требуется
            if not graceful and hasattr(context.neural_network, 'signals'):
                cleared_signals = len(context.neural_network.signals)
                context.neural_network.signals.clear()
                halted_items.append(f"signals:{cleared_signals}")
            
            return CommandResult(
                success=True,
                data={
                    'target': target,
                    'target_id': target_id,
                    'reason': reason,
                    'graceful': graceful,
                    'halted_items': halted_items,
                    'halt_count': len(halted_items),
                    'halt_time': time.time()
                },
                message=f"Halted {len(halted_items)} items: {', '.join(halted_items)}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to halt execution: {e}"
            )


class SpawnCommand(FlowControlCommand):
    """Создание новых процессов/потоков выполнения"""
    
    def __init__(self):
        super().__init__("spawn", "Create new execution processes or threads")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            process_type = kwargs.get('type', 'thread')  # 'thread', 'process', 'coroutine'
            target_function = kwargs.get('function')
            target_node = kwargs.get('node')
            process_name = kwargs.get('name')
            parameters = kwargs.get('params', {})
            priority = kwargs.get('priority', 5)
            
            if not target_function and not target_node:
                raise CommandError("Either function or node must be specified for spawning")
            
            # Генерация уникального ID процесса
            process_id = str(uuid.uuid4())
            if not process_name:
                process_name = f"spawn_{process_id[:8]}"
            
            # Создание контекста процесса
            process_context = {
                'id': process_id,
                'name': process_name,
                'type': process_type,
                'target_function': target_function,
                'target_node': target_node,
                'parameters': parameters,
                'priority': priority,
                'state': 'spawned',
                'created_at': time.time(),
                'parent_context': context,
                'execution_log': []
            }
            
            # Инициализация процессов в контексте
            if not hasattr(context, 'spawned_processes'):
                context.spawned_processes = {}
            
            spawned_result = None
            
            if process_type == 'thread':
                # Создание потока
                def thread_function():
                    thread_start = time.time()
                    process_context['state'] = 'running'
                    process_context['start_time'] = thread_start
                    
                    try:
                        if target_node and target_node in context.neural_network.nodes:
                            # Выполнение операций узла
                            node = context.neural_network.nodes[target_node]
                            result = {
                                'node_activation': target_node,
                                'timestamp': thread_start,
                                'thread_id': threading.current_thread().ident,
                                'parameters': parameters
                            }
                            
                            # Симуляция обработки узла
                            time.sleep(0.1)  # Имитация работы
                            node.metadata['last_thread_activation'] = thread_start
                            
                        elif target_function:
                            # Выполнение функции
                            result = {
                                'function_call': target_function,
                                'timestamp': thread_start,
                                'thread_id': threading.current_thread().ident,
                                'parameters': parameters
                            }
                        
                        process_context['state'] = 'completed'
                        process_context['end_time'] = time.time()
                        process_context['result'] = result
                        process_context['execution_log'].append({
                            'event': 'completed',
                            'timestamp': time.time(),
                            'result': result
                        })
                        
                    except Exception as e:
                        process_context['state'] = 'failed'
                        process_context['error'] = str(e)
                        process_context['end_time'] = time.time()
                        process_context['execution_log'].append({
                            'event': 'error',
                            'timestamp': time.time(),
                            'error': str(e)
                        })
                
                # Запуск потока
                thread = threading.Thread(target=thread_function, name=process_name)
                thread.daemon = True
                thread.start()
                
                process_context['thread'] = thread
                spawned_result = {
                    'thread_id': thread.ident,
                    'thread_name': thread.name,
                    'is_alive': thread.is_alive()
                }
            
            elif process_type == 'coroutine':
                # Создание корутины (асинхронной функции)
                async def coroutine_function():
                    coro_start = time.time()
                    process_context['state'] = 'running'
                    process_context['start_time'] = coro_start
                    
                    try:
                        if target_node and target_node in context.neural_network.nodes:
                            # Асинхронная обработка узла
                            await asyncio.sleep(0.1)  # Имитация асинхронной работы
                            
                            result = {
                                'async_node_activation': target_node,
                                'timestamp': coro_start,
                                'parameters': parameters
                            }
                        elif target_function:
                            await asyncio.sleep(0.1)
                            result = {
                                'async_function_call': target_function,
                                'timestamp': coro_start,
                                'parameters': parameters
                            }
                        
                        process_context['state'] = 'completed'
                        process_context['end_time'] = time.time()
                        process_context['result'] = result
                        
                    except Exception as e:
                        process_context['state'] = 'failed'
                        process_context['error'] = str(e)
                        process_context['end_time'] = time.time()
                
                # Сохранение корутины для будущего выполнения
                process_context['coroutine'] = coroutine_function()
                spawned_result = {
                    'coroutine_created': True,
                    'awaitable': True
                }
            
            else:  # process_type == 'process'
                # Создание процесса (симуляция)
                process_context['state'] = 'process_spawned'
                process_context['pid'] = f"sim_pid_{process_id[:8]}"
                spawned_result = {
                    'process_id': process_context['pid'],
                    'simulated': True
                }
            
            # Сохранение контекста процесса
            context.spawned_processes[process_id] = process_context
            
            return CommandResult(
                success=True,
                data={
                    'process_id': process_id,
                    'process_name': process_name,
                    'type': process_type,
                    'target_function': target_function,
                    'target_node': target_node,
                    'priority': priority,
                    'spawned_result': spawned_result,
                    'created_at': process_context['created_at']
                },
                message=f"Process '{process_name}' spawned as {process_type}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to spawn process: {e}"
            )


class JumpCommand(FlowControlCommand):
    """Переход к другому узлу или состоянию"""
    
    def __init__(self):
        super().__init__("jump", "Jump to another node or execution state")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('to')
            condition = kwargs.get('if')
            jump_type = kwargs.get('type', 'unconditional')  # 'unconditional', 'conditional', 'computed'
            current_node = kwargs.get('from')
            save_state = kwargs.get('save_state', True)
            
            if not target:
                raise CommandError("Jump target is required")
            
            # Инициализация системы переходов
            if not hasattr(context, 'jump_history'):
                context.jump_history = []
            
            if not hasattr(context, 'current_execution_node'):
                context.current_execution_node = None
            
            # Проверка условия перехода
            should_jump = True
            condition_result = None
            
            if jump_type == 'conditional' and condition:
                # Простая оценка условия
                try:
                    if isinstance(condition, str):
                        # Простые строковые условия
                        if 'true' in condition.lower():
                            condition_result = True
                        elif 'false' in condition.lower():
                            condition_result = False
                        else:
                            # Попытка оценить как выражение
                            condition_result = bool(eval(condition, {"__builtins__": {}}))
                    else:
                        condition_result = bool(condition)
                    
                    should_jump = condition_result
                    
                except Exception as e:
                    raise CommandError(f"Invalid jump condition: {e}")
            
            if not should_jump:
                return CommandResult(
                    success=True,
                    data={
                        'jumped': False,
                        'condition_result': condition_result,
                        'reason': 'condition_not_met'
                    },
                    message=f"Jump to '{target}' skipped - condition not met"
                )
            
            # Сохранение текущего состояния
            current_state = None
            if save_state:
                current_state = {
                    'previous_node': context.current_execution_node,
                    'timestamp': time.time(),
                    'variables': context.variables.copy() if hasattr(context, 'variables') else {},
                    'jump_reason': 'manual_jump'
                }
            
            # Проверка существования целевого узла
            if target in context.neural_network.nodes:
                target_node = context.neural_network.nodes[target]
                
                # Выполнение перехода
                old_node = context.current_execution_node
                context.current_execution_node = target
                
                # Обновление метаданных узла
                target_node.metadata['last_jump_arrival'] = time.time()
                target_node.metadata['jump_count'] = target_node.metadata.get('jump_count', 0) + 1
                
                # Запись в историю переходов
                jump_record = {
                    'id': str(uuid.uuid4()),
                    'from': old_node,
                    'to': target,
                    'type': jump_type,
                    'condition': condition,
                    'condition_result': condition_result,
                    'timestamp': time.time(),
                    'saved_state': current_state
                }
                
                context.jump_history.append(jump_record)
                
                # Активация целевого узла
                target_node.state = 'active'
                target_node.metadata['activation_source'] = 'jump_command'
                
                return CommandResult(
                    success=True,
                    data={
                        'jumped': True,
                        'from': old_node,
                        'to': target,
                        'type': jump_type,
                        'condition_result': condition_result,
                        'jump_id': jump_record['id'],
                        'node_activated': True,
                        'state_saved': save_state
                    },
                    message=f"Jumped from '{old_node}' to '{target}'"
                )
            
            else:
                # Целевой узел не существует - создать ссылку на будущий узел
                jump_record = {
                    'id': str(uuid.uuid4()),
                    'from': context.current_execution_node,
                    'to': target,
                    'type': jump_type,
                    'condition': condition,
                    'timestamp': time.time(),
                    'status': 'deferred',
                    'reason': 'target_node_not_found'
                }
                
                context.jump_history.append(jump_record)
                
                return CommandResult(
                    success=True,
                    data={
                        'jumped': False,
                        'from': context.current_execution_node,
                        'to': target,
                        'type': jump_type,
                        'status': 'deferred',
                        'jump_id': jump_record['id'],
                        'reason': 'target_node_not_found'
                    },
                    message=f"Jump to '{target}' deferred - target node will be created later"
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to execute jump: {e}"
            )


class WaitCommand(FlowControlCommand):
    """Ожидание события или условия"""
    
    def __init__(self):
        super().__init__("wait", "Wait for event, condition, or time duration")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            wait_type = kwargs.get('type', 'time')  # 'time', 'signal', 'condition', 'node', 'event'
            duration = kwargs.get('duration', 1.0)  # в секундах
            condition = kwargs.get('condition')
            target_node = kwargs.get('node')
            signal_type = kwargs.get('signal')
            timeout = kwargs.get('timeout', 30.0)  # максимальное время ожидания
            
            # Инициализация системы ожидания
            if not hasattr(context, 'wait_states'):
                context.wait_states = {}
            
            wait_id = str(uuid.uuid4())
            wait_start = time.time()
            
            wait_context = {
                'id': wait_id,
                'type': wait_type,
                'start_time': wait_start,
                'duration': duration,
                'condition': condition,
                'target_node': target_node,
                'signal_type': signal_type,
                'timeout': timeout,
                'state': 'waiting',
                'result': None
            }
            
            context.wait_states[wait_id] = wait_context
            
            if wait_type == 'time':
                # Ожидание определенного времени
                if duration > 0:
                    time.sleep(min(duration, 5.0))  # Ограничиваем максимальное ожидание
                
                wait_context['state'] = 'completed'
                wait_context['end_time'] = time.time()
                wait_context['actual_duration'] = wait_context['end_time'] - wait_start
                
                return CommandResult(
                    success=True,
                    data={
                        'wait_id': wait_id,
                        'type': wait_type,
                        'requested_duration': duration,
                        'actual_duration': wait_context['actual_duration'],
                        'completed': True
                    },
                    message=f"Waited for {wait_context['actual_duration']:.2f} seconds"
                )
            
            elif wait_type == 'signal':
                # Ожидание сигнала
                signal_found = False
                elapsed = 0.0
                check_interval = 0.1
                
                while elapsed < timeout:
                    # Проверка наличия сигналов
                    if hasattr(context.neural_network, 'signals'):
                        for signal_id, signal in context.neural_network.signals.items():
                            if signal_type and signal.signal_type.value == signal_type:
                                signal_found = True
                                wait_context['found_signal'] = signal_id
                                break
                            elif not signal_type:  # Любой сигнал
                                signal_found = True
                                wait_context['found_signal'] = signal_id
                                break
                    
                    if signal_found:
                        break
                    
                    time.sleep(check_interval)
                    elapsed += check_interval
                
                wait_context['state'] = 'completed' if signal_found else 'timeout'
                wait_context['end_time'] = time.time()
                wait_context['elapsed_time'] = elapsed
                
                return CommandResult(
                    success=signal_found,
                    data={
                        'wait_id': wait_id,
                        'type': wait_type,
                        'signal_found': signal_found,
                        'signal_type': signal_type,
                        'found_signal': wait_context.get('found_signal'),
                        'elapsed_time': elapsed,
                        'timeout': timeout
                    },
                    message=f"Signal wait {'completed' if signal_found else 'timed out'} after {elapsed:.2f}s"
                )
            
            elif wait_type == 'node':
                # Ожидание активности узла
                if not target_node:
                    raise CommandError("Node name required for node wait type")
                
                if target_node not in context.neural_network.nodes:
                    raise CommandError(f"Node '{target_node}' not found")
                
                node = context.neural_network.nodes[target_node]
                node_activated = False
                elapsed = 0.0
                check_interval = 0.1
                initial_activity = node.metadata.get('last_activity', 0)
                
                while elapsed < timeout:
                    current_activity = node.metadata.get('last_activity', 0)
                    if current_activity > initial_activity:
                        node_activated = True
                        wait_context['activity_time'] = current_activity
                        break
                    
                    time.sleep(check_interval)
                    elapsed += check_interval
                
                wait_context['state'] = 'completed' if node_activated else 'timeout'
                wait_context['end_time'] = time.time()
                wait_context['elapsed_time'] = elapsed
                
                return CommandResult(
                    success=node_activated,
                    data={
                        'wait_id': wait_id,
                        'type': wait_type,
                        'target_node': target_node,
                        'node_activated': node_activated,
                        'elapsed_time': elapsed,
                        'timeout': timeout,
                        'activity_time': wait_context.get('activity_time')
                    },
                    message=f"Node wait {'completed' if node_activated else 'timed out'} after {elapsed:.2f}s"
                )
            
            elif wait_type == 'condition':
                # Ожидание выполнения условия
                if not condition:
                    raise CommandError("Condition required for condition wait type")
                
                condition_met = False
                elapsed = 0.0
                check_interval = 0.1
                
                while elapsed < timeout:
                    try:
                        # Простая оценка условия
                        if isinstance(condition, str):
                            if 'true' in condition.lower():
                                condition_met = True
                            elif elapsed > duration:  # Условие времени
                                condition_met = True
                        else:
                            condition_met = bool(condition)
                        
                        if condition_met:
                            wait_context['condition_result'] = True
                            break
                            
                    except Exception:
                        pass  # Продолжаем ожидание при ошибке оценки
                    
                    time.sleep(check_interval)
                    elapsed += check_interval
                
                wait_context['state'] = 'completed' if condition_met else 'timeout'
                wait_context['end_time'] = time.time()
                wait_context['elapsed_time'] = elapsed
                
                return CommandResult(
                    success=condition_met,
                    data={
                        'wait_id': wait_id,
                        'type': wait_type,
                        'condition': condition,
                        'condition_met': condition_met,
                        'elapsed_time': elapsed,
                        'timeout': timeout
                    },
                    message=f"Condition wait {'completed' if condition_met else 'timed out'} after {elapsed:.2f}s"
                )
            
            else:
                raise CommandError(f"Unknown wait type: {wait_type}")
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to execute wait: {e}"
            )


# =============================================================================
# РЕГИСТРАЦИЯ КОМАНД
# =============================================================================

FLOW_CONTROL_COMMANDS = [
    PulseCommand(),
    DriftCommand(),
    EchoCommand(),
    ReflectCommand(),
    AbsorbCommand(),
    DiffuseCommand(),
    MergeCommand(),
    SplitCommand(),
    LoopCommand(),
    HaltCommand(),
    SpawnCommand(),
    JumpCommand(),
    WaitCommand()
] 