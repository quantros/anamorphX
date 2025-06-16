"""
Команды управления потоком AnamorphX

Реализация 11 недостающих команд управления потоком из спецификации.
"""

import asyncio
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Union, Callable

from .commands import (
    Command, CommandResult, CommandError, FlowControlCommand,
    NeuralEntity, SynapseConnection
)
from .runtime import ExecutionContext


# =============================================================================
# КОМАНДЫ УПРАВЛЕНИЯ ПОТОКОМ (11 команд)
# =============================================================================

class DriftCommand(FlowControlCommand):
    """Перенос данных между узлами"""
    
    def __init__(self):
        super().__init__("drift", "Transfer data between neural nodes")
    
    def execute(self, context: ExecutionContext, source: str, target: str,
                data: Any = None, drift_type: str = "gradual", **kwargs) -> CommandResult:
        """Перенести данные от источника к цели"""
        try:
            if not hasattr(context, 'neural_entities'):
                context.neural_entities = {}
            
            if source not in context.neural_entities:
                return CommandResult(success=False, error=f"Source node '{source}' not found")
            
            if target not in context.neural_entities:
                return CommandResult(success=False, error=f"Target node '{target}' not found")
            
            source_entity = context.neural_entities[source]
            target_entity = context.neural_entities[target]
            
            # Определяем данные для переноса
            if data is None:
                data = source_entity.state.get('output', 0.0)
            
            # Применяем тип переноса
            if drift_type == "gradual":
                # Постепенный перенос
                transfer_rate = kwargs.get('transfer_rate', 0.1)
                transferred_amount = data * transfer_rate
            elif drift_type == "complete":
                # Полный перенос
                transferred_amount = data
                source_entity.state['output'] = 0.0
            else:
                transferred_amount = data
            
            # Переносим данные
            if 'drift_input' not in target_entity.state:
                target_entity.state['drift_input'] = 0.0
            
            target_entity.state['drift_input'] += transferred_amount
            
            # Логируем перенос
            drift_log = {
                'source': source,
                'target': target,
                'amount': transferred_amount,
                'type': drift_type,
                'timestamp': time.time()
            }
            
            if not hasattr(context, 'drift_history'):
                context.drift_history = []
            context.drift_history.append(drift_log)
            
            return CommandResult(
                success=True,
                value=f"Drifted {transferred_amount} from '{source}' to '{target}'",
                side_effects=[f"Data transfer completed: {drift_type} drift"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class EchoCommand(FlowControlCommand):
    """Отражение сигналов"""
    
    def __init__(self):
        super().__init__("echo", "Echo signals back to source")
    
    def execute(self, context: ExecutionContext, signal: Any,
                echo_delay: float = 0.1, amplification: float = 1.0, **kwargs) -> CommandResult:
        """Отразить сигнал обратно к источнику"""
        try:
            # Создаем эхо-сигнал
            echo_signal = {
                'original_signal': signal,
                'amplification': amplification,
                'delay': echo_delay,
                'timestamp': time.time(),
                'echo_id': f"echo_{int(time.time() * 1000)}"
            }
            
            # Сохраняем в контексте для обработки
            if not hasattr(context, 'echo_signals'):
                context.echo_signals = deque()
            
            context.echo_signals.append(echo_signal)
            
            # Планируем обработку эхо с задержкой
            def process_echo():
                time.sleep(echo_delay)
                processed_signal = signal * amplification
                
                # Добавляем в очередь обработанных эхо
                if not hasattr(context, 'processed_echoes'):
                    context.processed_echoes = []
                context.processed_echoes.append({
                    'signal': processed_signal,
                    'processed_at': time.time(),
                    'echo_id': echo_signal['echo_id']
                })
            
            # Запускаем обработку в отдельном потоке
            threading.Thread(target=process_echo, daemon=True).start()
            
            return CommandResult(
                success=True,
                value=f"Echo scheduled with delay {echo_delay}s and amplification {amplification}x",
                side_effects=[f"Created echo signal {echo_signal['echo_id']}"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class ReflectCommand(FlowControlCommand):
    """Отражение состояния"""
    
    def __init__(self):
        super().__init__("reflect", "Reflect node state")
    
    def execute(self, context: ExecutionContext, node_name: str,
                reflection_type: str = "state", **kwargs) -> CommandResult:
        """Отразить состояние узла"""
        try:
            if not hasattr(context, 'neural_entities') or node_name not in context.neural_entities:
                return CommandResult(success=False, error=f"Node '{node_name}' not found")
            
            entity = context.neural_entities[node_name]
            
            # Создаем отражение согласно типу
            if reflection_type == "state":
                reflection = {
                    'node_name': node_name,
                    'state': entity.state.copy(),
                    'parameters': entity.parameters.copy(),
                    'type': 'state_reflection'
                }
            elif reflection_type == "activity":
                reflection = {
                    'node_name': node_name,
                    'activation_count': entity.activation_count,
                    'last_activation': entity.last_activation,
                    'age': time.time() - entity.created_at,
                    'type': 'activity_reflection'
                }
            elif reflection_type == "full":
                reflection = {
                    'node_name': node_name,
                    'full_info': entity.get_info(),
                    'type': 'full_reflection'
                }
            else:
                reflection = {'error': f"Unknown reflection type: {reflection_type}"}
            
            reflection['reflected_at'] = time.time()
            
            # Сохраняем отражение
            if not hasattr(context, 'reflections'):
                context.reflections = []
            context.reflections.append(reflection)
            
            return CommandResult(
                success=True,
                value=reflection,
                side_effects=[f"Created {reflection_type} reflection of {node_name}"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class AbsorbCommand(FlowControlCommand):
    """Поглощение сигналов"""
    
    def __init__(self):
        super().__init__("absorb", "Absorb incoming signals")
    
    def execute(self, context: ExecutionContext, node_name: str,
                absorption_rate: float = 1.0, capacity: int = 100, **kwargs) -> CommandResult:
        """Поглотить входящие сигналы"""
        try:
            if not hasattr(context, 'neural_entities') or node_name not in context.neural_entities:
                return CommandResult(success=False, error=f"Node '{node_name}' not found")
            
            entity = context.neural_entities[node_name]
            
            # Инициализируем систему поглощения
            if 'absorption' not in entity.state:
                entity.state['absorption'] = {
                    'absorbed_signals': [],
                    'total_absorbed': 0.0,
                    'capacity': capacity,
                    'rate': absorption_rate
                }
            
            absorption_system = entity.state['absorption']
            
            # Поглощаем доступные сигналы
            absorbed_count = 0
            if hasattr(context, 'pending_signals'):
                available_signals = [s for s in context.pending_signals if s.get('target') == node_name]
                
                for signal in available_signals[:capacity]:
                    if len(absorption_system['absorbed_signals']) < capacity:
                        absorbed_signal = {
                            'signal_data': signal,
                            'absorbed_at': time.time(),
                            'absorption_rate': absorption_rate
                        }
                        absorption_system['absorbed_signals'].append(absorbed_signal)
                        absorption_system['total_absorbed'] += absorption_rate
                        absorbed_count += 1
                        
                        # Удаляем поглощенный сигнал из очереди
                        context.pending_signals.remove(signal)
            
            return CommandResult(
                success=True,
                value=f"Absorbed {absorbed_count} signals into '{node_name}'",
                side_effects=[f"Node {node_name} absorbed {absorbed_count} signals"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class DiffuseCommand(FlowControlCommand):
    """Распространение сигналов"""
    
    def __init__(self):
        super().__init__("diffuse", "Diffuse signals across channels")
    
    def execute(self, context: ExecutionContext, signal: Any,
                diffusion_pattern: str = "radial", intensity: float = 1.0, **kwargs) -> CommandResult:
        """Распространить сигнал по каналам"""
        try:
            if not hasattr(context, 'neural_entities'):
                context.neural_entities = {}
            
            # Определяем паттерн распространения
            if diffusion_pattern == "radial":
                targets = self._get_radial_targets(context, kwargs.get('center_node'))
            elif diffusion_pattern == "linear":
                targets = self._get_linear_targets(context, kwargs.get('direction', 'forward'))
            elif diffusion_pattern == "random":
                targets = self._get_random_targets(context, kwargs.get('count', 5))
            else:
                targets = list(context.neural_entities.keys())
            
            # Распространяем сигнал
            diffused_signals = []
            for target in targets:
                if target in context.neural_entities:
                    entity = context.neural_entities[target]
                    
                    # Вычисляем интенсивность для этого узла
                    distance_factor = kwargs.get('distance_decay', 0.9)
                    target_intensity = intensity * (distance_factor ** len(diffused_signals))
                    
                    # Создаем диффузный сигнал
                    diffused_signal = {
                        'original_signal': signal,
                        'target': target,
                        'intensity': target_intensity,
                        'pattern': diffusion_pattern,
                        'timestamp': time.time()
                    }
                    
                    diffused_signals.append(diffused_signal)
                    
                    # Применяем сигнал к узлу
                    if 'diffused_input' not in entity.state:
                        entity.state['diffused_input'] = 0.0
                    entity.state['diffused_input'] += target_intensity
            
            # Сохраняем историю диффузии
            if not hasattr(context, 'diffusion_history'):
                context.diffusion_history = []
            
            context.diffusion_history.append({
                'signal': signal,
                'pattern': diffusion_pattern,
                'targets': len(diffused_signals),
                'total_intensity': sum(s['intensity'] for s in diffused_signals),
                'timestamp': time.time()
            })
            
            return CommandResult(
                success=True,
                value=f"Diffused signal to {len(diffused_signals)} targets using {diffusion_pattern} pattern",
                side_effects=[f"Signal diffused across {len(diffused_signals)} nodes"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    def _get_radial_targets(self, context: ExecutionContext, center_node: str = None) -> List[str]:
        """Получить цели для радиального распространения"""
        if not center_node or center_node not in context.neural_entities:
            # Выбираем случайный центр
            nodes = list(context.neural_entities.keys())
            center_node = nodes[0] if nodes else None
        
        if not center_node:
            return []
        
        # Возвращаем все узлы кроме центрального
        return [name for name in context.neural_entities.keys() if name != center_node]
    
    def _get_linear_targets(self, context: ExecutionContext, direction: str) -> List[str]:
        """Получить цели для линейного распространения"""
        nodes = list(context.neural_entities.keys())
        if direction == "reverse":
            nodes.reverse()
        return nodes
    
    def _get_random_targets(self, context: ExecutionContext, count: int) -> List[str]:
        """Получить случайные цели"""
        import random
        nodes = list(context.neural_entities.keys())
        return random.sample(nodes, min(count, len(nodes)))


class MergeCommand(FlowControlCommand):
    """Слияние потоков"""
    
    def __init__(self):
        super().__init__("merge", "Merge multiple data streams")
    
    def execute(self, context: ExecutionContext, streams: List[str],
                merge_type: str = "sum", output_name: str = None, **kwargs) -> CommandResult:
        """Слить несколько потоков данных"""
        try:
            if not streams:
                return CommandResult(success=False, error="No streams specified for merging")
            
            # Собираем данные из потоков
            stream_data = []
            for stream_name in streams:
                if hasattr(context, 'data_streams') and stream_name in context.data_streams:
                    stream_data.append(context.data_streams[stream_name])
                elif hasattr(context, 'neural_entities') and stream_name in context.neural_entities:
                    entity = context.neural_entities[stream_name]
                    stream_data.append(entity.state.get('output', 0.0))
                else:
                    stream_data.append(0.0)  # Значение по умолчанию
            
            # Применяем тип слияния
            if merge_type == "sum":
                merged_result = sum(stream_data)
            elif merge_type == "average":
                merged_result = sum(stream_data) / len(stream_data) if stream_data else 0.0
            elif merge_type == "max":
                merged_result = max(stream_data) if stream_data else 0.0
            elif merge_type == "min":
                merged_result = min(stream_data) if stream_data else 0.0
            elif merge_type == "product":
                merged_result = 1.0
                for value in stream_data:
                    merged_result *= value
            else:
                merged_result = stream_data  # Возвращаем как список
            
            # Сохраняем результат
            output_name = output_name or f"merged_{int(time.time())}"
            
            if not hasattr(context, 'merged_streams'):
                context.merged_streams = {}
            
            context.merged_streams[output_name] = {
                'result': merged_result,
                'source_streams': streams,
                'merge_type': merge_type,
                'merged_at': time.time()
            }
            
            return CommandResult(
                success=True,
                value=merged_result,
                side_effects=[f"Merged {len(streams)} streams into '{output_name}' using {merge_type}"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class SplitCommand(FlowControlCommand):
    """Разделение потоков"""
    
    def __init__(self):
        super().__init__("split", "Split data stream into multiple streams")
    
    def execute(self, context: ExecutionContext, stream_name: str,
                split_count: int = 2, split_type: str = "equal", **kwargs) -> CommandResult:
        """Разделить поток данных"""
        try:
            # Получаем исходные данные
            source_data = None
            if hasattr(context, 'data_streams') and stream_name in context.data_streams:
                source_data = context.data_streams[stream_name]
            elif hasattr(context, 'neural_entities') and stream_name in context.neural_entities:
                entity = context.neural_entities[stream_name]
                source_data = entity.state.get('output', 0.0)
            else:
                return CommandResult(success=False, error=f"Stream '{stream_name}' not found")
            
            # Применяем тип разделения
            split_results = []
            
            if split_type == "equal":
                # Равномерное разделение
                if isinstance(source_data, (int, float)):
                    split_value = source_data / split_count
                    split_results = [split_value] * split_count
                elif isinstance(source_data, list):
                    chunk_size = len(source_data) // split_count
                    for i in range(split_count):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size if i < split_count - 1 else len(source_data)
                        split_results.append(source_data[start_idx:end_idx])
                else:
                    split_results = [source_data] * split_count
            
            elif split_type == "weighted":
                # Взвешенное разделение
                weights = kwargs.get('weights', [1.0] * split_count)
                total_weight = sum(weights)
                
                if isinstance(source_data, (int, float)):
                    for weight in weights:
                        split_results.append(source_data * (weight / total_weight))
                else:
                    split_results = [source_data] * split_count
            
            elif split_type == "random":
                # Случайное разделение
                import random
                if isinstance(source_data, (int, float)):
                    random_splits = [random.random() for _ in range(split_count)]
                    total_random = sum(random_splits)
                    split_results = [source_data * (r / total_random) for r in random_splits]
                else:
                    split_results = [source_data] * split_count
            
            # Сохраняем результаты разделения
            if not hasattr(context, 'split_streams'):
                context.split_streams = {}
            
            split_names = []
            for i, result in enumerate(split_results):
                split_name = f"{stream_name}_split_{i}"
                context.split_streams[split_name] = {
                    'data': result,
                    'source_stream': stream_name,
                    'split_index': i,
                    'split_type': split_type,
                    'split_at': time.time()
                }
                split_names.append(split_name)
            
            return CommandResult(
                success=True,
                value=split_names,
                side_effects=[f"Split '{stream_name}' into {split_count} streams using {split_type} method"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class LoopCommand(FlowControlCommand):
    """Циклы"""
    
    def __init__(self):
        super().__init__("loop", "Execute loop operations")
    
    def execute(self, context: ExecutionContext, loop_type: str = "count",
                iterations: int = 10, condition: Callable = None, **kwargs) -> CommandResult:
        """Выполнить цикл"""
        try:
            loop_id = f"loop_{int(time.time() * 1000)}"
            
            if not hasattr(context, 'active_loops'):
                context.active_loops = {}
            
            # Создаем состояние цикла
            loop_state = {
                'id': loop_id,
                'type': loop_type,
                'iterations': iterations,
                'current_iteration': 0,
                'condition': condition,
                'started_at': time.time(),
                'status': 'running',
                'results': []
            }
            
            context.active_loops[loop_id] = loop_state
            
            # Выполняем цикл согласно типу
            if loop_type == "count":
                for i in range(iterations):
                    loop_state['current_iteration'] = i
                    result = self._execute_loop_iteration(context, loop_state, **kwargs)
                    loop_state['results'].append(result)
                    
                    # Проверяем условие прерывания
                    if kwargs.get('break_condition') and kwargs['break_condition'](result):
                        break
            
            elif loop_type == "while":
                while condition and condition(context, loop_state):
                    if loop_state['current_iteration'] >= iterations:  # Защита от бесконечного цикла
                        break
                    
                    result = self._execute_loop_iteration(context, loop_state, **kwargs)
                    loop_state['results'].append(result)
                    loop_state['current_iteration'] += 1
            
            elif loop_type == "foreach":
                items = kwargs.get('items', [])
                for i, item in enumerate(items):
                    loop_state['current_iteration'] = i
                    loop_state['current_item'] = item
                    result = self._execute_loop_iteration(context, loop_state, **kwargs)
                    loop_state['results'].append(result)
            
            loop_state['status'] = 'completed'
            loop_state['completed_at'] = time.time()
            
            return CommandResult(
                success=True,
                value=f"Loop completed: {loop_state['current_iteration']} iterations",
                side_effects=[f"Executed {loop_type} loop with {len(loop_state['results'])} results"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    def _execute_loop_iteration(self, context: ExecutionContext, loop_state: Dict, **kwargs) -> Any:
        """Выполнить одну итерацию цикла"""
        # Базовая логика итерации
        iteration_result = {
            'iteration': loop_state['current_iteration'],
            'timestamp': time.time()
        }
        
        # Выполняем пользовательскую логику итерации
        if 'iteration_function' in kwargs:
            try:
                user_result = kwargs['iteration_function'](context, loop_state)
                iteration_result['user_result'] = user_result
            except Exception as e:
                iteration_result['error'] = str(e)
        
        return iteration_result


class HaltCommand(FlowControlCommand):
    """Остановка выполнения"""
    
    def __init__(self):
        super().__init__("halt", "Halt execution")
    
    def execute(self, context: ExecutionContext, target: str = "current",
                halt_type: str = "immediate", **kwargs) -> CommandResult:
        """Остановить выполнение"""
        try:
            halt_info = {
                'target': target,
                'type': halt_type,
                'timestamp': time.time(),
                'reason': kwargs.get('reason', 'Manual halt')
            }
            
            if not hasattr(context, 'halt_requests'):
                context.halt_requests = []
            context.halt_requests.append(halt_info)
            
            # Применяем остановку согласно типу
            if halt_type == "immediate":
                context.execution_state = "halted"
                if hasattr(context, 'active_loops'):
                    for loop_id, loop_state in context.active_loops.items():
                        loop_state['status'] = 'halted'
            
            elif halt_type == "graceful":
                context.execution_state = "halting"
                # Позволяем текущим операциям завершиться
                
            elif halt_type == "conditional":
                condition = kwargs.get('condition')
                if condition and condition(context):
                    context.execution_state = "halted"
            
            return CommandResult(
                success=True,
                value=f"Halt request processed: {halt_type}",
                side_effects=[f"Execution halt requested for {target}"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class YieldCommand(FlowControlCommand):
    """Передача управления"""
    
    def __init__(self):
        super().__init__("yield", "Yield control to other processes")
    
    def execute(self, context: ExecutionContext, yield_type: str = "time",
                yield_duration: float = 0.1, **kwargs) -> CommandResult:
        """Передать управление"""
        try:
            yield_info = {
                'type': yield_type,
                'duration': yield_duration,
                'timestamp': time.time()
            }
            
            if yield_type == "time":
                # Временная передача управления
                time.sleep(yield_duration)
            
            elif yield_type == "priority":
                # Передача управления процессам с высоким приоритетом
                threading.Event().wait(yield_duration)
            
            elif yield_type == "resource":
                # Передача управления при нехватке ресурсов
                resource_threshold = kwargs.get('resource_threshold', 0.8)
                # Здесь можно добавить проверку ресурсов
                time.sleep(yield_duration)
            
            if not hasattr(context, 'yield_history'):
                context.yield_history = []
            context.yield_history.append(yield_info)
            
            return CommandResult(
                success=True,
                value=f"Yielded control for {yield_duration}s ({yield_type})",
                side_effects=[f"Control yielded using {yield_type} method"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class SpawnCommand(FlowControlCommand):
    """Создание процессов"""
    
    def __init__(self):
        super().__init__("spawn", "Spawn new processes")
    
    def execute(self, context: ExecutionContext, process_type: str = "thread",
                task: Callable = None, task_args: tuple = (), **kwargs) -> CommandResult:
        """Создать новый процесс"""
        try:
            process_id = f"process_{int(time.time() * 1000)}"
            
            if not hasattr(context, 'spawned_processes'):
                context.spawned_processes = {}
            
            process_info = {
                'id': process_id,
                'type': process_type,
                'task': task,
                'args': task_args,
                'spawned_at': time.time(),
                'status': 'starting'
            }
            
            context.spawned_processes[process_id] = process_info
            
            # Создаем процесс согласно типу
            if process_type == "thread":
                if task:
                    thread = threading.Thread(
                        target=self._execute_spawned_task,
                        args=(context, process_info, task, task_args),
                        daemon=True
                    )
                    thread.start()
                    process_info['thread'] = thread
                    process_info['status'] = 'running'
            
            elif process_type == "async":
                if task:
                    # Создаем асинхронную задачу
                    async def async_wrapper():
                        try:
                            result = await task(*task_args)
                            process_info['result'] = result
                            process_info['status'] = 'completed'
                        except Exception as e:
                            process_info['error'] = str(e)
                            process_info['status'] = 'failed'
                    
                    # Запускаем в event loop (если доступен)
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(async_wrapper())
                    except RuntimeError:
                        # Если event loop недоступен, используем thread
                        thread = threading.Thread(
                            target=lambda: asyncio.run(async_wrapper()),
                            daemon=True
                        )
                        thread.start()
                        process_info['thread'] = thread
                    
                    process_info['status'] = 'running'
            
            return CommandResult(
                success=True,
                value=process_id,
                side_effects=[f"Spawned {process_type} process {process_id}"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    def _execute_spawned_task(self, context: ExecutionContext, process_info: Dict,
                             task: Callable, args: tuple):
        """Выполнить задачу в отдельном процессе"""
        try:
            result = task(*args)
            process_info['result'] = result
            process_info['status'] = 'completed'
            process_info['completed_at'] = time.time()
        except Exception as e:
            process_info['error'] = str(e)
            process_info['status'] = 'failed'
            process_info['failed_at'] = time.time()


class JumpCommand(FlowControlCommand):
    """Переходы в коде"""
    
    def __init__(self):
        super().__init__("jump", "Jump to different execution points")
    
    def execute(self, context: ExecutionContext, target: str,
                jump_type: str = "absolute", **kwargs) -> CommandResult:
        """Выполнить переход"""
        try:
            jump_info = {
                'target': target,
                'type': jump_type,
                'timestamp': time.time(),
                'source': kwargs.get('source', 'unknown')
            }
            
            if not hasattr(context, 'jump_history'):
                context.jump_history = []
            context.jump_history.append(jump_info)
            
            # Применяем переход согласно типу
            if jump_type == "absolute":
                # Абсолютный переход к метке
                context.execution_pointer = target
            
            elif jump_type == "relative":
                # Относительный переход
                offset = kwargs.get('offset', 0)
                if hasattr(context, 'execution_pointer'):
                    context.execution_pointer = context.execution_pointer + offset
            
            elif jump_type == "conditional":
                # Условный переход
                condition = kwargs.get('condition')
                if condition and condition(context):
                    context.execution_pointer = target
            
            elif jump_type == "function":
                # Переход к функции
                if not hasattr(context, 'call_stack'):
                    context.call_stack = []
                context.call_stack.append(context.execution_pointer)
                context.execution_pointer = target
            
            return CommandResult(
                success=True,
                value=f"Jumped to '{target}' using {jump_type} jump",
                side_effects=[f"Execution pointer updated to {target}"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e))


class WaitCommand(FlowControlCommand):
    """Ожидание"""
    
    def __init__(self):
        super().__init__("wait", "Wait for conditions or time")
    
    def execute(self, context: ExecutionContext, wait_type: str = "time",
                duration: float = 1.0, condition: Callable = None, **kwargs) -> CommandResult:
        """Выполнить ожидание"""
        try:
            wait_start = time.time()
            
            wait_info = {
                'type': wait_type,
                'duration': duration,
                'started_at': wait_start
            }
            
            if wait_type == "time":
                # Ожидание по времени
                time.sleep(duration)
                wait_info['actual_duration'] = time.time() - wait_start
            
            elif wait_type == "condition":
                # Ожидание условия
                timeout = kwargs.get('timeout', 30.0)
                check_interval = kwargs.get('check_interval', 0.1)
                
                elapsed = 0.0
                while elapsed < timeout:
                    if condition and condition(context):
                        break
                    time.sleep(check_interval)
                    elapsed = time.time() - wait_start
                
                wait_info['actual_duration'] = elapsed
                wait_info['condition_met'] = condition(context) if condition else False
            
            elif wait_type == "signal":
                # Ожидание сигнала
                signal_name = kwargs.get('signal_name', 'default')
                timeout = kwargs.get('timeout', 30.0)
                
                if not hasattr(context, 'signals'):
                    context.signals = {}
                
                # Ждем появления сигнала
                elapsed = 0.0
                while elapsed < timeout:
                    if signal_name in context.signals:
                        break
                    time.sleep(0.1)
                    elapsed = time.time() - wait_start
                
                wait_info['actual_duration'] = elapsed
                wait_info['signal_received'] = signal_name in context.signals
            
            elif wait_type == "resource":
                # Ожидание доступности ресурса
                resource_name = kwargs.get('resource_name', 'default')
                timeout = kwargs.get('timeout', 30.0)
                
                if not hasattr(context, 'resources'):
                    context.resources = {}
                
                elapsed = 0.0
                while elapsed < timeout:
                    resource = context.resources.get(resource_name, {})
                    if resource.get('available', False):
                        break
                    time.sleep(0.1)
                    elapsed = time.time() - wait_start
                
                wait_info['actual_duration'] = elapsed
                wait_info['resource_available'] = context.resources.get(resource_name, {}).get('available', False)
            
            if not hasattr(context, 'wait_history'):
                context.wait_history = []
            context.wait_history.append(wait_info)
            
            return CommandResult(
                success=True,
                value=f"Wait completed: {wait_info.get('actual_duration', duration):.3f}s",
                side_effects=[f"Waited using {wait_type} method"]
            )
            
        except Exception as e:
            return CommandResult(success=False, error=str(e)) 