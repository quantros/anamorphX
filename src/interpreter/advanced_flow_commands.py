"""
Расширенные команды управления потоком для AnamorphX

Реализация оставшихся команд управления потоком данных и выполнением.
"""

import time
import threading
import queue
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from .commands import FlowControlCommand, CommandResult, CommandError, ExecutionContext


class FlowState(Enum):
    """Состояния потока выполнения"""
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    HALTED = "halted"
    YIELDED = "yielded"


@dataclass
class ProcessInfo:
    """Информация о процессе"""
    id: str
    name: str
    state: FlowState
    created_at: float
    parent_id: Optional[str] = None
    data: Dict[str, Any] = None


class ReflectCommand(FlowControlCommand):
    """Команда отражения состояния узлов"""
    
    def __init__(self):
        super().__init__(
            name="reflect",
            description="Отражает текущее состояние узлов и соединений",
            parameters={
                "target": "Целевой узел или группа узлов",
                "depth": "Глубина отражения (1-5)",
                "format": "Формат вывода (json, table, graph)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get("target", "all")
            depth = int(kwargs.get("depth", 1))
            format_type = kwargs.get("format", "json")
            
            reflection_data = self._reflect_state(context, target, depth)
            formatted_output = self._format_reflection(reflection_data, format_type)
            
            return CommandResult(
                success=True,
                message=f"Состояние отражено для {target}",
                data={
                    "reflection": reflection_data,
                    "formatted": formatted_output,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка отражения: {str(e)}",
                error=CommandError("REFLECTION_ERROR", str(e))
            )
    
    def _reflect_state(self, context: ExecutionContext, target: str, depth: int) -> Dict[str, Any]:
        """Отразить состояние системы"""
        reflection = {
            "target": target,
            "depth": depth,
            "timestamp": time.time(),
            "nodes": {},
            "connections": {},
            "memory": {},
            "performance": {}
        }
        
        # Отражение узлов
        if hasattr(context, 'neural_entities'):
            for entity_id, entity in context.neural_entities.items():
                if target == "all" or target in entity_id:
                    reflection["nodes"][entity_id] = {
                        "type": entity.entity_type,
                        "state": entity.state,
                        "parameters": entity.parameters,
                        "connections": len(entity.connections),
                        "last_activity": getattr(entity, 'last_activity', None)
                    }
        
        # Отражение соединений
        if hasattr(context, 'synapses'):
            for synapse_id, synapse in context.synapses.items():
                reflection["connections"][synapse_id] = {
                    "source": synapse.source_id,
                    "target": synapse.target_id,
                    "weight": synapse.weight,
                    "active": synapse.active,
                    "signal_count": getattr(synapse, 'signal_count', 0)
                }
        
        # Отражение памяти
        reflection["memory"] = {
            "total_entities": len(getattr(context, 'neural_entities', {})),
            "active_entities": len([e for e in getattr(context, 'neural_entities', {}).values() 
                                  if e.state == "active"]),
            "memory_usage": getattr(context, 'memory_usage', 0)
        }
        
        return reflection
    
    def _format_reflection(self, data: Dict[str, Any], format_type: str) -> str:
        """Форматировать данные отражения"""
        if format_type == "json":
            import json
            return json.dumps(data, indent=2, default=str)
        elif format_type == "table":
            return self._format_as_table(data)
        elif format_type == "graph":
            return self._format_as_graph(data)
        else:
            return str(data)
    
    def _format_as_table(self, data: Dict[str, Any]) -> str:
        """Форматировать как таблицу"""
        lines = ["=== REFLECTION REPORT ==="]
        lines.append(f"Target: {data['target']}")
        lines.append(f"Timestamp: {data['timestamp']}")
        lines.append("")
        
        # Таблица узлов
        if data["nodes"]:
            lines.append("NODES:")
            lines.append("-" * 60)
            lines.append(f"{'ID':<20} {'Type':<15} {'State':<10} {'Connections':<12}")
            lines.append("-" * 60)
            for node_id, node_info in data["nodes"].items():
                lines.append(f"{node_id:<20} {node_info['type']:<15} {node_info['state']:<10} {node_info['connections']:<12}")
        
        return "\n".join(lines)
    
    def _format_as_graph(self, data: Dict[str, Any]) -> str:
        """Форматировать как граф"""
        lines = ["digraph NetworkReflection {"]
        lines.append("  rankdir=LR;")
        
        # Узлы
        for node_id, node_info in data["nodes"].items():
            color = "green" if node_info["state"] == "active" else "gray"
            lines.append(f'  "{node_id}" [color={color}, label="{node_id}\\n{node_info["type"]}"];')
        
        # Соединения
        for conn_id, conn_info in data["connections"].items():
            style = "solid" if conn_info["active"] else "dashed"
            lines.append(f'  "{conn_info["source"]}" -> "{conn_info["target"]}" [style={style}];')
        
        lines.append("}")
        return "\n".join(lines)


class AbsorbCommand(FlowControlCommand):
    """Команда поглощения сигналов"""
    
    def __init__(self):
        super().__init__(
            name="absorb",
            description="Поглощает входящие сигналы в узле",
            parameters={
                "node": "Узел-поглотитель",
                "capacity": "Максимальная емкость поглощения",
                "filter": "Фильтр типов сигналов",
                "action": "Действие после поглощения (store, process, discard)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_id = kwargs.get("node")
            capacity = int(kwargs.get("capacity", 100))
            signal_filter = kwargs.get("filter", "all")
            action = kwargs.get("action", "store")
            
            if not node_id:
                return CommandResult(
                    success=False,
                    message="Не указан узел для поглощения",
                    error=CommandError("MISSING_NODE", "Node parameter required")
                )
            
            # Создаем поглотитель
            absorber = self._create_absorber(node_id, capacity, signal_filter, action)
            
            # Регистрируем в контексте
            if not hasattr(context, 'absorbers'):
                context.absorbers = {}
            context.absorbers[node_id] = absorber
            
            return CommandResult(
                success=True,
                message=f"Поглотитель создан для узла {node_id}",
                data={
                    "absorber_id": node_id,
                    "capacity": capacity,
                    "filter": signal_filter,
                    "action": action
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка создания поглотителя: {str(e)}",
                error=CommandError("ABSORBER_ERROR", str(e))
            )
    
    def _create_absorber(self, node_id: str, capacity: int, signal_filter: str, action: str) -> Dict[str, Any]:
        """Создать поглотитель сигналов"""
        return {
            "node_id": node_id,
            "capacity": capacity,
            "current_load": 0,
            "filter": signal_filter,
            "action": action,
            "absorbed_signals": [],
            "created_at": time.time(),
            "active": True
        }


class DiffuseCommand(FlowControlCommand):
    """Команда распространения сигналов"""
    
    def __init__(self):
        super().__init__(
            name="diffuse",
            description="Распространяет сигналы по сети",
            parameters={
                "source": "Источник сигнала",
                "pattern": "Паттерн распространения (radial, directional, random)",
                "strength": "Сила сигнала (0.0-1.0)",
                "decay": "Коэффициент затухания",
                "max_hops": "Максимальное количество переходов"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get("source")
            pattern = kwargs.get("pattern", "radial")
            strength = float(kwargs.get("strength", 1.0))
            decay = float(kwargs.get("decay", 0.1))
            max_hops = int(kwargs.get("max_hops", 5))
            
            if not source:
                return CommandResult(
                    success=False,
                    message="Не указан источник сигнала",
                    error=CommandError("MISSING_SOURCE", "Source parameter required")
                )
            
            # Выполняем диффузию
            diffusion_result = self._perform_diffusion(
                context, source, pattern, strength, decay, max_hops
            )
            
            return CommandResult(
                success=True,
                message=f"Сигнал распространен от {source}",
                data=diffusion_result
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка диффузии: {str(e)}",
                error=CommandError("DIFFUSION_ERROR", str(e))
            )
    
    def _perform_diffusion(self, context: ExecutionContext, source: str, 
                          pattern: str, strength: float, decay: float, max_hops: int) -> Dict[str, Any]:
        """Выполнить диффузию сигнала"""
        visited = set()
        signal_path = []
        current_strength = strength
        
        def diffuse_recursive(node_id: str, hop_count: int, current_str: float):
            if hop_count >= max_hops or current_str <= 0.01 or node_id in visited:
                return
            
            visited.add(node_id)
            signal_path.append({
                "node": node_id,
                "hop": hop_count,
                "strength": current_str,
                "timestamp": time.time()
            })
            
            # Получаем соседние узлы
            neighbors = self._get_neighbors(context, node_id, pattern)
            
            # Распространяем на соседей
            for neighbor in neighbors:
                new_strength = current_str * (1 - decay)
                diffuse_recursive(neighbor, hop_count + 1, new_strength)
        
        # Начинаем диффузию
        diffuse_recursive(source, 0, strength)
        
        return {
            "source": source,
            "pattern": pattern,
            "initial_strength": strength,
            "decay": decay,
            "max_hops": max_hops,
            "nodes_reached": len(visited),
            "signal_path": signal_path,
            "total_energy": sum(step["strength"] for step in signal_path)
        }
    
    def _get_neighbors(self, context: ExecutionContext, node_id: str, pattern: str) -> List[str]:
        """Получить соседние узлы для диффузии"""
        neighbors = []
        
        if hasattr(context, 'synapses'):
            for synapse in context.synapses.values():
                if synapse.source_id == node_id and synapse.active:
                    neighbors.append(synapse.target_id)
                elif pattern == "bidirectional" and synapse.target_id == node_id and synapse.active:
                    neighbors.append(synapse.source_id)
        
        return neighbors


class MergeCommand(FlowControlCommand):
    """Команда слияния потоков данных"""
    
    def __init__(self):
        super().__init__(
            name="merge",
            description="Сливает несколько потоков данных в один",
            parameters={
                "sources": "Список источников для слияния",
                "target": "Целевой узел",
                "strategy": "Стратегия слияния (concat, average, max, min, sum)",
                "weights": "Веса для источников (опционально)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            sources = kwargs.get("sources", "").split(",")
            target = kwargs.get("target")
            strategy = kwargs.get("strategy", "concat")
            weights_str = kwargs.get("weights", "")
            
            if not sources or not target:
                return CommandResult(
                    success=False,
                    message="Не указаны источники или цель для слияния",
                    error=CommandError("MISSING_PARAMS", "Sources and target required")
                )
            
            # Парсим веса
            weights = []
            if weights_str:
                try:
                    weights = [float(w.strip()) for w in weights_str.split(",")]
                except ValueError:
                    weights = [1.0] * len(sources)
            else:
                weights = [1.0] * len(sources)
            
            # Выполняем слияние
            merge_result = self._perform_merge(context, sources, target, strategy, weights)
            
            return CommandResult(
                success=True,
                message=f"Потоки слиты в {target}",
                data=merge_result
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка слияния: {str(e)}",
                error=CommandError("MERGE_ERROR", str(e))
            )
    
    def _perform_merge(self, context: ExecutionContext, sources: List[str], 
                      target: str, strategy: str, weights: List[float]) -> Dict[str, Any]:
        """Выполнить слияние потоков"""
        merged_data = []
        source_info = []
        
        # Собираем данные из источников
        for i, source in enumerate(sources):
            source = source.strip()
            weight = weights[i] if i < len(weights) else 1.0
            
            # Получаем данные из источника
            source_data = self._get_source_data(context, source)
            
            source_info.append({
                "source": source,
                "weight": weight,
                "data_size": len(source_data) if isinstance(source_data, (list, tuple)) else 1,
                "data_type": type(source_data).__name__
            })
            
            # Применяем вес
            if isinstance(source_data, (int, float)):
                weighted_data = source_data * weight
            elif isinstance(source_data, list):
                weighted_data = [x * weight for x in source_data if isinstance(x, (int, float))]
            else:
                weighted_data = source_data
            
            merged_data.append(weighted_data)
        
        # Применяем стратегию слияния
        final_result = self._apply_merge_strategy(merged_data, strategy)
        
        # Сохраняем результат в целевой узел
        self._store_merged_result(context, target, final_result)
        
        return {
            "sources": sources,
            "target": target,
            "strategy": strategy,
            "weights": weights,
            "source_info": source_info,
            "result_type": type(final_result).__name__,
            "result_size": len(final_result) if isinstance(final_result, (list, tuple)) else 1,
            "timestamp": time.time()
        }
    
    def _get_source_data(self, context: ExecutionContext, source: str) -> Any:
        """Получить данные из источника"""
        if hasattr(context, 'neural_entities') and source in context.neural_entities:
            entity = context.neural_entities[source]
            return getattr(entity, 'output_data', [])
        return []
    
    def _apply_merge_strategy(self, data_list: List[Any], strategy: str) -> Any:
        """Применить стратегию слияния"""
        if strategy == "concat":
            result = []
            for data in data_list:
                if isinstance(data, list):
                    result.extend(data)
                else:
                    result.append(data)
            return result
        
        elif strategy == "average":
            numeric_data = [d for d in data_list if isinstance(d, (int, float))]
            return sum(numeric_data) / len(numeric_data) if numeric_data else 0
        
        elif strategy == "max":
            numeric_data = [d for d in data_list if isinstance(d, (int, float))]
            return max(numeric_data) if numeric_data else 0
        
        elif strategy == "min":
            numeric_data = [d for d in data_list if isinstance(d, (int, float))]
            return min(numeric_data) if numeric_data else 0
        
        elif strategy == "sum":
            numeric_data = [d for d in data_list if isinstance(d, (int, float))]
            return sum(numeric_data)
        
        else:
            return data_list
    
    def _store_merged_result(self, context: ExecutionContext, target: str, result: Any):
        """Сохранить результат слияния в целевой узел"""
        if hasattr(context, 'neural_entities') and target in context.neural_entities:
            entity = context.neural_entities[target]
            entity.input_data = result
            entity.last_update = time.time()


class SplitCommand(FlowControlCommand):
    """Команда разделения потоков"""
    
    def __init__(self):
        super().__init__(
            name="split",
            description="Разделяет поток данных на несколько потоков",
            parameters={
                "source": "Источник данных",
                "targets": "Список целевых узлов",
                "strategy": "Стратегия разделения (equal, weighted, conditional)",
                "conditions": "Условия для разделения (опционально)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get("source")
            targets = kwargs.get("targets", "").split(",")
            strategy = kwargs.get("strategy", "equal")
            conditions = kwargs.get("conditions", "")
            
            if not source or not targets:
                return CommandResult(
                    success=False,
                    message="Не указан источник или цели для разделения",
                    error=CommandError("MISSING_PARAMS", "Source and targets required")
                )
            
            # Выполняем разделение
            split_result = self._perform_split(context, source, targets, strategy, conditions)
            
            return CommandResult(
                success=True,
                message=f"Поток разделен на {len(targets)} частей",
                data=split_result
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка разделения: {str(e)}",
                error=CommandError("SPLIT_ERROR", str(e))
            )
    
    def _perform_split(self, context: ExecutionContext, source: str, 
                      targets: List[str], strategy: str, conditions: str) -> Dict[str, Any]:
        """Выполнить разделение потока"""
        # Получаем исходные данные
        source_data = self._get_source_data(context, source)
        
        # Разделяем данные
        if strategy == "equal":
            split_data = self._split_equal(source_data, len(targets))
        elif strategy == "weighted":
            split_data = self._split_weighted(source_data, targets)
        elif strategy == "conditional":
            split_data = self._split_conditional(source_data, targets, conditions)
        else:
            split_data = [source_data] * len(targets)
        
        # Распределяем по целевым узлам
        distribution_info = []
        for i, target in enumerate(targets):
            target = target.strip()
            data_part = split_data[i] if i < len(split_data) else []
            
            self._store_split_result(context, target, data_part)
            
            distribution_info.append({
                "target": target,
                "data_size": len(data_part) if isinstance(data_part, (list, tuple)) else 1,
                "data_type": type(data_part).__name__
            })
        
        return {
            "source": source,
            "targets": targets,
            "strategy": strategy,
            "source_size": len(source_data) if isinstance(source_data, (list, tuple)) else 1,
            "distribution": distribution_info,
            "timestamp": time.time()
        }
    
    def _split_equal(self, data: Any, num_parts: int) -> List[Any]:
        """Равномерное разделение данных"""
        if isinstance(data, list):
            chunk_size = len(data) // num_parts
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        else:
            return [data] * num_parts
    
    def _split_weighted(self, data: Any, targets: List[str]) -> List[Any]:
        """Взвешенное разделение данных"""
        # Простая реализация - равномерное разделение
        return self._split_equal(data, len(targets))
    
    def _split_conditional(self, data: Any, targets: List[str], conditions: str) -> List[Any]:
        """Условное разделение данных"""
        # Простая реализация - равномерное разделение
        return self._split_equal(data, len(targets))
    
    def _store_split_result(self, context: ExecutionContext, target: str, data: Any):
        """Сохранить результат разделения в целевой узел"""
        if hasattr(context, 'neural_entities') and target in context.neural_entities:
            entity = context.neural_entities[target]
            entity.input_data = data
            entity.last_update = time.time()


class LoopCommand(FlowControlCommand):
    """Команда циклических операций"""
    
    def __init__(self):
        super().__init__(
            name="loop",
            description="Выполняет циклические операции",
            parameters={
                "type": "Тип цикла (for, while, until)",
                "condition": "Условие цикла",
                "iterations": "Количество итераций (для for)",
                "commands": "Команды для выполнения в цикле",
                "break_condition": "Условие прерывания"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            loop_type = kwargs.get("type", "for")
            condition = kwargs.get("condition", "true")
            iterations = int(kwargs.get("iterations", 1))
            commands = kwargs.get("commands", "").split(";")
            break_condition = kwargs.get("break_condition", "")
            
            # Выполняем цикл
            loop_result = self._execute_loop(
                context, loop_type, condition, iterations, commands, break_condition
            )
            
            return CommandResult(
                success=True,
                message=f"Цикл выполнен ({loop_result['iterations_completed']} итераций)",
                data=loop_result
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка выполнения цикла: {str(e)}",
                error=CommandError("LOOP_ERROR", str(e))
            )
    
    def _execute_loop(self, context: ExecutionContext, loop_type: str, condition: str,
                     iterations: int, commands: List[str], break_condition: str) -> Dict[str, Any]:
        """Выполнить цикл"""
        iteration_count = 0
        execution_log = []
        start_time = time.time()
        
        while True:
            # Проверяем условие прерывания
            if break_condition and self._evaluate_condition(context, break_condition):
                break
            
            # Проверяем основное условие цикла
            if loop_type == "while" and not self._evaluate_condition(context, condition):
                break
            elif loop_type == "until" and self._evaluate_condition(context, condition):
                break
            elif loop_type == "for" and iteration_count >= iterations:
                break
            
            # Выполняем команды итерации
            iteration_result = self._execute_iteration(context, commands, iteration_count)
            execution_log.append(iteration_result)
            
            iteration_count += 1
            
            # Защита от бесконечного цикла
            if iteration_count > 10000:
                break
        
        return {
            "loop_type": loop_type,
            "condition": condition,
            "iterations_completed": iteration_count,
            "execution_time": time.time() - start_time,
            "execution_log": execution_log[-10:],  # Последние 10 итераций
            "break_reason": self._determine_break_reason(loop_type, condition, iteration_count, iterations)
        }
    
    def _execute_iteration(self, context: ExecutionContext, commands: List[str], iteration: int) -> Dict[str, Any]:
        """Выполнить одну итерацию цикла"""
        iteration_start = time.time()
        command_results = []
        
        for cmd in commands:
            cmd = cmd.strip()
            if cmd:
                # Здесь должно быть выполнение команды через интерпретатор
                # Пока что просто логируем
                command_results.append({
                    "command": cmd,
                    "status": "simulated",
                    "timestamp": time.time()
                })
        
        return {
            "iteration": iteration,
            "commands_executed": len(command_results),
            "execution_time": time.time() - iteration_start,
            "results": command_results
        }
    
    def _evaluate_condition(self, context: ExecutionContext, condition: str) -> bool:
        """Оценить условие"""
        # Простая реализация - всегда возвращает True для "true"
        if condition.lower() in ["true", "1", "yes"]:
            return True
        elif condition.lower() in ["false", "0", "no"]:
            return False
        else:
            # Здесь должна быть более сложная логика оценки условий
            return True
    
    def _determine_break_reason(self, loop_type: str, condition: str, 
                               iteration_count: int, max_iterations: int) -> str:
        """Определить причину завершения цикла"""
        if loop_type == "for" and iteration_count >= max_iterations:
            return "max_iterations_reached"
        elif iteration_count > 10000:
            return "infinite_loop_protection"
        else:
            return "condition_met"


class HaltCommand(FlowControlCommand):
    """Команда остановки выполнения"""
    
    def __init__(self):
        super().__init__(
            name="halt",
            description="Останавливает выполнение программы или процесса",
            parameters={
                "target": "Цель остановки (all, process_id, node_id)",
                "reason": "Причина остановки",
                "graceful": "Корректная остановка (true/false)",
                "timeout": "Таймаут для корректной остановки (секунды)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get("target", "current")
            reason = kwargs.get("reason", "User requested halt")
            graceful = kwargs.get("graceful", "true").lower() == "true"
            timeout = int(kwargs.get("timeout", 5))
            
            # Выполняем остановку
            halt_result = self._perform_halt(context, target, reason, graceful, timeout)
            
            return CommandResult(
                success=True,
                message=f"Остановка выполнена для {target}",
                data=halt_result
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка остановки: {str(e)}",
                error=CommandError("HALT_ERROR", str(e))
            )
    
    def _perform_halt(self, context: ExecutionContext, target: str, 
                     reason: str, graceful: bool, timeout: int) -> Dict[str, Any]:
        """Выполнить остановку"""
        halt_start = time.time()
        halted_processes = []
        
        if target == "all":
            # Останавливаем все процессы
            if hasattr(context, 'processes'):
                for process_id, process_info in context.processes.items():
                    if process_info.state != FlowState.HALTED:
                        process_info.state = FlowState.HALTED
                        halted_processes.append(process_id)
        
        elif target.startswith("process_"):
            # Останавливаем конкретный процесс
            if hasattr(context, 'processes') and target in context.processes:
                context.processes[target].state = FlowState.HALTED
                halted_processes.append(target)
        
        else:
            # Останавливаем текущий процесс
            halted_processes.append("current")
        
        # Устанавливаем флаг остановки в контексте
        context.halt_requested = True
        context.halt_reason = reason
        context.halt_timestamp = time.time()
        
        return {
            "target": target,
            "reason": reason,
            "graceful": graceful,
            "timeout": timeout,
            "halted_processes": halted_processes,
            "halt_time": time.time() - halt_start,
            "timestamp": time.time()
        }


# Регистрируем все команды
ADVANCED_FLOW_COMMANDS = [
    ReflectCommand(),
    AbsorbCommand(),
    DiffuseCommand(),
    MergeCommand(),
    SplitCommand(),
    LoopCommand(),
    HaltCommand()
] 