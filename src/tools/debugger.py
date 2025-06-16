"""
Отладчик для языка Anamorph

Возможности:
- Точки останова (breakpoints)
- Пошаговое выполнение (step-by-step)
- Инспекция переменных и стека
- Визуализация состояния нейронной сети
- Трассировка выполнения
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import traceback

from ..syntax.nodes import *
from ..interpreter.environment import Environment
from ..interpreter.type_system import TypeSystem


class DebugState(Enum):
    """Состояния отладчика"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEP_INTO = "step_into"
    STEP_OVER = "step_over"
    STEP_OUT = "step_out"


class BreakpointType(Enum):
    """Типы точек останова"""
    LINE = "line"                    # По номеру строки
    FUNCTION = "function"            # При входе в функцию
    VARIABLE = "variable"            # При изменении переменной
    EXCEPTION = "exception"          # При исключении
    NEURAL_EVENT = "neural_event"    # При нейронных событиях


@dataclass
class Breakpoint:
    """Точка останова"""
    id: str
    type: BreakpointType
    file_path: str
    line: Optional[int] = None
    function_name: Optional[str] = None
    variable_name: Optional[str] = None
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    hit_condition: Optional[str] = None  # "== 3", ">= 5", etc.


@dataclass
class StackFrame:
    """Кадр стека выполнения"""
    function_name: str
    file_path: str
    line: int
    local_variables: Dict[str, Any]
    arguments: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'function_name': self.function_name,
            'file_path': self.file_path,
            'line': self.line,
            'local_variables': self._serialize_variables(self.local_variables),
            'arguments': self._serialize_variables(self.arguments)
        }
    
    def _serialize_variables(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Сериализация переменных для отображения"""
        result = {}
        for name, value in variables.items():
            try:
                if hasattr(value, '__dict__'):
                    # Объекты
                    result[name] = f"<{type(value).__name__}> {str(value)[:100]}"
                else:
                    result[name] = str(value)
            except:
                result[name] = f"<{type(value).__name__}>"
        return result


@dataclass
class DebugEvent:
    """Событие отладки"""
    type: str
    timestamp: float
    file_path: str
    line: int
    message: str
    data: Optional[Dict] = None


class DebugEventHandler(ABC):
    """Обработчик событий отладки"""
    
    @abstractmethod
    def on_breakpoint_hit(self, breakpoint: Breakpoint, context: Dict):
        """Обработка попадания в точку останова"""
        pass
    
    @abstractmethod
    def on_step_complete(self, context: Dict):
        """Обработка завершения шага"""
        pass
    
    @abstractmethod
    def on_variable_changed(self, name: str, old_value: Any, new_value: Any):
        """Обработка изменения переменной"""
        pass
    
    @abstractmethod
    def on_exception(self, exception: Exception, context: Dict):
        """Обработка исключения"""
        pass


class BreakpointManager:
    """Менеджер точек останова"""
    
    def __init__(self):
        self.breakpoints: Dict[str, Breakpoint] = {}
        self._next_id = 1
    
    def add_breakpoint(self, bp_type: BreakpointType, file_path: str, 
                      line: Optional[int] = None, **kwargs) -> str:
        """Добавить точку останова"""
        bp_id = f"bp_{self._next_id}"
        self._next_id += 1
        
        breakpoint = Breakpoint(
            id=bp_id,
            type=bp_type,
            file_path=file_path,
            line=line,
            **kwargs
        )
        
        self.breakpoints[bp_id] = breakpoint
        return bp_id
    
    def remove_breakpoint(self, bp_id: str) -> bool:
        """Удалить точку останова"""
        return self.breakpoints.pop(bp_id, None) is not None
    
    def enable_breakpoint(self, bp_id: str, enabled: bool = True):
        """Включить/выключить точку останова"""
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].enabled = enabled
    
    def get_breakpoints_for_line(self, file_path: str, line: int) -> List[Breakpoint]:
        """Получить точки останова для строки"""
        return [
            bp for bp in self.breakpoints.values()
            if bp.enabled and bp.type == BreakpointType.LINE 
            and bp.file_path == file_path and bp.line == line
        ]
    
    def get_function_breakpoints(self, function_name: str) -> List[Breakpoint]:
        """Получить точки останова для функции"""
        return [
            bp for bp in self.breakpoints.values()
            if bp.enabled and bp.type == BreakpointType.FUNCTION
            and bp.function_name == function_name
        ]
    
    def should_break(self, breakpoint: Breakpoint, context: Dict) -> bool:
        """Проверить, нужно ли останавливаться на точке"""
        if not breakpoint.enabled:
            return False
        
        breakpoint.hit_count += 1
        
        # Проверка условия попадания
        if breakpoint.hit_condition:
            try:
                condition = breakpoint.hit_condition.replace('hit_count', str(breakpoint.hit_count))
                if not eval(condition):
                    return False
            except:
                pass
        
        # Проверка пользовательского условия
        if breakpoint.condition:
            try:
                # Создаем локальное окружение для проверки условия
                local_env = context.get('variables', {})
                if not eval(breakpoint.condition, {}, local_env):
                    return False
            except:
                pass
        
        return True


class NeuralStateInspector:
    """Инспектор состояния нейронной сети"""
    
    def __init__(self):
        self.neural_objects: Dict[str, Any] = {}
        self.connection_graph: Dict[str, List[str]] = {}
        self.activation_history: List[Dict] = []
    
    def register_neural_object(self, name: str, obj: Any):
        """Регистрация нейронного объекта"""
        self.neural_objects[name] = obj
        
        # Анализ связей
        if hasattr(obj, 'connections'):
            self.connection_graph[name] = list(obj.connections.keys())
    
    def capture_activation_state(self, context: Dict):
        """Захват состояния активации"""
        state = {
            'timestamp': time.time(),
            'line': context.get('line', 0),
            'activations': {},
            'weights': {},
            'signals': {}
        }
        
        for name, obj in self.neural_objects.items():
            try:
                if hasattr(obj, 'activation'):
                    state['activations'][name] = obj.activation
                if hasattr(obj, 'weight'):
                    state['weights'][name] = obj.weight
                if hasattr(obj, 'signal_value'):
                    state['signals'][name] = obj.signal_value
            except:
                pass
        
        self.activation_history.append(state)
        
        # Ограничиваем историю
        if len(self.activation_history) > 1000:
            self.activation_history = self.activation_history[-1000:]
    
    def get_neural_graph(self) -> Dict:
        """Получить граф нейронной сети"""
        nodes = []
        edges = []
        
        for name, obj in self.neural_objects.items():
            # Узлы
            node_type = type(obj).__name__
            nodes.append({
                'id': name,
                'type': node_type,
                'properties': self._get_object_properties(obj)
            })
            
            # Связи
            if name in self.connection_graph:
                for target in self.connection_graph[name]:
                    edges.append({
                        'from': name,
                        'to': target,
                        'type': 'connection'
                    })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def _get_object_properties(self, obj: Any) -> Dict:
        """Получить свойства объекта"""
        properties = {}
        
        # Стандартные свойства
        for attr in ['activation', 'weight', 'bias', 'signal_value', 'threshold']:
            if hasattr(obj, attr):
                try:
                    properties[attr] = getattr(obj, attr)
                except:
                    properties[attr] = None
        
        return properties


class AnamorphDebugger:
    """Отладчик для языка Anamorph"""
    
    def __init__(self, interpreter=None):
        self.interpreter = interpreter
        self.state = DebugState.STOPPED
        self.breakpoint_manager = BreakpointManager()
        self.neural_inspector = NeuralStateInspector()
        
        # Стек выполнения
        self.call_stack: List[StackFrame] = []
        self.current_frame: Optional[StackFrame] = None
        
        # Обработчики событий
        self.event_handlers: List[DebugEventHandler] = []
        
        # История событий
        self.event_history: List[DebugEvent] = []
        
        # Текущее выполнение
        self.current_file: Optional[str] = None
        self.current_line: int = 0
        self.current_node: Optional[ASTNode] = None
        
        # Флаги пошагового выполнения
        self.step_mode = False
        self.step_level = 0  # Уровень вложенности для step over/out
        
        # Переменные для отслеживания
        self.watched_variables: Set[str] = set()
        self.variable_values: Dict[str, Any] = {}
        
        # Поток отладки
        self.debug_thread: Optional[threading.Thread] = None
        self.debug_lock = threading.Lock()
    
    def add_event_handler(self, handler: DebugEventHandler):
        """Добавить обработчик событий"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: DebugEventHandler):
        """Удалить обработчик событий"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    def start_debugging(self, code: str, file_path: str = "main.amph"):
        """Начать отладку"""
        self.state = DebugState.RUNNING
        self.current_file = file_path
        self.current_line = 1
        
        # Запуск в отдельном потоке
        self.debug_thread = threading.Thread(
            target=self._debug_execution,
            args=(code, file_path)
        )
        self.debug_thread.start()
    
    def _debug_execution(self, code: str, file_path: str):
        """Выполнение с отладкой"""
        try:
            if self.interpreter:
                # Интеграция с интерпретатором
                self.interpreter.set_debug_callback(self._debug_callback)
                self.interpreter.execute(code)
            else:
                # Простая эмуляция выполнения
                self._simulate_execution(code, file_path)
                
        except Exception as e:
            self._emit_event('exception', f"Исключение: {e}", {
                'exception': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Уведомление обработчиков
            for handler in self.event_handlers:
                try:
                    handler.on_exception(e, self._get_current_context())
                except:
                    pass
        
        self.state = DebugState.STOPPED
    
    def _debug_callback(self, node: ASTNode, context: Dict):
        """Callback для отладки выполнения"""
        self.current_node = node
        
        # Обновление позиции
        if hasattr(node, 'line'):
            self.current_line = node.line
        
        # Обновление стека
        self._update_call_stack(context)
        
        # Проверка переменных
        self._check_variable_changes(context)
        
        # Проверка точек останова
        if self._should_break_at_line():
            self.state = DebugState.PAUSED
            self._emit_event('breakpoint_hit', f"Точка останова на строке {self.current_line}")
            
            # Уведомление обработчиков
            breakpoints = self.breakpoint_manager.get_breakpoints_for_line(
                self.current_file, self.current_line
            )
            for bp in breakpoints:
                if self.breakpoint_manager.should_break(bp, context):
                    for handler in self.event_handlers:
                        try:
                            handler.on_breakpoint_hit(bp, self._get_current_context())
                        except:
                            pass
        
        # Пошаговое выполнение
        if self.step_mode:
            self._handle_step_execution()
        
        # Захват состояния нейронной сети
        self.neural_inspector.capture_activation_state(context)
        
        # Ожидание команд
        while self.state == DebugState.PAUSED:
            time.sleep(0.1)
    
    def _simulate_execution(self, code: str, file_path: str):
        """Симуляция выполнения для тестирования"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            self.current_line = i
            
            # Эмуляция контекста
            context = {
                'line': i,
                'variables': {'x': i, 'y': i * 2},
                'function': 'main' if i < 10 else 'helper'
            }
            
            # Проверки отладки
            self._debug_callback(None, context)
            
            # Пауза между строками
            time.sleep(0.1)
    
    def _should_break_at_line(self) -> bool:
        """Проверить, нужно ли останавливаться на текущей строке"""
        if not self.current_file:
            return False
        
        breakpoints = self.breakpoint_manager.get_breakpoints_for_line(
            self.current_file, self.current_line
        )
        
        return len(breakpoints) > 0
    
    def _handle_step_execution(self):
        """Обработка пошагового выполнения"""
        if self.state == DebugState.STEP_INTO:
            # Останавливаемся на каждой строке
            self.state = DebugState.PAUSED
            
        elif self.state == DebugState.STEP_OVER:
            # Останавливаемся только на том же уровне
            current_level = len(self.call_stack)
            if current_level <= self.step_level:
                self.state = DebugState.PAUSED
                
        elif self.state == DebugState.STEP_OUT:
            # Останавливаемся при выходе из функции
            current_level = len(self.call_stack)
            if current_level < self.step_level:
                self.state = DebugState.PAUSED
        
        # Уведомление обработчиков о завершении шага
        if self.state == DebugState.PAUSED:
            for handler in self.event_handlers:
                try:
                    handler.on_step_complete(self._get_current_context())
                except:
                    pass
    
    def _update_call_stack(self, context: Dict):
        """Обновление стека вызовов"""
        # Упрощенная логика обновления стека
        function_name = context.get('function', 'main')
        
        if not self.call_stack or self.call_stack[-1].function_name != function_name:
            # Новый вызов функции
            frame = StackFrame(
                function_name=function_name,
                file_path=self.current_file or '',
                line=self.current_line,
                local_variables=context.get('variables', {}),
                arguments=context.get('arguments', {})
            )
            self.call_stack.append(frame)
            self.current_frame = frame
        else:
            # Обновление текущего кадра
            if self.current_frame:
                self.current_frame.line = self.current_line
                self.current_frame.local_variables = context.get('variables', {})
    
    def _check_variable_changes(self, context: Dict):
        """Проверка изменений переменных"""
        variables = context.get('variables', {})
        
        for name, value in variables.items():
            if name in self.watched_variables:
                old_value = self.variable_values.get(name)
                
                if old_value != value:
                    # Переменная изменилась
                    self._emit_event('variable_changed', 
                                   f"Переменная {name} изменилась: {old_value} -> {value}")
                    
                    # Уведомление обработчиков
                    for handler in self.event_handlers:
                        try:
                            handler.on_variable_changed(name, old_value, value)
                        except:
                            pass
                    
                    self.variable_values[name] = value
    
    def _get_current_context(self) -> Dict:
        """Получить текущий контекст выполнения"""
        return {
            'file': self.current_file,
            'line': self.current_line,
            'state': self.state.value,
            'call_stack': [frame.to_dict() for frame in self.call_stack],
            'current_frame': self.current_frame.to_dict() if self.current_frame else None,
            'neural_state': self.neural_inspector.get_neural_graph()
        }
    
    def _emit_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Генерация события отладки"""
        event = DebugEvent(
            type=event_type,
            timestamp=time.time(),
            file_path=self.current_file or '',
            line=self.current_line,
            message=message,
            data=data
        )
        
        self.event_history.append(event)
        
        # Ограничиваем историю событий
        if len(self.event_history) > 10000:
            self.event_history = self.event_history[-10000:]
    
    # Управление отладчиком
    def pause(self):
        """Приостановить выполнение"""
        self.state = DebugState.PAUSED
    
    def resume(self):
        """Продолжить выполнение"""
        self.state = DebugState.RUNNING
        self.step_mode = False
    
    def step_into(self):
        """Шаг с заходом в функции"""
        self.state = DebugState.STEP_INTO
        self.step_mode = True
        self.step_level = len(self.call_stack)
    
    def step_over(self):
        """Шаг с пропуском функций"""
        self.state = DebugState.STEP_OVER
        self.step_mode = True
        self.step_level = len(self.call_stack)
    
    def step_out(self):
        """Шаг с выходом из функции"""
        self.state = DebugState.STEP_OUT
        self.step_mode = True
        self.step_level = len(self.call_stack)
    
    def stop(self):
        """Остановить отладку"""
        self.state = DebugState.STOPPED
        self.step_mode = False
        
        if self.debug_thread and self.debug_thread.is_alive():
            self.debug_thread.join(timeout=1.0)
    
    # Управление точками останова
    def add_line_breakpoint(self, file_path: str, line: int, condition: str = None) -> str:
        """Добавить точку останова на строке"""
        return self.breakpoint_manager.add_breakpoint(
            BreakpointType.LINE, file_path, line=line, condition=condition
        )
    
    def add_function_breakpoint(self, function_name: str, condition: str = None) -> str:
        """Добавить точку останова на функции"""
        return self.breakpoint_manager.add_breakpoint(
            BreakpointType.FUNCTION, '', function_name=function_name, condition=condition
        )
    
    def remove_breakpoint(self, bp_id: str) -> bool:
        """Удалить точку останова"""
        return self.breakpoint_manager.remove_breakpoint(bp_id)
    
    def list_breakpoints(self) -> List[Dict]:
        """Список всех точек останова"""
        return [
            {
                'id': bp.id,
                'type': bp.type.value,
                'file_path': bp.file_path,
                'line': bp.line,
                'function_name': bp.function_name,
                'condition': bp.condition,
                'enabled': bp.enabled,
                'hit_count': bp.hit_count
            }
            for bp in self.breakpoint_manager.breakpoints.values()
        ]
    
    # Инспекция переменных
    def add_watch(self, variable_name: str):
        """Добавить переменную для отслеживания"""
        self.watched_variables.add(variable_name)
    
    def remove_watch(self, variable_name: str):
        """Удалить переменную из отслеживания"""
        self.watched_variables.discard(variable_name)
    
    def get_variable_value(self, variable_name: str) -> Any:
        """Получить значение переменной"""
        if self.current_frame:
            return self.current_frame.local_variables.get(variable_name)
        return None
    
    def evaluate_expression(self, expression: str) -> Any:
        """Вычислить выражение в текущем контексте"""
        if not self.current_frame:
            return None
        
        try:
            local_vars = self.current_frame.local_variables
            return eval(expression, {}, local_vars)
        except Exception as e:
            return f"Ошибка: {e}"
    
    # Экспорт состояния
    def export_debug_session(self) -> Dict:
        """Экспорт сессии отладки"""
        return {
            'session_info': {
                'state': self.state.value,
                'current_file': self.current_file,
                'current_line': self.current_line,
                'timestamp': time.time()
            },
            'breakpoints': self.list_breakpoints(),
            'call_stack': [frame.to_dict() for frame in self.call_stack],
            'watched_variables': list(self.watched_variables),
            'event_history': [
                {
                    'type': event.type,
                    'timestamp': event.timestamp,
                    'file_path': event.file_path,
                    'line': event.line,
                    'message': event.message,
                    'data': event.data
                }
                for event in self.event_history[-100:]  # Последние 100 событий
            ],
            'neural_state': self.neural_inspector.get_neural_graph(),
            'activation_history': self.neural_inspector.activation_history[-50:]  # Последние 50 состояний
        }


# Консольный интерфейс отладчика
class ConsoleDebugHandler(DebugEventHandler):
    """Консольный обработчик событий отладки"""
    
    def on_breakpoint_hit(self, breakpoint: Breakpoint, context: Dict):
        print(f"\n🔴 Точка останова на строке {context['line']}")
        print(f"   Файл: {context['file']}")
        print(f"   Функция: {context.get('current_frame', {}).get('function_name', 'unknown')}")
        
        # Показать локальные переменные
        if 'current_frame' in context and context['current_frame']:
            variables = context['current_frame'].get('local_variables', {})
            if variables:
                print("   Переменные:")
                for name, value in variables.items():
                    print(f"     {name} = {value}")
    
    def on_step_complete(self, context: Dict):
        print(f"👣 Шаг выполнен: строка {context['line']}")
    
    def on_variable_changed(self, name: str, old_value: Any, new_value: Any):
        print(f"🔄 Переменная {name}: {old_value} -> {new_value}")
    
    def on_exception(self, exception: Exception, context: Dict):
        print(f"\n❌ Исключение: {exception}")
        print(f"   Строка: {context['line']}")
        print(f"   Файл: {context['file']}")


def create_debugger(interpreter=None) -> AnamorphDebugger:
    """Создать отладчик с консольным интерфейсом"""
    debugger = AnamorphDebugger(interpreter)
    debugger.add_event_handler(ConsoleDebugHandler())
    return debugger


if __name__ == "__main__":
    # Тестовый код
    test_code = '''
    def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n - 1)
    
    result = factorial(5)
    print(f"Факториал 5 = {result}")
    
    # Нейронные операции
    neuron test_neuron {
        activation: "relu"
        threshold: 0.5
    }
    
    signal input_signal {
        value: 0.8
    }
    
    # Активация нейрона
    test_neuron.activate(input_signal)
    '''
    
    # Создание отладчика
    debugger = create_debugger()
    
    # Добавление точек останова
    bp1 = debugger.add_line_breakpoint("test.amph", 3)
    bp2 = debugger.add_function_breakpoint("factorial")
    
    # Отслеживание переменных
    debugger.add_watch("n")
    debugger.add_watch("result")
    
    print("🐛 Запуск отладчика...")
    print(f"Точки останова: {debugger.list_breakpoints()}")
    
    # Запуск отладки
    debugger.start_debugging(test_code, "test.amph")
    
    # Интерактивные команды
    print("\nДоступные команды:")
    print("  c - продолжить")
    print("  s - шаг с заходом")
    print("  n - шаг с пропуском")
    print("  u - шаг с выходом")
    print("  q - выход")
    
    while debugger.state != DebugState.STOPPED:
        try:
            command = input("\n(debug) ").strip().lower()
            
            if command == 'c':
                debugger.resume()
            elif command == 's':
                debugger.step_into()
            elif command == 'n':
                debugger.step_over()
            elif command == 'u':
                debugger.step_out()
            elif command == 'q':
                debugger.stop()
                break
            elif command.startswith('p '):
                # Печать переменной
                var_name = command[2:]
                value = debugger.get_variable_value(var_name)
                print(f"{var_name} = {value}")
            elif command.startswith('eval '):
                # Вычисление выражения
                expression = command[5:]
                result = debugger.evaluate_expression(expression)
                print(f"=> {result}")
            else:
                print("Неизвестная команда")
                
        except KeyboardInterrupt:
            debugger.stop()
            break
    
    # Экспорт сессии
    session_data = debugger.export_debug_session()
    print(f"\n📊 Сессия отладки завершена")
    print(f"   События: {len(session_data['event_history'])}")
    print(f"   Активации: {len(session_data['activation_history'])}") 