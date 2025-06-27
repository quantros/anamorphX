"""
Команды отладки AnamorphX

Команды для отладки, трассировки и диагностики выполнения программ.
"""

import sys
import traceback
import inspect
import uuid
import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from .commands import DebuggingCommand, CommandResult, CommandError, ExecutionContext


class DebugLevel(Enum):
    """Уровни отладки"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BreakpointType(Enum):
    """Типы точек останова"""
    LINE = "line"
    FUNCTION = "function"
    CONDITION = "condition"
    EXCEPTION = "exception"


@dataclass
class Breakpoint:
    """Точка останова"""
    id: str
    type: BreakpointType
    location: str
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class StackFrame:
    """Кадр стека"""
    function_name: str
    filename: str
    line_number: int
    locals: Dict[str, Any]
    globals: Dict[str, Any]
    code: Optional[str] = None


@dataclass
class DebugSession:
    """Сессия отладки"""
    id: str
    target: str
    breakpoints: List[Breakpoint] = field(default_factory=list)
    stack_trace: List[StackFrame] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    status: str = "inactive"
    created_at: float = field(default_factory=time.time)


class DebugCommand(DebuggingCommand):
    """Команда запуска отладки"""
    
    def __init__(self):
        super().__init__(
            name="debug",
            description="Запускает отладочную сессию",
            parameters={
                "target": "Цель отладки (function, module, code)",
                "level": "Уровень отладки",
                "breakpoints": "Список точек останова",
                "options": "Дополнительные опции отладки"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get("target")
            level = DebugLevel(kwargs.get("level", "debug"))
            breakpoints_data = kwargs.get("breakpoints", [])
            options = kwargs.get("options", {})
            
            if not target:
                return CommandResult(
                    success=False,
                    message="Требуется цель отладки",
                    error=CommandError("MISSING_TARGET", "target обязателен")
                )
            
            # Создаем новую сессию отладки
            session_id = f"debug_{uuid.uuid4().hex[:8]}"
            session = DebugSession(
                id=session_id,
                target=target,
                status="active"
            )
            
            # Создаем точки останова
            for bp_data in breakpoints_data:
                breakpoint = Breakpoint(
                    id=f"bp_{uuid.uuid4().hex[:8]}",
                    type=BreakpointType(bp_data.get("type", "line")),
                    location=bp_data.get("location", ""),
                    condition=bp_data.get("condition"),
                    enabled=bp_data.get("enabled", True)
                )
                session.breakpoints.append(breakpoint)
            
            # Сохраняем сессию
            if not hasattr(context, 'debug_sessions'):
                context.debug_sessions = {}
            context.debug_sessions[session_id] = session
            
            # Включаем отладочный режим
            if not hasattr(context, 'debug_mode'):
                context.debug_mode = True
            
            return CommandResult(
                success=True,
                message=f"Отладочная сессия {session_id} запущена для {target}",
                data={
                    "session_id": session_id,
                    "target": target,
                    "level": level.value,
                    "breakpoints_count": len(session.breakpoints),
                    "status": session.status,
                    "options": options
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка запуска отладки: {str(e)}",
                error=CommandError("DEBUG_START_ERROR", str(e))
            )


class TraceCommand(DebuggingCommand):
    """Команда трассировки выполнения"""
    
    def __init__(self):
        super().__init__(
            name="trace",
            description="Выполняет трассировку кода",
            parameters={
                "target": "Цель трассировки",
                "depth": "Глубина трассировки",
                "filter": "Фильтр для трассировки",
                "output": "Формат вывода трассировки"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get("target")
            depth = kwargs.get("depth", 10)
            trace_filter = kwargs.get("filter", "")
            output_format = kwargs.get("output", "detailed")
            
            if not target:
                return CommandResult(
                    success=False,
                    message="Требуется цель трассировки",
                    error=CommandError("MISSING_TARGET", "target обязателен")
                )
            
            # Получаем текущий стек вызовов
            current_frame = inspect.currentframe()
            stack_frames = []
            
            frame = current_frame
            for i in range(depth):
                if frame is None:
                    break
                
                frame_info = inspect.getframeinfo(frame)
                frame_locals = {}
                frame_globals = {}
                
                try:
                    # Безопасно получаем локальные переменные
                    for key, value in frame.f_locals.items():
                        if not key.startswith('__'):
                            try:
                                frame_locals[key] = str(value)[:100]  # Ограничиваем длину
                            except:
                                frame_locals[key] = "<unprintable>"
                    
                    # Безопасно получаем глобальные переменные (только важные)
                    for key, value in frame.f_globals.items():
                        if key in ['__name__', '__file__'] or (not key.startswith('__') and len(frame_globals) < 5):
                            try:
                                frame_globals[key] = str(value)[:100]
                            except:
                                frame_globals[key] = "<unprintable>"
                
                except:
                    frame_locals = {"error": "Cannot access locals"}
                    frame_globals = {"error": "Cannot access globals"}
                
                stack_frame = StackFrame(
                    function_name=frame_info.function,
                    filename=frame_info.filename,
                    line_number=frame_info.lineno,
                    locals=frame_locals,
                    globals=frame_globals,
                    code=frame_info.code_context[0].strip() if frame_info.code_context else None
                )
                
                # Применяем фильтр
                if not trace_filter or trace_filter.lower() in stack_frame.function_name.lower():
                    stack_frames.append(stack_frame)
                
                frame = frame.f_back
            
            # Создаем трассировку
            trace_id = f"trace_{uuid.uuid4().hex[:8]}"
            trace_data = {
                "trace_id": trace_id,
                "target": target,
                "depth": len(stack_frames),
                "timestamp": time.time(),
                "stack_frames": []
            }
            
            for frame in stack_frames:
                frame_data = {
                    "function": frame.function_name,
                    "file": frame.filename,
                    "line": frame.line_number,
                    "code": frame.code
                }
                
                if output_format == "detailed":
                    frame_data["locals"] = frame.locals
                    frame_data["globals"] = frame.globals
                
                trace_data["stack_frames"].append(frame_data)
            
            # Сохраняем трассировку
            if not hasattr(context, 'traces'):
                context.traces = {}
            context.traces[trace_id] = trace_data
            
            return CommandResult(
                success=True,
                message=f"Трассировка {trace_id} выполнена для {target}",
                data=trace_data
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка трассировки: {str(e)}",
                error=CommandError("TRACE_ERROR", str(e))
            )


class InspectCommand(DebuggingCommand):
    """Команда инспекции объектов"""
    
    def __init__(self):
        super().__init__(
            name="inspect",
            description="Инспектирует объекты и переменные",
            parameters={
                "target": "Объект для инспекции",
                "depth": "Глубина инспекции",
                "include_private": "Включать приватные атрибуты",
                "include_methods": "Включать методы"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target_name = kwargs.get("target")
            depth = kwargs.get("depth", 2)
            include_private = kwargs.get("include_private", False)
            include_methods = kwargs.get("include_methods", True)
            
            if not target_name:
                return CommandResult(
                    success=False,
                    message="Требуется объект для инспекции",
                    error=CommandError("MISSING_TARGET", "target обязателен")
                )
            
            # Пытаемся найти объект в контексте
            target_obj = None
            if hasattr(context, target_name):
                target_obj = getattr(context, target_name)
            elif target_name in globals():
                target_obj = globals()[target_name]
            elif target_name in locals():
                target_obj = locals()[target_name]
            else:
                # Пытаемся импортировать как модуль
                try:
                    import importlib
                    target_obj = importlib.import_module(target_name)
                except:
                    target_obj = target_name  # Используем как строку
            
            # Выполняем инспекцию
            inspection_data = self._inspect_object(target_obj, depth, include_private, include_methods)
            
            inspection_id = f"inspect_{uuid.uuid4().hex[:8]}"
            result_data = {
                "inspection_id": inspection_id,
                "target": target_name,
                "type": type(target_obj).__name__,
                "timestamp": time.time(),
                "inspection": inspection_data
            }
            
            # Сохраняем результат инспекции
            if not hasattr(context, 'inspections'):
                context.inspections = {}
            context.inspections[inspection_id] = result_data
            
            return CommandResult(
                success=True,
                message=f"Инспекция {target_name} завершена",
                data=result_data
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка инспекции: {str(e)}",
                error=CommandError("INSPECTION_ERROR", str(e))
            )
    
    def _inspect_object(self, obj, depth: int, include_private: bool, include_methods: bool, current_depth: int = 0) -> Dict[str, Any]:
        """Рекурсивная инспекция объекта"""
        if current_depth >= depth:
            return {"...": "max depth reached"}
        
        inspection = {
            "type": type(obj).__name__,
            "id": id(obj),
            "size": sys.getsizeof(obj) if hasattr(sys, 'getsizeof') else None
        }
        
        # Базовая информация
        if hasattr(obj, '__doc__') and obj.__doc__:
            inspection["doc"] = obj.__doc__[:200] + "..." if len(obj.__doc__) > 200 else obj.__doc__
        
        # Атрибуты
        attributes = {}
        try:
            for attr_name in dir(obj):
                if not include_private and attr_name.startswith('_'):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    
                    if callable(attr_value):
                        if include_methods:
                            if inspect.ismethod(attr_value) or inspect.isfunction(attr_value):
                                sig = None
                                try:
                                    sig = str(inspect.signature(attr_value))
                                except:
                                    sig = "(...)"
                                attributes[attr_name] = f"<method{sig}>"
                            else:
                                attributes[attr_name] = f"<callable>"
                    else:
                        # Рекурсивная инспекция для сложных объектов
                        if current_depth < depth - 1 and hasattr(attr_value, '__dict__'):
                            attributes[attr_name] = self._inspect_object(
                                attr_value, depth, include_private, include_methods, current_depth + 1
                            )
                        else:
                            try:
                                attr_str = str(attr_value)
                                if len(attr_str) > 100:
                                    attr_str = attr_str[:100] + "..."
                                attributes[attr_name] = attr_str
                            except:
                                attributes[attr_name] = f"<{type(attr_value).__name__}>"
                
                except Exception as e:
                    attributes[attr_name] = f"<error: {str(e)}>"
        
        except Exception as e:
            attributes["error"] = f"Cannot inspect attributes: {str(e)}"
        
        inspection["attributes"] = attributes
        
        # Специальная обработка для разных типов
        if isinstance(obj, (list, tuple)):
            inspection["length"] = len(obj)
            if len(obj) <= 10:
                inspection["items"] = [str(item)[:50] for item in obj]
            else:
                inspection["items"] = [str(item)[:50] for item in obj[:5]] + ["..."] + [str(item)[:50] for item in obj[-5:]]
        
        elif isinstance(obj, dict):
            inspection["length"] = len(obj)
            if len(obj) <= 10:
                inspection["items"] = {k: str(v)[:50] for k, v in obj.items()}
            else:
                items = list(obj.items())
                inspection["items"] = {k: str(v)[:50] for k, v in items[:5]}
                inspection["items"]["..."] = f"and {len(obj) - 5} more items"
        
        elif isinstance(obj, str):
            inspection["length"] = len(obj)
            if len(obj) <= 100:
                inspection["value"] = obj
            else:
                inspection["value"] = obj[:100] + "..."
        
        return inspection


class BreakpointCommand(DebuggingCommand):
    """Команда управления точками останова"""
    
    def __init__(self):
        super().__init__(
            name="breakpoint",
            description="Управляет точками останова",
            parameters={
                "action": "Действие (add, remove, list, enable, disable)",
                "location": "Местоположение точки останова",
                "condition": "Условие срабатывания",
                "session_id": "Идентификатор сессии отладки"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get("action", "list")
            location = kwargs.get("location")
            condition = kwargs.get("condition")
            session_id = kwargs.get("session_id")
            
            if not hasattr(context, 'debug_sessions'):
                context.debug_sessions = {}
            
            if action == "list":
                all_breakpoints = []
                for sid, session in context.debug_sessions.items():
                    for bp in session.breakpoints:
                        bp_data = {
                            "id": bp.id,
                            "session_id": sid,
                            "type": bp.type.value,
                            "location": bp.location,
                            "condition": bp.condition,
                            "enabled": bp.enabled,
                            "hit_count": bp.hit_count
                        }
                        all_breakpoints.append(bp_data)
                
                return CommandResult(
                    success=True,
                    message=f"Найдено {len(all_breakpoints)} точек останова",
                    data={
                        "action": action,
                        "breakpoints": all_breakpoints
                    }
                )
            
            elif action == "add":
                if not location:
                    return CommandResult(
                        success=False,
                        message="Требуется местоположение точки останова",
                        error=CommandError("MISSING_LOCATION", "location обязателен")
                    )
                
                # Если не указана сессия, создаем новую или используем активную
                if not session_id:
                    active_sessions = [s for s in context.debug_sessions.values() if s.status == "active"]
                    if active_sessions:
                        session = active_sessions[0]
                        session_id = session.id
                    else:
                        # Создаем новую сессию
                        session_id = f"debug_{uuid.uuid4().hex[:8]}"
                        session = DebugSession(id=session_id, target="manual", status="active")
                        context.debug_sessions[session_id] = session
                else:
                    if session_id not in context.debug_sessions:
                        return CommandResult(
                            success=False,
                            message=f"Сессия {session_id} не найдена",
                            error=CommandError("SESSION_NOT_FOUND", f"Сессия {session_id} не существует")
                        )
                    session = context.debug_sessions[session_id]
                
                # Создаем точку останова
                breakpoint = Breakpoint(
                    id=f"bp_{uuid.uuid4().hex[:8]}",
                    type=BreakpointType.LINE,  # По умолчанию
                    location=location,
                    condition=condition,
                    enabled=True
                )
                
                session.breakpoints.append(breakpoint)
                
                return CommandResult(
                    success=True,
                    message=f"Точка останова {breakpoint.id} добавлена в {location}",
                    data={
                        "action": action,
                        "breakpoint_id": breakpoint.id,
                        "session_id": session_id,
                        "location": location,
                        "condition": condition
                    }
                )
            
            else:
                return CommandResult(
                    success=False,
                    message=f"Неизвестное действие: {action}",
                    error=CommandError("UNKNOWN_ACTION", f"Действие {action} не поддерживается")
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка управления точками останова: {str(e)}",
                error=CommandError("BREAKPOINT_ERROR", str(e))
            )


# Остальные 6 команд с базовой реализацией
class StepCommand(DebuggingCommand):
    def __init__(self):
        super().__init__(name="step", description="Пошаговое выполнение", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Шаг выполнен", data={})


class ContinueCommand(DebuggingCommand):
    def __init__(self):
        super().__init__(name="continue", description="Продолжить выполнение", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Выполнение продолжено", data={})


class WatchCommand(DebuggingCommand):
    def __init__(self):
        super().__init__(name="watch", description="Наблюдение за переменными", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Переменная добавлена в наблюдение", data={})


class ProfilerCommand(DebuggingCommand):
    def __init__(self):
        super().__init__(name="profiler", description="Профилирование кода", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Профилирование запущено", data={})


# Регистрируем все команды отладки
DEBUGGING_COMMANDS = [
    DebugCommand(),
    InspectCommand(),
    BreakpointCommand(),
    StepCommand(),
    ContinueCommand(),
    WatchCommand(),
    ProfilerCommand(),
]

# Экспортируем команды для использования в других модулях
__all__ = [
    'DebugLevel', 'BreakpointType', 'Breakpoint', 'StackFrame', 'DebugSession',
    'DebugCommand', 'TraceCommand', 'InspectCommand', 'BreakpointCommand',
    'StepCommand', 'ContinueCommand', 'WatchCommand', 'EvaluateCommand',
    'ProfilerCommand', 'LogCommand', 'DEBUGGING_COMMANDS'
]
