"""
Error Handling System for AnamorphX Interpreter

Комплексная система обработки ошибок для интерпретатора Anamorph:
- Иерархия исключений
- Стратегии восстановления после ошибок
- Детальные сообщения об ошибках
- Трассировка выполнения
- Система логирования ошибок
"""

import os
import sys
import time
import traceback
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


class ErrorSeverity(Enum):
    """Уровни серьезности ошибок."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()


class ErrorCategory(Enum):
    """Категории ошибок."""
    SYNTAX = auto()          # Синтаксические ошибки
    TYPE = auto()            # Ошибки типов
    NAME = auto()            # Ошибки имен/идентификаторов
    RUNTIME = auto()         # Ошибки времени выполнения
    NEURAL = auto()          # Ошибки нейронных операций
    SIGNAL = auto()          # Ошибки сигналов
    MEMORY = auto()          # Ошибки памяти
    SECURITY = auto()        # Ошибки безопасности
    SYSTEM = auto()          # Системные ошибки
    NETWORK = auto()         # Сетевые ошибки
    IO = auto()              # Ошибки ввода-вывода


@dataclass
class ErrorLocation:
    """Местоположение ошибки в коде."""
    filename: Optional[str] = None
    line: int = 0
    column: int = 0
    function_name: Optional[str] = None
    context: Optional[str] = None
    
    def __str__(self) -> str:
        location_parts = []
        if self.filename:
            location_parts.append(f"File '{self.filename}'")
        if self.line > 0:
            location_parts.append(f"line {self.line}")
        if self.column > 0:
            location_parts.append(f"column {self.column}")
        if self.function_name:
            location_parts.append(f"in {self.function_name}()")
        
        return ", ".join(location_parts) if location_parts else "unknown location"


@dataclass
class ErrorContext:
    """Контекст ошибки."""
    variables: Dict[str, Any] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    neural_state: Dict[str, Any] = field(default_factory=dict)
    signal_queue: List[Any] = field(default_factory=list)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    memory_usage: int = 0
    timestamp: float = field(default_factory=time.time)


class AnamorphError(Exception):
    """Базовый класс для всех ошибок Anamorph."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.RUNTIME,
                 severity: ErrorSeverity = ErrorSeverity.ERROR, 
                 location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.location = location or ErrorLocation()
        self.context = context or ErrorContext()
        self.cause = cause
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
    
    def _generate_error_id(self) -> str:
        """Генерация уникального ID ошибки."""
        import hashlib
        error_data = f"{self.category.name}_{self.message}_{self.timestamp}"
        return hashlib.md5(error_data.encode()).hexdigest()[:8]
    
    def __str__(self) -> str:
        return f"[{self.category.name}] {self.message}"
    
    def get_detailed_message(self) -> str:
        """Получение детального сообщения об ошибке."""
        lines = [
            f"Error ID: {self.error_id}",
            f"Category: {self.category.name}",
            f"Severity: {self.severity.name}",
            f"Message: {self.message}",
            f"Location: {self.location}",
            f"Timestamp: {time.ctime(self.timestamp)}"
        ]
        
        if self.context.call_stack:
            lines.append(f"Call Stack: {' -> '.join(self.context.call_stack)}")
        
        if self.context.variables:
            lines.append("Variables:")
            for var, value in self.context.variables.items():
                lines.append(f"  {var} = {value}")
        
        if self.cause:
            lines.append(f"Caused by: {self.cause}")
        
        return "\n".join(lines)


class SyntaxError(AnamorphError):
    """Ошибки синтаксиса."""
    
    def __init__(self, message: str, location: Optional[ErrorLocation] = None, 
                 context: Optional[ErrorContext] = None):
        super().__init__(message, ErrorCategory.SYNTAX, ErrorSeverity.ERROR, location, context)


class TypeError(AnamorphError):
    """Ошибки типов."""
    
    def __init__(self, message: str, expected_type: Optional[str] = None,
                 actual_type: Optional[str] = None, location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None):
        self.expected_type = expected_type
        self.actual_type = actual_type
        
        if expected_type and actual_type:
            message = f"{message} (expected {expected_type}, got {actual_type})"
        
        super().__init__(message, ErrorCategory.TYPE, ErrorSeverity.ERROR, location, context)


class NameError(AnamorphError):
    """Ошибки имен и идентификаторов."""
    
    def __init__(self, message: str, name: Optional[str] = None,
                 location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None):
        self.name = name
        super().__init__(message, ErrorCategory.NAME, ErrorSeverity.ERROR, location, context)


class RuntimeError(AnamorphError):
    """Ошибки времени выполнения."""
    
    def __init__(self, message: str, location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.RUNTIME, ErrorSeverity.ERROR, location, context, cause)


class NeuralError(AnamorphError):
    """Ошибки нейронных операций."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 neuron_name: Optional[str] = None, location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None):
        self.operation = operation
        self.neuron_name = neuron_name
        super().__init__(message, ErrorCategory.NEURAL, ErrorSeverity.ERROR, location, context)


class SignalError(AnamorphError):
    """Ошибки сигналов."""
    
    def __init__(self, message: str, signal_type: Optional[str] = None,
                 source: Optional[str] = None, target: Optional[str] = None,
                 location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None):
        self.signal_type = signal_type
        self.source = source
        self.target = target
        super().__init__(message, ErrorCategory.SIGNAL, ErrorSeverity.ERROR, location, context)


class MemoryError(AnamorphError):
    """Ошибки памяти."""
    
    def __init__(self, message: str, requested_size: Optional[int] = None,
                 available_size: Optional[int] = None, location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None):
        self.requested_size = requested_size
        self.available_size = available_size
        super().__init__(message, ErrorCategory.MEMORY, ErrorSeverity.CRITICAL, location, context)


class SecurityError(AnamorphError):
    """Ошибки безопасности."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 resource: Optional[str] = None, location: Optional[ErrorLocation] = None,
                 context: Optional[ErrorContext] = None):
        self.operation = operation
        self.resource = resource
        super().__init__(message, ErrorCategory.SECURITY, ErrorSeverity.CRITICAL, location, context)


class RecoveryStrategy(Enum):
    """Стратегии восстановления после ошибок."""
    IGNORE = auto()          # Игнорировать ошибку
    RETRY = auto()           # Повторить операцию
    FALLBACK = auto()        # Использовать резервный вариант
    SKIP = auto()            # Пропустить операцию
    ABORT = auto()           # Прервать выполнение
    RESTART = auto()         # Перезапустить компонент


@dataclass
class RecoveryAction:
    """Действие для восстановления после ошибки."""
    strategy: RecoveryStrategy
    handler: Optional[Callable] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_value: Any = None
    conditions: List[Callable] = field(default_factory=list)


class ErrorHandler:
    """Основной обработчик ошибок."""
    
    def __init__(self, log_level: int = logging.INFO, log_file: Optional[str] = None):
        self.recovery_strategies: Dict[ErrorCategory, RecoveryAction] = {}
        self.error_history: List[AnamorphError] = []
        self.max_history_size = 1000
        self.statistics = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
        
        # Настройка логирования
        self.logger = logging.getLogger('AnamorphErrorHandler')
        self.logger.setLevel(log_level)
        
        # Форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый обработчик (если указан)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Настройка стратегий восстановления по умолчанию."""
        # Синтаксические ошибки - прерывание
        self.recovery_strategies[ErrorCategory.SYNTAX] = RecoveryAction(
            strategy=RecoveryStrategy.ABORT
        )
        
        # Ошибки типов - попытка приведения типов
        self.recovery_strategies[ErrorCategory.TYPE] = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            fallback_value=None
        )
        
        # Ошибки имен - использование значения по умолчанию
        self.recovery_strategies[ErrorCategory.NAME] = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            fallback_value=0
        )
        
        # Ошибки времени выполнения - повтор с задержкой
        self.recovery_strategies[ErrorCategory.RUNTIME] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            max_retries=3,
            retry_delay=1.0
        )
        
        # Нейронные ошибки - пропуск операции
        self.recovery_strategies[ErrorCategory.NEURAL] = RecoveryAction(
            strategy=RecoveryStrategy.SKIP
        )
        
        # Ошибки сигналов - игнорирование
        self.recovery_strategies[ErrorCategory.SIGNAL] = RecoveryAction(
            strategy=RecoveryStrategy.IGNORE
        )
        
        # Ошибки памяти - принудительная очистка и повтор
        self.recovery_strategies[ErrorCategory.MEMORY] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            max_retries=1,
            handler=self._handle_memory_error
        )
        
        # Ошибки безопасности - прерывание
        self.recovery_strategies[ErrorCategory.SECURITY] = RecoveryAction(
            strategy=RecoveryStrategy.ABORT
        )
    
    def handle_error(self, error: AnamorphError) -> Tuple[bool, Any]:
        """
        Обработка ошибки с попыткой восстановления.
        
        Returns:
            Tuple[bool, Any]: (успешное восстановление, результат)
        """
        # Логирование ошибки
        self._log_error(error)
        
        # Обновление статистики
        self._update_statistics(error)
        
        # Сохранение в истории
        self._add_to_history(error)
        
        # Попытка восстановления
        recovery_action = self.recovery_strategies.get(error.category)
        if not recovery_action:
            return False, None
        
        return self._execute_recovery(error, recovery_action)
    
    def _log_error(self, error: AnamorphError):
        """Логирование ошибки."""
        log_level_map = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }
        
        log_level = log_level_map.get(error.severity, logging.ERROR)
        self.logger.log(log_level, error.get_detailed_message())
    
    def _update_statistics(self, error: AnamorphError):
        """Обновление статистики ошибок."""
        self.statistics['total_errors'] += 1
        
        # По категориям
        category_name = error.category.name
        self.statistics['errors_by_category'][category_name] = (
            self.statistics['errors_by_category'].get(category_name, 0) + 1
        )
        
        # По уровням серьезности
        severity_name = error.severity.name
        self.statistics['errors_by_severity'][severity_name] = (
            self.statistics['errors_by_severity'].get(severity_name, 0) + 1
        )
    
    def _add_to_history(self, error: AnamorphError):
        """Добавление ошибки в историю."""
        self.error_history.append(error)
        
        # Ограничение размера истории
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def _execute_recovery(self, error: AnamorphError, recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Выполнение стратегии восстановления."""
        self.statistics['recovery_attempts'] += 1
        
        try:
            if recovery_action.strategy == RecoveryStrategy.IGNORE:
                return True, None
            
            elif recovery_action.strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(error, recovery_action)
            
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                return True, recovery_action.fallback_value
            
            elif recovery_action.strategy == RecoveryStrategy.SKIP:
                self.logger.info(f"Skipping operation due to error: {error.message}")
                return True, None
            
            elif recovery_action.strategy == RecoveryStrategy.ABORT:
                self.logger.critical(f"Aborting execution due to error: {error.message}")
                return False, None
            
            elif recovery_action.strategy == RecoveryStrategy.RESTART:
                return self._restart_component(error, recovery_action)
            
            else:
                return False, None
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return False, None
    
    def _retry_operation(self, error: AnamorphError, recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Повтор операции."""
        if recovery_action.handler:
            try:
                result = recovery_action.handler(error)
                self.statistics['successful_recoveries'] += 1
                return True, result
            except Exception as e:
                self.logger.error(f"Custom recovery handler failed: {e}")
                return False, None
        
        # Базовый повтор (имитация)
        self.logger.info(f"Retrying operation after error: {error.message}")
        import time
        time.sleep(recovery_action.retry_delay)
        
        self.statistics['successful_recoveries'] += 1
        return True, None
    
    def _restart_component(self, error: AnamorphError, recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Перезапуск компонента."""
        self.logger.info(f"Restarting component due to error: {error.message}")
        
        if recovery_action.handler:
            try:
                result = recovery_action.handler(error)
                self.statistics['successful_recoveries'] += 1
                return True, result
            except Exception as e:
                self.logger.error(f"Component restart failed: {e}")
                return False, None
        
        return True, None
    
    def _handle_memory_error(self, error: MemoryError) -> Any:
        """Специальный обработчик ошибок памяти."""
        self.logger.warning("Handling memory error - attempting garbage collection")
        
        try:
            import gc
            gc.collect()
            self.logger.info("Garbage collection completed")
            return "memory_cleared"
        except Exception as e:
            self.logger.error(f"Garbage collection failed: {e}")
            raise
    
    def register_recovery_strategy(self, category: ErrorCategory, action: RecoveryAction):
        """Регистрация пользовательской стратегии восстановления."""
        self.recovery_strategies[category] = action
        self.logger.info(f"Registered recovery strategy for {category.name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Получение статистики ошибок."""
        return dict(self.statistics)
    
    def get_recent_errors(self, count: int = 10) -> List[AnamorphError]:
        """Получение последних ошибок."""
        return self.error_history[-count:] if self.error_history else []
    
    def clear_error_history(self):
        """Очистка истории ошибок."""
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def export_error_report(self, filename: str) -> bool:
        """Экспорт отчета об ошибках в файл."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("ANAMORPH ERROR REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Статистика
                f.write("STATISTICS:\n")
                for key, value in self.statistics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # История ошибок
                f.write("ERROR HISTORY:\n")
                for i, error in enumerate(self.error_history, 1):
                    f.write(f"\n{i}. {error.get_detailed_message()}\n")
                    f.write("-" * 30 + "\n")
            
            self.logger.info(f"Error report exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            return False


# Утилиты для создания контекста ошибок
def create_error_location(filename: str = None, line: int = 0, column: int = 0,
                         function_name: str = None, context: str = None) -> ErrorLocation:
    """Создание местоположения ошибки."""
    return ErrorLocation(filename, line, column, function_name, context)


def create_error_context(variables: Dict[str, Any] = None, call_stack: List[str] = None,
                        neural_state: Dict[str, Any] = None, signal_queue: List[Any] = None,
                        execution_stats: Dict[str, Any] = None, memory_usage: int = 0) -> ErrorContext:
    """Создание контекста ошибки."""
    return ErrorContext(
        variables or {},
        call_stack or [],
        neural_state or {},
        signal_queue or [],
        execution_stats or {},
        memory_usage
    )


# Декораторы для обработки ошибок
def error_handler(handler: ErrorHandler, category: ErrorCategory = ErrorCategory.RUNTIME):
    """Декоратор для автоматической обработки ошибок."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Создаем Anamorph ошибку
                error = AnamorphError(
                    message=str(e),
                    category=category,
                    location=create_error_location(function_name=func.__name__),
                    cause=e
                )
                
                # Обрабатываем ошибку
                recovered, result = handler.handle_error(error)
                if recovered:
                    return result
                else:
                    raise error
        
        return wrapper
    return decorator


# Экспорт основных классов
__all__ = [
    'ErrorHandler',
    'AnamorphError',
    'SyntaxError',
    'TypeError', 
    'NameError',
    'RuntimeError',
    'NeuralError',
    'SignalError',
    'MemoryError',
    'SecurityError',
    'ErrorLocation',
    'ErrorContext',
    'ErrorCategory',
    'ErrorSeverity',
    'RecoveryStrategy',
    'RecoveryAction',
    'create_error_location',
    'create_error_context',
    'error_handler'
]


if __name__ == "__main__":
    # Демонстрация системы обработки ошибок
    print("🛡️ ДЕМОНСТРАЦИЯ СИСТЕМЫ ОБРАБОТКИ ОШИБОК")
    print("=" * 50)
    
    # Создаем обработчик ошибок
    error_handler = ErrorHandler()
    
    # Тестовые ошибки
    test_errors = [
        SyntaxError("Unexpected token", 
                   create_error_location("test.amph", 10, 5)),
        TypeError("Type mismatch", "int", "string",
                 create_error_location("test.amph", 15, 10)),
        NameError("Undefined variable 'x'", "x",
                 create_error_location("test.amph", 20, 1)),
        NeuralError("Invalid activation function", "activate", "neuron1",
                   create_error_location("test.amph", 25, 3)),
        SignalError("Signal routing failed", "pulse", "input", "output",
                   create_error_location("test.amph", 30, 8))
    ]
    
    print("🔍 Обработка тестовых ошибок:")
    for i, error in enumerate(test_errors, 1):
        print(f"\n{i}. Обработка {error.category.name} ошибки:")
        recovered, result = error_handler.handle_error(error)
        print(f"   Восстановление: {'✅ Успешно' if recovered else '❌ Неудачно'}")
        if result is not None:
            print(f"   Результат: {result}")
    
    print(f"\n📊 СТАТИСТИКА ОШИБОК:")
    stats = error_handler.get_error_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n📚 ПОСЛЕДНИЕ ОШИБКИ:")
    recent_errors = error_handler.get_recent_errors(3)
    for i, error in enumerate(recent_errors, 1):
        print(f"  {i}. [{error.category.name}] {error.message}")
    
    # Экспорт отчета
    report_file = "error_report.txt"
    if error_handler.export_error_report(report_file):
        print(f"\n📄 Отчет экспортирован в {report_file}")
    
    print("\n✅ Демонстрация завершена") 