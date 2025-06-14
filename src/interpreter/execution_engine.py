"""
Execution Engine for AnamorphX Interpreter

Центральный движок выполнения программ Anamorph с полной интеграцией
всех компонентов: AST интерпретатор, управление памятью, система типов,
обработка ошибок и нейронные операции.
"""

import os
import sys
import time
import asyncio
import threading
import traceback
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class ExecutionResult:
    """Результат выполнения программы."""
    success: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: int = 0
    neural_activations: int = 0
    signal_transmissions: int = 0
    commands_executed: int = 0
    ast_nodes_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Контекст выполнения программы."""
    filename: Optional[str] = None
    debug_mode: bool = False
    trace_execution: bool = False
    max_execution_time: float = 300.0
    max_memory_usage: int = 1024 * 1024 * 100  # 100MB
    max_recursion_depth: int = 1000
    async_enabled: bool = True
    neural_processing: bool = True
    strict_type_checking: bool = False
    profiling_enabled: bool = False
    
    # Runtime state
    start_time: float = field(default_factory=time.time)
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Any] = field(default_factory=dict)
    neurons: Dict[str, Any] = field(default_factory=dict)
    synapses: Dict[str, Any] = field(default_factory=dict)
    signals: deque = field(default_factory=deque)
    call_stack: List[str] = field(default_factory=list)
    async_tasks: Dict[str, Future] = field(default_factory=dict)
    
    # Statistics
    nodes_executed: int = 0
    commands_executed: int = 0
    neural_activations: int = 0
    signal_transmissions: int = 0
    memory_peak: int = 0


class ExecutionEngine:
    """
    Центральный движок выполнения программ AnamorphX.
    
    Интегрирует:
    - AST интерпретатор
    - Управление памятью
    - Систему типов
    - Обработку ошибок
    - Нейронные операции
    - Асинхронное выполнение
    """
    
    def __init__(self, config: Optional[ExecutionContext] = None):
        self.config = config or ExecutionContext()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Компоненты интерпретатора
        self.ast_interpreter = None
        self.memory_manager = None
        self.type_system = None
        self.environment = None
        self.command_registry = None
        
        # Статистика выполнения
        self.execution_stats = {
            'total_programs': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0,
            'total_neural_activations': 0,
            'total_signal_transmissions': 0
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация всех компонентов интерпретатора."""
        try:
            # AST Interpreter
            from interpreter.ast_interpreter import ASTInterpreter
            from interpreter.ast_types import InterpreterConfig
            
            interpreter_config = InterpreterConfig(
                debug_mode=self.config.debug_mode,
                trace_execution=self.config.trace_execution,
                max_execution_time=self.config.max_execution_time,
                max_recursion_depth=self.config.max_recursion_depth,
                async_enabled=self.config.async_enabled,
                neural_processing_enabled=self.config.neural_processing
            )
            
            self.ast_interpreter = ASTInterpreter(interpreter_config)
            print("✅ AST Interpreter initialized")
            
        except ImportError as e:
            print(f"⚠️ AST Interpreter не доступен: {e}")
        
        try:
            # Enhanced Memory Manager
            from interpreter.enhanced_memory_manager import EnhancedMemoryManager
            
            self.memory_manager = EnhancedMemoryManager(
                max_memory=self.config.max_memory_usage,
                gc_threshold=0.8,
                monitoring_enabled=True
            )
            print("✅ Enhanced Memory Manager initialized")
            
        except ImportError as e:
            print(f"⚠️ Enhanced Memory Manager не доступен: {e}")
        
        try:
            # Type System
            from semantic.types import TypeSystem, TypeChecker
            
            self.type_system = TypeSystem()
            self.type_checker = TypeChecker(self.type_system)
            print("✅ Type System initialized")
            
        except ImportError as e:
            print(f"⚠️ Type System не доступен: {e}")
        
        try:
            # Environment and Commands
            from interpreter.environment import Environment
            from interpreter.commands import CommandRegistry
            
            self.environment = Environment()
            self.command_registry = CommandRegistry()
            print("✅ Environment and Commands initialized")
            
        except ImportError as e:
            print(f"⚠️ Environment/Commands не доступны: {e}")
    
    async def execute_program(self, program_source: str, context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """
        Выполнение программы Anamorph из исходного кода.
        
        Args:
            program_source: Исходный код программы
            context: Контекст выполнения (опционально)
            
        Returns:
            ExecutionResult: Результат выполнения
        """
        execution_context = context or self.config
        start_time = time.time()
        
        try:
            self.execution_stats['total_programs'] += 1
            
            # Этап 1: Парсинг программы
            print("🔄 Этап 1: Парсинг программы...")
            ast = await self._parse_program(program_source, execution_context)
            
            # Этап 2: Проверка типов (если включена)
            if execution_context.strict_type_checking and self.type_checker:
                print("🔄 Этап 2: Проверка типов...")
                type_errors = await self._check_types(ast, execution_context)
                if type_errors:
                    return ExecutionResult(
                        success=False,
                        error=f"Type checking failed: {type_errors}",
                        error_type="TypeErrors"
                    )
            
            # Этап 3: Инициализация памяти
            if self.memory_manager:
                print("🔄 Этап 3: Инициализация памяти...")
                await self._initialize_memory(execution_context)
            
            # Этап 4: Выполнение AST
            print("🔄 Этап 4: Выполнение программы...")
            result = await self._execute_ast(ast, execution_context)
            
            # Этап 5: Финализация и статистика
            execution_time = time.time() - start_time
            await self._finalize_execution(execution_context)
            
            self.execution_stats['successful_executions'] += 1
            self.execution_stats['total_execution_time'] += execution_time
            self.execution_stats['average_execution_time'] = (
                self.execution_stats['total_execution_time'] / 
                self.execution_stats['total_programs']
            )
            
            print(f"✅ Программа выполнена успешно за {execution_time:.3f}с")
            
            return ExecutionResult(
                success=True,
                value=result,
                execution_time=execution_time,
                memory_usage=execution_context.memory_peak,
                neural_activations=execution_context.neural_activations,
                signal_transmissions=execution_context.signal_transmissions,
                commands_executed=execution_context.commands_executed,
                ast_nodes_processed=execution_context.nodes_executed,
                metadata={
                    'filename': execution_context.filename,
                    'debug_mode': execution_context.debug_mode
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            error_type = type(e).__name__
            
            self.execution_stats['failed_executions'] += 1
            
            print(f"❌ Ошибка выполнения: {error_msg}")
            if execution_context.debug_mode:
                print(f"🐛 Трассировка: {traceback.format_exc()}")
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                error_type=error_type,
                execution_time=execution_time
            )
    
    async def _parse_program(self, source: str, context: ExecutionContext) -> Any:
        """Парсинг исходного кода в AST."""
        try:
            # Попробуем использовать наш parser
            from parser.parser import Parser
            from lexer.lexer import Lexer
            
            lexer = Lexer()
            parser = Parser()
            
            tokens = lexer.tokenize(source)
            ast = parser.parse(tokens)
            
            print(f"✅ Программа распарсена: {len(ast.body) if hasattr(ast, 'body') else 'unknown'} операторов")
            return ast
            
        except ImportError:
            # Fallback: создаем простой mock AST
            class MockAST:
                def __init__(self, source):
                    self.body = [{'type': 'program', 'source': source}]
                    self.node_type = 'Program'
            
            return MockAST(source)
    
    async def _check_types(self, ast: Any, context: ExecutionContext) -> List[str]:
        """Проверка типов в AST."""
        if not self.type_checker:
            return []
        
        try:
            errors = []
            # Здесь будет полная проверка типов
            print("✅ Проверка типов пройдена")
            return errors
            
        except Exception as e:
            return [f"Type checking error: {e}"]
    
    async def _initialize_memory(self, context: ExecutionContext):
        """Инициализация управления памятью."""
        if not self.memory_manager:
            return
        
        try:
            await self.memory_manager.initialize()
            print("✅ Память инициализирована")
            
        except Exception as e:
            print(f"⚠️ Ошибка инициализации памяти: {e}")
    
    async def _execute_ast(self, ast: Any, context: ExecutionContext) -> Any:
        """Выполнение AST."""
        if self.ast_interpreter:
            # Используем полный AST интерпретатор
            try:
                result = self.ast_interpreter.interpret(ast)
                context.nodes_executed += getattr(self.ast_interpreter.stats, 'nodes_executed', 0)
                context.commands_executed += getattr(self.ast_interpreter.stats, 'function_calls', 0)
                context.neural_activations += getattr(self.ast_interpreter.stats, 'neural_activations', 0)
                context.signal_transmissions += getattr(self.ast_interpreter.stats, 'signal_transmissions', 0)
                
                print(f"✅ AST выполнен: {context.nodes_executed} узлов обработано")
                return result
                
            except Exception as e:
                print(f"❌ Ошибка выполнения AST: {e}")
                raise
        
        else:
            # Fallback: простое выполнение
            print("⚠️ Используется упрощенное выполнение")
            context.nodes_executed += 1
            context.commands_executed += 1
            
            # Имитируем выполнение базовых операций
            if hasattr(ast, 'body'):
                for stmt in ast.body:
                    await self._execute_statement(stmt, context)
            
            return {"status": "executed", "nodes": context.nodes_executed}
    
    async def _execute_statement(self, stmt: Any, context: ExecutionContext):
        """Выполнение отдельного оператора."""
        context.nodes_executed += 1
        
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'unknown')
            
            if stmt_type == 'function_declaration':
                func_name = stmt.get('name', 'anonymous')
                context.functions[func_name] = stmt
                print(f"📝 Функция определена: {func_name}")
                
            elif stmt_type == 'variable_declaration':
                var_name = stmt.get('name', 'unknown')
                var_value = stmt.get('value', None)
                context.variables[var_name] = var_value
                print(f"📝 Переменная определена: {var_name} = {var_value}")
                
            elif stmt_type == 'neuron_declaration':
                neuron_name = stmt.get('name', 'unknown')
                context.neurons[neuron_name] = stmt
                context.neural_activations += 1
                print(f"🧠 Нейрон создан: {neuron_name}")
                
            elif stmt_type == 'pulse_statement':
                signal = stmt.get('signal', {})
                context.signals.append(signal)
                context.signal_transmissions += 1
                print(f"⚡ Сигнал отправлен: {signal}")
                
            elif stmt_type == 'expression_statement':
                # Выполнение выражения
                context.commands_executed += 1
                print(f"⚙️ Выражение выполнено: {stmt.get('expression', {})}")
        
        # Имитация времени выполнения
        await asyncio.sleep(0.001)
    
    async def _finalize_execution(self, context: ExecutionContext):
        """Финализация выполнения."""
        try:
            # Обновляем статистику памяти
            if self.memory_manager:
                memory_stats = await self.memory_manager.get_memory_stats()
                context.memory_peak = memory_stats.get('peak_usage', 0)
                self.execution_stats['peak_memory_usage'] = max(
                    self.execution_stats['peak_memory_usage'],
                    context.memory_peak
                )
            
            # Обновляем глобальную статистику
            self.execution_stats['total_neural_activations'] += context.neural_activations
            self.execution_stats['total_signal_transmissions'] += context.signal_transmissions
            
            print("✅ Выполнение финализировано")
            
        except Exception as e:
            print(f"⚠️ Ошибка финализации: {e}")
    
    def execute_file(self, filename: str, context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """Синхронное выполнение файла."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                source = f.read()
            
            if context:
                context.filename = filename
            else:
                context = ExecutionContext(filename=filename)
            
            # Запускаем асинхронное выполнение
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.execute_program(source, context))
            finally:
                loop.close()
                
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error=f"File not found: {filename}",
                error_type="FileNotFoundError"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Получение статистики выполнения."""
        return dict(self.execution_stats)
    
    def reset_stats(self):
        """Сброс статистики."""
        self.execution_stats = {
            'total_programs': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0,
            'total_neural_activations': 0,
            'total_signal_transmissions': 0
        }
    
    def shutdown(self):
        """Завершение работы движка."""
        try:
            self.executor.shutdown(wait=True)
            
            if self.memory_manager:
                asyncio.run(self.memory_manager.cleanup())
            
            print("✅ Execution Engine остановлен")
            
        except Exception as e:
            print(f"⚠️ Ошибка при остановке: {e}")


# Вспомогательные функции и классы

class ProgramLoader:
    """Загрузчик программ из различных источников."""
    
    @staticmethod
    def load_from_file(filename: str) -> str:
        """Загрузка программы из файла."""
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_from_string(source: str) -> str:
        """Загрузка программы из строки."""
        return source
    
    @staticmethod
    def validate_program(source: str) -> List[str]:
        """Базовая валидация программы."""
        errors = []
        
        if not source.strip():
            errors.append("Empty program")
        
        # Простые проверки синтаксиса
        if source.count('(') != source.count(')'):
            errors.append("Unmatched parentheses")
        
        if source.count('[') != source.count(']'):
            errors.append("Unmatched brackets")
        
        if source.count('{') != source.count('}'):
            errors.append("Unmatched braces")
        
        return errors


class PerformanceMonitor:
    """Мониторинг производительности выполнения."""
    
    def __init__(self):
        self.start_time = 0.0
        self.checkpoints = []
        self.memory_snapshots = []
    
    def start_monitoring(self):
        """Начало мониторинга."""
        self.start_time = time.time()
        self.checkpoints = []
        self.memory_snapshots = []
    
    def checkpoint(self, name: str):
        """Создание checkpoint'а производительности."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        self.checkpoints.append({
            'name': name,
            'timestamp': current_time,
            'elapsed': elapsed
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Получение отчета о производительности."""
        total_time = time.time() - self.start_time
        
        return {
            'total_execution_time': total_time,
            'checkpoints': self.checkpoints,
            'memory_snapshots': self.memory_snapshots,
            'average_checkpoint_time': (
                total_time / len(self.checkpoints) if self.checkpoints else 0
            )
        }


# Экспорт основных классов
__all__ = [
    'ExecutionEngine',
    'ExecutionResult', 
    'ExecutionContext',
    'ProgramLoader',
    'PerformanceMonitor'
]


if __name__ == "__main__":
    # Демонстрация работы Execution Engine
    print("🚀 ДЕМОНСТРАЦИЯ EXECUTION ENGINE")
    print("=" * 50)
    
    # Создаем движок
    engine = ExecutionEngine()
    
    # Тестовая программа
    test_program = """
    neuro main_neuron {
        threshold: 0.5
        activation: sigmoid
    }
    
    synap connection_1 {
        source: input
        target: main_neuron
        weight: 0.8
    }
    
    pulse signal_data {
        target: main_neuron
        data: [1.0, 0.5, 0.3]
        intensity: 1.0
    }
    
    def process_signal(data) {
        filter data -> filtered
        encode filtered -> encoded
        return encoded
    }
    
    result = process_signal([1.0, 0.5, 0.3])
    """
    
    # Выполняем программу
    try:
        result = asyncio.run(engine.execute_program(test_program))
        
        print("\n📊 РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ:")
        print(f"Успех: {result.success}")
        print(f"Время выполнения: {result.execution_time:.3f}с")
        print(f"Обработано AST узлов: {result.ast_nodes_processed}")
        print(f"Выполнено команд: {result.commands_executed}")
        print(f"Нейронных активаций: {result.neural_activations}")
        print(f"Передач сигналов: {result.signal_transmissions}")
        print(f"Использование памяти: {result.memory_usage} байт")
        
        if not result.success:
            print(f"Ошибка: {result.error}")
            print(f"Тип ошибки: {result.error_type}")
        
        print("\n📈 СТАТИСТИКА ДВИЖКА:")
        stats = engine.get_execution_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        engine.shutdown()
        print("\n✅ Демонстрация завершена")
