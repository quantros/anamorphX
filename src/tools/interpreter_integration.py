"""
Интеграция IDE с интерпретатором AnamorphX

Обеспечивает:
- Реальное выполнение кода Anamorph из IDE
- Автоматическое профилирование
- Встроенную отладку
- Асинхронное выполнение
- Управление сессиями
"""

import asyncio
import threading
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Импорты компонентов AnamorphX
from ..lexer.lexer import AnamorphLexer
from ..parser.parser import AnamorphParser
from ..semantic.analyzer import SemanticAnalyzer
from ..interpreter.ast_interpreter import ASTInterpreter
from ..codegen.python_codegen import PythonCodeGenerator

# Импорты инструментов разработки
from .debugger import AnamorphDebugger, DebugState
from .profiler import AnamorphProfiler, PerformanceAnalyzer


class ExecutionState(Enum):
    """Состояния выполнения"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Режимы выполнения"""
    INTERPRET = "interpret"      # Прямая интерпретация
    COMPILE_PYTHON = "compile_python"  # Компиляция в Python
    COMPILE_JS = "compile_js"    # Компиляция в JavaScript
    DEBUG = "debug"              # Отладка
    PROFILE = "profile"          # Профилирование


@dataclass
class ExecutionResult:
    """Результат выполнения"""
    success: bool
    execution_time: float
    output: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    return_value: Any = None
    profiling_data: Optional[Dict] = None
    debug_session: Optional[Dict] = None


@dataclass
class ExecutionContext:
    """Контекст выполнения"""
    file_path: str
    code: str
    mode: ExecutionMode
    debug_enabled: bool = False
    profiling_enabled: bool = False
    async_execution: bool = True
    timeout: Optional[float] = None
    environment: Dict[str, Any] = field(default_factory=dict)


class CodeExecutor:
    """Выполнитель кода Anamorph"""
    
    def __init__(self):
        # Компоненты компилятора
        self.lexer = AnamorphLexer()
        self.parser = AnamorphParser()
        self.semantic_analyzer = SemanticAnalyzer()
        self.interpreter = ASTInterpreter()
        self.python_codegen = PythonCodeGenerator()
        
        # Инструменты разработки
        self.debugger: Optional[AnamorphDebugger] = None
        self.profiler: Optional[AnamorphProfiler] = None
        
        # Состояние выполнения
        self.state = ExecutionState.IDLE
        self.current_execution: Optional[threading.Thread] = None
        self.execution_lock = threading.Lock()
        
        # Захват вывода
        self.output_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
        
        # Обработчики событий
        self.on_state_change: Optional[Callable[[ExecutionState], None]] = None
        self.on_output: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_completion: Optional[Callable[[ExecutionResult], None]] = None
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Выполнить код"""
        if self.state == ExecutionState.RUNNING:
            return ExecutionResult(
                success=False,
                execution_time=0.0,
                output="",
                errors=["Уже выполняется другая задача"]
            )
        
        with self.execution_lock:
            if context.async_execution:
                return self._execute_async(context)
            else:
                return self._execute_sync(context)
    
    def _execute_async(self, context: ExecutionContext) -> ExecutionResult:
        """Асинхронное выполнение"""
        result_container = {'result': None}
        
        def execution_thread():
            result_container['result'] = self._execute_sync(context)
        
        self.current_execution = threading.Thread(target=execution_thread)
        self.current_execution.daemon = True
        self.current_execution.start()
        
        # Возвращаем промежуточный результат
        return ExecutionResult(
            success=True,
            execution_time=0.0,
            output="Выполнение начато...",
            return_value="async_started"
        )
    
    def _execute_sync(self, context: ExecutionContext) -> ExecutionResult:
        """Синхронное выполнение"""
        start_time = time.perf_counter()
        
        try:
            self._set_state(ExecutionState.RUNNING)
            
            # Подготовка инструментов
            if context.debug_enabled:
                self.debugger = AnamorphDebugger(self.interpreter)
                self.debugger.start_debugging(context.code, context.file_path)
            
            if context.profiling_enabled:
                self.profiler = AnamorphProfiler()
                self.profiler.start_session("execution")
            
            # Выполнение по режиму
            if context.mode == ExecutionMode.INTERPRET:
                result = self._interpret_code(context)
            elif context.mode == ExecutionMode.COMPILE_PYTHON:
                result = self._compile_to_python(context)
            elif context.mode == ExecutionMode.COMPILE_JS:
                result = self._compile_to_javascript(context)
            elif context.mode == ExecutionMode.DEBUG:
                result = self._debug_code(context)
            elif context.mode == ExecutionMode.PROFILE:
                result = self._profile_code(context)
            else:
                raise ValueError(f"Неподдерживаемый режим: {context.mode}")
            
            # Завершение инструментов
            if self.profiler:
                profiling_data = self.profiler.stop_session("execution")
                result.profiling_data = profiling_data
            
            if self.debugger:
                debug_session = self.debugger.export_debug_session()
                result.debug_session = debug_session
                self.debugger.stop()
            
            execution_time = time.perf_counter() - start_time
            result.execution_time = execution_time
            
            self._set_state(ExecutionState.COMPLETED)
            
            # Уведомление о завершении
            if self.on_completion:
                self.on_completion(result)
            
            return result
            
        except Exception as e:
            self._set_state(ExecutionState.ERROR)
            
            error_msg = f"Ошибка выполнения: {e}"
            self._emit_error(error_msg)
            
            return ExecutionResult(
                success=False,
                execution_time=time.perf_counter() - start_time,
                output=self.output_buffer.getvalue(),
                errors=[error_msg, traceback.format_exc()]
            )
    
    def _interpret_code(self, context: ExecutionContext) -> ExecutionResult:
        """Интерпретация кода"""
        try:
            # Лексический анализ
            tokens = self.lexer.tokenize(context.code)
            
            # Синтаксический анализ
            ast = self.parser.parse(tokens)
            
            # Семантический анализ
            self.semantic_analyzer.analyze(ast)
            
            # Захват вывода
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                sys.stdout = self.output_buffer
                sys.stderr = self.error_buffer
                
                # Интерпретация
                result_value = self.interpreter.interpret(ast)
                
                output = self.output_buffer.getvalue()
                errors = self.error_buffer.getvalue()
                
                # Уведомление о выводе
                if output and self.on_output:
                    self.on_output(output)
                
                if errors and self.on_error:
                    self.on_error(errors)
                
                return ExecutionResult(
                    success=True,
                    execution_time=0.0,  # Будет установлено позже
                    output=output,
                    errors=[errors] if errors else [],
                    return_value=result_value
                )
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                execution_time=0.0,
                output=self.output_buffer.getvalue(),
                errors=[str(e), traceback.format_exc()]
            )
    
    def _compile_to_python(self, context: ExecutionContext) -> ExecutionResult:
        """Компиляция в Python"""
        try:
            # Лексический и синтаксический анализ
            tokens = self.lexer.tokenize(context.code)
            ast = self.parser.parse(tokens)
            
            # Семантический анализ
            self.semantic_analyzer.analyze(ast)
            
            # Генерация Python кода
            python_code = self.python_codegen.generate(ast)
            
            # Выполнение сгенерированного кода
            local_env = context.environment.copy()
            global_env = {}
            
            # Захват вывода
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                sys.stdout = self.output_buffer
                sys.stderr = self.error_buffer
                
                exec(python_code, global_env, local_env)
                
                output = self.output_buffer.getvalue()
                errors = self.error_buffer.getvalue()
                
                return ExecutionResult(
                    success=True,
                    execution_time=0.0,
                    output=output,
                    errors=[errors] if errors else [],
                    return_value=local_env.get('__result__')
                )
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                execution_time=0.0,
                output=self.output_buffer.getvalue(),
                errors=[str(e), traceback.format_exc()]
            )
    
    def _compile_to_javascript(self, context: ExecutionContext) -> ExecutionResult:
        """Компиляция в JavaScript"""
        # TODO: Реализовать после создания JS генератора
        return ExecutionResult(
            success=False,
            execution_time=0.0,
            output="",
            errors=["JavaScript генератор еще не реализован"]
        )
    
    def _debug_code(self, context: ExecutionContext) -> ExecutionResult:
        """Отладка кода"""
        context.debug_enabled = True
        return self._interpret_code(context)
    
    def _profile_code(self, context: ExecutionContext) -> ExecutionResult:
        """Профилирование кода"""
        context.profiling_enabled = True
        return self._interpret_code(context)
    
    def _set_state(self, new_state: ExecutionState):
        """Установить состояние выполнения"""
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(new_state)
    
    def _emit_output(self, output: str):
        """Отправить вывод"""
        if self.on_output:
            self.on_output(output)
    
    def _emit_error(self, error: str):
        """Отправить ошибку"""
        if self.on_error:
            self.on_error(error)
    
    def stop_execution(self):
        """Остановить выполнение"""
        if self.current_execution and self.current_execution.is_alive():
            # Python не позволяет принудительно остановить поток
            # Но мы можем установить флаг для проверки в коде
            self._set_state(ExecutionState.CANCELLED)
            
            # Остановить отладчик
            if self.debugger:
                self.debugger.stop()
            
            # Остановить профайлер
            if self.profiler:
                self.profiler.stop_session("execution")
    
    def pause_execution(self):
        """Приостановить выполнение"""
        if self.debugger and self.state == ExecutionState.RUNNING:
            self.debugger.pause()
            self._set_state(ExecutionState.PAUSED)
    
    def resume_execution(self):
        """Продолжить выполнение"""
        if self.debugger and self.state == ExecutionState.PAUSED:
            self.debugger.resume()
            self._set_state(ExecutionState.RUNNING)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Получить статус выполнения"""
        return {
            'state': self.state.value,
            'has_debugger': self.debugger is not None,
            'has_profiler': self.profiler is not None,
            'thread_alive': self.current_execution.is_alive() if self.current_execution else False
        }


class IDEIntegration:
    """Интеграция IDE с исполнительной системой"""
    
    def __init__(self, ide_components):
        self.ide = ide_components
        self.executor = CodeExecutor()
        
        # Настройка обработчиков
        self.executor.on_state_change = self._on_execution_state_change
        self.executor.on_output = self._on_execution_output
        self.executor.on_error = self._on_execution_error
        self.executor.on_completion = self._on_execution_completion
        
        # История выполнения
        self.execution_history: List[ExecutionResult] = []
        
        # Активные сессии
        self.active_sessions: Dict[str, ExecutionContext] = {}
    
    def execute_current_file(self, mode: ExecutionMode = ExecutionMode.INTERPRET):
        """Выполнить текущий файл"""
        if not hasattr(self.ide, 'code_editor'):
            return
        
        # Получение кода из редактора
        code = self.ide.code_editor.get_content()
        file_path = self.ide.code_editor.current_file or "untitled.amph"
        
        # Создание контекста выполнения
        context = ExecutionContext(
            file_path=file_path,
            code=code,
            mode=mode,
            debug_enabled=(mode == ExecutionMode.DEBUG),
            profiling_enabled=(mode == ExecutionMode.PROFILE),
            async_execution=True
        )
        
        # Запуск выполнения
        result = self.executor.execute(context)
        
        # Сохранение в историю
        if result.success or result.return_value != "async_started":
            self.execution_history.append(result)
            
            # Ограничиваем историю
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
    
    def _on_execution_state_change(self, state: ExecutionState):
        """Обработка изменения состояния выполнения"""
        # Обновление UI
        if hasattr(self.ide, 'status_label'):
            status_messages = {
                ExecutionState.IDLE: "Готов",
                ExecutionState.RUNNING: "⚡ Выполнение...",
                ExecutionState.PAUSED: "⏸️ Приостановлено",
                ExecutionState.COMPLETED: "✅ Выполнено",
                ExecutionState.ERROR: "❌ Ошибка",
                ExecutionState.CANCELLED: "🛑 Отменено"
            }
            self.ide.status_label.config(text=status_messages.get(state, ""))
        
        # Обновление кнопок в toolbar
        if hasattr(self.ide, 'toolbar'):
            self._update_toolbar_buttons(state)
    
    def _on_execution_output(self, output: str):
        """Обработка вывода выполнения"""
        if hasattr(self.ide, 'output_text'):
            self.ide.output_text.config(state='normal')
            self.ide.output_text.insert('end', output)
            self.ide.output_text.config(state='disabled')
            self.ide.output_text.see('end')
    
    def _on_execution_error(self, error: str):
        """Обработка ошибок выполнения"""
        if hasattr(self.ide, 'output_text'):
            self.ide.output_text.config(state='normal')
            self.ide.output_text.insert('end', f"❌ {error}\n", 'error')
            self.ide.output_text.config(state='disabled')
            self.ide.output_text.see('end')
    
    def _on_execution_completion(self, result: ExecutionResult):
        """Обработка завершения выполнения"""
        # Отображение результата
        if result.success:
            message = f"✅ Выполнение завершено за {result.execution_time:.3f}s"
            if result.return_value is not None:
                message += f"\nРезультат: {result.return_value}"
        else:
            message = f"❌ Выполнение завершено с ошибками"
            for error in result.errors:
                message += f"\n{error}"
        
        if hasattr(self.ide, 'output_text'):
            self.ide.output_text.config(state='normal')
            self.ide.output_text.insert('end', f"{message}\n")
            self.ide.output_text.config(state='disabled')
            self.ide.output_text.see('end')
        
        # Отображение данных профилирования
        if result.profiling_data:
            self._display_profiling_data(result.profiling_data)
        
        # Отображение сессии отладки
        if result.debug_session:
            self._display_debug_session(result.debug_session)
    
    def _update_toolbar_buttons(self, state: ExecutionState):
        """Обновление кнопок панели инструментов"""
        # TODO: Обновить состояние кнопок в зависимости от состояния выполнения
        pass
    
    def _display_profiling_data(self, profiling_data: Dict):
        """Отображение данных профилирования"""
        # TODO: Создать вкладку с данными профилирования
        pass
    
    def _display_debug_session(self, debug_session: Dict):
        """Отображение сессии отладки"""
        # TODO: Обновить панель отладки
        pass
    
    # Методы для интеграции с IDE
    def setup_ide_integration(self):
        """Настройка интеграции с IDE"""
        # Перехват команд выполнения
        if hasattr(self.ide, '_run_program'):
            original_run = self.ide._run_program
            
            def new_run_program():
                self.execute_current_file(ExecutionMode.INTERPRET)
            
            self.ide._run_program = new_run_program
        
        # Перехват команд отладки
        if hasattr(self.ide, '_debug_program'):
            original_debug = self.ide._debug_program
            
            def new_debug_program():
                self.execute_current_file(ExecutionMode.DEBUG)
            
            self.ide._debug_program = new_debug_program
    
    def get_execution_history(self) -> List[Dict]:
        """Получить историю выполнения"""
        return [
            {
                'success': result.success,
                'execution_time': result.execution_time,
                'output_length': len(result.output),
                'error_count': len(result.errors),
                'has_profiling': result.profiling_data is not None,
                'has_debug': result.debug_session is not None
            }
            for result in self.execution_history
        ]


# Функция для создания интеграции
def create_ide_integration(ide_components) -> IDEIntegration:
    """Создать интеграцию IDE с интерпретатором"""
    integration = IDEIntegration(ide_components)
    integration.setup_ide_integration()
    return integration


if __name__ == "__main__":
    # Тестирование интеграции
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
    
    test_neuron.activate(input_signal)
    '''
    
    # Создание и тестирование исполнителя
    executor = CodeExecutor()
    
    print("🔗 Тестирование интеграции с интерпретатором...")
    
    # Тестирование различных режимов
    modes = [
        ExecutionMode.INTERPRET,
        ExecutionMode.COMPILE_PYTHON,
        ExecutionMode.DEBUG,
        ExecutionMode.PROFILE
    ]
    
    for mode in modes:
        print(f"\n🎯 Тестирование режима: {mode.value}")
        
        context = ExecutionContext(
            file_path="test.amph",
            code=test_code,
            mode=mode,
            async_execution=False  # Синхронно для тестирования
        )
        
        result = executor.execute(context)
        
        print(f"  Успех: {result.success}")
        print(f"  Время: {result.execution_time:.3f}s")
        print(f"  Вывод: {len(result.output)} символов")
        print(f"  Ошибки: {len(result.errors)}")
        
        if result.profiling_data:
            print(f"  Профилирование: {len(result.profiling_data)} метрик")
        
        if result.debug_session:
            print(f"  Отладка: {len(result.debug_session.get('event_history', []))} событий")
    
    print("\n✅ Тестирование интеграции завершено") 