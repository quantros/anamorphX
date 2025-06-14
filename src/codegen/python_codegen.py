"""
Python Code Generator for AnamorphX

Генератор Python кода из AST программ Anamorph.
Поддерживает все конструкции языка включая нейронные операции.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from .base_codegen import (
    BaseCodeGenerator, CodeGenConfig, CodeGenResult, 
    TargetLanguage, OptimizationLevel, ASTVisitor
)


class PythonCodeGenerator(BaseCodeGenerator, ASTVisitor):
    """
    Генератор Python кода из AST Anamorph.
    
    Поддерживает:
    - Все базовые конструкции языка
    - Нейронные операции (neurons, synapses, signals, pulses)
    - Асинхронные операции
    - Оптимизации кода
    - Генерацию тестов
    """
    
    def __init__(self, config: CodeGenConfig = None):
        if config is None:
            config = CodeGenConfig(target_language=TargetLanguage.PYTHON)
        
        super().__init__(config)
        
        # Python-специфичные настройки
        self.python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'not', 'or', 'pass', 'print', 'raise', 'return', 'try',
            'while', 'with', 'yield', 'True', 'False', 'None'
        }
        
        # Маппинг типов Anamorph -> Python
        self.type_mapping = {
            'int': 'int',
            'float': 'float', 
            'string': 'str',
            'bool': 'bool',
            'list': 'List',
            'dict': 'Dict',
            'neuron': 'Neuron',
            'synapse': 'Synapse',
            'signal': 'Signal',
            'pulse': 'Pulse',
            'network': 'NeuralNetwork'
        }
        
        # Стандартные импорты для нейронных операций
        self.neural_imports = {
            'numpy': 'import numpy as np',
            'typing': 'from typing import List, Dict, Optional, Union, Any',
            'asyncio': 'import asyncio',
            'dataclasses': 'from dataclasses import dataclass, field',
            'abc': 'from abc import ABC, abstractmethod'
        }
    
    def get_target_language(self) -> TargetLanguage:
        """Возвращает целевой язык - Python."""
        return TargetLanguage.PYTHON
    
    def generate_program(self, ast) -> CodeGenResult:
        """Генерирует Python код для всей программы."""
        start_time = time.time()
        
        try:
            # Очищаем предыдущий код
            self.clear_generated_code()
            
            # Добавляем стандартные импорты
            self._generate_standard_imports()
            
            # Генерируем нейронные классы
            self._generate_neural_classes()
            
            # Обрабатываем AST
            if hasattr(ast, 'body') and ast.body:
                for node in ast.body:
                    self.visit(node)
                    self.stats['nodes_processed'] += 1
            else:
                # Простая демонстрация если AST пустой
                self._generate_demo_program()
            
            # Генерируем main функцию если нужно
            if self.config.generate_tests:
                self._generate_test_main()
            
            return self.create_result(start_time)
            
        except Exception as e:
            self.add_error(f"Code generation failed: {e}")
            return self.create_result(start_time)
    
    def _generate_standard_imports(self):
        """Генерирует стандартные импорты."""
        self.emit_line("#!/usr/bin/env python3")
        self.emit_line('"""')
        self.emit_line("Generated Anamorph Program")
        self.emit_line(f"Target: Python {self.config.python_version}")
        self.emit_line('"""')
        self.emit_blank_line()
        
        # Базовые импорты
        for import_stmt in self.neural_imports.values():
            self.emit_line(import_stmt)
        
        # Дополнительные импорты
        if self.config.neural_library == 'numpy':
            self.emit_line("import numpy as np")
        elif self.config.neural_library == 'tensorflow':
            self.emit_line("import tensorflow as tf")
        elif self.config.neural_library == 'pytorch':
            self.emit_line("import torch")
            self.emit_line("import torch.nn as nn")
        
        # Пользовательские импорты
        for custom_import in self.config.custom_imports:
            self.emit_line(custom_import)
        
        self.emit_blank_line()
    
    def _generate_neural_classes(self):
        """Генерирует базовые нейронные классы."""
        self.emit_line("# Neural Network Base Classes")
        self.emit_blank_line()
        
        # Класс Neuron
        self.emit_line("@dataclass")
        self.emit_line("class Neuron:")
        self.increase_indent()
        self.emit_line('"""Базовый класс нейрона."""')
        self.emit_line("id: str")
        self.emit_line("activation: float = 0.0")
        self.emit_line("threshold: float = 1.0")
        self.emit_line("connections: List['Synapse'] = field(default_factory=list)")
        self.emit_blank_line()
        
        self.emit_line("def activate(self, input_signal: float) -> float:")
        self.increase_indent()
        self.emit_line('"""Активирует нейрон."""')
        self.emit_line("self.activation = input_signal")
        self.emit_line("return self.activation if self.activation >= self.threshold else 0.0")
        self.decrease_indent()
        
        self.emit_line("def fire(self) -> bool:")
        self.increase_indent()
        self.emit_line('"""Проверяет, должен ли нейрон сработать."""')
        self.emit_line("return self.activation >= self.threshold")
        self.decrease_indent()
        self.decrease_indent()
        self.emit_blank_line()
        
        # Класс Synapse
        self.emit_line("@dataclass")
        self.emit_line("class Synapse:")
        self.increase_indent()
        self.emit_line('"""Синапс между нейронами."""')
        self.emit_line("source: Neuron")
        self.emit_line("target: Neuron")
        self.emit_line("weight: float = 1.0")
        self.emit_line("delay: float = 0.0")
        self.emit_blank_line()
        
        self.emit_line("def transmit(self, signal: float) -> float:")
        self.increase_indent()
        self.emit_line('"""Передает сигнал через синапс."""')
        self.emit_line("return signal * self.weight")
        self.decrease_indent()
        self.decrease_indent()
        self.emit_blank_line()
        
        # Класс Signal
        self.emit_line("@dataclass")
        self.emit_line("class Signal:")
        self.increase_indent()
        self.emit_line('"""Сигнал в нейронной сети."""')
        self.emit_line("value: float")
        self.emit_line("timestamp: float = 0.0")
        self.emit_line("source_id: Optional[str] = None")
        self.emit_line("metadata: Dict[str, Any] = field(default_factory=dict)")
        self.decrease_indent()
        self.emit_blank_line()
        
        # Класс Pulse
        self.emit_line("@dataclass")
        self.emit_line("class Pulse:")
        self.increase_indent()
        self.emit_line('"""Импульс в нейронной сети."""')
        self.emit_line("amplitude: float")
        self.emit_line("frequency: float")
        self.emit_line("duration: float")
        self.emit_line("phase: float = 0.0")
        self.decrease_indent()
        self.emit_blank_line()
        
        # Класс NeuralNetwork
        self.emit_line("class NeuralNetwork:")
        self.increase_indent()
        self.emit_line('"""Нейронная сеть."""')
        self.emit_blank_line()
        
        self.emit_line("def __init__(self):")
        self.increase_indent()
        self.emit_line("self.neurons: Dict[str, Neuron] = {}")
        self.emit_line("self.synapses: List[Synapse] = []")
        self.emit_line("self.signals: List[Signal] = []")
        self.decrease_indent()
        self.emit_blank_line()
        
        self.emit_line("def add_neuron(self, neuron: Neuron) -> None:")
        self.increase_indent()
        self.emit_line('"""Добавляет нейрон в сеть."""')
        self.emit_line("self.neurons[neuron.id] = neuron")
        self.decrease_indent()
        self.emit_blank_line()
        
        self.emit_line("def connect(self, source_id: str, target_id: str, weight: float = 1.0) -> None:")
        self.increase_indent()
        self.emit_line('"""Соединяет два нейрона."""')
        self.emit_line("if source_id in self.neurons and target_id in self.neurons:")
        self.increase_indent()
        self.emit_line("synapse = Synapse(")
        self.increase_indent()
        self.emit_line("source=self.neurons[source_id],")
        self.emit_line("target=self.neurons[target_id],")
        self.emit_line("weight=weight")
        self.decrease_indent()
        self.emit_line(")")
        self.emit_line("self.synapses.append(synapse)")
        self.emit_line("self.neurons[source_id].connections.append(synapse)")
        self.decrease_indent()
        self.decrease_indent()
        self.emit_blank_line()
        
        self.emit_line("async def process_signal(self, signal: Signal) -> List[Signal]:")
        self.increase_indent()
        self.emit_line('"""Обрабатывает сигнал в сети."""')
        self.emit_line("output_signals = []")
        self.emit_line("# Логика обработки сигнала")
        self.emit_line("return output_signals")
        self.decrease_indent()
        self.decrease_indent()
        self.emit_blank_line()
    
    def _generate_demo_program(self):
        """Генерирует демонстрационную программу."""
        self.emit_line("# Demo Anamorph Program")
        self.emit_blank_line()
        
        self.emit_line("def create_simple_network() -> NeuralNetwork:")
        self.increase_indent()
        self.emit_line('"""Создает простую нейронную сеть."""')
        self.emit_line("network = NeuralNetwork()")
        self.emit_blank_line()
        
        self.emit_line("# Создаем нейроны")
        self.emit_line('input_neuron = Neuron(id="input", threshold=0.5)')
        self.emit_line('hidden_neuron = Neuron(id="hidden", threshold=0.7)')
        self.emit_line('output_neuron = Neuron(id="output", threshold=0.8)')
        self.emit_blank_line()
        
        self.emit_line("# Добавляем в сеть")
        self.emit_line("network.add_neuron(input_neuron)")
        self.emit_line("network.add_neuron(hidden_neuron)")
        self.emit_line("network.add_neuron(output_neuron)")
        self.emit_blank_line()
        
        self.emit_line("# Создаем соединения")
        self.emit_line('network.connect("input", "hidden", weight=0.8)')
        self.emit_line('network.connect("hidden", "output", weight=0.9)')
        self.emit_blank_line()
        
        self.emit_line("return network")
        self.decrease_indent()
        self.emit_blank_line()
        
        self.stats['functions_generated'] += 1
    
    def _generate_test_main(self):
        """Генерирует main функцию для тестирования."""
        self.emit_line("async def main():")
        self.increase_indent()
        self.emit_line('"""Главная функция для тестирования."""')
        self.emit_line("print('AnamorphX Neural Network Demo')")
        self.emit_line("print('=' * 40)")
        self.emit_blank_line()
        
        self.emit_line("# Создаем сеть")
        self.emit_line("network = create_simple_network()")
        self.emit_line(f'print(f"Created network with {{len(network.neurons)}} neurons")')
        self.emit_blank_line()
        
        self.emit_line("# Тестируем активацию")
        self.emit_line('input_neuron = network.neurons["input"]')
        self.emit_line("activation = input_neuron.activate(1.0)")
        self.emit_line(f'print(f"Input neuron activation: {{activation}}")')
        self.emit_blank_line()
        
        self.emit_line("# Создаем и обрабатываем сигнал")
        self.emit_line("signal = Signal(value=1.5, timestamp=time.time())")
        self.emit_line("output_signals = await network.process_signal(signal)")
        self.emit_line(f'print(f"Processed signal, got {{len(output_signals)}} outputs")')
        self.emit_blank_line()
        
        self.emit_line("print('Demo completed successfully!')")
        self.decrease_indent()
        self.emit_blank_line()
        
        self.emit_line("if __name__ == '__main__':")
        self.increase_indent()
        self.emit_line("asyncio.run(main())")
        self.decrease_indent()
        
        self.stats['functions_generated'] += 1
    
    # Методы для генерации различных узлов AST
    
    def generate_function(self, node) -> str:
        """Генерирует Python функцию."""
        func_name = getattr(node, 'name', 'unnamed_function')
        
        # Проверяем на конфликт с ключевыми словами
        if func_name in self.python_keywords:
            func_name = f"anamorph_{func_name}"
            self.add_warning(f"Function name conflicts with Python keyword, renamed to {func_name}")
        
        # Регистрируем функцию
        self.register_symbol(func_name, 'function')
        
        # Генерируем сигнатуру
        params = getattr(node, 'parameters', [])
        param_list = []
        
        for param in params:
            param_name = getattr(param, 'name', 'param')
            param_type = getattr(param, 'type', None)
            
            if param_type and param_type in self.type_mapping:
                param_list.append(f"{param_name}: {self.type_mapping[param_type]}")
            else:
                param_list.append(param_name)
        
        # Определяем асинхронность
        is_async = getattr(node, 'is_async', False) or self.config.async_support
        async_keyword = "async " if is_async else ""
        
        # Генерируем функцию
        signature = f"{async_keyword}def {func_name}({', '.join(param_list)}):"
        self.emit_line(signature)
        
        self.increase_indent()
        
        # Добавляем docstring если есть
        if hasattr(node, 'docstring') and node.docstring:
            self.emit_line(f'"""{node.docstring}"""')
        
        # Генерируем тело функции
        if hasattr(node, 'body') and node.body:
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.emit_line("pass")
        
        self.decrease_indent()
        self.emit_blank_line()
        
        self.stats['functions_generated'] += 1
        return signature
    
    def generate_variable(self, node) -> str:
        """Генерирует объявление переменной."""
        var_name = getattr(node, 'name', 'unnamed_var')
        var_type = getattr(node, 'type', None)
        var_value = getattr(node, 'value', None)
        
        # Проверяем на конфликт с ключевыми словами
        if var_name in self.python_keywords:
            var_name = f"anamorph_{var_name}"
            self.add_warning(f"Variable name conflicts with Python keyword, renamed to {var_name}")
        
        # Регистрируем переменную
        self.register_symbol(var_name, var_type or 'unknown')
        
        # Генерируем объявление
        if var_value is not None:
            value_code = self.generate_expression(var_value)
            declaration = f"{var_name} = {value_code}"
        else:
            # Значение по умолчанию в зависимости от типа
            default_values = {
                'int': '0',
                'float': '0.0',
                'string': '""',
                'bool': 'False',
                'list': '[]',
                'dict': '{}',
                'neuron': 'Neuron(id="default")',
                'signal': 'Signal(value=0.0)'
            }
            default_value = default_values.get(var_type, 'None')
            declaration = f"{var_name} = {default_value}"
        
        self.emit_line(declaration)
        self.stats['variables_declared'] += 1
        return declaration
    
    def generate_expression(self, node) -> str:
        """Генерирует выражение."""
        if node is None:
            return "None"
        
        # Простые значения
        if isinstance(node, (int, float, str, bool)):
            if isinstance(node, str):
                return f'"{node}"'
            return str(node)
        
        # Если это объект с атрибутами
        if hasattr(node, 'type'):
            expr_type = node.type
            
            if expr_type == 'literal':
                value = getattr(node, 'value', None)
                if isinstance(value, str):
                    return f'"{value}"'
                return str(value)
            
            elif expr_type == 'identifier':
                return getattr(node, 'name', 'unknown')
            
            elif expr_type == 'binary_op':
                left = self.generate_expression(getattr(node, 'left', None))
                right = self.generate_expression(getattr(node, 'right', None))
                operator = getattr(node, 'operator', '+')
                return f"({left} {operator} {right})"
            
            elif expr_type == 'function_call':
                func_name = getattr(node, 'name', 'unknown')
                args = getattr(node, 'arguments', [])
                arg_list = [self.generate_expression(arg) for arg in args]
                return f"{func_name}({', '.join(arg_list)})"
        
        return str(node)
    
    def generate_neural_operation(self, node) -> str:
        """Генерирует нейронную операцию."""
        operation_type = getattr(node, 'operation', 'unknown')
        
        if operation_type == 'create_neuron':
            neuron_id = getattr(node, 'id', 'neuron')
            threshold = getattr(node, 'threshold', 1.0)
            return f'Neuron(id="{neuron_id}", threshold={threshold})'
        
        elif operation_type == 'create_synapse':
            source = getattr(node, 'source', 'source')
            target = getattr(node, 'target', 'target')
            weight = getattr(node, 'weight', 1.0)
            return f'Synapse(source={source}, target={target}, weight={weight})'
        
        elif operation_type == 'send_signal':
            signal_value = getattr(node, 'value', 0.0)
            target = getattr(node, 'target', 'target')
            return f'{target}.activate({signal_value})'
        
        elif operation_type == 'create_pulse':
            amplitude = getattr(node, 'amplitude', 1.0)
            frequency = getattr(node, 'frequency', 1.0)
            duration = getattr(node, 'duration', 1.0)
            return f'Pulse(amplitude={amplitude}, frequency={frequency}, duration={duration})'
        
        self.stats['neural_operations'] += 1
        return f"# Neural operation: {operation_type}"
    
    def _format_comment_block(self, lines: List[str]) -> str:
        """Форматирует блок комментариев для Python."""
        if not lines:
            return ""
        
        comment_lines = ['"""']
        comment_lines.extend(lines)
        comment_lines.append('"""')
        
        return '\n'.join(comment_lines)
    
    # Visitor методы для различных типов узлов
    
    def visit_program(self, node):
        """Посещает корневой узел программы."""
        if hasattr(node, 'statements'):
            for stmt in node.statements:
                self.visit(stmt)
    
    def visit_function_definition(self, node):
        """Посещает определение функции."""
        return self.generate_function(node)
    
    def visit_variable_declaration(self, node):
        """Посещает объявление переменной."""
        return self.generate_variable(node)
    
    def visit_assignment(self, node):
        """Посещает присваивание."""
        target = getattr(node, 'target', 'unknown')
        value = self.generate_expression(getattr(node, 'value', None))
        assignment = f"{target} = {value}"
        self.emit_line(assignment)
        return assignment
    
    def visit_expression_statement(self, node):
        """Посещает выражение-утверждение."""
        expr = self.generate_expression(getattr(node, 'expression', None))
        self.emit_line(expr)
        return expr
    
    def visit_neural_operation(self, node):
        """Посещает нейронную операцию."""
        operation = self.generate_neural_operation(node)
        self.emit_line(operation)
        return operation
    
    def visit_if_statement(self, node):
        """Посещает условное утверждение."""
        condition = self.generate_expression(getattr(node, 'condition', None))
        self.emit_line(f"if {condition}:")
        
        self.increase_indent()
        if hasattr(node, 'then_body'):
            for stmt in node.then_body:
                self.visit(stmt)
        else:
            self.emit_line("pass")
        self.decrease_indent()
        
        if hasattr(node, 'else_body') and node.else_body:
            self.emit_line("else:")
            self.increase_indent()
            for stmt in node.else_body:
                self.visit(stmt)
            self.decrease_indent()
    
    def visit_while_loop(self, node):
        """Посещает цикл while."""
        condition = self.generate_expression(getattr(node, 'condition', None))
        self.emit_line(f"while {condition}:")
        
        self.increase_indent()
        if hasattr(node, 'body'):
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.emit_line("pass")
        self.decrease_indent()
    
    def visit_for_loop(self, node):
        """Посещает цикл for."""
        variable = getattr(node, 'variable', 'i')
        iterable = self.generate_expression(getattr(node, 'iterable', None))
        self.emit_line(f"for {variable} in {iterable}:")
        
        self.increase_indent()
        if hasattr(node, 'body'):
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.emit_line("pass")
        self.decrease_indent()
    
    def visit_return_statement(self, node):
        """Посещает оператор return."""
        if hasattr(node, 'value') and node.value is not None:
            value = self.generate_expression(node.value)
            self.emit_line(f"return {value}")
        else:
            self.emit_line("return")


# Утилиты для быстрого использования
def generate_python_code(ast, config: CodeGenConfig = None) -> CodeGenResult:
    """Быстрая генерация Python кода из AST."""
    generator = PythonCodeGenerator(config)
    return generator.generate_program(ast)


def save_python_code(ast, filename: str, config: CodeGenConfig = None) -> bool:
    """Генерирует и сохраняет Python код в файл."""
    result = generate_python_code(ast, config)
    
    if result.success:
        generator = PythonCodeGenerator(config)
        return generator.save_to_file(result.generated_code, filename)
    
    return False


# Экспорт
__all__ = [
    'PythonCodeGenerator',
    'generate_python_code',
    'save_python_code'
] 