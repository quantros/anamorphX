"""
Base Code Generator for AnamorphX

Базовый класс для всех кодогенераторов с общей архитектурой и интерфейсами.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


class TargetLanguage(Enum):
    """Целевые языки для кодогенерации."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"


class OptimizationLevel(Enum):
    """Уровни оптимизации кода."""
    NONE = auto()       # Без оптимизации
    BASIC = auto()      # Базовые оптимизации
    ADVANCED = auto()   # Продвинутые оптимизации
    AGGRESSIVE = auto() # Агрессивные оптимизации


@dataclass
class CodeGenConfig:
    """Конфигурация кодогенерации."""
    target_language: TargetLanguage = TargetLanguage.PYTHON
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    output_directory: str = "generated"
    include_comments: bool = True
    include_debug_info: bool = False
    minify_output: bool = False
    generate_tests: bool = False
    async_support: bool = True
    neural_library: Optional[str] = None  # numpy, tensorflow, pytorch
    
    # Специфичные настройки
    python_version: str = "3.8+"
    javascript_target: str = "ES2020"
    cpp_standard: str = "C++17"
    
    # Дополнительные опции
    custom_imports: List[str] = field(default_factory=list)
    custom_headers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeGenResult:
    """Результат кодогенерации."""
    success: bool
    generated_code: str = ""
    output_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    lines_generated: int = 0
    optimizations_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCodeGenerator(ABC):
    """
    Базовый класс для всех кодогенераторов.
    
    Определяет общий интерфейс и базовую функциональность для
    генерации кода из AST программ Anamorph.
    """
    
    def __init__(self, config: CodeGenConfig):
        self.config = config
        self.generated_code = []
        self.indent_level = 0
        self.indent_size = 4
        self.current_scope = []
        self.symbol_table = {}
        self.imports = set()
        self.warnings = []
        self.errors = []
        self.optimizations = []
        
        # Статистика
        self.stats = {
            'nodes_processed': 0,
            'functions_generated': 0,
            'variables_declared': 0,
            'neural_operations': 0,
            'optimizations_applied': 0
        }
    
    @abstractmethod
    def get_target_language(self) -> TargetLanguage:
        """Возвращает целевой язык генератора."""
        pass
    
    @abstractmethod
    def generate_program(self, ast) -> CodeGenResult:
        """Генерирует код для всей программы."""
        pass
    
    @abstractmethod
    def generate_function(self, node) -> str:
        """Генерирует код функции."""
        pass
    
    @abstractmethod
    def generate_variable(self, node) -> str:
        """Генерирует объявление переменной."""
        pass
    
    @abstractmethod
    def generate_expression(self, node) -> str:
        """Генерирует выражение."""
        pass
    
    @abstractmethod
    def generate_neural_operation(self, node) -> str:
        """Генерирует нейронную операцию."""
        pass
    
    # Общие утилиты для всех генераторов
    
    def emit(self, code: str, indent: bool = True):
        """Добавляет строку кода с учетом отступов."""
        if indent:
            indentation = " " * (self.indent_level * self.indent_size)
            self.generated_code.append(f"{indentation}{code}")
        else:
            self.generated_code.append(code)
    
    def emit_line(self, code: str = ""):
        """Добавляет строку с переводом строки."""
        self.emit(code)
    
    def emit_blank_line(self):
        """Добавляет пустую строку."""
        self.generated_code.append("")
    
    def increase_indent(self):
        """Увеличивает уровень отступа."""
        self.indent_level += 1
    
    def decrease_indent(self):
        """Уменьшает уровень отступа."""
        self.indent_level = max(0, self.indent_level - 1)
    
    def enter_scope(self, scope_name: str):
        """Входит в новую область видимости."""
        self.current_scope.append(scope_name)
    
    def exit_scope(self):
        """Выходит из текущей области видимости."""
        if self.current_scope:
            self.current_scope.pop()
    
    def get_current_scope(self) -> str:
        """Возвращает текущую область видимости."""
        return ".".join(self.current_scope) if self.current_scope else "global"
    
    def add_import(self, import_statement: str):
        """Добавляет импорт."""
        self.imports.add(import_statement)
    
    def add_warning(self, message: str):
        """Добавляет предупреждение."""
        self.warnings.append(f"Warning: {message}")
    
    def add_error(self, message: str):
        """Добавляет ошибку."""
        self.errors.append(f"Error: {message}")
    
    def add_optimization(self, description: str):
        """Добавляет информацию об оптимизации."""
        self.optimizations.append(description)
        self.stats['optimizations_applied'] += 1
    
    def register_symbol(self, name: str, symbol_type: str, scope: str = None):
        """Регистрирует символ в таблице символов."""
        scope = scope or self.get_current_scope()
        full_name = f"{scope}.{name}" if scope != "global" else name
        self.symbol_table[full_name] = {
            'name': name,
            'type': symbol_type,
            'scope': scope,
            'full_name': full_name
        }
    
    def lookup_symbol(self, name: str) -> Optional[Dict]:
        """Ищет символ в таблице символов."""
        # Сначала ищем в текущей области
        current_scope = self.get_current_scope()
        full_name = f"{current_scope}.{name}" if current_scope != "global" else name
        
        if full_name in self.symbol_table:
            return self.symbol_table[full_name]
        
        # Затем ищем в глобальной области
        if name in self.symbol_table:
            return self.symbol_table[name]
        
        return None
    
    def generate_imports(self) -> List[str]:
        """Генерирует секцию импортов."""
        return sorted(list(self.imports))
    
    def get_generated_code(self) -> str:
        """Возвращает сгенерированный код как строку."""
        return "\n".join(self.generated_code)
    
    def clear_generated_code(self):
        """Очищает сгенерированный код."""
        self.generated_code.clear()
        self.indent_level = 0
    
    def apply_optimizations(self, code: str) -> str:
        """Применяет оптимизации к сгенерированному коду."""
        optimized_code = code
        
        if self.config.optimization_level == OptimizationLevel.NONE:
            return optimized_code
        
        # Базовые оптимизации
        if self.config.optimization_level.value >= OptimizationLevel.BASIC.value:
            optimized_code = self._apply_basic_optimizations(optimized_code)
        
        # Продвинутые оптимизации
        if self.config.optimization_level.value >= OptimizationLevel.ADVANCED.value:
            optimized_code = self._apply_advanced_optimizations(optimized_code)
        
        # Агрессивные оптимизации
        if self.config.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
            optimized_code = self._apply_aggressive_optimizations(optimized_code)
        
        return optimized_code
    
    def _apply_basic_optimizations(self, code: str) -> str:
        """Применяет базовые оптимизации."""
        # Удаление лишних пустых строк
        lines = code.split('\n')
        optimized_lines = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                optimized_lines.append(line)
            prev_empty = is_empty
        
        self.add_optimization("Removed redundant empty lines")
        return '\n'.join(optimized_lines)
    
    def _apply_advanced_optimizations(self, code: str) -> str:
        """Применяет продвинутые оптимизации."""
        # Здесь можно добавить более сложные оптимизации
        self.add_optimization("Applied advanced optimizations")
        return code
    
    def _apply_aggressive_optimizations(self, code: str) -> str:
        """Применяет агрессивные оптимизации."""
        # Минификация (если включена)
        if self.config.minify_output:
            # Простая минификация - удаление комментариев и лишних пробелов
            lines = code.split('\n')
            minified_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):  # Удаляем комментарии
                    minified_lines.append(stripped)
            
            self.add_optimization("Applied code minification")
            return '\n'.join(minified_lines)
        
        return code
    
    def generate_header_comment(self) -> str:
        """Генерирует заголовочный комментарий."""
        if not self.config.include_comments:
            return ""
        
        header = [
            f"Generated by AnamorphX Code Generator",
            f"Target Language: {self.get_target_language().value}",
            f"Generation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Optimization Level: {self.config.optimization_level.name}"
        ]
        
        return self._format_comment_block(header)
    
    @abstractmethod
    def _format_comment_block(self, lines: List[str]) -> str:
        """Форматирует блок комментариев для целевого языка."""
        pass
    
    def create_result(self, start_time: float) -> CodeGenResult:
        """Создает результат кодогенерации."""
        generated_code = self.get_generated_code()
        optimized_code = self.apply_optimizations(generated_code)
        
        # Добавляем заголовочный комментарий
        header = self.generate_header_comment()
        if header:
            final_code = f"{header}\n\n{optimized_code}"
        else:
            final_code = optimized_code
        
        return CodeGenResult(
            success=len(self.errors) == 0,
            generated_code=final_code,
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
            generation_time=time.time() - start_time,
            lines_generated=len(final_code.split('\n')),
            optimizations_applied=self.optimizations.copy(),
            metadata={
                'target_language': self.get_target_language().value,
                'stats': self.stats.copy(),
                'symbol_table_size': len(self.symbol_table),
                'imports_count': len(self.imports)
            }
        )
    
    def save_to_file(self, code: str, filename: str) -> bool:
        """Сохраняет код в файл."""
        try:
            # Создаем директорию если не существует
            os.makedirs(self.config.output_directory, exist_ok=True)
            
            # Полный путь к файлу
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Записываем файл
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            self.add_error(f"Failed to save file {filename}: {e}")
            return False


# Утилиты для работы с AST
class ASTVisitor:
    """Базовый visitor для обхода AST."""
    
    def visit(self, node):
        """Посещает узел AST."""
        method_name = f'visit_{type(node).__name__.lower()}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """Обработка узла по умолчанию."""
        return f"// Unhandled node: {type(node).__name__}"


# Экспорт основных классов
__all__ = [
    'BaseCodeGenerator',
    'CodeGenConfig', 
    'CodeGenResult',
    'TargetLanguage',
    'OptimizationLevel',
    'ASTVisitor'
] 