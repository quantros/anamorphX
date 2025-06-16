"""
Code Generation Module for AnamorphX

Модуль кодогенерации для компиляции программ Anamorph в различные целевые языки:
- Python (основной целевой язык)
- JavaScript (для веб-приложений)
- C++ (для высокопроизводительных приложений)
"""

from .python_codegen import PythonCodeGenerator
from .javascript_codegen import JavaScriptCodeGenerator
from .base_codegen import BaseCodeGenerator, CodeGenConfig, CodeGenResult

__version__ = "1.0.0"
__author__ = "AnamorphX Team"

__all__ = [
    'PythonCodeGenerator',
    'JavaScriptCodeGenerator', 
    'BaseCodeGenerator',
    'CodeGenConfig',
    'CodeGenResult'
] 