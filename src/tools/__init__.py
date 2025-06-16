"""
Development Tools for AnamorphX

Комплект профессиональных инструментов разработки для языка Anamorph:
- Syntax Highlighting (подсветка синтаксиса) 
- IDE Components (компоненты IDE)
- Debugger (отладчик)
- Profiler (профайлер)
"""

from .syntax_highlighter import AnamorphSyntaxHighlighter, HighlightTheme
from .ide_components import AnamorphIDE, CodeEditor, FileExplorer
from .debugger import AnamorphDebugger, BreakpointManager
from .profiler import AnamorphProfiler, PerformanceAnalyzer

__version__ = "1.0.0"
__author__ = "AnamorphX Development Team"

__all__ = [
    # Syntax Highlighting
    'AnamorphSyntaxHighlighter',
    'HighlightTheme',
    
    # IDE Components  
    'AnamorphIDE',
    'CodeEditor',
    'FileExplorer',
    
    # Debugger
    'AnamorphDebugger',
    'BreakpointManager',
    
    # Profiler
    'AnamorphProfiler', 
    'PerformanceAnalyzer'
] 