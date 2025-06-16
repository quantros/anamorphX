"""
AnamorphX Interpreter Module

Этот модуль содержит интерпретатор AnamorphX и все связанные компоненты.
"""

# Базовые классы
from .commands import (
    Command, CommandResult, CommandError, CommandCategory,
    StructuralCommand, FlowControlCommand, SecurityCommand,
    DataManagementCommand, MachineLearningCommand, CloudNetworkCommand,
    MonitoringCommand, SystemCommand
)

# Основные компоненты
from .commands import EnhancedCommandRegistry
from .runtime import ExecutionContext

# Сущности
from .commands import NeuralEntity, SynapseConnection

__all__ = [
    # Базовые классы
    'Command', 'CommandResult', 'CommandError', 'CommandCategory',
    'StructuralCommand', 'FlowControlCommand', 'SecurityCommand',
    'DataManagementCommand', 'MachineLearningCommand', 'CloudNetworkCommand',
    'MonitoringCommand', 'SystemCommand',
    
    # Основные компоненты
    'EnhancedCommandRegistry', 'ExecutionContext',
    
    # Сущности
    'NeuralEntity', 'SynapseConnection'
]

# Import all main classes and types
from .ast_types import (
    # Execution state and control flow
    ExecutionState,
    ControlFlowException,
    ReturnException,
    BreakException,
    ContinueException,
    PulseException,
    
    # Execution statistics and state
    ExecutionStats,
    ProgramState,
    DebugInfo,
    SignalData,
    InterpreterConfig,
    
    # Exception types
    InterpreterError,
    RuntimeInterpreterError,
    SyntaxInterpreterError,
    TypeInterpreterError,
    NameInterpreterError,
    
    # Utility types
    Literal
)

from .ast_builtins import BuiltinFunctions

from .ast_expressions import ExpressionEvaluator

from .ast_statements import StatementExecutor

from .ast_interpreter import ASTInterpreter

from .environment import Environment, VariableType

# Import memory manager if available
try:
    from .enhanced_memory_manager import EnhancedMemoryManager
except ImportError:
    EnhancedMemoryManager = None

# Version information
__version__ = "1.0.0"
__author__ = "Anamorph Development Team"

# Main exports
__all__ += [
    # Main interpreter class
    'ASTInterpreter',
    
    # Component classes
    'BuiltinFunctions',
    'ExpressionEvaluator',
    'StatementExecutor',
    'Environment',
    
    # Type definitions
    'ExecutionState',
    'ExecutionStats',
    'ProgramState',
    'DebugInfo',
    'SignalData',
    'InterpreterConfig',
    'VariableType',
    
    # Exception classes
    'ControlFlowException',
    'ReturnException',
    'BreakException',
    'ContinueException',
    'PulseException',
    'InterpreterError',
    'RuntimeInterpreterError',
    'SyntaxInterpreterError',
    'TypeInterpreterError',
    'NameInterpreterError',
    
    # Utility types
    'Literal',
    
    # Optional components
    'EnhancedMemoryManager',
]

# Package metadata
__package_info__ = {
    'name': 'ast_interpreter',
    'version': __version__,
    'description': 'Complete AST interpreter for Anamorph language',
    'features': [
        'Full AST node support',
        'Neural construct interpretation',
        'Signal processing',
        'Memory management',
        'Debug tracing',
        'Execution statistics',
        'Error handling',
        'Built-in functions',
        'Environment management',
        'Modular architecture'
    ],
    'components': {
        'ast_types': 'Core types and exceptions',
        'ast_builtins': 'Built-in functions',
        'ast_expressions': 'Expression evaluation',
        'ast_statements': 'Statement execution',
        'ast_interpreter': 'Main interpreter class',
        'environment': 'Variable environment management'
    }
}


def create_interpreter(config: InterpreterConfig = None) -> ASTInterpreter:
    """
    Create a new AST interpreter instance.
    
    Args:
        config: Optional interpreter configuration
        
    Returns:
        Configured ASTInterpreter instance
    """
    return ASTInterpreter(config)


def get_default_config() -> InterpreterConfig:
    """
    Get default interpreter configuration.
    
    Returns:
        Default InterpreterConfig instance
    """
    return InterpreterConfig()


def get_package_info() -> dict:
    """
    Get package information.
    
    Returns:
        Dictionary with package metadata
    """
    return __package_info__.copy() 