"""
AST Interpreter Types and Exceptions.

This module defines core types, enums, and exceptions used throughout
the AST interpreter system.
"""

import time
import sys
import os
import asyncio
import threading
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set, Iterator
from enum import Enum, auto
from dataclasses import dataclass, field

# Add parent directories to path for comprehensive imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try to import from syntax module if available
try:
    from syntax.nodes import (
        ASTNode, Expression, Statement, Literal as SyntaxLiteral,
        Identifier, BinaryOperation, UnaryOperation, FunctionCall,
        Assignment, IfStatement, WhileStatement, ForStatement,
        FunctionDeclaration, ReturnStatement, BreakStatement,
        ContinueStatement, TryStatement, CatchStatement,
        NeuralEntity, SynapseConnection, PulseStatement,
        ResonateStatement, SignalFlow, AsyncStatement
    )
    SYNTAX_AVAILABLE = True
except ImportError:
    # Create mock classes if syntax module is not available
    class ASTNode:
        def __init__(self):
            self.location = None
    
    class Expression(ASTNode):
        pass
    
    class Statement(ASTNode):
        pass
    
    class SyntaxLiteral(Expression):
        def __init__(self, value):
            super().__init__()
            self.value = value
    
    # Mock all other node types
    Identifier = type('Identifier', (Expression,), {})
    BinaryOperation = type('BinaryOperation', (Expression,), {})
    UnaryOperation = type('UnaryOperation', (Expression,), {})
    FunctionCall = type('FunctionCall', (Expression,), {})
    Assignment = type('Assignment', (Statement,), {})
    IfStatement = type('IfStatement', (Statement,), {})
    WhileStatement = type('WhileStatement', (Statement,), {})
    ForStatement = type('ForStatement', (Statement,), {})
    FunctionDeclaration = type('FunctionDeclaration', (Statement,), {})
    ReturnStatement = type('ReturnStatement', (Statement,), {})
    BreakStatement = type('BreakStatement', (Statement,), {})
    ContinueStatement = type('ContinueStatement', (Statement,), {})
    TryStatement = type('TryStatement', (Statement,), {})
    CatchStatement = type('CatchStatement', (Statement,), {})
    NeuralEntity = type('NeuralEntity', (Statement,), {})
    SynapseConnection = type('SynapseConnection', (Statement,), {})
    PulseStatement = type('PulseStatement', (Statement,), {})
    ResonateStatement = type('ResonateStatement', (Statement,), {})
    SignalFlow = type('SignalFlow', (Expression,), {})
    AsyncStatement = type('AsyncStatement', (Statement,), {})
    
    SYNTAX_AVAILABLE = False


# =============================================================================
# EXECUTION STATE AND CONTROL FLOW
# =============================================================================

class ExecutionState(Enum):
    """Execution state of the interpreter."""
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()
    COMPLETED = auto()


class ControlFlowException(Exception):
    """Base class for control flow exceptions."""
    pass


class ReturnException(ControlFlowException):
    """Exception for return statements."""
    def __init__(self, value: Any = None):
        self.value = value
        super().__init__(f"Return: {value}")


class BreakException(ControlFlowException):
    """Exception for break statements."""
    pass


class ContinueException(ControlFlowException):
    """Exception for continue statements."""
    pass


class PulseException(ControlFlowException):
    """Exception for neural pulse propagation."""
    def __init__(self, signal: Any, target: str = "broadcast", intensity: float = 1.0):
        self.signal = signal
        self.target = target
        self.intensity = intensity
        super().__init__(f"Pulse: {signal} -> {target}")


# =============================================================================
# EXECUTION STATISTICS AND STATE
# =============================================================================

@dataclass
class ExecutionStats:
    """Statistics for execution performance."""
    nodes_executed: int = 0
    execution_time: float = 0.0
    memory_usage: int = 0
    function_calls: int = 0
    neural_activations: int = 0
    signal_transmissions: int = 0
    errors: int = 0
    warnings: int = 0
    
    def reset(self):
        """Reset all statistics."""
        self.nodes_executed = 0
        self.execution_time = 0.0
        self.memory_usage = 0
        self.function_calls = 0
        self.neural_activations = 0
        self.signal_transmissions = 0
        self.errors = 0
        self.warnings = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            'nodes_executed': self.nodes_executed,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'function_calls': self.function_calls,
            'neural_activations': self.neural_activations,
            'signal_transmissions': self.signal_transmissions,
            'errors': self.errors,
            'warnings': self.warnings,
            'nodes_per_second': self.nodes_executed / max(self.execution_time, 0.001),
            'calls_per_second': self.function_calls / max(self.execution_time, 0.001)
        }


@dataclass
class ProgramState:
    """Complete program execution state."""
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Any] = field(default_factory=dict)  # FunctionDeclaration objects
    neurons: Dict[str, Any] = field(default_factory=dict)    # NeuralEntity objects
    synapses: Dict[str, Any] = field(default_factory=dict)   # SynapseConnection objects
    imports: Dict[str, Any] = field(default_factory=dict)
    exports: Dict[str, Any] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    signal_queue: deque = field(default_factory=deque)
    async_tasks: Dict[str, Any] = field(default_factory=dict)  # asyncio.Task
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current program state."""
        return {
            'variables': len(self.variables),
            'functions': len(self.functions),
            'neurons': len(self.neurons),
            'synapses': len(self.synapses),
            'call_stack_depth': len(self.call_stack),
            'pending_signals': len(self.signal_queue),
            'active_tasks': len(self.async_tasks)
        }
    
    def clear(self):
        """Clear all program state."""
        self.variables.clear()
        self.functions.clear()
        self.neurons.clear()
        self.synapses.clear()
        self.imports.clear()
        self.exports.clear()
        self.call_stack.clear()
        self.signal_queue.clear()
        self.async_tasks.clear()


@dataclass
class DebugInfo:
    """Debug information for execution tracing."""
    timestamp: float = field(default_factory=time.time)
    node_type: str = ""
    value: Any = None
    location: Optional[str] = None
    stack_depth: int = 0
    memory_usage: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'node_type': self.node_type,
            'value': str(self.value) if self.value is not None else None,
            'location': self.location,
            'stack_depth': self.stack_depth,
            'memory_usage': self.memory_usage
        }


# =============================================================================
# SIGNAL PROCESSING TYPES
# =============================================================================

@dataclass
class SignalData:
    """Neural signal data structure."""
    source: Any
    target: str
    signal: Any
    intensity: float = 1.0
    signal_type: str = "sync"  # sync, async, priority, streaming
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source': str(self.source),
            'target': self.target,
            'signal': self.signal,
            'intensity': self.intensity,
            'signal_type': self.signal_type,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


# =============================================================================
# INTERPRETER CONFIGURATION
# =============================================================================

@dataclass
class InterpreterConfig:
    """Configuration for AST interpreter."""
    debug_mode: bool = False
    async_enabled: bool = True
    max_execution_time: float = 300.0  # 5 minutes
    max_recursion_depth: int = 1000
    max_signal_queue_size: int = 10000
    trace_execution: bool = False
    enable_profiling: bool = False
    memory_limit: int = 512 * 1024 * 1024  # 512MB
    
    # Neural processing settings
    neural_processing_enabled: bool = True
    signal_processing_interval: float = 0.001  # 1ms
    max_neural_activations_per_cycle: int = 1000
    
    # Error handling
    strict_mode: bool = False
    continue_on_error: bool = False
    max_errors: int = 100
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        if self.max_execution_time <= 0:
            issues.append("max_execution_time must be positive")
        
        if self.max_recursion_depth <= 0:
            issues.append("max_recursion_depth must be positive")
        
        if self.memory_limit <= 0:
            issues.append("memory_limit must be positive")
        
        if self.signal_processing_interval <= 0:
            issues.append("signal_processing_interval must be positive")
        
        return issues


# =============================================================================
# UTILITY TYPES
# =============================================================================

class InterpreterError(Exception):
    """Base exception for interpreter errors."""
    def __init__(self, message: str, node: Optional[Any] = None):
        self.message = message
        self.node = node
        self.location = str(node.location) if node and hasattr(node, 'location') and node.location else None
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with location info."""
        if self.location:
            return f"{self.message} at {self.location}"
        return self.message


class RuntimeInterpreterError(InterpreterError):
    """Runtime error during interpretation."""
    pass


class SyntaxInterpreterError(InterpreterError):
    """Syntax error during interpretation."""
    pass


class TypeInterpreterError(InterpreterError):
    """Type error during interpretation."""
    pass


class NameInterpreterError(InterpreterError):
    """Name error during interpretation."""
    pass


# =============================================================================
# LITERAL TYPE FOR COMPATIBILITY
# =============================================================================

@dataclass
class Literal:
    """Generic literal for compatibility."""
    value: Any
    
    def __post_init__(self):
        self.location = None 