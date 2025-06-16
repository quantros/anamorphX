"""
Runtime execution engine for the AnamorphX interpreter.

This module provides the core runtime functionality including AST execution,
call stack management, and exception handling.
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Coroutine
from contextlib import contextmanager

from ..syntax.nodes import ASTNode
from .environment import Environment, EnvironmentError


class ExecutionState(Enum):
    """Execution state of the runtime."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ExecutionMode(Enum):
    """Execution mode for the runtime."""
    SYNCHRONOUS = "sync"
    ASYNCHRONOUS = "async"
    STEP_BY_STEP = "step"
    DEBUG = "debug"


@dataclass
class StackFrame:
    """Represents a single frame in the call stack."""
    function_name: str
    node: ASTNode
    local_vars: Dict[str, Any] = field(default_factory=dict)
    line_number: int = 0
    file_name: str = ""
    created_at: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return f"{self.function_name} at {self.file_name}:{self.line_number}"


@dataclass
class CallStack:
    """Manages the call stack for function calls."""
    frames: List[StackFrame] = field(default_factory=list)
    max_depth: int = 1000
    
    def push(self, frame: StackFrame) -> None:
        """Push a new frame onto the stack."""
        if len(self.frames) >= self.max_depth:
            raise RuntimeError(f"Maximum call stack depth ({self.max_depth}) exceeded")
        self.frames.append(frame)
    
    def pop(self) -> Optional[StackFrame]:
        """Pop the top frame from the stack."""
        return self.frames.pop() if self.frames else None
    
    def peek(self) -> Optional[StackFrame]:
        """Peek at the top frame without removing it."""
        return self.frames[-1] if self.frames else None
    
    def depth(self) -> int:
        """Get the current stack depth."""
        return len(self.frames)
    
    def clear(self) -> None:
        """Clear the entire stack."""
        self.frames.clear()
    
    def get_traceback(self) -> List[str]:
        """Get a traceback of the current stack."""
        return [str(frame) for frame in reversed(self.frames)]


@dataclass
class ExecutionContext:
    """Context for code execution."""
    environment: Environment
    call_stack: CallStack = field(default_factory=CallStack)
    execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    debug_mode: bool = False
    profiling_enabled: bool = False
    max_execution_time: Optional[float] = None
    max_memory_usage: Optional[int] = None
    
    # Execution statistics
    instructions_executed: int = 0
    execution_start_time: float = 0
    memory_usage: int = 0
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.instructions_executed = 0
        self.execution_start_time = time.time()
        self.memory_usage = 0


class ExecutionError(Exception):
    """Base class for execution errors."""
    
    def __init__(self, message: str, node: Optional[ASTNode] = None, 
                 context: Optional[ExecutionContext] = None):
        super().__init__(message)
        self.message = message
        self.node = node
        self.context = context
        self.traceback_info = self._get_traceback_info(context)
    
    def _get_traceback_info(self, context: Optional[ExecutionContext]) -> List[str]:
        """Get traceback information from context."""
        if context and context.call_stack:
            return context.call_stack.get_traceback()
        return []


class RuntimeError(ExecutionError):
    """Runtime execution error."""
    pass


class TimeoutError(ExecutionError):
    """Execution timeout error."""
    pass


class MemoryError(ExecutionError):
    """Memory limit exceeded error."""
    pass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    value: Any = None
    success: bool = True
    error: Optional[ExecutionError] = None
    execution_time: float = 0.0
    instructions_executed: int = 0
    memory_used: int = 0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def failed(self) -> bool:
        """Check if execution failed."""
        return not self.success


class Runtime:
    """Core runtime execution engine."""
    
    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment or Environment()
        self.state = ExecutionState.IDLE
        self.context: Optional[ExecutionContext] = None
        
        # Execution control
        self._should_stop = False
        self._step_mode = False
        self._breakpoints: set = set()
        
        # Performance monitoring
        self._execution_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'total_instructions': 0,
            'errors': 0
        }
        
        # Built-in function registry
        self._builtin_functions: Dict[str, Callable] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self) -> None:
        """Register built-in functions."""
        self._builtin_functions.update({
            'print': self._builtin_print,
            'len': self._builtin_len,
            'type': self._builtin_type,
            'str': self._builtin_str,
            'int': self._builtin_int,
            'float': self._builtin_float,
            'bool': self._builtin_bool,
            'list': self._builtin_list,
            'dict': self._builtin_dict,
        })
    
    # Built-in function implementations
    def _builtin_print(self, *args) -> None:
        """Built-in print function."""
        print(*args)
    
    def _builtin_len(self, obj) -> int:
        """Built-in len function."""
        return len(obj)
    
    def _builtin_type(self, obj) -> str:
        """Built-in type function."""
        return type(obj).__name__
    
    def _builtin_str(self, obj) -> str:
        """Built-in str function."""
        return str(obj)
    
    def _builtin_int(self, obj) -> int:
        """Built-in int function."""
        return int(obj)
    
    def _builtin_float(self, obj) -> float:
        """Built-in float function."""
        return float(obj)
    
    def _builtin_bool(self, obj) -> bool:
        """Built-in bool function."""
        return bool(obj)
    
    def _builtin_list(self, *args) -> list:
        """Built-in list function."""
        return list(args) if args else []
    
    def _builtin_dict(self, **kwargs) -> dict:
        """Built-in dict function."""
        return dict(kwargs)
    
    def execute(self, node: ASTNode, 
                execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS,
                **kwargs) -> ExecutionResult:
        """Execute an AST node."""
        if execution_mode == ExecutionMode.ASYNCHRONOUS:
            return asyncio.run(self.execute_async(node, **kwargs))
        else:
            return self._execute_sync(node, **kwargs)
    
    async def execute_async(self, node: ASTNode, **kwargs) -> ExecutionResult:
        """Execute an AST node asynchronously."""
        context = self._create_execution_context(ExecutionMode.ASYNCHRONOUS, **kwargs)
        
        try:
            self.state = ExecutionState.RUNNING
            self.context = context
            
            # Start execution timer
            start_time = time.time()
            context.execution_start_time = start_time
            
            # Execute the node
            result_value = await self._execute_node_async(node, context)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create successful result
            result = ExecutionResult(
                value=result_value,
                success=True,
                execution_time=execution_time,
                instructions_executed=context.instructions_executed,
                memory_used=context.memory_usage
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - context.execution_start_time
            error = self._create_execution_error(e, node, context)
            
            result = ExecutionResult(
                success=False,
                error=error,
                execution_time=execution_time,
                instructions_executed=context.instructions_executed,
                memory_used=context.memory_usage
            )
            
            self._update_stats(result)
            return result
            
        finally:
            self.state = ExecutionState.IDLE
            self.context = None
    
    def _execute_sync(self, node: ASTNode, **kwargs) -> ExecutionResult:
        """Execute an AST node synchronously."""
        context = self._create_execution_context(ExecutionMode.SYNCHRONOUS, **kwargs)
        
        try:
            self.state = ExecutionState.RUNNING
            self.context = context
            
            # Start execution timer
            start_time = time.time()
            context.execution_start_time = start_time
            
            # Execute the node
            result_value = self._execute_node(node, context)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create successful result
            result = ExecutionResult(
                value=result_value,
                success=True,
                execution_time=execution_time,
                instructions_executed=context.instructions_executed,
                memory_used=context.memory_usage
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - context.execution_start_time
            error = self._create_execution_error(e, node, context)
            
            result = ExecutionResult(
                success=False,
                error=error,
                execution_time=execution_time,
                instructions_executed=context.instructions_executed,
                memory_used=context.memory_usage
            )
            
            self._update_stats(result)
            return result
            
        finally:
            self.state = ExecutionState.IDLE
            self.context = None
    
    def _create_execution_context(self, mode: ExecutionMode, **kwargs) -> ExecutionContext:
        """Create execution context."""
        context = ExecutionContext(
            environment=self.environment,
            execution_mode=mode,
            debug_mode=kwargs.get('debug_mode', False),
            profiling_enabled=kwargs.get('profiling_enabled', False),
            max_execution_time=kwargs.get('max_execution_time'),
            max_memory_usage=kwargs.get('max_memory_usage')
        )
        context.reset_stats()
        return context
    
    def _execute_node(self, node: ASTNode, context: ExecutionContext) -> Any:
        """Execute a single AST node synchronously."""
        # Check execution limits
        self._check_execution_limits(context)
        
        # Increment instruction counter
        context.instructions_executed += 1
        
        # Check for breakpoints in debug mode
        if context.debug_mode and id(node) in self._breakpoints:
            self._handle_breakpoint(node, context)
        
        # Dispatch to appropriate handler based on node type
        handler_name = f"_execute_{node.__class__.__name__.lower()}"
        handler = getattr(self, handler_name, self._execute_generic)
        
        return handler(node, context)
    
    async def _execute_node_async(self, node: ASTNode, context: ExecutionContext) -> Any:
        """Execute a single AST node asynchronously."""
        # Check execution limits
        self._check_execution_limits(context)
        
        # Increment instruction counter
        context.instructions_executed += 1
        
        # Check for breakpoints in debug mode
        if context.debug_mode and id(node) in self._breakpoints:
            self._handle_breakpoint(node, context)
        
        # Dispatch to appropriate handler based on node type
        handler_name = f"_execute_{node.__class__.__name__.lower()}_async"
        async_handler = getattr(self, handler_name, None)
        
        if async_handler:
            return await async_handler(node, context)
        else:
            # Fall back to synchronous execution
            handler_name = f"_execute_{node.__class__.__name__.lower()}"
            handler = getattr(self, handler_name, self._execute_generic)
            return handler(node, context)
    
    def _execute_generic(self, node: ASTNode, context: ExecutionContext) -> Any:
        """Generic node execution handler."""
        raise RuntimeError(f"No handler found for node type: {node.__class__.__name__}")
    
    def _check_execution_limits(self, context: ExecutionContext) -> None:
        """Check execution time and memory limits."""
        current_time = time.time()
        
        # Check execution time limit
        if (context.max_execution_time and 
            current_time - context.execution_start_time > context.max_execution_time):
            raise TimeoutError(f"Execution timeout after {context.max_execution_time} seconds")
        
        # Check memory limit (simplified)
        if (context.max_memory_usage and 
            context.memory_usage > context.max_memory_usage):
            raise MemoryError(f"Memory usage exceeded limit: {context.max_memory_usage} bytes")
        
        # Check stop flag
        if self._should_stop:
            raise RuntimeError("Execution stopped by user")
    
    def _handle_breakpoint(self, node: ASTNode, context: ExecutionContext) -> None:
        """Handle breakpoint in debug mode."""
        self.state = ExecutionState.PAUSED
        # In a real implementation, this would trigger debugger interface
        print(f"Breakpoint hit at node: {node}")
    
    def _create_execution_error(self, exception: Exception, node: ASTNode, 
                              context: ExecutionContext) -> ExecutionError:
        """Create appropriate execution error from exception."""
        if isinstance(exception, ExecutionError):
            return exception
        elif isinstance(exception, EnvironmentError):
            return RuntimeError(str(exception), node, context)
        else:
            return RuntimeError(f"Unexpected error: {str(exception)}", node, context)
    
    def _update_stats(self, result: ExecutionResult) -> None:
        """Update execution statistics."""
        self._execution_stats['total_executions'] += 1
        self._execution_stats['total_time'] += result.execution_time
        self._execution_stats['total_instructions'] += result.instructions_executed
        
        if result.failed:
            self._execution_stats['errors'] += 1
    
    # Control methods
    def stop(self) -> None:
        """Stop execution."""
        self._should_stop = True
    
    def resume(self) -> None:
        """Resume execution."""
        self._should_stop = False
        if self.state == ExecutionState.PAUSED:
            self.state = ExecutionState.RUNNING
    
    def add_breakpoint(self, node: ASTNode) -> None:
        """Add a breakpoint at a node."""
        self._breakpoints.add(id(node))
    
    def remove_breakpoint(self, node: ASTNode) -> None:
        """Remove a breakpoint from a node."""
        self._breakpoints.discard(id(node))
    
    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()
    
    # Utility methods
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._execution_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'total_instructions': 0,
            'errors': 0
        }
    
    @contextmanager
    def execution_context(self, **kwargs):
        """Context manager for execution."""
        old_context = self.context
        try:
            context = self._create_execution_context(ExecutionMode.SYNCHRONOUS, **kwargs)
            self.context = context
            yield context
        finally:
            self.context = old_context 