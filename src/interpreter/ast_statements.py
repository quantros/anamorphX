"""
Statement Handlers for AST Interpreter.

This module handles all statement execution in the AST interpreter.
"""

import time
import sys
import os
import asyncio
import threading
from typing import Any, List, Dict, Optional, Union, Callable, Tuple, Set
from collections import defaultdict, deque
from functools import reduce, partial

# Add parent directories to path for comprehensive imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try comprehensive imports with fallbacks
try:
    from interpreter.ast_types import (
        ProgramState, SignalData, ExecutionState, ExecutionStats,
        DebugInfo, InterpreterConfig, ReturnException, BreakException,
        ContinueException, PulseException, RuntimeInterpreterError,
        NameInterpreterError, TypeInterpreterError, Literal,
        ASTNode, Expression, Statement, SyntaxLiteral, Identifier,
        BinaryOperation, UnaryOperation, FunctionCall, Assignment,
        SignalFlow, NeuralEntity, SynapseConnection
    )
    AST_TYPES_AVAILABLE = True
except ImportError:
    try:
        from ast_types import (
            ProgramState, SignalData, ExecutionState, ExecutionStats,
            DebugInfo, InterpreterConfig, ReturnException, BreakException,
            ContinueException, PulseException, RuntimeInterpreterError,
            NameInterpreterError, TypeInterpreterError, Literal,
            ASTNode, Expression, Statement, SyntaxLiteral, Identifier,
            BinaryOperation, UnaryOperation, FunctionCall, Assignment,
            SignalFlow, NeuralEntity, SynapseConnection
        )
        AST_TYPES_AVAILABLE = True
    except ImportError:
        # Mock exceptions for fallback
        class ReturnException(Exception):
            def __init__(self, value=None):
                self.value = value
        
        class BreakException(Exception):
            pass
        
        class ContinueException(Exception):
            pass
        
        class PulseException(Exception):
            def __init__(self, signal, target="broadcast", intensity=1.0):
                self.signal = signal
                self.target = target
                self.intensity = intensity
        
        class RuntimeInterpreterError(Exception):
            pass
        
        class NameInterpreterError(Exception):
            pass
        
        class TypeInterpreterError(Exception):
            pass
        
        class SignalData:
            def __init__(self, source, target, signal, intensity=1.0):
                self.source = source
                self.target = target
                self.signal = signal
                self.intensity = intensity
        
        AST_TYPES_AVAILABLE = False

# Try to import syntax nodes with comprehensive fallbacks
try:
    # Try various import paths
    try:
        from syntax.nodes import (
            IfStatement, WhileStatement, ForStatement, TryStatement,
            CatchClause, ReturnStatement, BreakStatement, ContinueStatement,
            ExpressionStatement, BlockStatement, PulseStatement, ResonateStatement,
            Identifier, FunctionDeclaration, VariableDeclaration, ImportStatement,
            ExportStatement
        )
    except ImportError:
        # Try with src prefix
        from src.syntax.nodes import (
            IfStatement, WhileStatement, ForStatement, TryStatement,
            CatchClause, ReturnStatement, BreakStatement, ContinueStatement,
            ExpressionStatement, BlockStatement, PulseStatement, ResonateStatement,
            Identifier, FunctionDeclaration, VariableDeclaration, ImportStatement,
            ExportStatement
        )
    SYNTAX_AVAILABLE = True
    print("✅ Используются настоящие AST ноды из syntax.nodes")
except ImportError:
    print("⚠ Используются мок-классы вместо настоящих AST нод")
    SYNTAX_AVAILABLE = False
    
    # Create mock classes for fallback
    class MockNode:
        """Mock AST node for compatibility."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.location = None

    class IfStatement(MockNode):
        def __init__(self, test=None, consequent=None, alternate=None):
            super().__init__()
            self.test = test
            self.consequent = consequent
            self.alternate = alternate

    class WhileStatement(MockNode):
        def __init__(self, test=None, body=None):
            super().__init__()
            self.test = test
            self.body = body

    class ForStatement(MockNode):
        def __init__(self, init=None, test=None, update=None, body=None):
            super().__init__()
            self.init = init
            self.test = test
            self.update = update
            self.body = body

    class TryStatement(MockNode):
        def __init__(self, block=None, handlers=None, finalizer=None):
            super().__init__()
            self.block = block
            self.handlers = handlers or []
            self.finalizer = finalizer

    class CatchClause(MockNode):
        def __init__(self, param=None, body=None):
            super().__init__()
            self.param = param
            self.body = body

    class ReturnStatement(MockNode):
        def __init__(self, argument=None):
            super().__init__()
            self.argument = argument

    class BreakStatement(MockNode):
        pass

    class ContinueStatement(MockNode):
        pass

    class ExpressionStatement(MockNode):
        def __init__(self, expression=None):
            super().__init__()
            self.expression = expression

    class BlockStatement(MockNode):
        def __init__(self, body=None):
            super().__init__()
            self.body = body or []

    class PulseStatement(MockNode):
        def __init__(self, signal=None, target=None, intensity=None):
            super().__init__()
            self.signal = signal
            self.target = target
            self.intensity = intensity

    class ResonateStatement(MockNode):
        def __init__(self, frequency=None, duration=None):
            super().__init__()
            self.frequency = frequency
            self.duration = duration

    class Identifier(MockNode):
        def __init__(self, name=None):
            super().__init__()
            self.name = name

    class FunctionDeclaration(MockNode):
        def __init__(self, name=None, params=None, body=None):
            super().__init__()
            self.name = name
            self.params = params or []
            self.body = body

    class VariableDeclaration(MockNode):
        def __init__(self, declarations=None):
            super().__init__()
            self.declarations = declarations or []

    class ImportStatement(MockNode):
        def __init__(self, source=None, specifiers=None):
            super().__init__()
            self.source = source
            self.specifiers = specifiers or []

    class ExportStatement(MockNode):
        def __init__(self, declaration=None):
            super().__init__()
            self.declaration = declaration

# Try to import environment module
try:
    from interpreter.environment import Environment, VariableType
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    try:
        from environment import Environment, VariableType
        ENVIRONMENT_AVAILABLE = True
    except ImportError:
        ENVIRONMENT_AVAILABLE = False


class StatementExecutor:
    """Handles execution of all statement types."""
    
    def __init__(self, interpreter):
        """Initialize with interpreter reference."""
        self.interpreter = interpreter
    
    def execute_if_statement(self, node: IfStatement) -> Any:
        """Execute if statement."""
        self.interpreter._trace_execution(node, "if_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        test = self.interpreter.visit(getattr(node, 'test', None))
        
        if self._is_truthy(test):
            return self.interpreter.visit(getattr(node, 'consequent', None))
        elif hasattr(node, 'alternate') and getattr(node, 'alternate', None):
            return self.interpreter.visit(getattr(node, 'alternate', None))
        
        return None
    
    def execute_while_statement(self, node: WhileStatement) -> Any:
        """Execute while loop."""
        self.interpreter._trace_execution(node, "while_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        result = None
        
        try:
            while True:
                test = self.interpreter.visit(getattr(node, 'test', None))
                if not self._is_truthy(test):
                    break
                
                try:
                    result = self.interpreter.visit(getattr(node, 'body', None))
                except ContinueException:
                    continue
                except BreakException:
                    break
        except BreakException:
            pass
        
        return result
    
    def execute_for_statement(self, node: ForStatement) -> Any:
        """Execute for loop."""
        self.interpreter._trace_execution(node, "for_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        # Initialize
        if hasattr(node, 'init') and getattr(node, 'init', None):
            self.interpreter.visit(getattr(node, 'init', None))
        
        result = None
        
        try:
            while True:
                # Test condition
                if hasattr(node, 'test') and getattr(node, 'test', None):
                    test = self.interpreter.visit(getattr(node, 'test', None))
                    if not self._is_truthy(test):
                        break
                
                # Execute body
                try:
                    result = self.interpreter.visit(getattr(node, 'body', None))
                except ContinueException:
                    pass
                except BreakException:
                    break
                
                # Update
                if hasattr(node, 'update') and getattr(node, 'update', None):
                    self.interpreter.visit(getattr(node, 'update', None))
        except BreakException:
            pass
        
        return result
    
    def execute_try_statement(self, node: TryStatement) -> Any:
        """Execute try-catch statement."""
        self.interpreter._trace_execution(node, "try_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        try:
            return self.interpreter.visit(getattr(node, 'block', None))
        except Exception as e:
            # Handle catch clauses
            for handler in getattr(node, 'handlers', []):
                if self._exception_matches(e, handler):
                    # Bind exception to parameter if specified
                    param = getattr(handler, 'param', None)
                    if param and isinstance(param, Identifier):
                        param_name = getattr(param, 'name', 'error')
                        self.interpreter.current_environment.define(param_name, e, "variable")
                    
                    return self.interpreter.visit(getattr(handler, 'body', None))
            
            # No matching handler, re-raise
            raise
        finally:
            # Execute finally block if present
            if hasattr(node, 'finalizer') and getattr(node, 'finalizer', None):
                self.interpreter.visit(getattr(node, 'finalizer', None))
    
    def execute_return_statement(self, node: ReturnStatement) -> Any:
        """Execute return statement."""
        self.interpreter._trace_execution(node, "return_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        value = None
        if hasattr(node, 'argument') and getattr(node, 'argument', None):
            value = self.interpreter.visit(getattr(node, 'argument', None))
        
        raise ReturnException(value)
    
    def execute_break_statement(self, node: BreakStatement) -> Any:
        """Execute break statement."""
        self.interpreter._trace_execution(node, "break_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        raise BreakException()
    
    def execute_continue_statement(self, node: ContinueStatement) -> Any:
        """Execute continue statement."""
        self.interpreter._trace_execution(node, "continue_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        raise ContinueException()
    
    def execute_expression_statement(self, node: ExpressionStatement) -> Any:
        """Execute expression statement."""
        self.interpreter._trace_execution(node, "expression_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        return self.interpreter.visit(getattr(node, 'expression', None))
    
    def execute_block_statement(self, node: BlockStatement) -> Any:
        """Execute block statement."""
        self.interpreter._trace_execution(node, "block_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        
        # Create new scope
        self.interpreter.current_environment = self.interpreter.current_environment.create_child()
        
        try:
            result = None
            for stmt in getattr(node, 'body', []):
                result = self.interpreter.visit(stmt)
            return result
        finally:
            # Restore parent scope
            self.interpreter.current_environment = self.interpreter.current_environment.parent
    
    def execute_pulse_statement(self, node: PulseStatement) -> Any:
        """Execute neural pulse statement."""
        self.interpreter._trace_execution(node, "pulse_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        self.interpreter.execution_stats.neural_activations += 1
        
        # Get signal value
        signal = None
        if hasattr(node, 'signal') and getattr(node, 'signal', None):
            signal = self.interpreter.visit(getattr(node, 'signal', None))
        
        # Get target
        target = "broadcast"
        if hasattr(node, 'target') and getattr(node, 'target', None):
            target_node = getattr(node, 'target', None)
            if isinstance(target_node, Identifier):
                target = getattr(target_node, 'name', 'broadcast')
            else:
                target = str(self.interpreter.visit(target_node))
        
        # Get intensity
        intensity = 1.0
        if hasattr(node, 'intensity') and getattr(node, 'intensity', None):
            intensity = float(self.interpreter.visit(getattr(node, 'intensity', None)))
        
        # Create and queue signal
        signal_data = SignalData(
            source="pulse_statement",
            target=target,
            signal=signal,
            intensity=intensity,
            signal_type="sync"
        )
        
        self.interpreter.program_state.signal_queue.append(signal_data)
        
        # Also raise pulse exception for immediate handling
        raise PulseException(signal, target, intensity)
    
    def execute_resonate_statement(self, node: ResonateStatement) -> Any:
        """Execute neural resonate statement."""
        self.interpreter._trace_execution(node, "resonate_statement")
        self.interpreter.execution_stats.nodes_executed += 1
        self.interpreter.execution_stats.neural_activations += 1
        
        # Get frequency
        frequency = 1.0
        if hasattr(node, 'frequency') and getattr(node, 'frequency', None):
            frequency = float(self.interpreter.visit(getattr(node, 'frequency', None)))
        
        # Get duration
        duration = 1.0
        if hasattr(node, 'duration') and getattr(node, 'duration', None):
            duration = float(self.interpreter.visit(getattr(node, 'duration', None)))
        
        # Get target neurons
        targets = []
        if hasattr(node, 'targets') and getattr(node, 'targets', None):
            targets_node = getattr(node, 'targets', None)
            if isinstance(targets_node, list):
                for target_node in targets_node:
                    if isinstance(target_node, Identifier):
                        targets.append(getattr(target_node, 'name', 'unknown'))
                    else:
                        targets.append(str(self.interpreter.visit(target_node)))
            else:
                if isinstance(targets_node, Identifier):
                    targets.append(getattr(targets_node, 'name', 'unknown'))
                else:
                    targets.append(str(self.interpreter.visit(targets_node)))
        
        # Apply resonance to all neurons if no specific targets
        if not targets:
            targets = list(self.interpreter.program_state.neurons.keys())
        
        # Create resonance signals
        for target in targets:
            signal_data = SignalData(
                source="resonate_statement",
                target=target,
                signal=frequency,
                intensity=duration,
                signal_type="resonance"
            )
            self.interpreter.program_state.signal_queue.append(signal_data)
        
        return len(targets)  # Return number of affected neurons
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _is_truthy(self, value: Any) -> bool:
        """Check if value is truthy."""
        if value is None or value is False:
            return False
        if isinstance(value, (int, float)) and value == 0:
            return False
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            return False
        return True
    
    def _exception_matches(self, exception: Exception, handler: CatchClause) -> bool:
        """Check if exception matches catch handler."""
        # If no specific type specified, catch all
        if not hasattr(handler, 'type') or not getattr(handler, 'type', None):
            return True
        
        # Get exception type name
        exception_type = type(exception).__name__
        
        # Get handler type
        handler_type_node = getattr(handler, 'type', None)
        if isinstance(handler_type_node, Identifier):
            handler_type = getattr(handler_type_node, 'name', 'Exception')
        else:
            handler_type = str(self.interpreter.visit(handler_type_node))
        
        # Simple string matching for now
        return exception_type == handler_type or handler_type == "Exception" 