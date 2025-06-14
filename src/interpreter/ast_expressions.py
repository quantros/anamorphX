"""
Expression Handlers for AST Interpreter.

This module handles all expression evaluation in the AST interpreter.
"""

import time
import sys
import os
import math
import operator
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
        # Create minimal fallback classes
        class ProgramState:
            def __init__(self):
                self.variables = {}
                self.functions = {}
                self.neurons = {}
                self.synapses = {}
                self.signal_queue = deque()
        
        class SignalData:
            def __init__(self, source, target, signal, intensity=1.0):
                self.source = source
                self.target = target
                self.signal = signal
                self.intensity = intensity
        
        class ASTNode:
            def __init__(self):
                self.location = None
        
        class Expression(ASTNode):
            pass
        
        class Statement(ASTNode):
            pass
        
        class Literal(Expression):
            def __init__(self, value):
                super().__init__()
                self.value = value
        
        class SyntaxLiteral(Literal):
            pass
        
        class Identifier(Expression):
            def __init__(self, name):
                super().__init__()
                self.name = name
        
        class BinaryOperation(Expression):
            def __init__(self, left, operator, right):
                super().__init__()
                self.left = left
                self.operator = operator
                self.right = right
        
        class UnaryOperation(Expression):
            def __init__(self, operator, operand):
                super().__init__()
                self.operator = operator
                self.operand = operand
        
        class FunctionCall(Expression):
            def __init__(self, function, arguments):
                super().__init__()
                self.function = function
                self.arguments = arguments
        
        class Assignment(Expression):
            def __init__(self, target, value):
                super().__init__()
                self.target = target
                self.value = value
        
        class SignalFlow(Expression):
            def __init__(self, source, target, signal):
                super().__init__()
                self.source = source
                self.target = target
                self.signal = signal
        
        class NeuralEntity(Statement):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.state = {'activation': 0.0}
        
        class SynapseConnection(Statement):
            def __init__(self, source, target, weight=1.0):
                super().__init__()
                self.source = source
                self.target = target
                self.weight = weight
        
        # Mock exceptions
        class RuntimeInterpreterError(Exception):
            pass
        
        class NameInterpreterError(Exception):
            pass
        
        class TypeInterpreterError(Exception):
            pass
        
        AST_TYPES_AVAILABLE = False

# Try to import environment module
try:
    from interpreter.environment import Environment, VariableType
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    try:
        from environment import Environment, VariableType
        ENVIRONMENT_AVAILABLE = True
    except ImportError:
        # Mock environment
        class Environment:
            def __init__(self, parent=None):
                self.parent = parent
                self.variables = {}
            
            def get(self, name):
                if name in self.variables:
                    return self.variables[name]
                elif self.parent:
                    return self.parent.get(name)
                else:
                    raise NameError(f"Variable '{name}' not defined")
            
            def set(self, name, value):
                self.variables[name] = value
            
            def define(self, name, value, var_type=None):
                self.variables[name] = value
        
        ENVIRONMENT_AVAILABLE = False

# Try to import builtin functions
try:
    from interpreter.ast_builtins import BuiltinFunctions
    BUILTINS_AVAILABLE = True
except ImportError:
    try:
        from ast_builtins import BuiltinFunctions
        BUILTINS_AVAILABLE = True
    except ImportError:
        BUILTINS_AVAILABLE = False

# Try to import syntax nodes
try:
    from syntax.nodes import (
        ASTNode as SyntaxASTNode, Expression as SyntaxExpression,
        Statement as SyntaxStatement, Literal as SyntaxLiteral,
        Identifier as SyntaxIdentifier, BinaryOperation as SyntaxBinaryOperation,
        UnaryOperation as SyntaxUnaryOperation, FunctionCall as SyntaxFunctionCall,
        Assignment as SyntaxAssignment, SignalFlow as SyntaxSignalFlow,
        MemberAccess, ArrayAccess, ConditionalExpression, LambdaExpression
    )
    SYNTAX_AVAILABLE = True
except ImportError:
    # Mock additional syntax classes
    class MemberAccess(Expression):
        def __init__(self, object, member):
            super().__init__()
            self.object = object
            self.member = member
    
    class ArrayAccess(Expression):
        def __init__(self, array, index):
            super().__init__()
            self.array = array
            self.index = index
    
    class ConditionalExpression(Expression):
        def __init__(self, condition, true_expr, false_expr):
            super().__init__()
            self.condition = condition
            self.true_expr = true_expr
            self.false_expr = false_expr
    
    class LambdaExpression(Expression):
        def __init__(self, params, body):
            super().__init__()
            self.params = params
            self.body = body
    
    SYNTAX_AVAILABLE = False

class ExpressionEvaluator:
    """Handles evaluation of all expression types."""
    
    def __init__(self, interpreter):
        """Initialize with interpreter reference."""
        self.interpreter = interpreter
    
    def evaluate_binary_expression(self, node: Any) -> Any:
        """Evaluate binary expression."""
        self.interpreter._trace_execution(node, "binary_expression", getattr(node, 'operator', 'unknown'))
        self.interpreter.execution_stats.nodes_executed += 1
        
        # Handle assignment separately
        if getattr(node, 'operator', None) == BinaryOperator.ASSIGN:
            return self._handle_assignment(node)
        
        # Handle signal flow operators
        if getattr(node, 'operator', None) in (BinaryOperator.ARROW, BinaryOperator.DOUBLE_ARROW):
            return self._handle_signal_flow(node)
        
        # Evaluate operands
        left = self.interpreter.visit(node.left)
        
        # Short-circuit evaluation for logical operators
        if getattr(node, 'operator', None) == BinaryOperator.AND:
            if not self._is_truthy(left):
                return left
            return self.interpreter.visit(node.right)
        elif getattr(node, 'operator', None) == BinaryOperator.OR:
            if self._is_truthy(left):
                return left
            return self.interpreter.visit(node.right)
        
        right = self.interpreter.visit(node.right)
        
        # Arithmetic operations
        operator = getattr(node, 'operator', None)
        if operator == BinaryOperator.ADD:
            return left + right
        elif operator == BinaryOperator.SUBTRACT:
            return left - right
        elif operator == BinaryOperator.MULTIPLY:
            return left * right
        elif operator == BinaryOperator.DIVIDE:
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        elif operator == BinaryOperator.MODULO:
            return left % right
        elif operator == BinaryOperator.POWER:
            return left ** right
        
        # Comparison operations
        elif operator == BinaryOperator.EQUAL:
            return left == right
        elif operator == BinaryOperator.NOT_EQUAL:
            return left != right
        elif operator == BinaryOperator.LESS_THAN:
            return left < right
        elif operator == BinaryOperator.LESS_EQUAL:
            return left <= right
        elif operator == BinaryOperator.GREATER_THAN:
            return left > right
        elif operator == BinaryOperator.GREATER_EQUAL:
            return left >= right
        
        # Bitwise operations
        elif operator == BinaryOperator.BIT_AND:
            return left & right
        elif operator == BinaryOperator.BIT_OR:
            return left | right
        elif operator == BinaryOperator.BIT_XOR:
            return left ^ right
        elif operator == BinaryOperator.LEFT_SHIFT:
            return left << right
        elif operator == BinaryOperator.RIGHT_SHIFT:
            return left >> right
        
        else:
            raise NotImplementedError(f"Binary operator not implemented: {operator}")
    
    def evaluate_unary_expression(self, node: Any) -> Any:
        """Evaluate unary expression."""
        self.interpreter._trace_execution(node, "unary_expression", getattr(node, 'operator', 'unknown'))
        self.interpreter.execution_stats.nodes_executed += 1
        
        operand = self.interpreter.visit(node.operand)
        operator = getattr(node, 'operator', None)
        
        if operator == UnaryOperator.PLUS:
            return +operand
        elif operator == UnaryOperator.MINUS:
            return -operand
        elif operator == UnaryOperator.NOT:
            return not self._is_truthy(operand)
        elif operator == UnaryOperator.BIT_NOT:
            return ~operand
        else:
            raise NotImplementedError(f"Unary operator not implemented: {operator}")
    
    def evaluate_call_expression(self, node: Any) -> Any:
        """Evaluate function call expression."""
        self.interpreter._trace_execution(node, "call_expression")
        self.interpreter.execution_stats.nodes_executed += 1
        self.interpreter.execution_stats.function_calls += 1
        
        # Get the function
        callee = self.interpreter.visit(node.callee)
        
        # Evaluate arguments
        args = []
        for arg in getattr(node, 'arguments', []):
            args.append(self.interpreter.visit(arg))
        
        # Handle different types of callables
        if callable(callee):
            # Built-in function or Python callable
            try:
                return callee(*args)
            except Exception as e:
                raise RuntimeError(f"Function call error: {e}")
        
        elif hasattr(callee, 'id') and hasattr(callee, 'params'):
            # User-defined function
            return self.interpreter._call_user_function(callee, args)
        
        elif hasattr(callee, 'name') and hasattr(callee, 'activation_function'):
            # Neural function
            return self.interpreter._call_neural_function(callee, args)
        
        else:
            raise TypeError(f"Object is not callable: {type(callee)}")
    
    def evaluate_member_expression(self, node: Any) -> Any:
        """Evaluate member access expression."""
        self.interpreter._trace_execution(node, "member_expression")
        self.interpreter.execution_stats.nodes_executed += 1
        
        obj = self.interpreter.visit(getattr(node, 'object', None))
        
        if getattr(node, 'computed', False):
            # obj[property]
            prop = self.interpreter.visit(getattr(node, 'property', None))
        else:
            # obj.property
            property_node = getattr(node, 'property', None)
            if hasattr(property_node, 'name'):
                prop = getattr(property_node, 'name', 'unknown')
            else:
                prop = self.interpreter.visit(property_node)
        
        # Handle different object types
        if isinstance(obj, dict):
            return obj.get(prop)
        elif hasattr(obj, prop):
            return getattr(obj, prop)
        elif isinstance(obj, (list, tuple)) and isinstance(prop, int):
            return obj[prop]
        else:
            raise AttributeError(f"Object has no attribute: {prop}")
    
    def evaluate_index_expression(self, node: Any) -> Any:
        """Evaluate index expression."""
        self.interpreter._trace_execution(node, "index_expression")
        self.interpreter.execution_stats.nodes_executed += 1
        
        obj = self.interpreter.visit(getattr(node, 'object', None))
        index = self.interpreter.visit(getattr(node, 'index', None))
        
        try:
            return obj[index]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Index error: {e}")
    
    def evaluate_conditional_expression(self, node: Any) -> Any:
        """Evaluate conditional (ternary) expression."""
        self.interpreter._trace_execution(node, "conditional_expression")
        self.interpreter.execution_stats.nodes_executed += 1
        
        test = self.interpreter.visit(getattr(node, 'test', None))
        
        if self._is_truthy(test):
            return self.interpreter.visit(getattr(node, 'consequent', None))
        else:
            return self.interpreter.visit(getattr(node, 'alternate', None))
    
    def evaluate_signal_expression(self, node: Any) -> Any:
        """Evaluate signal flow expression (Anamorph-specific)."""
        signal_type = getattr(node, 'signal_type', 'sync')
        self.interpreter._trace_execution(node, "signal_expression", signal_type)
        self.interpreter.execution_stats.nodes_executed += 1
        
        source = self.interpreter.visit(getattr(node, 'source', None))
        target_node = getattr(node, 'target', None)
        target_name = None
        
        if hasattr(target_node, 'name'):
            target_name = getattr(target_node, 'name', 'broadcast')
        else:
            target_name = str(self.interpreter.visit(target_node))
        
        signal_value = self.interpreter.visit(getattr(node, 'signal', None))
        intensity = getattr(node, 'intensity', 1.0)
        
        # Create signal data
        if AST_TYPES_AVAILABLE:
            signal_data = SignalData(
                source=source,
                target=target_name,
                signal=signal_value,
                intensity=intensity,
                signal_type=signal_type
            )
        else:
            # Fallback signal data
            signal_data = {
                'source': source,
                'target': target_name,
                'signal': signal_value,
                'intensity': intensity,
                'signal_type': signal_type
            }
        
        # Add to signal queue
        self.interpreter.program_state.signal_queue.append(signal_data)
        self.interpreter.execution_stats.signal_transmissions += 1
        
        return signal_value

    def evaluate_array_literal(self, node: Any) -> List[Any]:
        """Evaluate array literal."""
        self.interpreter._trace_execution(node, "array_literal")
        self.interpreter.execution_stats.nodes_executed += 1
        
        elements = []
        for element in getattr(node, 'elements', []):
            elements.append(self.interpreter.visit(element))
        
        return elements

    def evaluate_object_literal(self, node: Any) -> Dict[Any, Any]:
        """Evaluate object literal."""
        self.interpreter._trace_execution(node, "object_literal")
        self.interpreter.execution_stats.nodes_executed += 1
        
        obj = {}
        for prop in getattr(node, 'properties', []):
            key = self.interpreter.visit(getattr(prop, 'key', None))
            value = self.interpreter.visit(getattr(prop, 'value', None))
            obj[key] = value
        
        return obj

    def _handle_assignment(self, node: Any) -> Any:
        """Handle assignment expression."""
        target = getattr(node, 'left', None)
        value = self.interpreter.visit(getattr(node, 'right', None))
        
        if hasattr(target, 'name'):
            # Simple identifier assignment
            var_name = getattr(target, 'name', 'unknown')
            self.interpreter.current_environment.set(var_name, value)
            return value
        
        elif hasattr(target, 'object') and hasattr(target, 'property'):
            # Member assignment (obj.prop = value)
            obj = self.interpreter.visit(getattr(target, 'object', None))
            
            if getattr(target, 'computed', False):
                # obj[prop] = value
                prop = self.interpreter.visit(getattr(target, 'property', None))
            else:
                # obj.prop = value
                property_node = getattr(target, 'property', None)
                if hasattr(property_node, 'name'):
                    prop = getattr(property_node, 'name', 'unknown')
                else:
                    prop = self.interpreter.visit(property_node)
            
            if isinstance(obj, dict):
                obj[prop] = value
            else:
                setattr(obj, prop, value)
            
            return value
        
        else:
            raise RuntimeError(f"Invalid assignment target: {type(target)}")

    def _handle_signal_flow(self, node: Any) -> Any:
        """Handle signal flow expression (-> and =>)."""
        source = self.interpreter.visit(getattr(node, 'left', None))
        target_node = getattr(node, 'right', None)
        
        if hasattr(target_node, 'name'):
            target_name = getattr(target_node, 'name', 'broadcast')
        else:
            target_name = str(self.interpreter.visit(target_node))
        
        operator = getattr(node, 'operator', None)
        signal_type = "async" if operator == BinaryOperator.DOUBLE_ARROW else "sync"
        
        # Create and queue signal
        if AST_TYPES_AVAILABLE:
            signal_data = SignalData(
                source="expression",
                target=target_name,
                signal=source,
                intensity=1.0,
                signal_type=signal_type
            )
        else:
            signal_data = {
                'source': "expression",
                'target': target_name,
                'signal': source,
                'intensity': 1.0,
                'signal_type': signal_type
            }
        
        self.interpreter.program_state.signal_queue.append(signal_data)
        self.interpreter.execution_stats.signal_transmissions += 1
        
        return source

    def _is_truthy(self, value: Any) -> bool:
        """Check if a value is truthy."""
        if value is None or value is False:
            return False
        if isinstance(value, (int, float)) and value == 0:
            return False
        if isinstance(value, (str, list, dict, tuple)) and len(value) == 0:
            return False
        return True 