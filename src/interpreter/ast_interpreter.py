"""
Main AST Interpreter Module.

This module provides the main ASTInterpreter class that coordinates
all components of the AST interpretation system.
"""

import time
import asyncio
import sys
import os
import threading
import json
import math
import operator
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
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
        # Create comprehensive fallback classes
        class ProgramState:
            def __init__(self):
                self.variables = {}
                self.functions = {}
                self.neurons = {}
                self.synapses = {}
                self.signal_queue = deque()
                self.imports = {}
                self.exports = {}
                self.call_stack = []
                self.async_tasks = {}
        
        class ExecutionStats:
            def __init__(self):
                self.nodes_executed = 0
                self.execution_time = 0.0
                self.function_calls = 0
                self.neural_activations = 0
                self.signal_transmissions = 0
        
        class InterpreterConfig:
            def __init__(self, **kwargs):
                self.debug_mode = kwargs.get('debug_mode', False)
                self.async_enabled = kwargs.get('async_enabled', True)
                self.max_execution_time = kwargs.get('max_execution_time', 300.0)
                self.max_recursion_depth = kwargs.get('max_recursion_depth', 1000)
                self.trace_execution = kwargs.get('trace_execution', False)
                self.enable_profiling = kwargs.get('enable_profiling', False)
                self.neural_processing_enabled = kwargs.get('neural_processing_enabled', True)
                self.strict_mode = kwargs.get('strict_mode', False)
        
        # Mock exceptions
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
            
            def create_child(self):
                return Environment(self)
        
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

# Try to import expression and statement handlers
try:
    from interpreter.ast_expressions import ExpressionEvaluator
    from interpreter.ast_statements import StatementExecutor
    HANDLERS_AVAILABLE = True
except ImportError:
    try:
        from ast_expressions import ExpressionEvaluator
        from ast_statements import StatementExecutor
        HANDLERS_AVAILABLE = True
    except ImportError:
        HANDLERS_AVAILABLE = False

# Import all AST node types from syntax module with comprehensive fallbacks
try:
    from syntax.nodes import (
        # Base classes
        ASTNode, Statement, Expression, Declaration, Literal,
        
        # Literals
        IntegerLiteral, FloatLiteral, StringLiteral, BooleanLiteral,
        ArrayLiteral, ObjectLiteral,
        
        # Identifiers and Types
        Identifier, TypeAnnotation,
        
        # Expressions
        BinaryExpression, UnaryExpression, CallExpression,
        MemberExpression, IndexExpression, ConditionalExpression,
        SignalExpression,
        
        # Statements
        IfStatement, WhileStatement, ForStatement, TryStatement,
        CatchClause, ReturnStatement, BreakStatement, ContinueStatement,
        ExpressionStatement, BlockStatement,
        
        # Declarations
        VariableDeclaration, FunctionDeclaration, NeuronDeclaration,
        SynapseDeclaration, TypeDeclaration,
        
        # Neural constructs
        PulseStatement, ResonateStatement, NeuralEntity,
        SynapseConnection, ActivationFunction,
        
        # Program structure
        Program, ImportDeclaration, ExportDeclaration,
        
        # Operators
        BinaryOperator, UnaryOperator
    )
    SYNTAX_AVAILABLE = True
except ImportError:
    # Comprehensive fallback mock classes if syntax module not available
    class MockNode:
        """Mock AST node for compatibility."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.location = None

    # Base classes
    class ASTNode(MockNode):
        pass

    class Statement(MockNode):
        pass

    class Expression(MockNode):
        pass

    class Declaration(MockNode):
        pass

    class Literal(MockNode):
        pass

    # Literals
    class IntegerLiteral(MockNode):
        def __init__(self, value=0, **kwargs):
            super().__init__(**kwargs)
            self.value = value

    class FloatLiteral(MockNode):
        def __init__(self, value=0.0, **kwargs):
            super().__init__(**kwargs)
            self.value = value

    class StringLiteral(MockNode):
        def __init__(self, value="", **kwargs):
            super().__init__(**kwargs)
            self.value = value

    class BooleanLiteral(MockNode):
        def __init__(self, value=False, **kwargs):
            super().__init__(**kwargs)
            self.value = value

    class ArrayLiteral(MockNode):
        def __init__(self, elements=None, **kwargs):
            super().__init__(**kwargs)
            self.elements = elements or []

    class ObjectLiteral(MockNode):
        def __init__(self, properties=None, **kwargs):
            super().__init__(**kwargs)
            self.properties = properties or []

    # Identifiers and Types
    class Identifier(MockNode):
        def __init__(self, name="unknown", **kwargs):
            super().__init__(**kwargs)
            self.name = name

    class TypeAnnotation(MockNode):
        def __init__(self, type_name="any", **kwargs):
            super().__init__(**kwargs)
            self.type_name = type_name

    # Expressions
    class BinaryExpression(MockNode):
        def __init__(self, left=None, operator=None, right=None, **kwargs):
            super().__init__(**kwargs)
            self.left = left
            self.operator = operator
            self.right = right

    class UnaryExpression(MockNode):
        def __init__(self, operator=None, operand=None, **kwargs):
            super().__init__(**kwargs)
            self.operator = operator
            self.operand = operand

    class CallExpression(MockNode):
        def __init__(self, callee=None, arguments=None, **kwargs):
            super().__init__(**kwargs)
            self.callee = callee
            self.arguments = arguments or []

    class MemberExpression(MockNode):
        def __init__(self, object=None, property=None, computed=False, **kwargs):
            super().__init__(**kwargs)
            self.object = object
            self.property = property
            self.computed = computed

    class IndexExpression(MockNode):
        def __init__(self, object=None, index=None, **kwargs):
            super().__init__(**kwargs)
            self.object = object
            self.index = index

    class ConditionalExpression(MockNode):
        def __init__(self, test=None, consequent=None, alternate=None, **kwargs):
            super().__init__(**kwargs)
            self.test = test
            self.consequent = consequent
            self.alternate = alternate

    class SignalExpression(MockNode):
        def __init__(self, source=None, target=None, signal=None, **kwargs):
            super().__init__(**kwargs)
            self.source = source
            self.target = target
            self.signal = signal

    # Statements
    class IfStatement(MockNode):
        def __init__(self, test=None, consequent=None, alternate=None, **kwargs):
            super().__init__(**kwargs)
            self.test = test
            self.consequent = consequent
            self.alternate = alternate

    class WhileStatement(MockNode):
        def __init__(self, test=None, body=None, **kwargs):
            super().__init__(**kwargs)
            self.test = test
            self.body = body

    class ForStatement(MockNode):
        def __init__(self, init=None, test=None, update=None, body=None, **kwargs):
            super().__init__(**kwargs)
            self.init = init
            self.test = test
            self.update = update
            self.body = body

    class TryStatement(MockNode):
        def __init__(self, block=None, handler=None, finalizer=None, **kwargs):
            super().__init__(**kwargs)
            self.block = block
            self.handler = handler
            self.finalizer = finalizer

    class CatchClause(MockNode):
        def __init__(self, param=None, body=None, **kwargs):
            super().__init__(**kwargs)
            self.param = param
            self.body = body

    class ReturnStatement(MockNode):
        def __init__(self, argument=None, **kwargs):
            super().__init__(**kwargs)
            self.argument = argument

    class BreakStatement(MockNode):
        pass

    class ContinueStatement(MockNode):
        pass

    class ExpressionStatement(MockNode):
        def __init__(self, expression=None, **kwargs):
            super().__init__(**kwargs)
            self.expression = expression

    class BlockStatement(MockNode):
        def __init__(self, body=None, **kwargs):
            super().__init__(**kwargs)
            self.body = body or []

    # Declarations
    class VariableDeclaration(MockNode):
        def __init__(self, declarations=None, **kwargs):
            super().__init__(**kwargs)
            self.declarations = declarations or []

    class FunctionDeclaration(MockNode):
        def __init__(self, id=None, params=None, body=None, **kwargs):
            super().__init__(**kwargs)
            self.id = id
            self.params = params or []
            self.body = body

    class NeuronDeclaration(MockNode):
        def __init__(self, id=None, activation_function=None, threshold=0.5, **kwargs):
            super().__init__(**kwargs)
            self.id = id
            self.activation_function = activation_function
            self.threshold = threshold

    class SynapseDeclaration(MockNode):
        def __init__(self, source=None, target=None, weight=1.0, **kwargs):
            super().__init__(**kwargs)
            self.source = source
            self.target = target
            self.weight = weight

    class TypeDeclaration(MockNode):
        def __init__(self, id=None, type_annotation=None, **kwargs):
            super().__init__(**kwargs)
            self.id = id
            self.type_annotation = type_annotation

    # Neural constructs
    class PulseStatement(MockNode):
        def __init__(self, signal=None, target=None, intensity=1.0, **kwargs):
            super().__init__(**kwargs)
            self.signal = signal
            self.target = target
            self.intensity = intensity

    class ResonateStatement(MockNode):
        def __init__(self, frequency=None, duration=1.0, **kwargs):
            super().__init__(**kwargs)
            self.frequency = frequency
            self.duration = duration

    class NeuralEntity(MockNode):
        def __init__(self, name="neuron", activation_function=None, threshold=0.5, state=None, **kwargs):
            super().__init__(**kwargs)
            self.name = name
            self.activation_function = activation_function or "linear"
            self.threshold = threshold
            self.state = state or {'activation': 0.0, 'last_signal': 0.0}
        
        def activate(self, signal):
            """Activate the neuron with given signal."""
            self.state['last_signal'] = signal
            if signal >= self.threshold:
                self.state['activation'] = min(1.0, signal)
                return self.state['activation']
            return 0.0
        
        def get_info(self):
            """Get neuron information."""
            return {
                'name': self.name,
                'activation_function': self.activation_function,
                'threshold': self.threshold,
                'state': self.state.copy()
            }

    class SynapseConnection(MockNode):
        def __init__(self, source="", target="", weight=1.0, connection_type="excitatory", **kwargs):
            super().__init__(**kwargs)
            self.source = source
            self.target = target
            self.weight = weight
            self.connection_type = connection_type

    class ActivationFunction(MockNode):
        def __init__(self, function_type="linear", **kwargs):
            super().__init__(**kwargs)
            self.function_type = function_type

    # Program structure
    class Program(MockNode):
        def __init__(self, body=None, **kwargs):
            super().__init__(**kwargs)
            self.body = body or []

    class ImportDeclaration(MockNode):
        def __init__(self, source=None, specifiers=None, **kwargs):
            super().__init__(**kwargs)
            self.source = source
            self.specifiers = specifiers or []

    class ExportDeclaration(MockNode):
        def __init__(self, declaration=None, **kwargs):
            super().__init__(**kwargs)
            self.declaration = declaration

    # Operators
    class BinaryOperator:
        ASSIGN = "="
        ARROW = "->"
        DOUBLE_ARROW = "=>"
        AND = "&&"
        OR = "||"
        ADD = "+"
        SUBTRACT = "-"
        MULTIPLY = "*"
        DIVIDE = "/"
        MODULO = "%"
        POWER = "**"
        EQUAL = "=="
        NOT_EQUAL = "!="
        LESS_THAN = "<"
        LESS_EQUAL = "<="
        GREATER_THAN = ">"
        GREATER_EQUAL = ">="
        BIT_AND = "&"
        BIT_OR = "|"
        BIT_XOR = "^"
        LEFT_SHIFT = "<<"
        RIGHT_SHIFT = ">>"

    class UnaryOperator:
        PLUS = "+"
        MINUS = "-"
        NOT = "!"
        BIT_NOT = "~"

    SYNTAX_AVAILABLE = False

# Create a generic literal class for compatibility
class GenericLiteral(Literal):
    """Generic literal for compatibility with different literal types."""
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

class ASTInterpreter:
    """Main AST interpreter class."""
    
    def __init__(self, config: Optional[InterpreterConfig] = None):
        """Initialize the interpreter."""
        self.config = config or InterpreterConfig()
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ValueError(f"Invalid configuration: {', '.join(config_issues)}")
        
        # Initialize state
        self.execution_state = ExecutionState.READY
        self.execution_stats = ExecutionStats()
        self.program_state = ProgramState()
        self.debug_trace = deque(maxlen=1000)
        
        # Initialize environment
        self.global_environment = Environment()
        self.current_environment = self.global_environment
        
        # Initialize components
        self.builtin_functions = BuiltinFunctions(self.program_state)
        self.expression_evaluator = ExpressionEvaluator(self)
        self.statement_executor = StatementExecutor(self)
        
        # Setup built-in functions
        self._setup_builtins()
        
        # Execution control
        self.start_time = None
        self.max_recursion_depth = self.config.max_recursion_depth
        self.current_recursion_depth = 0
    
    def interpret(self, ast: Program) -> Any:
        """Interpret an AST program."""
        try:
            self.execution_state = ExecutionState.RUNNING
            self.start_time = time.time()
            self.execution_stats.reset()
            
            # Visit the program
            result = self.visit(ast)
            
            self.execution_state = ExecutionState.COMPLETED
            self.execution_stats.execution_time = time.time() - self.start_time
            
            return result
            
        except Exception as e:
            self.execution_state = ExecutionState.ERROR
            self.execution_stats.errors += 1
            
            if self.config.continue_on_error:
                print(f"Error during interpretation: {e}")
                return None
            else:
                raise RuntimeInterpreterError(f"Interpretation failed: {e}")
    
    def visit(self, node: ASTNode) -> Any:
        """Visit an AST node using the visitor pattern."""
        if node is None:
            return None
        
        # Check execution limits
        self._check_execution_limits()
        
        # Get visitor method name
        method_name = f"visit_{type(node).__name__.lower()}"
        
        # Get visitor method
        visitor = getattr(self, method_name, self.generic_visit)
        
        # Call visitor method
        return visitor(node)
    
    def generic_visit(self, node: ASTNode) -> Any:
        """Generic visitor for unknown node types."""
        raise NotImplementedError(f"No visitor method for {type(node).__name__}")
    
    # =========================================================================
    # PROGRAM AND STRUCTURE VISITORS
    # =========================================================================
    
    def visit_program(self, node: Program) -> Any:
        """Visit program node."""
        self._trace_execution(node, "program")
        self.execution_stats.nodes_executed += 1
        
        result = None
        for stmt in node.body:
            result = self.visit(stmt)
        
        return result
    
    def visit_blockstatement(self, node: BlockStatement) -> Any:
        """Visit block statement."""
        return self.statement_executor.execute_block_statement(node)
    
    # =========================================================================
    # LITERAL VISITORS
    # =========================================================================
    
    def visit_integerliteral(self, node: IntegerLiteral) -> int:
        """Visit integer literal."""
        self._trace_execution(node, "integer_literal", node.value)
        self.execution_stats.nodes_executed += 1
        return node.value
    
    def visit_floatliteral(self, node: FloatLiteral) -> float:
        """Visit float literal."""
        self._trace_execution(node, "float_literal", node.value)
        self.execution_stats.nodes_executed += 1
        return node.value
    
    def visit_stringliteral(self, node: StringLiteral) -> str:
        """Visit string literal."""
        self._trace_execution(node, "string_literal", node.value)
        self.execution_stats.nodes_executed += 1
        return node.value
    
    def visit_booleanliteral(self, node: BooleanLiteral) -> bool:
        """Visit boolean literal."""
        self._trace_execution(node, "boolean_literal", node.value)
        self.execution_stats.nodes_executed += 1
        return node.value
    
    def visit_arrayliteral(self, node: ArrayLiteral) -> List[Any]:
        """Visit array literal."""
        return self.expression_evaluator.evaluate_array_literal(node)
    
    def visit_objectliteral(self, node: ObjectLiteral) -> Dict[Any, Any]:
        """Visit object literal."""
        return self.expression_evaluator.evaluate_object_literal(node)
    
    def visit_literal(self, node: GenericLiteral) -> Any:
        """Visit generic literal."""
        self._trace_execution(node, "literal", node.value)
        self.execution_stats.nodes_executed += 1
        return node.value
    
    # =========================================================================
    # IDENTIFIER AND TYPE VISITORS
    # =========================================================================
    
    def visit_identifier(self, node: Identifier) -> Any:
        """Visit identifier."""
        self._trace_execution(node, "identifier", node.name)
        self.execution_stats.nodes_executed += 1
        
        if self.current_environment.has(node.name):
            return self.current_environment.get(node.name)
        else:
            raise NameInterpreterError(f"Undefined variable: {node.name}", node)
    
    def visit_typeannotation(self, node: TypeAnnotation) -> str:
        """Visit type annotation."""
        self._trace_execution(node, "type_annotation", node.type_name)
        self.execution_stats.nodes_executed += 1
        return node.type_name
    
    # =========================================================================
    # EXPRESSION VISITORS
    # =========================================================================
    
    def visit_binaryexpression(self, node: BinaryExpression) -> Any:
        """Visit binary expression."""
        return self.expression_evaluator.evaluate_binary_expression(node)
    
    def visit_unaryexpression(self, node: UnaryExpression) -> Any:
        """Visit unary expression."""
        return self.expression_evaluator.evaluate_unary_expression(node)
    
    def visit_callexpression(self, node: CallExpression) -> Any:
        """Visit call expression."""
        return self.expression_evaluator.evaluate_call_expression(node)
    
    def visit_memberexpression(self, node: MemberExpression) -> Any:
        """Visit member expression."""
        return self.expression_evaluator.evaluate_member_expression(node)
    
    def visit_indexexpression(self, node: IndexExpression) -> Any:
        """Visit index expression."""
        return self.expression_evaluator.evaluate_index_expression(node)
    
    def visit_conditionalexpression(self, node: ConditionalExpression) -> Any:
        """Visit conditional expression."""
        return self.expression_evaluator.evaluate_conditional_expression(node)
    
    def visit_signalexpression(self, node: SignalExpression) -> Any:
        """Visit signal expression."""
        return self.expression_evaluator.evaluate_signal_expression(node)
    
    # =========================================================================
    # STATEMENT VISITORS
    # =========================================================================
    
    def visit_ifstatement(self, node: IfStatement) -> Any:
        """Visit if statement."""
        return self.statement_executor.execute_if_statement(node)
    
    def visit_whilestatement(self, node: WhileStatement) -> Any:
        """Visit while statement."""
        return self.statement_executor.execute_while_statement(node)
    
    def visit_forstatement(self, node: ForStatement) -> Any:
        """Visit for statement."""
        return self.statement_executor.execute_for_statement(node)
    
    def visit_trystatement(self, node: TryStatement) -> Any:
        """Visit try statement."""
        return self.statement_executor.execute_try_statement(node)
    
    def visit_returnstatement(self, node: ReturnStatement) -> Any:
        """Visit return statement."""
        return self.statement_executor.execute_return_statement(node)
    
    def visit_breakstatement(self, node: BreakStatement) -> Any:
        """Visit break statement."""
        return self.statement_executor.execute_break_statement(node)
    
    def visit_continuestatement(self, node: ContinueStatement) -> Any:
        """Visit continue statement."""
        return self.statement_executor.execute_continue_statement(node)
    
    def visit_expressionstatement(self, node: ExpressionStatement) -> Any:
        """Visit expression statement."""
        return self.statement_executor.execute_expression_statement(node)
    
    # =========================================================================
    # DECLARATION VISITORS
    # =========================================================================
    
    def visit_variabledeclaration(self, node: VariableDeclaration) -> Any:
        """Visit variable declaration."""
        self._trace_execution(node, "variable_declaration", node.name.name)
        self.execution_stats.nodes_executed += 1
        
        # Get initial value
        value = None
        if node.initializer:
            value = self.visit(node.initializer)
        
        # Define variable
        self.current_environment.define(node.name.name, value, VariableType.VARIABLE)
        
        return value
    
    def visit_functiondeclaration(self, node: FunctionDeclaration) -> Any:
        """Visit function declaration."""
        self._trace_execution(node, "function_declaration", node.name.name)
        self.execution_stats.nodes_executed += 1
        
        # Store function in environment
        self.current_environment.define(node.name.name, node, VariableType.FUNCTION)
        
        # Also store in program state
        self.program_state.functions[node.name.name] = node
        
        return node
    
    def visit_neurondeclaration(self, node: NeuronDeclaration) -> Any:
        """Visit neuron declaration."""
        self._trace_execution(node, "neuron_declaration", node.name.name)
        self.execution_stats.nodes_executed += 1
        
        # Create neural entity
        neural_entity = NeuralEntity(
            name=node.name.name,
            activation_function=node.activation_function,
            threshold=getattr(node, 'threshold', 0.5),
            state={}
        )
        
        # Store in environment and program state
        self.current_environment.define(node.name.name, neural_entity, VariableType.NEURON)
        self.program_state.neurons[node.name.name] = neural_entity
        
        return neural_entity
    
    def visit_synapsedeclaration(self, node: SynapseDeclaration) -> Any:
        """Visit synapse declaration."""
        self._trace_execution(node, "synapse_declaration")
        self.execution_stats.nodes_executed += 1
        
        # Create synapse connection
        synapse = SynapseConnection(
            source=node.source.name,
            target=node.target.name,
            weight=node.weight if hasattr(node, 'weight') else 1.0,
            connection_type=getattr(node, 'connection_type', 'excitatory')
        )
        
        # Store in program state
        synapse_key = f"{node.source.name}->{node.target.name}"
        self.program_state.synapses[synapse_key] = synapse
        
        return synapse
    
    def visit_typedeclaration(self, node: TypeDeclaration) -> Any:
        """Visit type declaration."""
        self._trace_execution(node, "type_declaration", node.name.name)
        self.execution_stats.nodes_executed += 1
        
        # For now, just store the type definition
        self.current_environment.define(node.name.name, node, VariableType.TYPE)
        
        return node
    
    # =========================================================================
    # NEURAL CONSTRUCT VISITORS
    # =========================================================================
    
    def visit_pulsestatement(self, node: PulseStatement) -> Any:
        """Visit pulse statement."""
        return self.statement_executor.execute_pulse_statement(node)
    
    def visit_resonatestatement(self, node: ResonateStatement) -> Any:
        """Visit resonate statement."""
        return self.statement_executor.execute_resonate_statement(node)
    
    def visit_neuralentity(self, node: NeuralEntity) -> Any:
        """Visit neural entity."""
        self._trace_execution(node, "neural_entity", node.name)
        self.execution_stats.nodes_executed += 1
        return node
    
    def visit_synapseconnection(self, node: SynapseConnection) -> Any:
        """Visit synapse connection."""
        self._trace_execution(node, "synapse_connection")
        self.execution_stats.nodes_executed += 1
        return node
    
    def visit_activationfunction(self, node: ActivationFunction) -> Any:
        """Visit activation function."""
        self._trace_execution(node, "activation_function", node.function_type)
        self.execution_stats.nodes_executed += 1
        return node
    
    # =========================================================================
    # IMPORT/EXPORT VISITORS
    # =========================================================================
    
    def visit_importdeclaration(self, node: ImportDeclaration) -> Any:
        """Visit import declaration."""
        self._trace_execution(node, "import_declaration", node.source.value)
        self.execution_stats.nodes_executed += 1
        
        # Load module (simplified implementation)
        module_name = node.source.value
        module = self._load_module(module_name)
        
        # Handle different import types
        if hasattr(node, 'specifiers') and node.specifiers:
            for spec in node.specifiers:
                if hasattr(spec, 'imported') and hasattr(spec, 'local'):
                    # Named import: import { foo as bar } from 'module'
                    imported_name = spec.imported.name
                    local_name = spec.local.name
                    if hasattr(module, imported_name):
                        self.current_environment.define(local_name, getattr(module, imported_name), VariableType.IMPORT)
                else:
                    # Default import: import foo from 'module'
                    local_name = spec.local.name
                    self.current_environment.define(local_name, module, VariableType.IMPORT)
        else:
            # Import entire module
            self.current_environment.define(module_name, module, VariableType.IMPORT)
        
        self.program_state.imports[module_name] = module
        return module
    
    def visit_exportdeclaration(self, node: ExportDeclaration) -> Any:
        """Visit export declaration."""
        self._trace_execution(node, "export_declaration")
        self.execution_stats.nodes_executed += 1
        
        # Handle different export types
        if hasattr(node, 'declaration') and node.declaration:
            # export function foo() {}
            result = self.visit(node.declaration)
            if hasattr(node.declaration, 'name'):
                export_name = node.declaration.name.name
                self.program_state.exports[export_name] = result
            return result
        elif hasattr(node, 'specifiers') and node.specifiers:
            # export { foo, bar }
            for spec in node.specifiers:
                local_name = spec.local.name
                exported_name = spec.exported.name if hasattr(spec, 'exported') else local_name
                if self.current_environment.has(local_name):
                    value = self.current_environment.get(local_name)
                    self.program_state.exports[exported_name] = value
        
        return None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _setup_builtins(self):
        """Setup built-in functions in global environment."""
        builtins = self.builtin_functions.get_all_builtins()
        for name, func in builtins.items():
            self.global_environment.define(name, func, VariableType.BUILTIN)
    
    def _trace_execution(self, node: ASTNode, node_type: str, value: Any = None):
        """Trace execution for debugging."""
        if not self.config.trace_execution:
            return
        
        debug_info = DebugInfo(
            node_type=node_type,
            value=value,
            location=str(node.location) if hasattr(node, 'location') and node.location else None,
            stack_depth=len(self.program_state.call_stack),
            memory_usage=0  # TODO: implement memory tracking
        )
        
        self.debug_trace.append(debug_info)
    
    def _check_execution_limits(self):
        """Check execution limits and timeouts."""
        # Check execution time
        if self.start_time and self.config.max_execution_time > 0:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.max_execution_time:
                raise RuntimeError(f"Execution timeout: {elapsed:.2f}s > {self.config.max_execution_time}s")
        
        # Check recursion depth
        if len(self.program_state.call_stack) > self.max_recursion_depth:
            raise RuntimeError(f"Maximum recursion depth exceeded: {len(self.program_state.call_stack)}")
        
        # Check error count
        if self.execution_stats.errors > self.config.max_errors:
            raise RuntimeError(f"Too many errors: {self.execution_stats.errors}")
    
    def _call_user_function(self, func_node: FunctionDeclaration, args: List[Any]) -> Any:
        """Call user-defined function."""
        # Create new environment for function
        func_env = self.current_environment.create_child()
        
        # Bind parameters
        if hasattr(func_node, 'parameters'):
            for i, param in enumerate(func_node.parameters):
                param_name = param.name.name if hasattr(param, 'name') else str(param)
                value = args[i] if i < len(args) else None
                func_env.define(param_name, value, VariableType.PARAMETER)
        
        # Execute function body
        old_env = self.current_environment
        self.current_environment = func_env
        
        try:
            self.program_state.call_stack.append(func_node.name.name)
            result = self.visit(func_node.body)
            return result
        except ReturnException as ret:
            return ret.value
        finally:
            self.program_state.call_stack.pop()
            self.current_environment = old_env
    
    def _call_neural_function(self, neuron_node: NeuronDeclaration, args: List[Any]) -> Any:
        """Call neural function."""
        # Get or create neural entity
        neuron_name = neuron_node.name.name
        if neuron_name in self.program_state.neurons:
            neuron = self.program_state.neurons[neuron_name]
        else:
            neuron = NeuralEntity(
                name=neuron_name,
                activation_function=neuron_node.activation_function,
                threshold=getattr(neuron_node, 'threshold', 0.5),
                state={}
            )
            self.program_state.neurons[neuron_name] = neuron
        
        # Activate neuron with input signal
        input_signal = args[0] if args else 0.0
        return neuron.activate(input_signal)
    
    def _load_module(self, module_name: str) -> Any:
        """Load external module (simplified implementation)."""
        # This is a simplified implementation
        # In a real system, this would load actual modules
        class MockModule:
            def __init__(self, name):
                self.name = name
                self.loaded = True
        
        return MockModule(module_name)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            'state': self.execution_state.name,
            'stats': self.execution_stats.get_summary(),
            'program_state': self.program_state.get_snapshot(),
            'debug_trace_length': len(self.debug_trace)
        }