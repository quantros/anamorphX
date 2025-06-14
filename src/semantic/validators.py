"""
Semantic Validators for AnamorphX

This module provides comprehensive validation for semantic analysis,
including declaration validation, expression validation, and neural construct validation.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from abc import ABC, abstractmethod
from ..syntax.nodes import ASTNode, SourceLocation, NodeType
from .symbols import Symbol, SymbolType, SymbolResolver
from .types import Type, TypeSystem, TypeChecker
from .scopes import ScopeManager, ScopeType
from .errors import SemanticError, SemanticErrorType, TypeError, ScopeError, NeuralError


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    is_valid: bool
    errors: List[SemanticError] = field(default_factory=list)
    warnings: List[SemanticError] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: SemanticError):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: SemanticError):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.context.update(other.context)
        if not other.is_valid:
            self.is_valid = False


class ValidationRule(ABC):
    """Base class for validation rules."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.enabled = True
    
    @abstractmethod
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        """Validate a node according to this rule."""
        pass


class SemanticValidator(ABC):
    """Base class for semantic validators."""
    
    def __init__(self, type_system: TypeSystem, scope_manager: ScopeManager, 
                 symbol_resolver: SymbolResolver):
        self.type_system = type_system
        self.scope_manager = scope_manager
        self.symbol_resolver = symbol_resolver
        self.type_checker = TypeChecker(type_system)
        self.rules: List[ValidationRule] = []
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def validate_node(self, node: ASTNode) -> ValidationResult:
        """Validate a node using all applicable rules."""
        result = ValidationResult(is_valid=True)
        
        for rule in self.rules:
            if rule.enabled:
                rule_result = rule.validate(node, self._get_context())
                result.merge(rule_result)
        
        return result
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current validation context."""
        return {
            'type_system': self.type_system,
            'scope_manager': self.scope_manager,
            'symbol_resolver': self.symbol_resolver,
            'type_checker': self.type_checker,
            'current_scope': self.scope_manager.current_scope
        }


class DeclarationValidator(SemanticValidator):
    """Validates variable and function declarations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup declaration validation rules."""
        self.add_rule(VariableDeclarationRule())
        self.add_rule(FunctionDeclarationRule())
        self.add_rule(ParameterDeclarationRule())
        self.add_rule(RedefinitionRule())
        self.add_rule(InitializationRule())
    
    def validate_variable_declaration(self, node: ASTNode) -> ValidationResult:
        """Validate variable declaration."""
        result = ValidationResult(is_valid=True)
        
        # Check if variable name is valid
        if not self._is_valid_identifier(node.name):
            result.add_error(SemanticError(
                SemanticErrorType.INVALID_DECLARATION,
                f"Invalid variable name '{node.name}'",
                location=node.location
            ))
        
        # Check type annotation
        if hasattr(node, 'type_annotation') and node.type_annotation:
            declared_type = self.type_system.get_type(node.type_annotation)
            if not declared_type:
                result.add_error(TypeError(
                    f"Unknown type '{node.type_annotation}'",
                    location=node.location
                ))
        
        # Check initializer type compatibility
        if hasattr(node, 'initializer') and node.initializer:
            initializer_type = self.type_checker.check_node(node.initializer)
            if hasattr(node, 'type_annotation') and node.type_annotation:
                declared_type = self.type_system.get_type(node.type_annotation)
                if declared_type and initializer_type:
                    if not declared_type.is_compatible_with(initializer_type):
                        result.add_error(TypeError(
                            f"Cannot initialize {declared_type} with {initializer_type}",
                            expected_type=str(declared_type),
                            actual_type=str(initializer_type),
                            location=node.location
                        ))
        
        return result
    
    def validate_function_declaration(self, node: ASTNode) -> ValidationResult:
        """Validate function declaration."""
        result = ValidationResult(is_valid=True)
        
        # Check function name
        if not self._is_valid_identifier(node.name):
            result.add_error(SemanticError(
                SemanticErrorType.INVALID_DECLARATION,
                f"Invalid function name '{node.name}'",
                location=node.location
            ))
        
        # Check return type
        if hasattr(node, 'return_type') and node.return_type:
            return_type = self.type_system.get_type(node.return_type)
            if not return_type:
                result.add_error(TypeError(
                    f"Unknown return type '{node.return_type}'",
                    location=node.location
                ))
        
        # Check parameters
        if hasattr(node, 'parameters'):
            param_names = set()
            for param in node.parameters:
                # Check for duplicate parameter names
                if param.name in param_names:
                    result.add_error(SemanticError(
                        SemanticErrorType.DUPLICATE_DECLARATION,
                        f"Duplicate parameter name '{param.name}'",
                        location=param.location
                    ))
                param_names.add(param.name)
                
                # Validate parameter type
                if hasattr(param, 'type_annotation') and param.type_annotation:
                    param_type = self.type_system.get_type(param.type_annotation)
                    if not param_type:
                        result.add_error(TypeError(
                            f"Unknown parameter type '{param.type_annotation}'",
                            location=param.location
                        ))
        
        return result
    
    def _is_valid_identifier(self, name: str) -> bool:
        """Check if identifier name is valid."""
        if not name or not name.isidentifier():
            return False
        
        # Check against reserved keywords
        reserved = {
            'if', 'else', 'while', 'for', 'function', 'return', 'break', 'continue',
            'try', 'catch', 'finally', 'throw', 'neuron', 'synapse', 'pulse', 'resonate',
            'signal', 'true', 'false', 'null', 'undefined', 'var', 'let', 'const'
        }
        
        return name not in reserved


class ExpressionValidator(SemanticValidator):
    """Validates expressions and their types."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup expression validation rules."""
        self.add_rule(BinaryExpressionRule())
        self.add_rule(UnaryExpressionRule())
        self.add_rule(CallExpressionRule())
        self.add_rule(MemberAccessRule())
        self.add_rule(ArrayAccessRule())
    
    def validate_binary_expression(self, node: ASTNode) -> ValidationResult:
        """Validate binary expression."""
        result = ValidationResult(is_valid=True)
        
        left_type = self.type_checker.check_node(node.left)
        right_type = self.type_checker.check_node(node.right)
        
        if not left_type or not right_type:
            result.add_error(TypeError(
                "Cannot determine operand types",
                location=node.location
            ))
            return result
        
        # Check operator compatibility
        if not self._is_operator_compatible(node.operator, left_type, right_type):
            result.add_error(TypeError(
                f"Operator '{node.operator}' not supported for types {left_type} and {right_type}",
                location=node.location
            ))
        
        return result
    
    def validate_call_expression(self, node: ASTNode) -> ValidationResult:
        """Validate function call expression."""
        result = ValidationResult(is_valid=True)
        
        # Resolve function symbol
        if hasattr(node, 'callee') and hasattr(node.callee, 'name'):
            function_symbol = self.symbol_resolver.resolve_symbol(
                node.callee.name, node.location
            )
            
            if not function_symbol:
                result.add_error(SemanticError(
                    SemanticErrorType.FUNCTION_NOT_FOUND,
                    f"Function '{node.callee.name}' not found",
                    location=node.location
                ))
                return result
            
            if function_symbol.symbol_type != SymbolType.FUNCTION:
                result.add_error(SemanticError(
                    SemanticErrorType.INVALID_SYMBOL_USAGE,
                    f"'{node.callee.name}' is not a function",
                    location=node.location
                ))
                return result
        
        # Check argument count and types
        if hasattr(node, 'arguments'):
            # This would need function signature information
            pass
        
        return result
    
    def _is_operator_compatible(self, operator: str, left_type: Type, right_type: Type) -> bool:
        """Check if operator is compatible with operand types."""
        from .types import PrimitiveType, TypeCompatibility
        
        # Arithmetic operators
        if operator in ['+', '-', '*', '/', '%']:
            return (TypeCompatibility.is_numeric(left_type) and 
                   TypeCompatibility.is_numeric(right_type))
        
        # Comparison operators
        elif operator in ['==', '!=']:
            return left_type.is_compatible_with(right_type)
        
        elif operator in ['<', '>', '<=', '>=']:
            return (TypeCompatibility.is_numeric(left_type) and 
                   TypeCompatibility.is_numeric(right_type))
        
        # Logical operators
        elif operator in ['&&', '||']:
            return (isinstance(left_type, PrimitiveType) and left_type.name == 'bool' and
                   isinstance(right_type, PrimitiveType) and right_type.name == 'bool')
        
        return False


class StatementValidator(SemanticValidator):
    """Validates statements and control flow."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup statement validation rules."""
        self.add_rule(ReturnStatementRule())
        self.add_rule(BreakContinueRule())
        self.add_rule(IfStatementRule())
        self.add_rule(LoopStatementRule())
    
    def validate_return_statement(self, node: ASTNode) -> ValidationResult:
        """Validate return statement."""
        result = ValidationResult(is_valid=True)
        
        # Check if in function
        function_scope = self.scope_manager.find_enclosing_function()
        if not function_scope:
            result.add_error(ScopeError(
                "Return statement outside of function",
                location=node.location
            ))
            return result
        
        # Check return value type
        if hasattr(node, 'value') and node.value:
            return_type = self.type_checker.check_node(node.value)
            if function_scope.return_type:
                expected_type = self.type_system.get_type(function_scope.return_type)
                if expected_type and return_type:
                    if not expected_type.is_compatible_with(return_type):
                        result.add_error(TypeError(
                            f"Return type mismatch: expected {expected_type}, got {return_type}",
                            expected_type=str(expected_type),
                            actual_type=str(return_type),
                            location=node.location
                        ))
        elif function_scope.return_type and function_scope.return_type != 'void':
            result.add_error(SemanticError(
                SemanticErrorType.MISSING_RETURN_VALUE,
                "Function must return a value",
                location=node.location
            ))
        
        return result
    
    def validate_break_continue(self, node: ASTNode) -> ValidationResult:
        """Validate break/continue statements."""
        result = ValidationResult(is_valid=True)
        
        is_break = node.node_type == NodeType.BREAK_STATEMENT
        is_continue = node.node_type == NodeType.CONTINUE_STATEMENT
        
        if is_break and not self.scope_manager.can_break():
            result.add_error(ScopeError(
                "Break statement outside of loop",
                location=node.location
            ))
        elif is_continue and not self.scope_manager.can_continue():
            result.add_error(ScopeError(
                "Continue statement outside of loop",
                location=node.location
            ))
        
        return result


class NeuralValidator(SemanticValidator):
    """Validates neural constructs and neural networks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup neural validation rules."""
        self.add_rule(NeuronDeclarationRule())
        self.add_rule(SynapseDeclarationRule())
        self.add_rule(PulseStatementRule())
        self.add_rule(SignalExpressionRule())
        self.add_rule(NeuralNetworkRule())
    
    def validate_neuron_declaration(self, node: ASTNode) -> ValidationResult:
        """Validate neuron declaration."""
        result = ValidationResult(is_valid=True)
        
        # Check neuron name
        if not self._is_valid_identifier(node.name):
            result.add_error(NeuralError(
                f"Invalid neuron name '{node.name}'",
                neural_type='neuron',
                location=node.location
            ))
        
        # Check neuron type
        if hasattr(node, 'neuron_type'):
            valid_types = {'basic', 'perceptron', 'lstm', 'gru', 'cnn'}
            if node.neuron_type not in valid_types:
                result.add_error(NeuralError(
                    f"Unknown neuron type '{node.neuron_type}'",
                    neural_type='neuron',
                    location=node.location
                ))
        
        # Check activation function
        if hasattr(node, 'activation_function'):
            valid_functions = {'sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax'}
            if node.activation_function not in valid_functions:
                result.add_error(NeuralError(
                    f"Unknown activation function '{node.activation_function}'",
                    neural_type='neuron',
                    location=node.location
                ))
        
        return result
    
    def validate_synapse_declaration(self, node: ASTNode) -> ValidationResult:
        """Validate synapse declaration."""
        result = ValidationResult(is_valid=True)
        
        # Check source neuron exists
        source_symbol = self.symbol_resolver.resolve_symbol(node.source, node.location)
        if not source_symbol:
            result.add_error(NeuralError(
                f"Source neuron '{node.source}' not found",
                neural_type='synapse',
                location=node.location
            ))
        elif source_symbol.symbol_type != SymbolType.NEURON:
            result.add_error(NeuralError(
                f"'{node.source}' is not a neuron",
                neural_type='synapse',
                location=node.location
            ))
        
        # Check target neuron exists
        target_symbol = self.symbol_resolver.resolve_symbol(node.target, node.location)
        if not target_symbol:
            result.add_error(NeuralError(
                f"Target neuron '{node.target}' not found",
                neural_type='synapse',
                location=node.location
            ))
        elif target_symbol.symbol_type != SymbolType.NEURON:
            result.add_error(NeuralError(
                f"'{node.target}' is not a neuron",
                neural_type='synapse',
                location=node.location
            ))
        
        # Check weight range
        if hasattr(node, 'weight') and node.weight is not None:
            if not isinstance(node.weight, (int, float)) or abs(node.weight) > 10:
                result.add_warning(NeuralError(
                    f"Synapse weight {node.weight} may be too large",
                    neural_type='synapse',
                    location=node.location
                ))
        
        return result
    
    def validate_pulse_statement(self, node: ASTNode) -> ValidationResult:
        """Validate pulse statement."""
        result = ValidationResult(is_valid=True)
        
        # Check if in neuron context
        neuron_scope = self.scope_manager.find_enclosing_neuron()
        if not neuron_scope:
            result.add_error(NeuralError(
                "Pulse statement must be inside a neuron",
                neural_type='pulse',
                location=node.location
            ))
        
        # Check pulse parameters
        if hasattr(node, 'frequency') and node.frequency is not None:
            if not isinstance(node.frequency, (int, float)) or node.frequency <= 0:
                result.add_error(NeuralError(
                    f"Invalid pulse frequency {node.frequency}",
                    neural_type='pulse',
                    location=node.location
                ))
        
        return result
    
    def _is_valid_identifier(self, name: str) -> bool:
        """Check if identifier is valid for neural constructs."""
        return name and name.isidentifier() and not name.startswith('_')


class FlowValidator(SemanticValidator):
    """Validates control flow and reachability."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reachable_nodes: Set[ASTNode] = set()
        self.unreachable_nodes: Set[ASTNode] = set()
    
    def validate_reachability(self, node: ASTNode) -> ValidationResult:
        """Validate code reachability."""
        result = ValidationResult(is_valid=True)
        
        # Perform reachability analysis
        self._analyze_reachability(node)
        
        # Report unreachable code
        for unreachable in self.unreachable_nodes:
            result.add_warning(SemanticError(
                SemanticErrorType.UNREACHABLE_CODE,
                "Unreachable code detected",
                location=unreachable.location
            ))
        
        return result
    
    def _analyze_reachability(self, node: ASTNode, reachable: bool = True):
        """Analyze code reachability recursively."""
        if reachable:
            self.reachable_nodes.add(node)
        else:
            self.unreachable_nodes.add(node)
        
        # Handle different node types
        if node.node_type == NodeType.RETURN_STATEMENT:
            # Code after return is unreachable
            reachable = False
        elif node.node_type == NodeType.IF_STATEMENT:
            # Analyze branches
            if hasattr(node, 'then_statement'):
                self._analyze_reachability(node.then_statement, reachable)
            if hasattr(node, 'else_statement'):
                self._analyze_reachability(node.else_statement, reachable)
        elif node.node_type == NodeType.BLOCK_STATEMENT:
            # Analyze statements in sequence
            for stmt in getattr(node, 'statements', []):
                self._analyze_reachability(stmt, reachable)
                if stmt.node_type == NodeType.RETURN_STATEMENT:
                    reachable = False


# Specific validation rules
class VariableDeclarationRule(ValidationRule):
    """Rule for validating variable declarations."""
    
    def __init__(self):
        super().__init__("variable_declaration", "Validates variable declarations")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if node.node_type != NodeType.VARIABLE_DECLARATION:
            return result
        
        # Implementation would go here
        return result


class FunctionDeclarationRule(ValidationRule):
    """Rule for validating function declarations."""
    
    def __init__(self):
        super().__init__("function_declaration", "Validates function declarations")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if node.node_type != NodeType.FUNCTION_DECLARATION:
            return result
        
        # Implementation would go here
        return result


class ParameterDeclarationRule(ValidationRule):
    """Rule for validating parameter declarations."""
    
    def __init__(self):
        super().__init__("parameter_declaration", "Validates parameter declarations")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class RedefinitionRule(ValidationRule):
    """Rule for checking symbol redefinitions."""
    
    def __init__(self):
        super().__init__("redefinition", "Checks for symbol redefinitions")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class InitializationRule(ValidationRule):
    """Rule for checking variable initialization."""
    
    def __init__(self):
        super().__init__("initialization", "Checks variable initialization")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class BinaryExpressionRule(ValidationRule):
    """Rule for validating binary expressions."""
    
    def __init__(self):
        super().__init__("binary_expression", "Validates binary expressions")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class UnaryExpressionRule(ValidationRule):
    """Rule for validating unary expressions."""
    
    def __init__(self):
        super().__init__("unary_expression", "Validates unary expressions")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class CallExpressionRule(ValidationRule):
    """Rule for validating call expressions."""
    
    def __init__(self):
        super().__init__("call_expression", "Validates call expressions")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class MemberAccessRule(ValidationRule):
    """Rule for validating member access."""
    
    def __init__(self):
        super().__init__("member_access", "Validates member access")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class ArrayAccessRule(ValidationRule):
    """Rule for validating array access."""
    
    def __init__(self):
        super().__init__("array_access", "Validates array access")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class ReturnStatementRule(ValidationRule):
    """Rule for validating return statements."""
    
    def __init__(self):
        super().__init__("return_statement", "Validates return statements")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class BreakContinueRule(ValidationRule):
    """Rule for validating break/continue statements."""
    
    def __init__(self):
        super().__init__("break_continue", "Validates break/continue statements")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class IfStatementRule(ValidationRule):
    """Rule for validating if statements."""
    
    def __init__(self):
        super().__init__("if_statement", "Validates if statements")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class LoopStatementRule(ValidationRule):
    """Rule for validating loop statements."""
    
    def __init__(self):
        super().__init__("loop_statement", "Validates loop statements")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class NeuronDeclarationRule(ValidationRule):
    """Rule for validating neuron declarations."""
    
    def __init__(self):
        super().__init__("neuron_declaration", "Validates neuron declarations")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class SynapseDeclarationRule(ValidationRule):
    """Rule for validating synapse declarations."""
    
    def __init__(self):
        super().__init__("synapse_declaration", "Validates synapse declarations")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class PulseStatementRule(ValidationRule):
    """Rule for validating pulse statements."""
    
    def __init__(self):
        super().__init__("pulse_statement", "Validates pulse statements")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class SignalExpressionRule(ValidationRule):
    """Rule for validating signal expressions."""
    
    def __init__(self):
        super().__init__("signal_expression", "Validates signal expressions")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result


class NeuralNetworkRule(ValidationRule):
    """Rule for validating neural network structure."""
    
    def __init__(self):
        super().__init__("neural_network", "Validates neural network structure")
    
    def validate(self, node: ASTNode, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        # Implementation would go here
        return result 