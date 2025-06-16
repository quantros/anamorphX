"""
Visitor Pattern Implementation for Anamorph AST.

This module provides visitor classes for traversing and transforming
Abstract Syntax Tree nodes using the visitor pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic

# Forward declaration for type hints
if False:  # TYPE_CHECKING
    from .nodes import *

T = TypeVar('T')


class ASTVisitor(ABC):
    """
    Abstract base class for AST visitors.
    
    Implements the visitor pattern for traversing AST nodes.
    Subclasses should override visit methods for specific node types.
    """
    
    def visit(self, node: 'ASTNode') -> Any:
        """Visit a node by dispatching to the appropriate visit method."""
        return node.accept(self)
    
    def generic_visit(self, node: 'ASTNode') -> Any:
        """Default visit method for nodes without specific handlers."""
        for child in node.get_children():
            self.visit(child)
    
    # Literals
    def visit_integer_literal(self, node: 'IntegerLiteral') -> Any:
        return self.generic_visit(node)
    
    def visit_float_literal(self, node: 'FloatLiteral') -> Any:
        return self.generic_visit(node)
    
    def visit_string_literal(self, node: 'StringLiteral') -> Any:
        return self.generic_visit(node)
    
    def visit_boolean_literal(self, node: 'BooleanLiteral') -> Any:
        return self.generic_visit(node)
    
    def visit_array_literal(self, node: 'ArrayLiteral') -> Any:
        for element in node.elements:
            self.visit(element)
        return self.generic_visit(node)
    
    def visit_object_literal(self, node: 'ObjectLiteral') -> Any:
        for key, value in node.properties:
            self.visit(key)
            self.visit(value)
        return self.generic_visit(node)
    
    # Identifiers and types
    def visit_identifier(self, node: 'Identifier') -> Any:
        return self.generic_visit(node)
    
    def visit_type_annotation(self, node: 'TypeAnnotation') -> Any:
        if node.generic_args:
            for arg in node.generic_args:
                self.visit(arg)
        return self.generic_visit(node)
    
    # Expressions
    def visit_binary_expression(self, node: 'BinaryExpression') -> Any:
        self.visit(node.left)
        self.visit(node.right)
        return self.generic_visit(node)
    
    def visit_unary_expression(self, node: 'UnaryExpression') -> Any:
        self.visit(node.operand)
        return self.generic_visit(node)
    
    def visit_call_expression(self, node: 'CallExpression') -> Any:
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)
        return self.generic_visit(node)
    
    def visit_member_expression(self, node: 'MemberExpression') -> Any:
        self.visit(node.object)
        self.visit(node.property)
        return self.generic_visit(node)
    
    def visit_index_expression(self, node: 'IndexExpression') -> Any:
        self.visit(node.object)
        self.visit(node.index)
        return self.generic_visit(node)
    
    def visit_conditional_expression(self, node: 'ConditionalExpression') -> Any:
        self.visit(node.test)
        self.visit(node.consequent)
        self.visit(node.alternate)
        return self.generic_visit(node)
    
    def visit_signal_expression(self, node: 'SignalExpression') -> Any:
        self.visit(node.source)
        self.visit(node.target)
        return self.generic_visit(node)
    
    # Statements
    def visit_expression_statement(self, node: 'ExpressionStatement') -> Any:
        self.visit(node.expression)
        return self.generic_visit(node)
    
    def visit_block_statement(self, node: 'BlockStatement') -> Any:
        for stmt in node.statements:
            self.visit(stmt)
        return self.generic_visit(node)
    
    def visit_if_statement(self, node: 'IfStatement') -> Any:
        self.visit(node.test)
        self.visit(node.consequent)
        if node.alternate:
            self.visit(node.alternate)
        return self.generic_visit(node)
    
    def visit_while_statement(self, node: 'WhileStatement') -> Any:
        self.visit(node.test)
        self.visit(node.body)
        return self.generic_visit(node)
    
    def visit_for_statement(self, node: 'ForStatement') -> Any:
        if node.init:
            self.visit(node.init)
        if node.test:
            self.visit(node.test)
        if node.update:
            self.visit(node.update)
        self.visit(node.body)
        return self.generic_visit(node)
    
    def visit_return_statement(self, node: 'ReturnStatement') -> Any:
        if node.argument:
            self.visit(node.argument)
        return self.generic_visit(node)
    
    def visit_break_statement(self, node: 'BreakStatement') -> Any:
        return self.generic_visit(node)
    
    def visit_continue_statement(self, node: 'ContinueStatement') -> Any:
        return self.generic_visit(node)
    
    def visit_try_statement(self, node: 'TryStatement') -> Any:
        self.visit(node.block)
        if node.handler:
            self.visit(node.handler)
        if node.finalizer:
            self.visit(node.finalizer)
        return self.generic_visit(node)
    
    def visit_catch_clause(self, node: 'CatchClause') -> Any:
        if node.param:
            self.visit(node.param)
        self.visit(node.body)
        return self.generic_visit(node)
    
    def visit_throw_statement(self, node: 'ThrowStatement') -> Any:
        self.visit(node.argument)
        return self.generic_visit(node)
    
    # Declarations
    def visit_variable_declaration(self, node: 'VariableDeclaration') -> Any:
        for declarator in node.declarations:
            self.visit(declarator)
        return self.generic_visit(node)
    
    def visit_variable_declarator(self, node: 'VariableDeclarator') -> Any:
        self.visit(node.id)
        if node.init:
            self.visit(node.init)
        if node.type_annotation:
            self.visit(node.type_annotation)
        return self.generic_visit(node)
    
    def visit_function_declaration(self, node: 'FunctionDeclaration') -> Any:
        self.visit(node.id)
        for param in node.params:
            self.visit(param)
        self.visit(node.body)
        if node.return_type:
            self.visit(node.return_type)
        return self.generic_visit(node)
    
    def visit_parameter(self, node: 'Parameter') -> Any:
        self.visit(node.id)
        if node.type_annotation:
            self.visit(node.type_annotation)
        if node.default_value:
            self.visit(node.default_value)
        return self.generic_visit(node)
    
    # Neural constructs
    def visit_neuron_declaration(self, node: 'NeuronDeclaration') -> Any:
        self.visit(node.id)
        for param in node.params:
            self.visit(param)
        self.visit(node.body)
        if node.return_type:
            self.visit(node.return_type)
        return self.generic_visit(node)
    
    def visit_synapse_declaration(self, node: 'SynapseDeclaration') -> Any:
        for declarator in node.declarations:
            self.visit(declarator)
        return self.generic_visit(node)
    
    def visit_pulse_statement(self, node: 'PulseStatement') -> Any:
        self.visit(node.signal)
        if node.target:
            self.visit(node.target)
        if node.condition:
            self.visit(node.condition)
        return self.generic_visit(node)
    
    def visit_resonate_statement(self, node: 'ResonateStatement') -> Any:
        self.visit(node.id)
        for param in node.params:
            self.visit(param)
        self.visit(node.body)
        return self.generic_visit(node)
    
    # Program structure
    def visit_program(self, node: 'Program') -> Any:
        for item in node.body:
            self.visit(item)
        return self.generic_visit(node)
    
    def visit_import_statement(self, node: 'ImportStatement') -> Any:
        for spec in node.specifiers:
            self.visit(spec)
        self.visit(node.source)
        return self.generic_visit(node)
    
    def visit_import_specifier(self, node: 'ImportSpecifier') -> Any:
        self.visit(node.imported)
        if node.local:
            self.visit(node.local)
        return self.generic_visit(node)
    
    def visit_export_statement(self, node: 'ExportStatement') -> Any:
        if node.declaration:
            self.visit(node.declaration)
        if node.specifiers:
            for spec in node.specifiers:
                self.visit(spec)
        if node.source:
            self.visit(node.source)
        return self.generic_visit(node)
    
    def visit_export_specifier(self, node: 'ExportSpecifier') -> Any:
        self.visit(node.local)
        if node.exported:
            self.visit(node.exported)
        return self.generic_visit(node)


class ASTTransformer(ASTVisitor, Generic[T]):
    """
    AST transformer that can modify nodes during traversal.
    
    Unlike ASTVisitor, this class returns potentially modified nodes,
    allowing for AST transformations.
    """
    
    def visit(self, node: 'ASTNode') -> T:
        """Visit and potentially transform a node."""
        return node.accept(self)
    
    def generic_visit(self, node: 'ASTNode') -> T:
        """Default transformation - return the node unchanged."""
        return node
    
    # Override visit methods to return transformed nodes
    # (Implementation would be similar to ASTVisitor but returning nodes)


class NodeVisitor:
    """
    Simple node visitor that calls visit_<node_type> methods.
    
    This is a simpler alternative to ASTVisitor that doesn't require
    implementing the accept method in nodes.
    """
    
    def visit(self, node: 'ASTNode') -> Any:
        """Visit a node by calling the appropriate visit method."""
        method_name = f"visit_{node.node_type.name.lower()}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: 'ASTNode') -> Any:
        """Default visit method."""
        for child in node.get_children():
            self.visit(child)


# ============================================================================
# Utility Visitors
# ============================================================================

class ASTDumper(ASTVisitor):
    """Visitor that creates a string representation of the AST."""
    
    def __init__(self, indent: str = "  "):
        self.indent = indent
        self.level = 0
        self.output = []
    
    def dump(self, node: 'ASTNode') -> str:
        """Dump the AST to a string."""
        self.output = []
        self.level = 0
        self.visit(node)
        return "\n".join(self.output)
    
    def _add_line(self, text: str):
        """Add a line with proper indentation."""
        self.output.append(self.indent * self.level + text)
    
    def generic_visit(self, node: 'ASTNode') -> Any:
        """Add node information and visit children."""
        node_info = f"{node.node_type.name}"
        
        # Add node-specific information
        if hasattr(node, 'value'):
            node_info += f": {repr(node.value)}"
        elif hasattr(node, 'name'):
            node_info += f": {node.name}"
        elif hasattr(node, 'operator'):
            node_info += f": {node.operator.value}"
        
        self._add_line(node_info)
        
        # Visit children with increased indentation
        self.level += 1
        for child in node.get_children():
            self.visit(child)
        self.level -= 1


class ASTValidator(ASTVisitor):
    """Visitor that validates AST structure and semantics."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate(self, node: 'ASTNode') -> tuple[list, list]:
        """Validate the AST and return errors and warnings."""
        self.errors = []
        self.warnings = []
        self.visit(node)
        return self.errors, self.warnings
    
    def _add_error(self, message: str, node: 'ASTNode'):
        """Add a validation error."""
        location = node.location or "unknown location"
        self.errors.append(f"Error at {location}: {message}")
    
    def _add_warning(self, message: str, node: 'ASTNode'):
        """Add a validation warning."""
        location = node.location or "unknown location"
        self.warnings.append(f"Warning at {location}: {message}")
    
    def visit_binary_expression(self, node: 'BinaryExpression') -> Any:
        """Validate binary expressions."""
        # Check for type compatibility (basic validation)
        if hasattr(node.left, 'value') and hasattr(node.right, 'value'):
            left_type = type(node.left.value)
            right_type = type(node.right.value)
            
            # Check arithmetic operations on compatible types
            if node.operator in [BinaryOperator.ADD, BinaryOperator.SUBTRACT, 
                               BinaryOperator.MULTIPLY, BinaryOperator.DIVIDE]:
                if not (left_type in [int, float] and right_type in [int, float]):
                    if not (left_type == str and right_type == str and node.operator == BinaryOperator.ADD):
                        self._add_warning(
                            f"Potential type mismatch in {node.operator.value} operation",
                            node
                        )
        
        return super().visit_binary_expression(node)
    
    def visit_function_declaration(self, node: 'FunctionDeclaration') -> Any:
        """Validate function declarations."""
        # Check for duplicate parameter names
        param_names = [param.id.name for param in node.params]
        if len(param_names) != len(set(param_names)):
            self._add_error("Duplicate parameter names in function declaration", node)
        
        return super().visit_function_declaration(node)
    
    def visit_neuron_declaration(self, node: 'NeuronDeclaration') -> Any:
        """Validate neuron declarations."""
        # Similar validation to function declarations
        param_names = [param.id.name for param in node.params]
        if len(param_names) != len(set(param_names)):
            self._add_error("Duplicate parameter names in neuron declaration", node)
        
        return super().visit_neuron_declaration(node) 