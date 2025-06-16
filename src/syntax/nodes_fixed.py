"""
AST Node Definitions for Anamorph Language.

This module defines all Abstract Syntax Tree nodes for the Anamorph language
including expressions, statements, declarations, and neural-specific constructs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from enum import Enum, auto

# Flexible import for TokenType with fallbacks to solve relative import issues
try:
    from lexer.tokens import TokenType
except ImportError:
    try:
        from src.lexer.tokens import TokenType
    except ImportError:
        # Create a simple fallback TokenType for compatibility
        class TokenType(Enum):
            """Fallback TokenType for compatibility."""
            IDENTIFIER = "IDENTIFIER"
            STRING = "STRING"
            NUMBER = "NUMBER"
            KEYWORD = "KEYWORD"
            OPERATOR = "OPERATOR"
            EOF = "EOF"
            NEWLINE = "NEWLINE"
            INDENT = "INDENT"
            DEDENT = "DEDENT"
        print("⚠️ Используется fallback TokenType в syntax.nodes")


class NodeType(Enum):
    """Enumeration of all AST node types."""
    
    # Base types
    PROGRAM = auto()
    
    # Literals
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    ARRAY_LITERAL = auto()
    OBJECT_LITERAL = auto()
    
    # Identifiers
    IDENTIFIER = auto()
    TYPE_ANNOTATION = auto()
    
    # Expressions
    BINARY_EXPRESSION = auto()
    UNARY_EXPRESSION = auto()
    CALL_EXPRESSION = auto()
    MEMBER_EXPRESSION = auto()
    INDEX_EXPRESSION = auto()
    CONDITIONAL_EXPRESSION = auto()
    SIGNAL_EXPRESSION = auto()
    
    # Statements
    EXPRESSION_STATEMENT = auto()
    BLOCK_STATEMENT = auto()
    IF_STATEMENT = auto()
    WHILE_STATEMENT = auto()
    FOR_STATEMENT = auto()
    RETURN_STATEMENT = auto()
    BREAK_STATEMENT = auto()
    CONTINUE_STATEMENT = auto()
    TRY_STATEMENT = auto()
    THROW_STATEMENT = auto()
    
    # Declarations
    VARIABLE_DECLARATION = auto()
    FUNCTION_DECLARATION = auto()
    
    # Neural constructs
    NEURON_DECLARATION = auto()
    SYNAPSE_DECLARATION = auto()
    PULSE_STATEMENT = auto()
    RESONATE_STATEMENT = auto()
    
    # Import/Export
    IMPORT_STATEMENT = auto()
    EXPORT_STATEMENT = auto()


class BinaryOperator(Enum):
    """Binary operators in Anamorph."""
    
    # Arithmetic
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"
    
    # Comparison
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    
    # Logical
    AND = "&&"
    OR = "||"
    
    # Bitwise
    BIT_AND = "&"
    BIT_OR = "|"
    BIT_XOR = "^"
    LEFT_SHIFT = "<<"
    RIGHT_SHIFT = ">>"
    
    # Assignment
    ASSIGN = "="
    
    # Signal flow
    ARROW = "->"
    DOUBLE_ARROW = "=>"


class UnaryOperator(Enum):
    """Unary operators in Anamorph."""
    
    PLUS = "+"
    MINUS = "-"
    NOT = "!"
    BIT_NOT = "~"


@dataclass
class SourceLocation:
    """Source code location information."""
    
    line: int
    column: int
    position: int
    length: int
    source_file: Optional[str] = None
    
    def __str__(self) -> str:
        location = f"{self.source_file}:" if self.source_file else ""
        return f"{location}{self.line}:{self.column}"


@dataclass
class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    # Remove default values to fix dataclass inheritance ordering
    location: Optional[SourceLocation] = field(default=None, init=False)
    parent: Optional['ASTNode'] = field(default=None, repr=False, init=False)
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if not hasattr(self, 'location'):
            self.location = None
        if not hasattr(self, 'parent'):
            self.parent = None
    
    @abstractmethod
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Accept a visitor for the visitor pattern."""
        pass
    
    @property
    def node_type(self) -> NodeType:
        return getattr(self, '_node_type', NodeType.PROGRAM)
    
    def get_children(self) -> List['ASTNode']:
        """Get all child nodes."""
        children = []
        for field_name, field_value in self.__dict__.items():
            if field_name in ('parent', 'location', 'node_type', '_node_type'):
                continue
            
            if isinstance(field_value, ASTNode):
                children.append(field_value)
            elif isinstance(field_value, list):
                children.extend(node for node in field_value if isinstance(node, ASTNode))
        
        return children
    
    def set_parent(self, parent: 'ASTNode'):
        """Set the parent node."""
        self.parent = parent
        for child in self.get_children():
            child.set_parent(self)


# ============================================================================
# Base Categories
# ============================================================================

@dataclass
class Expression(ASTNode):
    """Base class for all expressions."""
    pass


@dataclass
class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass
class Declaration(ASTNode):
    """Base class for all declarations."""
    pass


# ============================================================================
# Literals
# ============================================================================

@dataclass
class IntegerLiteral(Expression):
    """Integer literal node."""
    
    value: int
    
    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.INTEGER_LITERAL
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_integer_literal(self)


@dataclass
class FloatLiteral(Expression):
    """Float literal node."""
    
    value: float

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.FLOAT_LITERAL
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_float_literal(self)


@dataclass
class StringLiteral(Expression):
    """String literal node."""
    
    value: str

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.STRING_LITERAL
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_string_literal(self)


@dataclass
class BooleanLiteral(Expression):
    """Boolean literal node."""
    
    value: bool

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.BOOLEAN_LITERAL
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_boolean_literal(self)


@dataclass
class ArrayLiteral(Expression):
    """Array literal node."""
    
    elements: List[Expression]

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.ARRAY_LITERAL
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_array_literal(self)


@dataclass
class ObjectLiteral(Expression):
    """Object literal node."""
    
    properties: List[tuple[Expression, Expression]]  # (key, value) pairs

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.OBJECT_LITERAL
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_object_literal(self)


# ============================================================================
# Identifiers and Types
# ============================================================================

@dataclass
class Identifier(Expression):
    """Identifier node."""
    
    name: str

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.IDENTIFIER
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_identifier(self)


@dataclass
class TypeAnnotation(ASTNode):
    """Type annotation node."""
    
    type_name: str
    generic_args: Optional[List['TypeAnnotation']] = None

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.TYPE_ANNOTATION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_type_annotation(self)


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class BinaryExpression(Expression):
    """Binary expression node."""
    
    left: Expression
    operator: BinaryOperator
    right: Expression

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.BINARY_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_binary_expression(self)


@dataclass
class UnaryExpression(Expression):
    """Unary expression node."""
    
    operator: UnaryOperator
    operand: Expression

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.UNARY_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_unary_expression(self)


@dataclass
class CallExpression(Expression):
    """Function call expression node."""
    
    callee: Expression
    arguments: List[Expression]

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.CALL_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_call_expression(self)


@dataclass
class MemberExpression(Expression):
    """Member access expression node."""
    
    object: Expression
    property: Expression
    computed: bool = False  # True for obj[prop], False for obj.prop

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.MEMBER_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_member_expression(self)


@dataclass
class IndexExpression(Expression):
    """Array/object index expression node."""
    
    object: Expression
    index: Expression

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.INDEX_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_index_expression(self)


@dataclass
class ConditionalExpression(Expression):
    """Conditional (ternary) expression node."""
    
    test: Expression
    consequent: Expression
    alternate: Expression

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.CONDITIONAL_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_conditional_expression(self)


@dataclass
class SignalExpression(Expression):
    """Signal flow expression node (unique to Anamorph)."""
    
    source: Expression
    target: Expression
    signal_type: str = "sync"  # sync, async, priority, streaming

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.SIGNAL_EXPRESSION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_signal_expression(self)


# ============================================================================
# Statements
# ============================================================================

@dataclass
class ExpressionStatement(Statement):
    """Expression statement node."""
    
    expression: Expression

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.EXPRESSION_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_expression_statement(self)


@dataclass
class BlockStatement(Statement):
    """Block statement node."""
    
    statements: List[Statement]

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.BLOCK_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_block_statement(self)


@dataclass
class IfStatement(Statement):
    """If statement node."""
    
    test: Expression
    consequent: Statement
    alternate: Optional[Statement] = None

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.IF_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_if_statement(self)


@dataclass
class WhileStatement(Statement):
    """While loop statement node."""
    
    test: Expression
    body: Statement

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.WHILE_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_while_statement(self)


@dataclass
class ForStatement(Statement):
    """For loop statement node."""
    
    init: Optional[Union[Declaration, Expression]]
    test: Optional[Expression]
    update: Optional[Expression]
    body: Statement

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.FOR_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_for_statement(self)


@dataclass
class ReturnStatement(Statement):
    """Return statement node."""
    
    argument: Optional[Expression] = None

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.RETURN_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_return_statement(self)


@dataclass
class BreakStatement(Statement):
    """Break statement node."""
    

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.BREAK_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_break_statement(self)


@dataclass
class ContinueStatement(Statement):
    """Continue statement node."""
    

    def __post_init__(self):
        super().__init__()
        self._node_type = NodeType.CONTINUE_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_continue_statement(self)


@dataclass
class TryStatement(Statement):
    """Try-catch statement node."""
    
    block: BlockStatement
    handler: Optional['CatchClause'] = None
    finalizer: Optional[BlockStatement] = None

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.TRY_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_try_statement(self)


@dataclass
class CatchClause(ASTNode):
    """Catch clause for try statement."""
    
    param: Optional[Identifier]
    body: BlockStatement

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_catch_clause(self)


@dataclass
class ThrowStatement(Statement):
    """Throw statement node."""
    
    argument: Expression

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.THROW_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_throw_statement(self)


# ============================================================================
# Declarations
# ============================================================================

@dataclass
class VariableDeclaration(Declaration):
    """Variable declaration node."""
    
    declarations: List['VariableDeclarator']
    kind: str = "synap"  # synap, const, let

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.VARIABLE_DECLARATION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_variable_declaration(self)


@dataclass
class VariableDeclarator(ASTNode):
    """Variable declarator node."""
    
    id: Identifier
    init: Optional[Expression] = None
    type_annotation: Optional[TypeAnnotation] = None

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_variable_declarator(self)


@dataclass
class FunctionDeclaration(Declaration):
    """Function declaration node."""
    
    id: Identifier
    params: List['Parameter']
    body: BlockStatement
    return_type: Optional[TypeAnnotation] = None
    is_async: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.FUNCTION_DECLARATION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_function_declaration(self)


@dataclass
class Parameter(ASTNode):
    """Function parameter node."""
    
    id: Identifier
    type_annotation: Optional[TypeAnnotation] = None
    default_value: Optional[Expression] = None

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_parameter(self)


# ============================================================================
# Neural Constructs (Unique to Anamorph)
# ============================================================================

@dataclass
class NeuronDeclaration(Declaration):
    """Neuron declaration node (unique to Anamorph)."""
    
    id: Identifier
    params: List[Parameter]
    body: BlockStatement
    return_type: Optional[TypeAnnotation] = None
    is_async: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.NEURON_DECLARATION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_neuron_declaration(self)


@dataclass
class SynapseDeclaration(Declaration):
    """Synapse declaration node (unique to Anamorph)."""
    
    declarations: List[VariableDeclarator]

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.SYNAPSE_DECLARATION
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_synapse_declaration(self)


@dataclass
class PulseStatement(Statement):
    """Pulse statement node (unique to Anamorph)."""
    
    signal: Expression
    target: Optional[Expression] = None
    condition: Optional[Expression] = None

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.PULSE_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_pulse_statement(self)


@dataclass
class ResonateStatement(Statement):
    """Resonate statement node (unique to Anamorph)."""
    
    id: Identifier
    params: List[Parameter]
    body: BlockStatement
    signal_type: str = "sync"

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.RESONATE_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_resonate_statement(self)


# ============================================================================
# Program and Modules
# ============================================================================

@dataclass
class Program(ASTNode):
    """Root program node."""
    
    body: List[Union[Statement, Declaration]]
    source_type: str = "module"

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.PROGRAM
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_program(self)


@dataclass
class ImportStatement(Statement):
    """Import statement node."""
    
    specifiers: List['ImportSpecifier']
    source: StringLiteral

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.IMPORT_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_import_statement(self)


@dataclass
class ImportSpecifier(ASTNode):
    """Import specifier node."""
    
    imported: Identifier
    local: Optional[Identifier] = None

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_import_specifier(self)


@dataclass
class ExportStatement(Statement):
    """Export statement node."""
    
    declaration: Optional[Declaration] = None
    specifiers: Optional[List['ExportSpecifier']] = None
    source: Optional[StringLiteral] = None

    def __post_init__(self):
        super().__post_init__()
        self._node_type = NodeType.EXPORT_STATEMENT
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_export_statement(self)


@dataclass
class ExportSpecifier(ASTNode):
    """Export specifier node."""
    
    local: Identifier
    exported: Optional[Identifier] = None

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_export_specifier(self)


# Forward references for ASTVisitor
class ASTVisitor:
    """Forward reference for visitor pattern."""
    pass 