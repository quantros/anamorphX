"""
Anamorph Abstract Syntax Tree (AST) Package.

This package provides AST node definitions for the Anamorph language
including expressions, statements, declarations, and neural constructs.
"""

from .nodes import (
    # Base classes
    ASTNode,
    Expression,
    Statement,
    Declaration,
    
    # Literals
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BooleanLiteral,
    ArrayLiteral,
    ObjectLiteral,
    
    # Identifiers and types
    Identifier,
    TypeAnnotation,
    
    # Expressions
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    MemberExpression,
    IndexExpression,
    ConditionalExpression,
    
    # Statements
    ExpressionStatement,
    BlockStatement,
    IfStatement,
    WhileStatement,
    ForStatement,
    ReturnStatement,
    BreakStatement,
    ContinueStatement,
    TryStatement,
    ThrowStatement,
    
    # Declarations
    VariableDeclaration,
    FunctionDeclaration,
    
    # Neural constructs
    NeuronDeclaration,
    SynapseDeclaration,
    PulseStatement,
    ResonateStatement,
    SignalExpression,
    
    # Program structure
    Program,
    ImportStatement,
    ExportStatement,
)

from .visitor import (
    ASTVisitor,
    ASTTransformer,
    NodeVisitor,
)

# TODO: Implement builder and utils modules
# from .builder import (
#     ASTBuilder,
#     ExpressionBuilder,
#     StatementBuilder,
# )

# from .utils import (
#     ast_to_dict,
#     dict_to_ast,
#     pretty_print_ast,
#     validate_ast,
# )

# Package version
__version__ = "0.1.0"

# Main exports
__all__ = [
    # Base classes
    'ASTNode',
    'Expression',
    'Statement', 
    'Declaration',
    
    # Literals
    'IntegerLiteral',
    'FloatLiteral',
    'StringLiteral',
    'BooleanLiteral',
    'ArrayLiteral',
    'ObjectLiteral',
    
    # Identifiers and types
    'Identifier',
    'TypeAnnotation',
    
    # Expressions
    'BinaryExpression',
    'UnaryExpression',
    'CallExpression',
    'MemberExpression',
    'IndexExpression',
    'ConditionalExpression',
    
    # Statements
    'ExpressionStatement',
    'BlockStatement',
    'IfStatement',
    'WhileStatement',
    'ForStatement',
    'ReturnStatement',
    'BreakStatement',
    'ContinueStatement',
    'TryStatement',
    'ThrowStatement',
    
    # Declarations
    'VariableDeclaration',
    'FunctionDeclaration',
    
    # Neural constructs
    'NeuronDeclaration',
    'SynapseDeclaration',
    'PulseStatement',
    'ResonateStatement',
    'SignalExpression',
    
    # Program structure
    'Program',
    'ImportStatement',
    'ExportStatement',
    
    # Visitor pattern
    'ASTVisitor',
    'ASTTransformer',
    'NodeVisitor',
    
    # Builder pattern (TODO: implement)
    # 'ASTBuilder',
    # 'ExpressionBuilder',
    # 'StatementBuilder',
    
    # Utilities (TODO: implement)
    # 'ast_to_dict',
    # 'dict_to_ast',
    # 'pretty_print_ast',
    # 'validate_ast',
] 