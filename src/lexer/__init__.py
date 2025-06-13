"""
Anamorph Lexer Package.

This package provides lexical analysis capabilities for the Anamorph language
including tokenization, error handling, and performance optimization.
"""

from .tokens import (
    Token,
    TokenType,
    NEURAL_COMMANDS,
    KEYWORDS,
    TYPE_KEYWORDS,
    OPERATORS,
    DELIMITERS,
    RESERVED_WORDS,
    TOKEN_PATTERNS,
    TOKEN_CATEGORIES,
    is_neural_command,
    is_keyword,
    is_operator,
    is_literal,
    get_token_category
)

from .lexer import (
    AnamorphLexer,
    LexerState,
    tokenize,
    tokenize_async,
    tokenize_file,
    tokenize_file_async
)

from .errors import (
    LexerError,
    ErrorContext,
    ErrorSeverity,
    ErrorCode,
    ErrorHandler,
    ErrorRecoveryStrategy,
    RecoveryAction,
    create_invalid_character_error,
    create_unterminated_string_error,
    create_invalid_number_error
)

# Package version
__version__ = "0.1.0"

# Main exports
__all__ = [
    # Core classes
    'AnamorphLexer',
    'Token',
    'TokenType',
    'LexerState',
    
    # Error handling
    'LexerError',
    'ErrorContext',
    'ErrorSeverity',
    'ErrorCode',
    'ErrorHandler',
    'ErrorRecoveryStrategy',
    'RecoveryAction',
    
    # Token definitions
    'NEURAL_COMMANDS',
    'KEYWORDS',
    'TYPE_KEYWORDS',
    'OPERATORS',
    'DELIMITERS',
    'RESERVED_WORDS',
    'TOKEN_PATTERNS',
    'TOKEN_CATEGORIES',
    
    # Utility functions
    'tokenize',
    'tokenize_async',
    'tokenize_file',
    'tokenize_file_async',
    'is_neural_command',
    'is_keyword',
    'is_operator',
    'is_literal',
    'get_token_category',
    'create_invalid_character_error',
    'create_unterminated_string_error',
    'create_invalid_number_error',
] 