"""
Anamorph Parser Package.

This package provides parsing capabilities for the Anamorph language
including recursive descent parsing, AST generation, and error recovery.
"""

from .parser import (
    AnamorphParser,
    ParseResult,
    parse,
    parse_async,
    parse_file,
    parse_file_async,
)

from .errors import (
    ParseError,
    ParseErrorCode,
    ParseErrorSeverity,
    ParseContext,
    ErrorRecoveryStrategy,
    RecoveryAction,
    ParseErrorHandler,
)

# TODO: Implement grammar module
# from .grammar import (
#     GrammarRule,
#     ProductionRule,
#     GrammarValidator,
#     validate_grammar,
# )

# Package version
__version__ = "0.1.0"

# Main exports
__all__ = [
    # Core parser
    'AnamorphParser',
    'ParseResult',
    'parse',
    'parse_async',
    'parse_file',
    'parse_file_async',
    
    # Error handling
    'ParseError',
    'ParseErrorCode',
    'ParseErrorSeverity',
    'ParseContext',
    'ErrorRecoveryStrategy',
    'RecoveryAction',
    'ParseErrorHandler',
    
    # Grammar validation (TODO: implement)
    # 'GrammarRule',
    # 'ProductionRule',
    # 'GrammarValidator',
    # 'validate_grammar',
] 