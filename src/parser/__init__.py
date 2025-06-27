"""
Anamorph Parser Package.

This package provides parsing capabilities for the Anamorph language
including recursive descent parsing, AST generation, and error recovery.
"""

from .parser import (
    AnamorphParser,
    ParseResult,
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

from .grammar import (
    GrammarRule,
    ProductionRule,
    GrammarValidator,
    validate_grammar,
)

# Package version
__version__ = "0.1.0"

# Main exports
__all__ = [
    # Core parser
    'AnamorphParser',
    'ParseResult',
    
    # Error handling
    'ParseError',
    'ParseErrorCode',
    'ParseErrorSeverity',
    'ParseContext',
    'ErrorRecoveryStrategy',
    'RecoveryAction',
    'ParseErrorHandler',
    
    # Grammar validation helpers
    'GrammarRule',
    'ProductionRule',
    'GrammarValidator',
    'validate_grammar',
]
