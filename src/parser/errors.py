"""
Error handling and recovery for Anamorph Parser.

This module provides comprehensive error handling capabilities including
error recovery strategies, detailed reporting, and context information.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from ..lexer.tokens import Token, TokenType
from ..syntax.nodes import ASTNode, SourceLocation


class ParseErrorSeverity(Enum):
    """Severity levels for parser errors."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ParseErrorCode(Enum):
    """Specific error codes for different types of parser errors."""
    
    # Syntax errors
    UNEXPECTED_TOKEN = "PAR001"
    EXPECTED_TOKEN = "PAR002"
    MISSING_TOKEN = "PAR003"
    INVALID_SYNTAX = "PAR004"
    
    # Expression errors
    INVALID_EXPRESSION = "PAR101"
    INCOMPLETE_EXPRESSION = "PAR102"
    INVALID_OPERATOR = "PAR103"
    MISMATCHED_PARENTHESES = "PAR104"
    
    # Statement errors
    INVALID_STATEMENT = "PAR201"
    INCOMPLETE_STATEMENT = "PAR202"
    INVALID_BLOCK = "PAR203"
    
    # Declaration errors
    INVALID_DECLARATION = "PAR301"
    DUPLICATE_DECLARATION = "PAR302"
    INVALID_PARAMETER = "PAR303"
    
    # Neural construct errors
    INVALID_NEURON = "PAR401"
    INVALID_SYNAPSE = "PAR402"
    INVALID_PULSE = "PAR403"
    INVALID_RESONATE = "PAR404"
    INVALID_SIGNAL = "PAR405"
    
    # Type errors
    INVALID_TYPE = "PAR501"
    TYPE_MISMATCH = "PAR502"
    UNKNOWN_TYPE = "PAR503"
    
    # General errors
    INTERNAL_ERROR = "PAR901"
    TIMEOUT_ERROR = "PAR902"
    MEMORY_ERROR = "PAR903"


class ErrorRecoveryStrategy(Enum):
    """Strategies for recovering from parser errors."""
    
    # Panic mode recovery
    PANIC_MODE = auto()           # Skip tokens until synchronization point
    
    # Phrase level recovery
    PHRASE_LEVEL = auto()         # Insert/delete/replace tokens
    
    # Error productions
    ERROR_PRODUCTIONS = auto()    # Use error productions in grammar
    
    # Global correction
    GLOBAL_CORRECTION = auto()    # Minimal distance correction
    
    # Custom recovery
    CUSTOM = auto()               # Custom recovery strategy


@dataclass
class ParseContext:
    """Detailed context information for a parser error."""
    
    tokens: List[Token]
    current_position: int
    expected_tokens: List[TokenType]
    current_rule: Optional[str] = None
    call_stack: List[str] = None
    source_file: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.call_stack is None:
            self.call_stack = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def get_current_token(self) -> Optional[Token]:
        """Get the current token."""
        if 0 <= self.current_position < len(self.tokens):
            return self.tokens[self.current_position]
        return None
    
    def get_surrounding_tokens(self, context_size: int = 3) -> List[Token]:
        """Get tokens around the current position."""
        start = max(0, self.current_position - context_size)
        end = min(len(self.tokens), self.current_position + context_size + 1)
        return self.tokens[start:end]
    
    def get_source_snippet(self, context_lines: int = 2) -> str:
        """Get source code snippet around the error location."""
        current_token = self.get_current_token()
        if not current_token or not current_token.source_file:
            return ""
        
        try:
            with open(current_token.source_file, 'r') as f:
                lines = f.readlines()
            
            start_line = max(0, current_token.line - context_lines - 1)
            end_line = min(len(lines), current_token.line + context_lines)
            
            snippet_lines = []
            for i in range(start_line, end_line):
                line_num = i + 1
                line_content = lines[i].rstrip()
                
                if line_num == current_token.line:
                    # Highlight the error position
                    pointer = " " * (current_token.column - 1) + "^"
                    snippet_lines.append(f"{line_num:4d} | {line_content}")
                    snippet_lines.append(f"     | {pointer}")
                else:
                    snippet_lines.append(f"{line_num:4d} | {line_content}")
            
            return "\n".join(snippet_lines)
        except (FileNotFoundError, IOError):
            return "Source file not available"


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    
    strategy: ErrorRecoveryStrategy
    skip_count: int = 0
    insert_tokens: List[TokenType] = None
    replace_token: Optional[TokenType] = None
    sync_tokens: List[TokenType] = None
    message: str = ""
    
    def __post_init__(self):
        if self.insert_tokens is None:
            self.insert_tokens = []
        if self.sync_tokens is None:
            self.sync_tokens = []


@dataclass
class ParseError(Exception):
    """Comprehensive parser error with detailed context."""
    
    code: ParseErrorCode
    message: str
    severity: ParseErrorSeverity
    context: ParseContext
    suggestions: List[str] = None
    related_errors: List['ParseError'] = None
    recovery_action: Optional[RecoveryAction] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.related_errors is None:
            self.related_errors = []
    
    def __str__(self) -> str:
        current_token = self.context.get_current_token()
        location = ""
        if current_token:
            if self.context.source_file:
                location = f"{self.context.source_file}:"
            location += f"{current_token.line}:{current_token.column}"
        
        return f"{self.severity.name} {self.code.value} at {location}: {self.message}"
    
    def get_detailed_report(self) -> str:
        """Generate a detailed error report."""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append("ANAMORPH PARSER ERROR REPORT")
        report.append("=" * 60)
        
        # Basic information
        report.append(f"Error Code: {self.code.value}")
        report.append(f"Severity: {self.severity.name}")
        report.append(f"Message: {self.message}")
        report.append(f"Timestamp: {self.context.timestamp.isoformat()}")
        
        # Location information
        current_token = self.context.get_current_token()
        if current_token:
            if self.context.source_file:
                report.append(f"File: {self.context.source_file}")
            report.append(f"Line: {current_token.line}")
            report.append(f"Column: {current_token.column}")
            report.append(f"Token: {current_token.type.name} '{current_token.value}'")
        
        # Expected tokens
        if self.context.expected_tokens:
            expected = [t.name for t in self.context.expected_tokens]
            report.append(f"Expected: {', '.join(expected)}")
        
        # Current parsing rule
        if self.context.current_rule:
            report.append(f"Current rule: {self.context.current_rule}")
        
        # Source code snippet
        snippet = self.context.get_source_snippet()
        if snippet:
            report.append("\nSource Code Context:")
            report.append("-" * 30)
            report.append(snippet)
        
        # Surrounding tokens
        surrounding = self.context.get_surrounding_tokens()
        if surrounding:
            report.append("\nSurrounding Tokens:")
            for i, token in enumerate(surrounding):
                marker = " -> " if i == len(surrounding) // 2 else "    "
                report.append(f"{marker}{token.type.name}: '{token.value}'")
        
        # Suggestions
        if self.suggestions:
            report.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                report.append(f"  {i}. {suggestion}")
        
        # Recovery action
        if self.recovery_action:
            report.append(f"\nRecovery Strategy: {self.recovery_action.strategy.name}")
            if self.recovery_action.message:
                report.append(f"Recovery Action: {self.recovery_action.message}")
        
        # Call stack
        if self.context.call_stack:
            report.append("\nParsing Call Stack:")
            for frame in self.context.call_stack:
                report.append(f"  {frame}")
        
        # Related errors
        if self.related_errors:
            report.append("\nRelated Errors:")
            for error in self.related_errors:
                report.append(f"  - {error}")
        
        report.append("=" * 60)
        return "\n".join(report)


class ParseErrorHandler:
    """Handles parser errors with recovery strategies."""
    
    def __init__(self, max_errors: int = 100, enable_recovery: bool = True):
        self.max_errors = max_errors
        self.enable_recovery = enable_recovery
        self.errors: List[ParseError] = []
        self.warnings: List[ParseError] = []
        self.error_count_by_type: Dict[ParseErrorCode, int] = {}
        
        # Synchronization tokens for panic mode recovery
        self.sync_tokens = {
            TokenType.SEMICOLON,
            TokenType.LEFT_BRACE,
            TokenType.RIGHT_BRACE,
            TokenType.NEURO,
            TokenType.SYNAP,
            TokenType.IF,
            TokenType.WHILE,
            TokenType.FOR,
            TokenType.RETURN,
            TokenType.EOF,
        }
    
    def report_error(self,
                    code: ParseErrorCode,
                    message: str,
                    context: ParseContext,
                    severity: ParseErrorSeverity = ParseErrorSeverity.ERROR,
                    suggestions: List[str] = None) -> ParseError:
        """Report a new parser error."""
        
        error = ParseError(
            code=code,
            message=message,
            severity=severity,
            context=context,
            suggestions=suggestions or []
        )
        
        # Add automatic suggestions based on error type
        self._add_automatic_suggestions(error)
        
        # Determine recovery action
        if self.enable_recovery:
            error.recovery_action = self._get_recovery_action(error)
        
        # Track error statistics
        self.error_count_by_type[code] = self.error_count_by_type.get(code, 0) + 1
        
        if severity in (ParseErrorSeverity.ERROR, ParseErrorSeverity.CRITICAL):
            self.errors.append(error)
        else:
            self.warnings.append(error)
        
        return error
    
    def _add_automatic_suggestions(self, error: ParseError):
        """Add automatic suggestions based on error type."""
        
        if error.code == ParseErrorCode.EXPECTED_TOKEN:
            if error.context.expected_tokens:
                expected = [t.name for t in error.context.expected_tokens]
                error.suggestions.extend([
                    f"Insert one of: {', '.join(expected)}",
                    "Check for missing punctuation",
                    "Verify syntax according to language specification"
                ])
        
        elif error.code == ParseErrorCode.UNEXPECTED_TOKEN:
            current_token = error.context.get_current_token()
            if current_token:
                error.suggestions.extend([
                    f"Remove unexpected '{current_token.value}'",
                    "Check for typos in keywords",
                    "Verify correct token sequence"
                ])
        
        elif error.code == ParseErrorCode.MISMATCHED_PARENTHESES:
            error.suggestions.extend([
                "Check for missing opening or closing parentheses",
                "Verify balanced brackets and braces",
                "Use proper nesting of expressions"
            ])
        
        elif error.code == ParseErrorCode.INVALID_NEURON:
            error.suggestions.extend([
                "Check neuron declaration syntax: neuro name(params) { body }",
                "Verify parameter list format",
                "Ensure proper block structure"
            ])
        
        elif error.code == ParseErrorCode.INVALID_PULSE:
            error.suggestions.extend([
                "Check pulse statement syntax: pulse signal -> target",
                "Verify signal expression format",
                "Use proper arrow operator (->)"
            ])
    
    def _get_recovery_action(self, error: ParseError) -> RecoveryAction:
        """Determine recovery action for an error."""
        
        if len(self.errors) >= self.max_errors:
            return RecoveryAction(
                strategy=ErrorRecoveryStrategy.PANIC_MODE,
                message="Too many errors, stopping recovery"
            )
        
        # Recovery strategies based on error type
        if error.code == ParseErrorCode.UNEXPECTED_TOKEN:
            return RecoveryAction(
                strategy=ErrorRecoveryStrategy.PANIC_MODE,
                sync_tokens=list(self.sync_tokens),
                message="Skip to next synchronization point"
            )
        
        elif error.code == ParseErrorCode.EXPECTED_TOKEN:
            if error.context.expected_tokens:
                # Try phrase-level recovery by inserting expected token
                return RecoveryAction(
                    strategy=ErrorRecoveryStrategy.PHRASE_LEVEL,
                    insert_tokens=error.context.expected_tokens[:1],
                    message=f"Insert missing {error.context.expected_tokens[0].name}"
                )
        
        elif error.code == ParseErrorCode.MISSING_TOKEN:
            return RecoveryAction(
                strategy=ErrorRecoveryStrategy.PHRASE_LEVEL,
                skip_count=1,
                message="Skip problematic token"
            )
        
        else:
            return RecoveryAction(
                strategy=ErrorRecoveryStrategy.PANIC_MODE,
                sync_tokens=list(self.sync_tokens),
                message="Default panic mode recovery"
            )
    
    def has_errors(self) -> bool:
        """Check if any errors were reported."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings were reported."""
        return len(self.warnings) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and warnings."""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'error_types': dict(self.error_count_by_type),
            'critical_errors': len([e for e in self.errors if e.severity == ParseErrorSeverity.CRITICAL]),
            'has_fatal_errors': any(e.severity == ParseErrorSeverity.CRITICAL for e in self.errors)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive error report."""
        report = []
        
        summary = self.get_error_summary()
        
        report.append("ANAMORPH PARSER ERROR SUMMARY")
        report.append("=" * 40)
        report.append(f"Total Errors: {summary['total_errors']}")
        report.append(f"Total Warnings: {summary['total_warnings']}")
        report.append(f"Critical Errors: {summary['critical_errors']}")
        
        if summary['error_types']:
            report.append("\nError Types:")
            for error_type, count in summary['error_types'].items():
                report.append(f"  {error_type.value}: {count}")
        
        if self.errors:
            report.append("\nDETAILED ERRORS:")
            report.append("-" * 20)
            for i, error in enumerate(self.errors, 1):
                report.append(f"\n{i}. {error}")
                if error.suggestions:
                    report.append("   Suggestions:")
                    for suggestion in error.suggestions:
                        report.append(f"   - {suggestion}")
        
        if self.warnings:
            report.append("\nWARNINGS:")
            report.append("-" * 10)
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"{i}. {warning}")
        
        return "\n".join(report)
    
    def clear(self):
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
        self.error_count_by_type.clear()


# Utility functions for creating common errors
def create_unexpected_token_error(token: Token, context: ParseContext) -> ParseError:
    """Create an unexpected token error."""
    return ParseError(
        code=ParseErrorCode.UNEXPECTED_TOKEN,
        message=f"Unexpected token: {token.type.name} '{token.value}'",
        severity=ParseErrorSeverity.ERROR,
        context=context
    )


def create_expected_token_error(expected: List[TokenType], got: Token, context: ParseContext) -> ParseError:
    """Create an expected token error."""
    expected_names = [t.name for t in expected]
    return ParseError(
        code=ParseErrorCode.EXPECTED_TOKEN,
        message=f"Expected {', '.join(expected_names)}, got {got.type.name} '{got.value}'",
        severity=ParseErrorSeverity.ERROR,
        context=context
    )


def create_missing_token_error(expected: TokenType, context: ParseContext) -> ParseError:
    """Create a missing token error."""
    return ParseError(
        code=ParseErrorCode.MISSING_TOKEN,
        message=f"Missing required token: {expected.name}",
        severity=ParseErrorSeverity.ERROR,
        context=context
    ) 