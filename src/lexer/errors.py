"""
Error handling and reporting for Anamorph Lexer.

This module provides comprehensive error handling capabilities including
error recovery, detailed reporting, and context information.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import traceback
from datetime import datetime


class ErrorSeverity(Enum):
    """Severity levels for lexer errors."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ErrorCode(Enum):
    """Specific error codes for different types of lexer errors."""
    
    # Character and encoding errors
    INVALID_CHARACTER = "LEX001"
    INVALID_ENCODING = "LEX002"
    UNEXPECTED_EOF = "LEX003"
    
    # String literal errors
    UNTERMINATED_STRING = "LEX101"
    INVALID_ESCAPE_SEQUENCE = "LEX102"
    STRING_TOO_LONG = "LEX103"
    
    # Numeric literal errors
    INVALID_NUMBER_FORMAT = "LEX201"
    NUMBER_OUT_OF_RANGE = "LEX202"
    INVALID_FLOAT_FORMAT = "LEX203"
    INVALID_SCIENTIFIC_NOTATION = "LEX204"
    
    # Identifier errors
    INVALID_IDENTIFIER = "LEX301"
    IDENTIFIER_TOO_LONG = "LEX302"
    RESERVED_WORD_AS_IDENTIFIER = "LEX303"
    
    # Comment errors
    UNTERMINATED_COMMENT = "LEX401"
    
    # General tokenization errors
    TOKENIZATION_FAILED = "LEX501"
    INTERNAL_ERROR = "LEX502"
    MEMORY_ERROR = "LEX503"
    TIMEOUT_ERROR = "LEX504"


@dataclass
class ErrorContext:
    """Detailed context information for an error."""
    
    source_code: str
    line: int
    column: int
    position: int
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    call_stack: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.call_stack is None:
            self.call_stack = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def get_source_snippet(self, context_lines: int = 2) -> str:
        """Get source code snippet around the error location."""
        lines = self.source_code.split('\n')
        start_line = max(0, self.line - context_lines - 1)
        end_line = min(len(lines), self.line + context_lines)
        
        snippet_lines = []
        for i in range(start_line, end_line):
            line_num = i + 1
            line_content = lines[i] if i < len(lines) else ""
            
            if line_num == self.line:
                # Highlight the error line
                pointer = " " * (self.column - 1) + "^"
                snippet_lines.append(f"{line_num:4d} | {line_content}")
                snippet_lines.append(f"     | {pointer}")
            else:
                snippet_lines.append(f"{line_num:4d} | {line_content}")
        
        return "\n".join(snippet_lines)


@dataclass
class LexerError(Exception):
    """Comprehensive lexer error with detailed context."""
    
    code: ErrorCode
    message: str
    severity: ErrorSeverity
    context: ErrorContext
    suggestions: List[str] = None
    related_errors: List['LexerError'] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.related_errors is None:
            self.related_errors = []
    
    def __str__(self) -> str:
        location = ""
        if self.context.file_path:
            location = f"{self.context.file_path}:"
        location += f"{self.context.line}:{self.context.column}"
        
        return f"{self.severity.name} {self.code.value} at {location}: {self.message}"
    
    def get_detailed_report(self) -> str:
        """Generate a detailed error report."""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append(f"ANAMORPH LEXER ERROR REPORT")
        report.append("=" * 60)
        
        # Basic information
        report.append(f"Error Code: {self.code.value}")
        report.append(f"Severity: {self.severity.name}")
        report.append(f"Message: {self.message}")
        report.append(f"Timestamp: {self.context.timestamp.isoformat()}")
        
        # Location information
        if self.context.file_path:
            report.append(f"File: {self.context.file_path}")
        report.append(f"Line: {self.context.line}")
        report.append(f"Column: {self.context.column}")
        report.append(f"Position: {self.context.position}")
        
        # Source code snippet
        report.append("\nSource Code Context:")
        report.append("-" * 30)
        report.append(self.context.get_source_snippet())
        
        # Suggestions
        if self.suggestions:
            report.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                report.append(f"  {i}. {suggestion}")
        
        # Related errors
        if self.related_errors:
            report.append("\nRelated Errors:")
            for error in self.related_errors:
                report.append(f"  - {error}")
        
        # Call stack
        if self.context.call_stack:
            report.append("\nCall Stack:")
            for frame in self.context.call_stack:
                report.append(f"  {frame}")
        
        report.append("=" * 60)
        return "\n".join(report)


class ErrorRecoveryStrategy(Enum):
    """Strategies for recovering from lexer errors."""
    SKIP_CHARACTER = auto()      # Skip the problematic character
    SKIP_TO_DELIMITER = auto()   # Skip to next delimiter
    SKIP_TO_NEWLINE = auto()     # Skip to next newline
    INSERT_MISSING = auto()      # Insert missing character/token
    ABORT = auto()               # Abort tokenization


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    strategy: ErrorRecoveryStrategy
    skip_count: int = 0
    insert_text: str = ""
    message: str = ""


class ErrorHandler:
    """Handles lexer errors with recovery strategies."""
    
    def __init__(self, max_errors: int = 100, enable_recovery: bool = True):
        self.max_errors = max_errors
        self.enable_recovery = enable_recovery
        self.errors: List[LexerError] = []
        self.warnings: List[LexerError] = []
        self.error_count_by_type: Dict[ErrorCode, int] = {}
    
    def report_error(self, 
                    code: ErrorCode,
                    message: str,
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    suggestions: List[str] = None) -> LexerError:
        """Report a new error."""
        
        error = LexerError(
            code=code,
            message=message,
            severity=severity,
            context=context,
            suggestions=suggestions or []
        )
        
        # Add automatic suggestions based on error type
        self._add_automatic_suggestions(error)
        
        # Track error statistics
        self.error_count_by_type[code] = self.error_count_by_type.get(code, 0) + 1
        
        if severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL):
            self.errors.append(error)
        else:
            self.warnings.append(error)
        
        return error
    
    def _add_automatic_suggestions(self, error: LexerError):
        """Add automatic suggestions based on error type."""
        
        if error.code == ErrorCode.UNTERMINATED_STRING:
            error.suggestions.extend([
                "Add closing quote to terminate the string",
                "Check for unescaped quotes within the string",
                "Use escape sequences for quotes: \\\" or \\'"
            ])
        
        elif error.code == ErrorCode.INVALID_ESCAPE_SEQUENCE:
            error.suggestions.extend([
                "Use valid escape sequences: \\n, \\t, \\r, \\\\, \\\", \\'",
                "Escape backslashes with double backslash: \\\\",
                "Check escape sequence documentation"
            ])
        
        elif error.code == ErrorCode.INVALID_NUMBER_FORMAT:
            error.suggestions.extend([
                "Check numeric literal format",
                "Use valid integer format: 123",
                "Use valid float format: 123.45 or 1.23e-4"
            ])
        
        elif error.code == ErrorCode.INVALID_CHARACTER:
            error.suggestions.extend([
                "Remove or replace the invalid character",
                "Check character encoding (UTF-8 expected)",
                "Use valid Anamorph syntax characters"
            ])
        
        elif error.code == ErrorCode.RESERVED_WORD_AS_IDENTIFIER:
            error.suggestions.extend([
                "Use a different identifier name",
                "Add prefix or suffix to make it unique",
                "Check reserved words list in documentation"
            ])
    
    def get_recovery_action(self, error: LexerError) -> RecoveryAction:
        """Determine recovery action for an error."""
        
        if not self.enable_recovery:
            return RecoveryAction(ErrorRecoveryStrategy.ABORT)
        
        if len(self.errors) >= self.max_errors:
            return RecoveryAction(
                ErrorRecoveryStrategy.ABORT,
                message="Too many errors, aborting tokenization"
            )
        
        # Recovery strategies based on error type
        if error.code == ErrorCode.INVALID_CHARACTER:
            return RecoveryAction(
                ErrorRecoveryStrategy.SKIP_CHARACTER,
                skip_count=1,
                message="Skipping invalid character"
            )
        
        elif error.code == ErrorCode.UNTERMINATED_STRING:
            return RecoveryAction(
                ErrorRecoveryStrategy.SKIP_TO_NEWLINE,
                message="Skipping to next line after unterminated string"
            )
        
        elif error.code in (ErrorCode.INVALID_NUMBER_FORMAT, ErrorCode.INVALID_IDENTIFIER):
            return RecoveryAction(
                ErrorRecoveryStrategy.SKIP_TO_DELIMITER,
                message="Skipping to next delimiter"
            )
        
        else:
            return RecoveryAction(
                ErrorRecoveryStrategy.SKIP_CHARACTER,
                skip_count=1,
                message="Default recovery: skip character"
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
            'critical_errors': len([e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]),
            'has_fatal_errors': any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive error report."""
        report = []
        
        summary = self.get_error_summary()
        
        report.append("ANAMORPH LEXER ERROR SUMMARY")
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
def create_invalid_character_error(char: str, context: ErrorContext) -> LexerError:
    """Create an invalid character error."""
    return LexerError(
        code=ErrorCode.INVALID_CHARACTER,
        message=f"Invalid character: '{char}' (Unicode: U+{ord(char):04X})",
        severity=ErrorSeverity.ERROR,
        context=context
    )


def create_unterminated_string_error(context: ErrorContext) -> LexerError:
    """Create an unterminated string error."""
    return LexerError(
        code=ErrorCode.UNTERMINATED_STRING,
        message="String literal is not properly terminated",
        severity=ErrorSeverity.ERROR,
        context=context
    )


def create_invalid_number_error(number_str: str, context: ErrorContext) -> LexerError:
    """Create an invalid number format error."""
    return LexerError(
        code=ErrorCode.INVALID_NUMBER_FORMAT,
        message=f"Invalid number format: '{number_str}'",
        severity=ErrorSeverity.ERROR,
        context=context
    ) 