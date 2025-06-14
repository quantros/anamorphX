"""
Semantic Error Handling for AnamorphX

This module provides comprehensive error handling for semantic analysis,
including error types, severity levels, diagnostics, and recovery strategies.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from ..syntax.nodes import ASTNode, SourceLocation


class SemanticErrorType(Enum):
    """Types of semantic errors that can occur during analysis."""
    
    # Type errors
    TYPE_MISMATCH = auto()
    INCOMPATIBLE_TYPES = auto()
    UNDEFINED_TYPE = auto()
    INVALID_TYPE_OPERATION = auto()
    TYPE_INFERENCE_FAILED = auto()
    CIRCULAR_TYPE_DEPENDENCY = auto()
    
    # Symbol errors
    UNDEFINED_SYMBOL = auto()
    REDEFINED_SYMBOL = auto()
    SYMBOL_NOT_ACCESSIBLE = auto()
    INVALID_SYMBOL_USAGE = auto()
    SYMBOL_SHADOWING = auto()
    
    # Scope errors
    UNDEFINED_VARIABLE = auto()
    VARIABLE_NOT_INITIALIZED = auto()
    INVALID_SCOPE_ACCESS = auto()
    SCOPE_VIOLATION = auto()
    UNREACHABLE_CODE = auto()
    
    # Function errors
    FUNCTION_NOT_FOUND = auto()
    INVALID_ARGUMENT_COUNT = auto()
    INVALID_ARGUMENT_TYPE = auto()
    MISSING_RETURN_VALUE = auto()
    UNREACHABLE_RETURN = auto()
    RECURSIVE_FUNCTION_LIMIT = auto()
    
    # Neural construct errors
    INVALID_NEURON_DECLARATION = auto()
    UNDEFINED_NEURON = auto()
    INVALID_SYNAPSE_CONNECTION = auto()
    SIGNAL_TYPE_MISMATCH = auto()
    PULSE_FREQUENCY_ERROR = auto()
    RESONANCE_CONFLICT = auto()
    NEURAL_NETWORK_CYCLE = auto()
    INVALID_NEURAL_OPERATION = auto()
    
    # Control flow errors
    BREAK_OUTSIDE_LOOP = auto()
    CONTINUE_OUTSIDE_LOOP = auto()
    INVALID_CONTROL_FLOW = auto()
    MISSING_EXCEPTION_HANDLER = auto()
    UNREACHABLE_CATCH_BLOCK = auto()
    
    # Declaration errors
    INVALID_DECLARATION = auto()
    DUPLICATE_DECLARATION = auto()
    MISSING_INITIALIZER = auto()
    INVALID_MODIFIER = auto()
    CONFLICTING_DECLARATIONS = auto()
    
    # Expression errors
    INVALID_EXPRESSION = auto()
    DIVISION_BY_ZERO = auto()
    ARRAY_INDEX_OUT_OF_BOUNDS = auto()
    NULL_POINTER_ACCESS = auto()
    INVALID_MEMBER_ACCESS = auto()
    
    # Import/Export errors
    MODULE_NOT_FOUND = auto()
    CIRCULAR_IMPORT = auto()
    INVALID_EXPORT = auto()
    EXPORT_NOT_FOUND = auto()
    IMPORT_CONFLICT = auto()


class SemanticErrorSeverity(Enum):
    """Severity levels for semantic errors."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SemanticDiagnostic:
    """Diagnostic information for semantic errors."""
    
    message: str
    suggestion: Optional[str] = None
    related_locations: List[SourceLocation] = field(default_factory=list)
    code_snippet: Optional[str] = None
    fix_suggestions: List[str] = field(default_factory=list)
    documentation_link: Optional[str] = None


@dataclass
class SemanticError(Exception):
    """Comprehensive semantic error with detailed information."""
    
    error_type: SemanticErrorType
    message: str
    location: Optional[SourceLocation] = None
    severity: SemanticErrorSeverity = SemanticErrorSeverity.ERROR
    node: Optional[ASTNode] = None
    diagnostic: Optional[SemanticDiagnostic] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    
    def __post_init__(self):
        """Initialize error code and diagnostic if not provided."""
        if not self.error_code:
            self.error_code = f"SE{self.error_type.value:04d}"
        
        if not self.diagnostic:
            self.diagnostic = self._create_diagnostic()
    
    def _create_diagnostic(self) -> SemanticDiagnostic:
        """Create diagnostic information based on error type."""
        suggestions = self._get_suggestions()
        code_snippet = self._get_code_snippet()
        
        return SemanticDiagnostic(
            message=self.message,
            suggestion=suggestions[0] if suggestions else None,
            code_snippet=code_snippet,
            fix_suggestions=suggestions,
            documentation_link=self._get_documentation_link()
        )
    
    def _get_suggestions(self) -> List[str]:
        """Get fix suggestions based on error type."""
        suggestions = {
            SemanticErrorType.UNDEFINED_SYMBOL: [
                "Check if the symbol is declared in the current scope",
                "Verify the symbol name spelling",
                "Import the symbol if it's from another module"
            ],
            SemanticErrorType.TYPE_MISMATCH: [
                "Check the expected type for this operation",
                "Use explicit type conversion if needed",
                "Verify the variable declaration type"
            ],
            SemanticErrorType.UNDEFINED_NEURON: [
                "Declare the neuron before using it",
                "Check the neuron name spelling",
                "Verify the neuron is in the correct scope"
            ],
            SemanticErrorType.INVALID_SYNAPSE_CONNECTION: [
                "Ensure both neurons exist before creating synapse",
                "Check synapse connection syntax",
                "Verify neuron compatibility for connection"
            ]
        }
        
        return suggestions.get(self.error_type, [])
    
    def _get_code_snippet(self) -> Optional[str]:
        """Extract code snippet from location if available."""
        if not self.location or not self.location.source:
            return None
        
        lines = self.location.source.split('\n')
        start_line = max(0, self.location.line - 2)
        end_line = min(len(lines), self.location.line + 2)
        
        snippet_lines = []
        for i in range(start_line, end_line):
            line_num = i + 1
            line_content = lines[i] if i < len(lines) else ""
            marker = ">>> " if line_num == self.location.line else "    "
            snippet_lines.append(f"{marker}{line_num:3d}: {line_content}")
        
        return '\n'.join(snippet_lines)
    
    def _get_documentation_link(self) -> Optional[str]:
        """Get documentation link for error type."""
        base_url = "https://docs.anamorphx.dev/errors"
        return f"{base_url}/{self.error_code.lower()}"
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.location:
            parts.append(f"at {self.location}")
        
        if self.diagnostic and self.diagnostic.suggestion:
            parts.append(f"Suggestion: {self.diagnostic.suggestion}")
        
        return " ".join(parts)


# Specific error types
class TypeError(SemanticError):
    """Type-related semantic error."""
    
    def __init__(self, message: str, expected_type: str = None, 
                 actual_type: str = None, **kwargs):
        super().__init__(
            error_type=SemanticErrorType.TYPE_MISMATCH,
            message=message,
            **kwargs
        )
        if expected_type and actual_type:
            self.context.update({
                'expected_type': expected_type,
                'actual_type': actual_type
            })


class ScopeError(SemanticError):
    """Scope-related semantic error."""
    
    def __init__(self, message: str, scope_name: str = None, **kwargs):
        super().__init__(
            error_type=SemanticErrorType.SCOPE_VIOLATION,
            message=message,
            **kwargs
        )
        if scope_name:
            self.context['scope_name'] = scope_name


class SymbolError(SemanticError):
    """Symbol-related semantic error."""
    
    def __init__(self, message: str, symbol_name: str = None, **kwargs):
        super().__init__(
            error_type=SemanticErrorType.UNDEFINED_SYMBOL,
            message=message,
            **kwargs
        )
        if symbol_name:
            self.context['symbol_name'] = symbol_name


class NeuralError(SemanticError):
    """Neural construct-related semantic error."""
    
    def __init__(self, message: str, neural_type: str = None, **kwargs):
        super().__init__(
            error_type=SemanticErrorType.INVALID_NEURAL_OPERATION,
            message=message,
            **kwargs
        )
        if neural_type:
            self.context['neural_type'] = neural_type


@dataclass
class ErrorRecoveryStrategy:
    """Strategy for recovering from semantic errors."""
    
    name: str
    description: str
    applicable_errors: List[SemanticErrorType]
    recovery_action: callable
    confidence: float = 0.5  # 0.0 to 1.0


class SemanticErrorHandler:
    """Handles semantic errors with recovery strategies."""
    
    def __init__(self):
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        self.max_errors = 100
        self.error_threshold = 10
        
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup default error recovery strategies."""
        self.recovery_strategies = [
            ErrorRecoveryStrategy(
                name="symbol_suggestion",
                description="Suggest similar symbol names",
                applicable_errors=[SemanticErrorType.UNDEFINED_SYMBOL],
                recovery_action=self._suggest_similar_symbols,
                confidence=0.8
            ),
            ErrorRecoveryStrategy(
                name="type_coercion",
                description="Attempt automatic type conversion",
                applicable_errors=[SemanticErrorType.TYPE_MISMATCH],
                recovery_action=self._attempt_type_coercion,
                confidence=0.6
            ),
            ErrorRecoveryStrategy(
                name="scope_resolution",
                description="Search in parent scopes",
                applicable_errors=[SemanticErrorType.UNDEFINED_VARIABLE],
                recovery_action=self._resolve_in_parent_scope,
                confidence=0.7
            )
        ]
    
    def add_error(self, error: SemanticError) -> bool:
        """Add a semantic error and attempt recovery."""
        if len(self.errors) >= self.max_errors:
            return False
        
        if error.severity == SemanticErrorSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.errors.append(error)
        
        # Attempt recovery
        self._attempt_recovery(error)
        
        return len(self.errors) < self.error_threshold
    
    def _attempt_recovery(self, error: SemanticError):
        """Attempt to recover from the error."""
        for strategy in self.recovery_strategies:
            if error.error_type in strategy.applicable_errors:
                try:
                    strategy.recovery_action(error)
                except Exception:
                    # Recovery failed, continue with other strategies
                    continue
    
    def _suggest_similar_symbols(self, error: SemanticError):
        """Suggest similar symbol names."""
        # Implementation would use fuzzy matching
        pass
    
    def _attempt_type_coercion(self, error: SemanticError):
        """Attempt automatic type conversion."""
        # Implementation would try safe type conversions
        pass
    
    def _resolve_in_parent_scope(self, error: SemanticError):
        """Try to resolve symbol in parent scopes."""
        # Implementation would search parent scopes
        pass
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and warnings."""
        return {
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'critical_errors': len([e for e in self.errors 
                                  if e.severity == SemanticErrorSeverity.CRITICAL]),
            'error_types': list(set(e.error_type for e in self.errors)),
            'can_continue': len(self.errors) < self.error_threshold
        }
    
    def format_errors(self, include_warnings: bool = True) -> str:
        """Format all errors and warnings for display."""
        lines = []
        
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  {error}")
                if error.diagnostic and error.diagnostic.code_snippet:
                    lines.append(f"    {error.diagnostic.code_snippet}")
        
        if include_warnings and self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings:
                lines.append(f"  {warning}")
        
        return '\n'.join(lines)
    
    def clear(self):
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear() 