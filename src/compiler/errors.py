"""
Code generation error handling for AnamorphX.

This module provides comprehensive error handling for code generation,
including specific error types and diagnostic capabilities.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from ..syntax.nodes import ASTNode, SourceLocation


class CodeGenerationErrorType(Enum):
    """Types of code generation errors."""
    
    # Target platform errors
    TARGET_NOT_SUPPORTED = auto()
    TARGET_FEATURE_MISSING = auto()
    TARGET_INCOMPATIBLE = auto()
    
    # Template errors
    TEMPLATE_NOT_FOUND = auto()
    TEMPLATE_SYNTAX_ERROR = auto()
    TEMPLATE_VARIABLE_MISSING = auto()
    TEMPLATE_COMPILATION_ERROR = auto()
    
    # Code generation errors
    UNSUPPORTED_CONSTRUCT = auto()
    INVALID_SYMBOL_NAME = auto()
    MISSING_DEPENDENCY = auto()
    CIRCULAR_DEPENDENCY = auto()
    
    # Neural-specific errors
    NEURAL_CONSTRUCT_ERROR = auto()
    INVALID_NEURON_TYPE = auto()
    SYNAPSE_CONNECTION_ERROR = auto()
    SIGNAL_TYPE_MISMATCH = auto()
    
    # Optimization errors
    OPTIMIZATION_FAILED = auto()
    INVALID_OPTIMIZATION_PASS = auto()
    OPTIMIZATION_CONFLICT = auto()
    
    # Output errors
    FILE_WRITE_ERROR = auto()
    PERMISSION_DENIED = auto()
    DISK_SPACE_ERROR = auto()
    
    # Internal errors
    INTERNAL_ERROR = auto()
    ASSERTION_FAILED = auto()
    MEMORY_ERROR = auto()


class CodeGenerationErrorSeverity(Enum):
    """Severity levels for code generation errors."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CodeGenerationDiagnostic:
    """Diagnostic information for code generation errors."""
    
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    related_locations: List[SourceLocation] = field(default_factory=list)
    documentation_url: Optional[str] = None
    fix_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic to dictionary."""
        return {
            'message': self.message,
            'suggestion': self.suggestion,
            'code_snippet': self.code_snippet,
            'related_locations': [str(loc) for loc in self.related_locations],
            'documentation_url': self.documentation_url,
            'fix_available': self.fix_available
        }


@dataclass
class CodeGenerationError(Exception):
    """Base class for code generation errors."""
    
    error_type: CodeGenerationErrorType
    message: str
    location: Optional[SourceLocation] = None
    node: Optional[ASTNode] = None
    severity: CodeGenerationErrorSeverity = CodeGenerationErrorSeverity.ERROR
    target_platform: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    diagnostic: Optional[CodeGenerationDiagnostic] = None
    
    def __post_init__(self):
        """Initialize error after creation."""
        super().__init__(self.message)
        if not self.diagnostic:
            self.diagnostic = self._create_diagnostic()
    
    def _create_diagnostic(self) -> CodeGenerationDiagnostic:
        """Create diagnostic information for this error."""
        suggestion = self._get_suggestion()
        code_snippet = self._get_code_snippet()
        documentation_url = self._get_documentation_url()
        
        return CodeGenerationDiagnostic(
            message=self.message,
            suggestion=suggestion,
            code_snippet=code_snippet,
            documentation_url=documentation_url,
            fix_available=suggestion is not None
        )
    
    def _get_suggestion(self) -> Optional[str]:
        """Get suggestion for fixing this error."""
        suggestions = {
            CodeGenerationErrorType.TARGET_NOT_SUPPORTED: 
                "Try using a supported target platform like 'python', 'javascript', or 'cpp'",
            CodeGenerationErrorType.TEMPLATE_NOT_FOUND:
                "Check if the template file exists and is in the correct location",
            CodeGenerationErrorType.UNSUPPORTED_CONSTRUCT:
                "This language construct may not be supported in the target platform",
            CodeGenerationErrorType.INVALID_SYMBOL_NAME:
                "Use valid identifier names according to the target language conventions",
            CodeGenerationErrorType.MISSING_DEPENDENCY:
                "Add the required dependency to your project configuration",
            CodeGenerationErrorType.NEURAL_CONSTRUCT_ERROR:
                "Check the neural construct syntax and ensure proper neuron/synapse definitions",
        }
        return suggestions.get(self.error_type)
    
    def _get_code_snippet(self) -> Optional[str]:
        """Get code snippet around the error location."""
        if not self.location:
            return None
        
        # This would be implemented to extract code around the error location
        # For now, return a placeholder
        return f"Line {self.location.line}: <code snippet>"
    
    def _get_documentation_url(self) -> Optional[str]:
        """Get documentation URL for this error type."""
        base_url = "https://docs.anamorphx.dev/errors/codegen"
        error_codes = {
            CodeGenerationErrorType.TARGET_NOT_SUPPORTED: "target-not-supported",
            CodeGenerationErrorType.TEMPLATE_NOT_FOUND: "template-not-found",
            CodeGenerationErrorType.UNSUPPORTED_CONSTRUCT: "unsupported-construct",
            CodeGenerationErrorType.NEURAL_CONSTRUCT_ERROR: "neural-construct-error",
        }
        
        if self.error_type in error_codes:
            return f"{base_url}/{error_codes[self.error_type]}"
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.error_type.name,
            'message': self.message,
            'location': str(self.location) if self.location else None,
            'severity': self.severity.value,
            'target_platform': self.target_platform,
            'context': self.context,
            'diagnostic': self.diagnostic.to_dict() if self.diagnostic else None
        }


class TargetNotSupportedError(CodeGenerationError):
    """Error for unsupported target platforms."""
    
    def __init__(self, target: str, supported_targets: List[str], **kwargs):
        message = f"Target platform '{target}' is not supported. Supported targets: {', '.join(supported_targets)}"
        super().__init__(
            error_type=CodeGenerationErrorType.TARGET_NOT_SUPPORTED,
            message=message,
            target_platform=target,
            context={'supported_targets': supported_targets},
            **kwargs
        )


class TemplateError(CodeGenerationError):
    """Error for template-related issues."""
    
    def __init__(self, template_name: str, template_error: str, **kwargs):
        message = f"Template error in '{template_name}': {template_error}"
        super().__init__(
            error_type=CodeGenerationErrorType.TEMPLATE_SYNTAX_ERROR,
            message=message,
            context={'template_name': template_name, 'template_error': template_error},
            **kwargs
        )


class OptimizationError(CodeGenerationError):
    """Error for optimization-related issues."""
    
    def __init__(self, optimization_pass: str, error_message: str, **kwargs):
        message = f"Optimization error in pass '{optimization_pass}': {error_message}"
        super().__init__(
            error_type=CodeGenerationErrorType.OPTIMIZATION_FAILED,
            message=message,
            context={'optimization_pass': optimization_pass, 'error_message': error_message},
            **kwargs
        )


class NeuralConstructError(CodeGenerationError):
    """Error for neural construct generation issues."""
    
    def __init__(self, construct_type: str, construct_name: str, error_message: str, **kwargs):
        message = f"Neural construct error in {construct_type} '{construct_name}': {error_message}"
        super().__init__(
            error_type=CodeGenerationErrorType.NEURAL_CONSTRUCT_ERROR,
            message=message,
            context={
                'construct_type': construct_type,
                'construct_name': construct_name,
                'error_message': error_message
            },
            **kwargs
        )


class CodeGenerationErrorHandler:
    """Handles and manages code generation errors."""
    
    def __init__(self):
        self.errors: List[CodeGenerationError] = []
        self.warnings: List[CodeGenerationError] = []
        self.max_errors = 100
        self.error_recovery_enabled = True
    
    def add_error(self, error: CodeGenerationError):
        """Add an error to the handler."""
        if error.severity in [CodeGenerationErrorSeverity.ERROR, CodeGenerationErrorSeverity.CRITICAL]:
            self.errors.append(error)
        else:
            self.warnings.append(error)
        
        # Limit number of errors to prevent memory issues
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return any(error.severity == CodeGenerationErrorSeverity.CRITICAL for error in self.errors)
    
    def get_error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors)
    
    def get_warning_count(self) -> int:
        """Get total number of warnings."""
        return len(self.warnings)
    
    def clear_errors(self):
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and warnings."""
        error_types = {}
        for error in self.errors + self.warnings:
            error_type = error.error_type.name
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'error_types': error_types,
            'has_critical': self.has_critical_errors()
        }
    
    def format_errors(self, include_diagnostics: bool = True) -> str:
        """Format all errors and warnings for display."""
        lines = []
        
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error.message}")
                if include_diagnostics and error.diagnostic and error.diagnostic.suggestion:
                    lines.append(f"     Suggestion: {error.diagnostic.suggestion}")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning.message}")
        
        return "\n".join(lines) if lines else "No errors or warnings." 