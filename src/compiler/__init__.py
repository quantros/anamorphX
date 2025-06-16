"""
AnamorphX Code Generator Package.

This package provides code generation capabilities for the AnamorphX language,
supporting multiple target platforms and neural-specific optimizations.
"""

from .generator import (
    CodeGenerator,
    GenerationResult,
    GenerationContext,
    generate_code,
    generate_code_async,
)

from .targets import (
    TargetPlatform,
    PythonTarget,
    JavaScriptTarget,
    CppTarget,
    LLVMTarget,
    TargetRegistry,
)

from .templates import (
    CodeTemplate,
    TemplateEngine,
    TemplateRegistry,
    PythonTemplates,
    JavaScriptTemplates,
    CppTemplates,
)

from .optimizers import (
    CodeOptimizer,
    OptimizationLevel,
    OptimizationPass,
    DeadCodeElimination,
    ConstantFolding,
    NeuralOptimizer,
)

from .emitters import (
    CodeEmitter,
    PythonEmitter,
    JavaScriptEmitter,
    CppEmitter,
    LLVMEmitter,
)

from .neural import (
    NeuralCodeGenerator,
    NeuronGenerator,
    SynapseGenerator,
    SignalGenerator,
    PulseGenerator,
)

from .utils import (
    CodeFormatter,
    SymbolMangler,
    DependencyResolver,
    ImportManager,
)

from .errors import (
    CodeGenerationError,
    TargetNotSupportedError,
    TemplateError,
    OptimizationError,
)

# Package version
__version__ = "0.1.0"

# Package metadata
__author__ = "AnamorphX Team"
__description__ = "Multi-target code generator for AnamorphX neural programming language"

# Main exports
__all__ = [
    # Core generator
    'CodeGenerator',
    'GenerationResult',
    'GenerationContext',
    'generate_code',
    'generate_code_async',
    
    # Target platforms
    'TargetPlatform',
    'PythonTarget',
    'JavaScriptTarget',
    'CppTarget',
    'LLVMTarget',
    'TargetRegistry',
    
    # Template system
    'CodeTemplate',
    'TemplateEngine',
    'TemplateRegistry',
    'PythonTemplates',
    'JavaScriptTemplates',
    'CppTemplates',
    
    # Optimization
    'CodeOptimizer',
    'OptimizationLevel',
    'OptimizationPass',
    'DeadCodeElimination',
    'ConstantFolding',
    'NeuralOptimizer',
    
    # Code emitters
    'CodeEmitter',
    'PythonEmitter',
    'JavaScriptEmitter',
    'CppEmitter',
    'LLVMEmitter',
    
    # Neural code generation
    'NeuralCodeGenerator',
    'NeuronGenerator',
    'SynapseGenerator',
    'SignalGenerator',
    'PulseGenerator',
    
    # Utilities
    'CodeFormatter',
    'SymbolMangler',
    'DependencyResolver',
    'ImportManager',
    
    # Error handling
    'CodeGenerationError',
    'TargetNotSupportedError',
    'TemplateError',
    'OptimizationError',
] 