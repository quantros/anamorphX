"""
Semantic Analysis Package for AnamorphX

This package provides comprehensive semantic analysis capabilities for the Anamorph
neural programming language, including type checking, scope analysis, symbol resolution,
and neural construct validation.

Components:
- analyzer: Main semantic analyzer
- symbols: Symbol table and symbol management
- types: Type system and type checking
- scopes: Scope management and resolution
- validators: Semantic validation rules
- errors: Semantic error handling
- neural: Neural construct semantic analysis
"""

from .analyzer import (
    SemanticAnalyzer,
    AnalysisResult,
    AnalysisContext,
    AnalysisPhase,
    SemanticPass
)

from .symbols import (
    Symbol,
    SymbolTable,
    SymbolType,
    SymbolScope,
    VariableSymbol,
    FunctionSymbol,
    NeuronSymbol,
    SynapseSymbol,
    SymbolResolver,
    SymbolCollector
)

from .types import (
    Type,
    TypeSystem,
    TypeChecker,
    TypeInference,
    PrimitiveType,
    ArrayType,
    ObjectType,
    FunctionType,
    NeuralType,
    SignalType,
    PulseType,
    TypeCompatibility,
    TypeCoercion,
    TypeValidator
)

from .scopes import (
    Scope,
    ScopeManager,
    ScopeType,
    GlobalScope,
    FunctionScope,
    BlockScope,
    NeuronScope,
    ScopeResolver,
    ScopeValidator
)

from .validators import (
    SemanticValidator,
    DeclarationValidator,
    ExpressionValidator,
    StatementValidator,
    NeuralValidator,
    FlowValidator,
    ValidationRule,
    ValidationResult
)

from .errors import (
    SemanticError,
    SemanticErrorType,
    SemanticErrorSeverity,
    TypeError,
    ScopeError,
    SymbolError,
    NeuralError,
    SemanticErrorHandler,
    SemanticDiagnostic
)

from .neural import (
    NeuralAnalyzer,
    NeuronAnalyzer,
    SynapseAnalyzer,
    SignalAnalyzer,
    PulseAnalyzer,
    ResonanceAnalyzer,
    NeuralNetworkValidator,
    NeuralFlowAnalyzer
)

__all__ = [
    # Main analyzer
    'SemanticAnalyzer',
    'AnalysisResult',
    'AnalysisContext',
    'AnalysisPhase',
    'SemanticPass',
    
    # Symbols
    'Symbol',
    'SymbolTable',
    'SymbolType',
    'SymbolScope',
    'VariableSymbol',
    'FunctionSymbol',
    'NeuronSymbol',
    'SynapseSymbol',
    'SymbolResolver',
    'SymbolCollector',
    
    # Types
    'Type',
    'TypeSystem',
    'TypeChecker',
    'TypeInference',
    'PrimitiveType',
    'ArrayType',
    'ObjectType',
    'FunctionType',
    'NeuralType',
    'SignalType',
    'PulseType',
    'TypeCompatibility',
    'TypeCoercion',
    'TypeValidator',
    
    # Scopes
    'Scope',
    'ScopeManager',
    'ScopeType',
    'GlobalScope',
    'FunctionScope',
    'BlockScope',
    'NeuronScope',
    'ScopeResolver',
    'ScopeValidator',
    
    # Validators
    'SemanticValidator',
    'DeclarationValidator',
    'ExpressionValidator',
    'StatementValidator',
    'NeuralValidator',
    'FlowValidator',
    'ValidationRule',
    'ValidationResult',
    
    # Errors
    'SemanticError',
    'SemanticErrorType',
    'SemanticErrorSeverity',
    'TypeError',
    'ScopeError',
    'SymbolError',
    'NeuralError',
    'SemanticErrorHandler',
    'SemanticDiagnostic',
    
    # Neural analysis
    'NeuralAnalyzer',
    'NeuronAnalyzer',
    'SynapseAnalyzer',
    'SignalAnalyzer',
    'PulseAnalyzer',
    'ResonanceAnalyzer',
    'NeuralNetworkValidator',
    'NeuralFlowAnalyzer'
]

# Version info
__version__ = '1.0.0'
__author__ = 'AnamorphX Team'
__description__ = 'Semantic Analysis for Anamorph Neural Programming Language' 