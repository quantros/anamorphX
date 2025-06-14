"""
Main Semantic Analyzer for AnamorphX

This module provides the main semantic analyzer that orchestrates
all semantic analysis phases for the Anamorph programming language.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
import asyncio
import time
from ..syntax.nodes import ASTNode, Program
from ..syntax.visitor import ASTVisitor
from .symbols import SymbolResolver, SymbolCollector, SymbolTable
from .types import TypeSystem, TypeChecker, TypeInference
from .scopes import ScopeManager, ScopeResolver, ScopeValidator
from .validators import (
    DeclarationValidator, ExpressionValidator, StatementValidator,
    NeuralValidator, FlowValidator, ValidationResult
)
from .neural import NeuralAnalyzer
from .errors import SemanticError, SemanticErrorHandler


class AnalysisPhase(Enum):
    """Phases of semantic analysis."""
    
    SYMBOL_COLLECTION = auto()
    SCOPE_RESOLUTION = auto()
    TYPE_CHECKING = auto()
    DECLARATION_VALIDATION = auto()
    EXPRESSION_VALIDATION = auto()
    STATEMENT_VALIDATION = auto()
    NEURAL_ANALYSIS = auto()
    FLOW_ANALYSIS = auto()
    FINAL_VALIDATION = auto()


@dataclass
class AnalysisContext:
    """Context information for semantic analysis."""
    
    source_file: str
    module_name: str
    imports: Dict[str, str] = field(default_factory=dict)
    exports: Set[str] = field(default_factory=set)
    options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of semantic analysis."""
    
    success: bool
    ast: Optional[ASTNode] = None
    symbol_table: Optional[SymbolTable] = None
    errors: List[SemanticError] = field(default_factory=list)
    warnings: List[SemanticError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    neural_analysis: Dict[str, Any] = field(default_factory=dict)
    analysis_time: float = 0.0
    
    def add_error(self, error: SemanticError):
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: SemanticError):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def merge_validation_result(self, validation_result: ValidationResult):
        """Merge a validation result."""
        self.errors.extend(validation_result.errors)
        self.warnings.extend(validation_result.warnings)
        if not validation_result.is_valid:
            self.success = False


class SemanticPass:
    """Base class for semantic analysis passes."""
    
    def __init__(self, name: str, phase: AnalysisPhase):
        self.name = name
        self.phase = phase
        self.enabled = True
        self.dependencies: List[AnalysisPhase] = []
    
    def run(self, ast: ASTNode, context: AnalysisContext, 
            analyzer: 'SemanticAnalyzer') -> bool:
        """Run the semantic pass."""
        raise NotImplementedError


class SemanticAnalyzer(ASTVisitor):
    """Main semantic analyzer orchestrating all analysis phases."""
    
    def __init__(self, options: Dict[str, Any] = None):
        super().__init__()
        self.options = options or {}
        
        # Core components
        self.type_system = TypeSystem()
        self.scope_manager = ScopeManager()
        self.symbol_resolver = SymbolResolver()
        self.error_handler = SemanticErrorHandler()
        
        # Analysis components
        self.type_checker = TypeChecker(self.type_system)
        self.type_inference = TypeInference(self.type_system, self.type_checker)
        self.scope_resolver = ScopeResolver(self.scope_manager)
        self.scope_validator = ScopeValidator(self.scope_manager)
        self.symbol_collector = SymbolCollector(self.symbol_resolver)
        
        # Validators
        self.declaration_validator = DeclarationValidator(
            self.type_system, self.scope_manager, self.symbol_resolver
        )
        self.expression_validator = ExpressionValidator(
            self.type_system, self.scope_manager, self.symbol_resolver
        )
        self.statement_validator = StatementValidator(
            self.type_system, self.scope_manager, self.symbol_resolver
        )
        self.neural_validator = NeuralValidator(
            self.type_system, self.scope_manager, self.symbol_resolver
        )
        self.flow_validator = FlowValidator(
            self.type_system, self.scope_manager, self.symbol_resolver
        )
        
        # Neural analyzer
        self.neural_analyzer = NeuralAnalyzer(
            self.type_system, self.scope_manager, self.symbol_resolver
        )
        
        # Analysis state
        self.current_context: Optional[AnalysisContext] = None
        self.current_result: Optional[AnalysisResult] = None
        self.analysis_passes: List[SemanticPass] = []
        
        self._setup_analysis_passes()
    
    def _setup_analysis_passes(self):
        """Setup semantic analysis passes."""
        self.analysis_passes = [
            SymbolCollectionPass(),
            ScopeResolutionPass(),
            TypeCheckingPass(),
            DeclarationValidationPass(),
            ExpressionValidationPass(),
            StatementValidationPass(),
            NeuralAnalysisPass(),
            FlowAnalysisPass(),
            FinalValidationPass()
        ]
    
    def analyze(self, ast: ASTNode, context: AnalysisContext = None) -> AnalysisResult:
        """Perform complete semantic analysis."""
        start_time = time.time()
        
        # Initialize analysis
        self.current_context = context or AnalysisContext(
            source_file="<unknown>",
            module_name="main"
        )
        self.current_result = AnalysisResult(success=True, ast=ast)
        
        try:
            # Run analysis passes
            for pass_obj in self.analysis_passes:
                if pass_obj.enabled:
                    if not self._run_pass(pass_obj):
                        self.current_result.success = False
                        if self.options.get('fail_fast', False):
                            break
            
            # Finalize analysis
            self._finalize_analysis()
            
        except Exception as e:
            self.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                message=f"Analysis failed: {e}",
                context={'exception': str(e)}
            ))
        
        # Record timing
        self.current_result.analysis_time = time.time() - start_time
        
        return self.current_result
    
    async def analyze_async(self, ast: ASTNode, context: AnalysisContext = None) -> AnalysisResult:
        """Perform asynchronous semantic analysis."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.analyze, ast, context
        )
    
    def _run_pass(self, pass_obj: SemanticPass) -> bool:
        """Run a single analysis pass."""
        try:
            # Check dependencies
            for dep_phase in pass_obj.dependencies:
                if not self._is_phase_completed(dep_phase):
                    self.current_result.add_error(SemanticError(
                        error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                        message=f"Pass {pass_obj.name} dependency {dep_phase.name} not satisfied"
                    ))
                    return False
            
            # Run the pass
            success = pass_obj.run(
                self.current_result.ast,
                self.current_context,
                self
            )
            
            # Record pass completion
            self.current_result.metrics[f"{pass_obj.name}_completed"] = True
            
            return success
            
        except Exception as e:
            self.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                message=f"Pass {pass_obj.name} failed: {e}"
            ))
            return False
    
    def _is_phase_completed(self, phase: AnalysisPhase) -> bool:
        """Check if an analysis phase is completed."""
        for pass_obj in self.analysis_passes:
            if pass_obj.phase == phase:
                return self.current_result.metrics.get(f"{pass_obj.name}_completed", False)
        return False
    
    def _finalize_analysis(self):
        """Finalize semantic analysis."""
        # Collect all errors from components
        self.current_result.errors.extend(self.error_handler.errors)
        self.current_result.warnings.extend(self.error_handler.warnings)
        
        # Set symbol table
        self.current_result.symbol_table = self.symbol_resolver.global_table
        
        # Collect metrics
        self.current_result.metrics.update({
            'symbol_count': len(self.symbol_resolver.global_table.get_all_symbols()),
            'scope_depth': self.scope_manager.get_scope_depth(),
            'neural_neurons': len(self.neural_analyzer.neuron_analyzer.neurons),
            'neural_synapses': len(self.neural_analyzer.synapse_analyzer.synapses),
            'error_count': len(self.current_result.errors),
            'warning_count': len(self.current_result.warnings)
        })
        
        # Neural analysis results
        if self.neural_analyzer.neuron_analyzer.neurons:
            self.current_result.neural_analysis = self.neural_analyzer.analyze_signal_flow()
    
    # AST Visitor methods for traversal
    def visit_program(self, node: Program):
        """Visit program node."""
        # Enter global scope
        self.scope_manager.enter_scope(self.scope_manager.global_scope)
        
        # Visit all statements
        for stmt in node.statements:
            self.visit(stmt)
        
        # Exit global scope
        self.scope_manager.exit_scope()
    
    def visit_function_declaration(self, node):
        """Visit function declaration."""
        # Collect function symbol
        symbols = self.symbol_collector.collect_from_node(node)
        for symbol in symbols:
            self.symbol_resolver.define_symbol(symbol)
        
        # Enter function scope
        function_scope = self.scope_manager.create_function_scope(
            node.name,
            getattr(node, 'return_type', None)
        )
        
        # Visit function body
        if hasattr(node, 'body') and node.body:
            self.visit(node.body)
        
        # Exit function scope
        self.scope_manager.exit_scope()
    
    def visit_variable_declaration(self, node):
        """Visit variable declaration."""
        # Collect variable symbol
        symbols = self.symbol_collector.collect_from_node(node)
        for symbol in symbols:
            self.symbol_resolver.define_symbol(symbol)
        
        # Visit initializer if present
        if hasattr(node, 'initializer') and node.initializer:
            self.visit(node.initializer)
    
    def visit_neuron_declaration(self, node):
        """Visit neuron declaration."""
        # Analyze neural construct
        self.neural_analyzer.analyze_neural_construct(node)
        
        # Collect neuron symbol
        symbols = self.symbol_collector.collect_from_node(node)
        for symbol in symbols:
            self.symbol_resolver.define_symbol(symbol)
        
        # Enter neuron scope
        neuron_scope = self.scope_manager.create_neuron_scope(
            node.name,
            getattr(node, 'neuron_type', 'basic')
        )
        
        # Visit neuron body
        if hasattr(node, 'body') and node.body:
            self.visit(node.body)
        
        # Exit neuron scope
        self.scope_manager.exit_scope()
    
    def visit_synapse_declaration(self, node):
        """Visit synapse declaration."""
        # Analyze neural construct
        self.neural_analyzer.analyze_neural_construct(node)
        
        # Collect synapse symbol
        symbols = self.symbol_collector.collect_from_node(node)
        for symbol in symbols:
            self.symbol_resolver.define_symbol(symbol)
    
    def visit_block_statement(self, node):
        """Visit block statement."""
        # Enter block scope
        self.scope_manager.create_block_scope()
        
        # Visit all statements
        for stmt in getattr(node, 'statements', []):
            self.visit(stmt)
        
        # Exit block scope
        self.scope_manager.exit_scope()


# Analysis pass implementations
class SymbolCollectionPass(SemanticPass):
    """Pass for collecting symbols from AST."""
    
    def __init__(self):
        super().__init__("symbol_collection", AnalysisPhase.SYMBOL_COLLECTION)
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run symbol collection pass."""
        try:
            # Visit AST to collect symbols
            analyzer.visit(ast)
            return True
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                message=f"Symbol collection failed: {e}"
            ))
            return False


class ScopeResolutionPass(SemanticPass):
    """Pass for resolving scopes and symbol references."""
    
    def __init__(self):
        super().__init__("scope_resolution", AnalysisPhase.SCOPE_RESOLUTION)
        self.dependencies = [AnalysisPhase.SYMBOL_COLLECTION]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run scope resolution pass."""
        try:
            # Validate scopes
            success = analyzer.scope_validator.validate_all_scopes()
            
            # Merge errors
            analyzer.current_result.errors.extend(analyzer.scope_validator.errors)
            
            return success
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.SCOPE_VIOLATION,
                message=f"Scope resolution failed: {e}"
            ))
            return False


class TypeCheckingPass(SemanticPass):
    """Pass for type checking and inference."""
    
    def __init__(self):
        super().__init__("type_checking", AnalysisPhase.TYPE_CHECKING)
        self.dependencies = [AnalysisPhase.SYMBOL_COLLECTION, AnalysisPhase.SCOPE_RESOLUTION]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run type checking pass."""
        try:
            # Type check the AST
            analyzer.type_checker.check_node(ast)
            
            # Merge errors
            analyzer.current_result.errors.extend(analyzer.type_checker.errors)
            
            return len(analyzer.type_checker.errors) == 0
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.TYPE_MISMATCH,
                message=f"Type checking failed: {e}"
            ))
            return False


class DeclarationValidationPass(SemanticPass):
    """Pass for validating declarations."""
    
    def __init__(self):
        super().__init__("declaration_validation", AnalysisPhase.DECLARATION_VALIDATION)
        self.dependencies = [AnalysisPhase.TYPE_CHECKING]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run declaration validation pass."""
        try:
            # Validate declarations
            result = analyzer.declaration_validator.validate_node(ast)
            analyzer.current_result.merge_validation_result(result)
            
            return result.is_valid
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_DECLARATION,
                message=f"Declaration validation failed: {e}"
            ))
            return False


class ExpressionValidationPass(SemanticPass):
    """Pass for validating expressions."""
    
    def __init__(self):
        super().__init__("expression_validation", AnalysisPhase.EXPRESSION_VALIDATION)
        self.dependencies = [AnalysisPhase.TYPE_CHECKING]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run expression validation pass."""
        try:
            # Validate expressions
            result = analyzer.expression_validator.validate_node(ast)
            analyzer.current_result.merge_validation_result(result)
            
            return result.is_valid
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                message=f"Expression validation failed: {e}"
            ))
            return False


class StatementValidationPass(SemanticPass):
    """Pass for validating statements."""
    
    def __init__(self):
        super().__init__("statement_validation", AnalysisPhase.STATEMENT_VALIDATION)
        self.dependencies = [AnalysisPhase.EXPRESSION_VALIDATION]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run statement validation pass."""
        try:
            # Validate statements
            result = analyzer.statement_validator.validate_node(ast)
            analyzer.current_result.merge_validation_result(result)
            
            return result.is_valid
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                message=f"Statement validation failed: {e}"
            ))
            return False


class NeuralAnalysisPass(SemanticPass):
    """Pass for neural construct analysis."""
    
    def __init__(self):
        super().__init__("neural_analysis", AnalysisPhase.NEURAL_ANALYSIS)
        self.dependencies = [AnalysisPhase.STATEMENT_VALIDATION]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run neural analysis pass."""
        try:
            # Validate neural network
            success = analyzer.neural_analyzer.validate_neural_network()
            
            # Merge errors
            analyzer.current_result.errors.extend(analyzer.neural_analyzer.errors)
            
            return success
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_NEURAL_OPERATION,
                message=f"Neural analysis failed: {e}"
            ))
            return False


class FlowAnalysisPass(SemanticPass):
    """Pass for control flow analysis."""
    
    def __init__(self):
        super().__init__("flow_analysis", AnalysisPhase.FLOW_ANALYSIS)
        self.dependencies = [AnalysisPhase.NEURAL_ANALYSIS]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run flow analysis pass."""
        try:
            # Validate reachability
            result = analyzer.flow_validator.validate_reachability(ast)
            analyzer.current_result.merge_validation_result(result)
            
            return result.is_valid
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.UNREACHABLE_CODE,
                message=f"Flow analysis failed: {e}"
            ))
            return False


class FinalValidationPass(SemanticPass):
    """Final validation pass."""
    
    def __init__(self):
        super().__init__("final_validation", AnalysisPhase.FINAL_VALIDATION)
        self.dependencies = [AnalysisPhase.FLOW_ANALYSIS]
    
    def run(self, ast: ASTNode, context: AnalysisContext, analyzer: SemanticAnalyzer) -> bool:
        """Run final validation pass."""
        try:
            # Perform final checks
            success = True
            
            # Check for unresolved symbols
            if analyzer.symbol_resolver.unresolved_references:
                for name, location in analyzer.symbol_resolver.unresolved_references:
                    analyzer.current_result.add_error(SemanticError(
                        error_type=SemanticError.SemanticErrorType.UNDEFINED_SYMBOL,
                        message=f"Unresolved symbol '{name}'",
                        location=location
                    ))
                    success = False
            
            return success
        except Exception as e:
            analyzer.current_result.add_error(SemanticError(
                error_type=SemanticError.SemanticErrorType.INVALID_EXPRESSION,
                message=f"Final validation failed: {e}"
            ))
            return False 