"""
Main code generator for AnamorphX.

This module provides the primary code generation interface that orchestrates
the entire compilation process from AST to target code.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, IO
from pathlib import Path
import json

from ..syntax.nodes import ASTNode, Program
from ..semantic.analyzer import SemanticAnalyzer, AnalysisResult
from .targets import TargetPlatform, target_registry
from .templates import TemplateEngine, TemplateContext, template_registry
from .optimizers import CodeOptimizer, OptimizationLevel, OptimizationResult
from .errors import (
    CodeGenerationError, 
    CodeGenerationErrorHandler,
    TargetNotSupportedError,
    CodeGenerationErrorType
)


@dataclass
class GenerationOptions:
    """Options for code generation."""
    
    # Target configuration
    target_platform: str = "python"
    target_version: str = "latest"
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    enable_optimizations: bool = True
    
    # Output settings
    output_directory: Optional[str] = None
    output_filename: Optional[str] = None
    include_debug_info: bool = False
    include_source_maps: bool = False
    
    # Neural settings
    neural_runtime: str = "builtin"
    signal_implementation: str = "async"
    
    # Code style
    format_code: bool = True
    include_comments: bool = True
    include_type_annotations: bool = True
    
    # Advanced options
    parallel_generation: bool = False
    max_workers: int = 4
    memory_limit: int = 1024 * 1024 * 1024  # 1GB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary."""
        return {
            'target_platform': self.target_platform,
            'target_version': self.target_version,
            'optimization_level': self.optimization_level.name,
            'enable_optimizations': self.enable_optimizations,
            'output_directory': self.output_directory,
            'output_filename': self.output_filename,
            'include_debug_info': self.include_debug_info,
            'include_source_maps': self.include_source_maps,
            'neural_runtime': self.neural_runtime,
            'signal_implementation': self.signal_implementation,
            'format_code': self.format_code,
            'include_comments': self.include_comments,
            'include_type_annotations': self.include_type_annotations,
            'parallel_generation': self.parallel_generation,
            'max_workers': self.max_workers,
            'memory_limit': self.memory_limit,
        }


@dataclass
class GenerationMetrics:
    """Metrics for code generation process."""
    
    # Timing metrics
    total_time: float = 0.0
    semantic_analysis_time: float = 0.0
    optimization_time: float = 0.0
    template_rendering_time: float = 0.0
    code_formatting_time: float = 0.0
    file_writing_time: float = 0.0
    
    # Size metrics
    input_ast_nodes: int = 0
    output_code_lines: int = 0
    output_code_size: int = 0
    
    # Generation metrics
    functions_generated: int = 0
    classes_generated: int = 0
    neurons_generated: int = 0
    synapses_generated: int = 0
    
    # Optimization metrics
    optimization_passes: int = 0
    nodes_optimized: int = 0
    performance_gain: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_time': self.total_time,
            'semantic_analysis_time': self.semantic_analysis_time,
            'optimization_time': self.optimization_time,
            'template_rendering_time': self.template_rendering_time,
            'code_formatting_time': self.code_formatting_time,
            'file_writing_time': self.file_writing_time,
            'input_ast_nodes': self.input_ast_nodes,
            'output_code_lines': self.output_code_lines,
            'output_code_size': self.output_code_size,
            'functions_generated': self.functions_generated,
            'classes_generated': self.classes_generated,
            'neurons_generated': self.neurons_generated,
            'synapses_generated': self.synapses_generated,
            'optimization_passes': self.optimization_passes,
            'nodes_optimized': self.nodes_optimized,
            'performance_gain': self.performance_gain,
        }


@dataclass
class GenerationResult:
    """Result of code generation process."""
    
    success: bool
    generated_code: str = ""
    output_files: List[str] = field(default_factory=list)
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    semantic_result: Optional[AnalysisResult] = None
    optimization_result: Optional[OptimizationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'generated_code': self.generated_code,
            'output_files': self.output_files,
            'metrics': self.metrics.to_dict(),
            'warnings': self.warnings,
            'errors': self.errors,
            'has_semantic_result': self.semantic_result is not None,
            'has_optimization_result': self.optimization_result is not None,
        }


@dataclass
class GenerationContext:
    """Context for code generation process."""
    
    ast: ASTNode
    target: TargetPlatform
    template_engine: TemplateEngine
    options: GenerationOptions
    error_handler: CodeGenerationErrorHandler
    
    # Generation state
    current_scope: str = "global"
    symbol_table: Dict[str, Any] = field(default_factory=dict)
    import_statements: List[str] = field(default_factory=list)
    generated_functions: List[str] = field(default_factory=list)
    generated_classes: List[str] = field(default_factory=list)
    generated_neurons: List[str] = field(default_factory=list)
    
    def add_import(self, import_statement: str):
        """Add import statement."""
        if import_statement not in self.import_statements:
            self.import_statements.append(import_statement)
    
    def add_symbol(self, name: str, symbol_info: Any):
        """Add symbol to symbol table."""
        self.symbol_table[name] = symbol_info
    
    def get_symbol(self, name: str) -> Optional[Any]:
        """Get symbol from symbol table."""
        return self.symbol_table.get(name)


class CodeGenerator:
    """Main code generator for AnamorphX."""
    
    def __init__(self, options: Optional[GenerationOptions] = None):
        self.options = options or GenerationOptions()
        self.error_handler = CodeGenerationErrorHandler()
        self.semantic_analyzer = SemanticAnalyzer()
        self.optimizer = CodeOptimizer(self.options.optimization_level)
        
        # Initialize components
        self.target: Optional[TargetPlatform] = None
        self.template_engine: Optional[TemplateEngine] = None
        self._initialize_target()
        self._initialize_templates()
    
    def _initialize_target(self):
        """Initialize target platform."""
        try:
            self.target = target_registry.get_target(
                self.options.target_platform,
                version=self.options.target_version
            )
        except ValueError as e:
            error = TargetNotSupportedError(
                self.options.target_platform,
                target_registry.list_targets()
            )
            self.error_handler.add_error(error)
            raise error
    
    def _initialize_templates(self):
        """Initialize template engine."""
        self.template_engine = template_registry.get_engine(self.options.target_platform)
        if not self.template_engine:
            error = CodeGenerationError(
                error_type=CodeGenerationErrorType.TEMPLATE_NOT_FOUND,
                message=f"No templates found for target '{self.options.target_platform}'",
                target_platform=self.options.target_platform
            )
            self.error_handler.add_error(error)
            raise error
    
    def generate(self, ast: ASTNode) -> GenerationResult:
        """Generate code from AST."""
        start_time = time.time()
        result = GenerationResult(success=False)
        
        try:
            # Phase 1: Semantic Analysis
            semantic_start = time.time()
            semantic_result = self._run_semantic_analysis(ast)
            result.semantic_result = semantic_result
            result.metrics.semantic_analysis_time = time.time() - semantic_start
            
            if semantic_result.has_errors():
                result.errors.extend([str(error) for error in semantic_result.errors])
                return result
            
            # Phase 2: Optimization
            optimization_start = time.time()
            if self.options.enable_optimizations:
                optimization_result = self._run_optimization(semantic_result.analyzed_ast)
                result.optimization_result = optimization_result
                ast = optimization_result.optimized_ast
                result.metrics.optimization_passes = len(optimization_result.metrics)
                result.metrics.nodes_optimized = optimization_result.get_total_nodes_removed()
                result.metrics.performance_gain = optimization_result.get_total_performance_gain()
            result.metrics.optimization_time = time.time() - optimization_start
            
            # Phase 3: Code Generation
            generation_context = self._create_generation_context(ast)
            
            # Generate code
            template_start = time.time()
            generated_code = self._generate_code(generation_context)
            result.metrics.template_rendering_time = time.time() - template_start
            
            # Phase 4: Code Formatting
            format_start = time.time()
            if self.options.format_code:
                generated_code = self._format_code(generated_code)
            result.metrics.code_formatting_time = time.time() - format_start
            
            # Phase 5: Output Writing
            write_start = time.time()
            output_files = self._write_output(generated_code)
            result.output_files = output_files
            result.metrics.file_writing_time = time.time() - write_start
            
            # Update result
            result.success = True
            result.generated_code = generated_code
            result.metrics.total_time = time.time() - start_time
            result.metrics.input_ast_nodes = self._count_ast_nodes(ast)
            result.metrics.output_code_lines = len(generated_code.split('\n'))
            result.metrics.output_code_size = len(generated_code.encode('utf-8'))
            
            # Collect warnings
            if self.error_handler.get_warning_count() > 0:
                result.warnings = [str(warning) for warning in self.error_handler.warnings]
            
        except Exception as e:
            result.errors.append(str(e))
            result.metrics.total_time = time.time() - start_time
        
        return result
    
    async def generate_async(self, ast: ASTNode) -> GenerationResult:
        """Generate code asynchronously."""
        if not self.options.parallel_generation:
            return self.generate(ast)
        
        # Run in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, ast)
    
    def _run_semantic_analysis(self, ast: ASTNode) -> AnalysisResult:
        """Run semantic analysis on AST."""
        return self.semantic_analyzer.analyze(ast)
    
    def _run_optimization(self, ast: ASTNode) -> OptimizationResult:
        """Run optimization passes on AST."""
        return self.optimizer.optimize(ast)
    
    def _create_generation_context(self, ast: ASTNode) -> GenerationContext:
        """Create generation context."""
        return GenerationContext(
            ast=ast,
            target=self.target,
            template_engine=self.template_engine,
            options=self.options,
            error_handler=self.error_handler
        )
    
    def _generate_code(self, context: GenerationContext) -> str:
        """Generate code using templates."""
        # Create template context
        template_context = TemplateContext(
            target_platform=self.options.target_platform,
            optimization_level=self.options.optimization_level.value
        )
        
        # Add target-specific imports
        neural_imports = context.target.get_neural_runtime_imports()
        for import_stmt in neural_imports:
            template_context.add_import(import_stmt)
        
        # Generate code sections
        code_sections = []
        
        # Module header
        header_code = self._generate_module_header(context, template_context)
        if header_code:
            code_sections.append(header_code)
        
        # Imports
        imports_code = self._generate_imports(context, template_context)
        if imports_code:
            code_sections.append(imports_code)
        
        # Main code
        main_code = self._generate_main_code(context, template_context)
        if main_code:
            code_sections.append(main_code)
        
        return '\n\n'.join(code_sections)
    
    def _generate_module_header(self, context: GenerationContext, template_context: TemplateContext) -> str:
        """Generate module header."""
        template_context.set_variable('target_version', self.options.target_version)
        template_context.set_variable('generation_time', time.strftime('%Y-%m-%d %H:%M:%S'))
        
        try:
            return context.template_engine.render_template('module_header', template_context)
        except Exception:
            return f'# Generated AnamorphX code for {self.options.target_platform}\n'
    
    def _generate_imports(self, context: GenerationContext, template_context: TemplateContext) -> str:
        """Generate import statements."""
        return '\n'.join(template_context.imports)
    
    def _generate_main_code(self, context: GenerationContext, template_context: TemplateContext) -> str:
        """Generate main code from AST."""
        from .emitters import PythonEmitter, JavaScriptEmitter, CppEmitter
        
        # Select appropriate emitter
        if self.options.target_platform == 'python':
            emitter = PythonEmitter(context)
        elif self.options.target_platform in ['javascript', 'js']:
            emitter = JavaScriptEmitter(context)
        elif self.options.target_platform in ['cpp', 'c++']:
            emitter = CppEmitter(context)
        else:
            # Fallback to Python emitter
            emitter = PythonEmitter(context)
        
        return emitter.emit(context.ast)
    
    def _format_code(self, code: str) -> str:
        """Format generated code."""
        if self.target:
            return self.target.format_code(code)
        return code
    
    def _write_output(self, code: str) -> List[str]:
        """Write generated code to files."""
        output_files = []
        
        if self.options.output_directory or self.options.output_filename:
            # Write to specified location
            if self.options.output_filename:
                output_path = Path(self.options.output_filename)
            else:
                filename = f"generated{self.target.get_file_extension()}"
                output_path = Path(self.options.output_directory) / filename
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            output_files.append(str(output_path))
        
        return output_files
    
    def _count_ast_nodes(self, ast: ASTNode) -> int:
        """Count nodes in AST."""
        if not ast:
            return 0
        
        count = 1
        for attr_name in dir(ast):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(ast, attr_name)
            
            if isinstance(attr_value, ASTNode):
                count += self._count_ast_nodes(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, ASTNode):
                        count += self._count_ast_nodes(item)
        
        return count
    
    def set_options(self, options: GenerationOptions):
        """Update generation options."""
        self.options = options
        self.optimizer.set_optimization_level(options.optimization_level)
        self._initialize_target()
        self._initialize_templates()
    
    def get_supported_targets(self) -> List[str]:
        """Get list of supported target platforms."""
        return target_registry.list_targets()
    
    def validate_options(self) -> List[str]:
        """Validate generation options."""
        errors = []
        
        # Check target platform
        if not target_registry.is_supported(self.options.target_platform):
            errors.append(f"Unsupported target platform: {self.options.target_platform}")
        
        # Check output settings
        if self.options.output_directory and not Path(self.options.output_directory).exists():
            try:
                Path(self.options.output_directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {e}")
        
        # Check memory limit
        if self.options.memory_limit < 1024 * 1024:  # 1MB minimum
            errors.append("Memory limit too low (minimum 1MB)")
        
        return errors


# Convenience functions
def generate_code(
    ast: ASTNode,
    target: str = "python",
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
    **kwargs
) -> GenerationResult:
    """Generate code with default settings."""
    options = GenerationOptions(
        target_platform=target,
        optimization_level=optimization_level,
        **kwargs
    )
    
    generator = CodeGenerator(options)
    return generator.generate(ast)


async def generate_code_async(
    ast: ASTNode,
    target: str = "python",
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
    **kwargs
) -> GenerationResult:
    """Generate code asynchronously."""
    options = GenerationOptions(
        target_platform=target,
        optimization_level=optimization_level,
        parallel_generation=True,
        **kwargs
    )
    
    generator = CodeGenerator(options)
    return await generator.generate_async(ast) 