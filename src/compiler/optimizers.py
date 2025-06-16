"""
Code optimization system for AnamorphX.

This module provides various optimization passes for improving
generated code performance and neural network efficiency.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Callable
import time
from ..syntax.nodes import ASTNode, Program, FunctionDeclaration, NeuronDeclaration


class OptimizationLevel(Enum):
    """Optimization levels."""
    
    NONE = 0      # No optimizations
    BASIC = 1     # Basic optimizations (safe)
    STANDARD = 2  # Standard optimizations
    AGGRESSIVE = 3  # Aggressive optimizations (may change semantics)


class OptimizationCategory(Enum):
    """Categories of optimizations."""
    
    DEAD_CODE = auto()
    CONSTANT_FOLDING = auto()
    LOOP_OPTIMIZATION = auto()
    FUNCTION_INLINING = auto()
    NEURAL_OPTIMIZATION = auto()
    MEMORY_OPTIMIZATION = auto()
    VECTORIZATION = auto()
    PARALLELIZATION = auto()


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    
    pass_name: str
    execution_time: float = 0.0
    nodes_processed: int = 0
    nodes_removed: int = 0
    nodes_modified: int = 0
    memory_saved: int = 0
    performance_gain: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'pass_name': self.pass_name,
            'execution_time': self.execution_time,
            'nodes_processed': self.nodes_processed,
            'nodes_removed': self.nodes_removed,
            'nodes_modified': self.nodes_modified,
            'memory_saved': self.memory_saved,
            'performance_gain': self.performance_gain
        }


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    
    optimized_ast: ASTNode
    metrics: List[OptimizationMetrics] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_total_execution_time(self) -> float:
        """Get total optimization execution time."""
        return sum(metric.execution_time for metric in self.metrics)
    
    def get_total_nodes_removed(self) -> int:
        """Get total number of nodes removed."""
        return sum(metric.nodes_removed for metric in self.metrics)
    
    def get_total_performance_gain(self) -> float:
        """Get estimated total performance gain."""
        return sum(metric.performance_gain for metric in self.metrics)


class OptimizationPass(ABC):
    """Abstract base class for optimization passes."""
    
    def __init__(self, name: str, category: OptimizationCategory, level: OptimizationLevel):
        self.name = name
        self.category = category
        self.level = level
        self.enabled = True
        self.metrics = OptimizationMetrics(name)
    
    @abstractmethod
    def optimize(self, ast: ASTNode) -> ASTNode:
        """Apply optimization to AST."""
        pass
    
    def can_run(self, optimization_level: OptimizationLevel) -> bool:
        """Check if this pass can run at given optimization level."""
        return self.enabled and self.level.value <= optimization_level.value
    
    def reset_metrics(self):
        """Reset optimization metrics."""
        self.metrics = OptimizationMetrics(self.name)


class DeadCodeElimination(OptimizationPass):
    """Remove unreachable and unused code."""
    
    def __init__(self):
        super().__init__("Dead Code Elimination", OptimizationCategory.DEAD_CODE, OptimizationLevel.BASIC)
        self.used_symbols: Set[str] = set()
        self.reachable_nodes: Set[ASTNode] = set()
    
    def optimize(self, ast: ASTNode) -> ASTNode:
        """Remove dead code from AST."""
        start_time = time.time()
        
        # Phase 1: Mark reachable code
        self._mark_reachable(ast)
        
        # Phase 2: Remove unreachable code
        optimized_ast = self._remove_unreachable(ast)
        
        # Update metrics
        self.metrics.execution_time = time.time() - start_time
        self.metrics.nodes_processed = self._count_nodes(ast)
        self.metrics.nodes_removed = self.metrics.nodes_processed - self._count_nodes(optimized_ast)
        
        return optimized_ast
    
    def _mark_reachable(self, node: ASTNode):
        """Mark reachable nodes starting from entry points."""
        if node in self.reachable_nodes:
            return
        
        self.reachable_nodes.add(node)
        
        # Mark children as reachable
        for child in self._get_children(node):
            self._mark_reachable(child)
    
    def _remove_unreachable(self, node: ASTNode) -> ASTNode:
        """Remove unreachable nodes from AST."""
        if node not in self.reachable_nodes:
            return None
        
        # Process children
        if hasattr(node, 'body') and isinstance(node.body, list):
            node.body = [child for child in node.body if child in self.reachable_nodes]
        
        return node
    
    def _get_children(self, node: ASTNode) -> List[ASTNode]:
        """Get child nodes."""
        children = []
        
        if hasattr(node, 'body') and isinstance(node.body, list):
            children.extend(node.body)
        elif hasattr(node, 'body') and node.body:
            children.append(node.body)
        
        if hasattr(node, 'statements') and isinstance(node.statements, list):
            children.extend(node.statements)
        
        return children
    
    def _count_nodes(self, node: ASTNode) -> int:
        """Count total nodes in AST."""
        if not node:
            return 0
        
        count = 1
        for child in self._get_children(node):
            count += self._count_nodes(child)
        
        return count


class ConstantFolding(OptimizationPass):
    """Fold constant expressions at compile time."""
    
    def __init__(self):
        super().__init__("Constant Folding", OptimizationCategory.CONSTANT_FOLDING, OptimizationLevel.BASIC)
    
    def optimize(self, ast: ASTNode) -> ASTNode:
        """Fold constant expressions."""
        start_time = time.time()
        
        optimized_ast = self._fold_constants(ast)
        
        # Update metrics
        self.metrics.execution_time = time.time() - start_time
        self.metrics.nodes_processed = self._count_nodes(ast)
        
        return optimized_ast
    
    def _fold_constants(self, node: ASTNode) -> ASTNode:
        """Recursively fold constant expressions."""
        if not node:
            return node
        
        # Handle binary expressions
        if hasattr(node, 'operator') and hasattr(node, 'left') and hasattr(node, 'right'):
            left = self._fold_constants(node.left)
            right = self._fold_constants(node.right)
            
            # Try to evaluate constant expression
            if self._is_constant(left) and self._is_constant(right):
                result = self._evaluate_binary_expression(node.operator, left, right)
                if result is not None:
                    self.metrics.nodes_modified += 1
                    return result
            
            node.left = left
            node.right = right
        
        # Handle unary expressions
        elif hasattr(node, 'operator') and hasattr(node, 'operand'):
            operand = self._fold_constants(node.operand)
            
            if self._is_constant(operand):
                result = self._evaluate_unary_expression(node.operator, operand)
                if result is not None:
                    self.metrics.nodes_modified += 1
                    return result
            
            node.operand = operand
        
        # Process other node types recursively
        else:
            self._fold_node_children(node)
        
        return node
    
    def _is_constant(self, node: ASTNode) -> bool:
        """Check if node represents a constant value."""
        return hasattr(node, 'value') and hasattr(node, '_node_type')
    
    def _evaluate_binary_expression(self, operator, left: ASTNode, right: ASTNode) -> Optional[ASTNode]:
        """Evaluate binary expression with constant operands."""
        try:
            left_val = left.value
            right_val = right.value
            
            # Arithmetic operations
            if operator.value == '+':
                result = left_val + right_val
            elif operator.value == '-':
                result = left_val - right_val
            elif operator.value == '*':
                result = left_val * right_val
            elif operator.value == '/':
                if right_val == 0:
                    return None  # Don't fold division by zero
                result = left_val / right_val
            elif operator.value == '%':
                if right_val == 0:
                    return None
                result = left_val % right_val
            elif operator.value == '**':
                result = left_val ** right_val
            
            # Comparison operations
            elif operator.value == '==':
                result = left_val == right_val
            elif operator.value == '!=':
                result = left_val != right_val
            elif operator.value == '<':
                result = left_val < right_val
            elif operator.value == '<=':
                result = left_val <= right_val
            elif operator.value == '>':
                result = left_val > right_val
            elif operator.value == '>=':
                result = left_val >= right_val
            
            # Logical operations
            elif operator.value == '&&':
                result = left_val and right_val
            elif operator.value == '||':
                result = left_val or right_val
            
            else:
                return None
            
            # Create appropriate literal node
            return self._create_literal_node(result)
            
        except Exception:
            return None
    
    def _evaluate_unary_expression(self, operator, operand: ASTNode) -> Optional[ASTNode]:
        """Evaluate unary expression with constant operand."""
        try:
            operand_val = operand.value
            
            if operator.value == '-':
                result = -operand_val
            elif operator.value == '+':
                result = +operand_val
            elif operator.value == '!':
                result = not operand_val
            else:
                return None
            
            return self._create_literal_node(result)
            
        except Exception:
            return None
    
    def _create_literal_node(self, value: Any) -> ASTNode:
        """Create appropriate literal node for value."""
        # This would create the appropriate AST node type
        # For now, return a placeholder
        from ..syntax.nodes import IntegerLiteral, FloatLiteral, BooleanLiteral, StringLiteral
        
        if isinstance(value, int):
            return IntegerLiteral(value=value)
        elif isinstance(value, float):
            return FloatLiteral(value=value)
        elif isinstance(value, bool):
            return BooleanLiteral(value=value)
        elif isinstance(value, str):
            return StringLiteral(value=value)
        else:
            return None
    
    def _fold_node_children(self, node: ASTNode):
        """Recursively fold children of a node."""
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(node, attr_name)
            
            if isinstance(attr_value, ASTNode):
                setattr(node, attr_name, self._fold_constants(attr_value))
            elif isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    if isinstance(item, ASTNode):
                        attr_value[i] = self._fold_constants(item)
    
    def _count_nodes(self, node: ASTNode) -> int:
        """Count nodes in AST."""
        if not node:
            return 0
        
        count = 1
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(node, attr_name)
            
            if isinstance(attr_value, ASTNode):
                count += self._count_nodes(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, ASTNode):
                        count += self._count_nodes(item)
        
        return count


class FunctionInlining(OptimizationPass):
    """Inline small functions to reduce call overhead."""
    
    def __init__(self):
        super().__init__("Function Inlining", OptimizationCategory.FUNCTION_INLINING, OptimizationLevel.STANDARD)
        self.inline_threshold = 10  # Maximum statements to inline
        self.inlined_functions: Set[str] = set()
    
    def optimize(self, ast: ASTNode) -> ASTNode:
        """Inline eligible functions."""
        start_time = time.time()
        
        # Find functions eligible for inlining
        eligible_functions = self._find_eligible_functions(ast)
        
        # Inline function calls
        optimized_ast = self._inline_functions(ast, eligible_functions)
        
        # Update metrics
        self.metrics.execution_time = time.time() - start_time
        self.metrics.nodes_processed = len(eligible_functions)
        self.metrics.nodes_modified = len(self.inlined_functions)
        
        return optimized_ast
    
    def _find_eligible_functions(self, ast: ASTNode) -> Dict[str, FunctionDeclaration]:
        """Find functions that can be inlined."""
        eligible = {}
        
        if isinstance(ast, Program):
            for stmt in ast.body:
                if isinstance(stmt, FunctionDeclaration):
                    if self._is_inlinable(stmt):
                        eligible[stmt.name.name] = stmt
        
        return eligible
    
    def _is_inlinable(self, func: FunctionDeclaration) -> bool:
        """Check if function is eligible for inlining."""
        # Don't inline recursive functions
        if self._is_recursive(func):
            return False
        
        # Don't inline large functions
        if self._count_statements(func.body) > self.inline_threshold:
            return False
        
        # Don't inline functions with complex control flow
        if self._has_complex_control_flow(func.body):
            return False
        
        return True
    
    def _is_recursive(self, func: FunctionDeclaration) -> bool:
        """Check if function is recursive."""
        # Simple check - look for calls to self
        return self._contains_self_call(func.body, func.name.name)
    
    def _contains_self_call(self, node: ASTNode, func_name: str) -> bool:
        """Check if node contains call to specified function."""
        # This would be implemented to traverse the AST
        # For now, return False
        return False
    
    def _count_statements(self, node: ASTNode) -> int:
        """Count statements in function body."""
        if hasattr(node, 'statements'):
            return len(node.statements)
        return 1
    
    def _has_complex_control_flow(self, node: ASTNode) -> bool:
        """Check if node has complex control flow."""
        # This would check for loops, try/catch, etc.
        return False
    
    def _inline_functions(self, ast: ASTNode, eligible_functions: Dict[str, FunctionDeclaration]) -> ASTNode:
        """Inline function calls in AST."""
        # This would implement the actual inlining logic
        return ast


class NeuralOptimizer(OptimizationPass):
    """Neural network specific optimizations."""
    
    def __init__(self):
        super().__init__("Neural Optimization", OptimizationCategory.NEURAL_OPTIMIZATION, OptimizationLevel.STANDARD)
        self.optimized_neurons: Set[str] = set()
        self.merged_synapses: int = 0
    
    def optimize(self, ast: ASTNode) -> ASTNode:
        """Apply neural-specific optimizations."""
        start_time = time.time()
        
        # Optimize neuron connections
        optimized_ast = self._optimize_neural_connections(ast)
        
        # Merge redundant synapses
        optimized_ast = self._merge_synapses(optimized_ast)
        
        # Optimize signal processing
        optimized_ast = self._optimize_signal_processing(optimized_ast)
        
        # Update metrics
        self.metrics.execution_time = time.time() - start_time
        self.metrics.nodes_modified = len(self.optimized_neurons)
        self.metrics.performance_gain = self.merged_synapses * 0.1  # Estimated gain
        
        return optimized_ast
    
    def _optimize_neural_connections(self, ast: ASTNode) -> ASTNode:
        """Optimize neural network connections."""
        if isinstance(ast, Program):
            for stmt in ast.body:
                if isinstance(stmt, NeuronDeclaration):
                    self._optimize_neuron(stmt)
        
        return ast
    
    def _optimize_neuron(self, neuron: NeuronDeclaration):
        """Optimize individual neuron."""
        self.optimized_neurons.add(neuron.name.name)
        
        # Optimize neuron parameters
        self._optimize_neuron_parameters(neuron)
        
        # Optimize activation functions
        self._optimize_activation_functions(neuron)
    
    def _optimize_neuron_parameters(self, neuron: NeuronDeclaration):
        """Optimize neuron parameters."""
        # This would implement parameter optimization
        pass
    
    def _optimize_activation_functions(self, neuron: NeuronDeclaration):
        """Optimize activation functions."""
        # This would implement activation function optimization
        pass
    
    def _merge_synapses(self, ast: ASTNode) -> ASTNode:
        """Merge redundant synapses."""
        # This would implement synapse merging logic
        self.merged_synapses = 5  # Placeholder
        return ast
    
    def _optimize_signal_processing(self, ast: ASTNode) -> ASTNode:
        """Optimize signal processing logic."""
        # This would implement signal processing optimization
        return ast


class LoopOptimization(OptimizationPass):
    """Loop-specific optimizations."""
    
    def __init__(self):
        super().__init__("Loop Optimization", OptimizationCategory.LOOP_OPTIMIZATION, OptimizationLevel.STANDARD)
        self.unrolled_loops: int = 0
        self.vectorized_loops: int = 0
    
    def optimize(self, ast: ASTNode) -> ASTNode:
        """Apply loop optimizations."""
        start_time = time.time()
        
        # Loop unrolling
        optimized_ast = self._unroll_loops(ast)
        
        # Loop vectorization
        optimized_ast = self._vectorize_loops(optimized_ast)
        
        # Update metrics
        self.metrics.execution_time = time.time() - start_time
        self.metrics.nodes_modified = self.unrolled_loops + self.vectorized_loops
        self.metrics.performance_gain = self.unrolled_loops * 0.2 + self.vectorized_loops * 0.5
        
        return optimized_ast
    
    def _unroll_loops(self, ast: ASTNode) -> ASTNode:
        """Unroll small loops."""
        # This would implement loop unrolling
        self.unrolled_loops = 2  # Placeholder
        return ast
    
    def _vectorize_loops(self, ast: ASTNode) -> ASTNode:
        """Vectorize loops where possible."""
        # This would implement loop vectorization
        self.vectorized_loops = 1  # Placeholder
        return ast


class CodeOptimizer:
    """Main code optimizer that manages optimization passes."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        self.passes: List[OptimizationPass] = []
        self.enabled_categories: Set[OptimizationCategory] = set()
        self._register_default_passes()
    
    def _register_default_passes(self):
        """Register default optimization passes."""
        self.passes = [
            DeadCodeElimination(),
            ConstantFolding(),
            FunctionInlining(),
            LoopOptimization(),
            NeuralOptimizer(),
        ]
        
        # Enable all categories by default
        self.enabled_categories = set(OptimizationCategory)
    
    def add_pass(self, optimization_pass: OptimizationPass):
        """Add an optimization pass."""
        self.passes.append(optimization_pass)
    
    def remove_pass(self, pass_name: str):
        """Remove an optimization pass by name."""
        self.passes = [p for p in self.passes if p.name != pass_name]
    
    def enable_category(self, category: OptimizationCategory):
        """Enable optimization category."""
        self.enabled_categories.add(category)
    
    def disable_category(self, category: OptimizationCategory):
        """Disable optimization category."""
        self.enabled_categories.discard(category)
    
    def optimize(self, ast: ASTNode) -> OptimizationResult:
        """Run all optimization passes on AST."""
        current_ast = ast
        metrics = []
        warnings = []
        errors = []
        
        for pass_instance in self.passes:
            # Check if pass should run
            if not pass_instance.can_run(self.optimization_level):
                continue
            
            if pass_instance.category not in self.enabled_categories:
                continue
            
            try:
                # Reset pass metrics
                pass_instance.reset_metrics()
                
                # Run optimization pass
                current_ast = pass_instance.optimize(current_ast)
                
                # Collect metrics
                metrics.append(pass_instance.metrics)
                
            except Exception as e:
                error_msg = f"Error in optimization pass '{pass_instance.name}': {e}"
                errors.append(error_msg)
                
                # Continue with other passes
                continue
        
        return OptimizationResult(
            optimized_ast=current_ast,
            metrics=metrics,
            warnings=warnings,
            errors=errors
        )
    
    def get_pass_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered passes."""
        return [
            {
                'name': p.name,
                'category': p.category.name,
                'level': p.level.name,
                'enabled': p.enabled
            }
            for p in self.passes
        ]
    
    def set_optimization_level(self, level: OptimizationLevel):
        """Set optimization level."""
        self.optimization_level = level 