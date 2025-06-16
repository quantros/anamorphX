"""
Semantic Analyzer Demonstration for AnamorphX

This script demonstrates the capabilities of the semantic analyzer
for the Anamorph neural programming language.
"""

import sys
import os
import time
import asyncio
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic import (
    SemanticAnalyzer, AnalysisContext, AnalysisResult,
    TypeSystem, SymbolResolver, ScopeManager,
    SemanticError, TypeError, ScopeError, NeuralError
)
from semantic.symbols import VariableSymbol, FunctionSymbol, NeuronSymbol
from semantic.types import PrimitiveType, NeuralType, SignalType
from ast.nodes import (
    Program, VariableDeclaration, FunctionDeclaration, NeuronDeclaration,
    SynapseDeclaration, PulseStatement, BlockStatement,
    IntegerLiteral, FloatLiteral, StringLiteral, BinaryExpression,
    BinaryOperator, Identifier, SourceLocation
)
from parser import AnamorphParser


def create_sample_program() -> Program:
    """Create a sample Anamorph program for analysis."""
    
    # Variable declarations
    var1 = VariableDeclaration(
        name='learning_rate',
        type_annotation='float',
        initializer=FloatLiteral(value=0.01),
        location=SourceLocation(line=1, column=1, source="var learning_rate: float = 0.01;")
    )
    
    var2 = VariableDeclaration(
        name='max_iterations',
        type_annotation='int',
        initializer=IntegerLiteral(value=1000),
        location=SourceLocation(line=2, column=1, source="var max_iterations: int = 1000;")
    )
    
    # Function declaration
    func_decl = FunctionDeclaration(
        name='train_network',
        parameters=[],
        return_type='bool',
        body=BlockStatement(statements=[]),
        location=SourceLocation(line=4, column=1, source="function train_network(): bool { }")
    )
    
    # Neuron declarations
    input_neuron = NeuronDeclaration(
        name='input_layer',
        neuron_type='basic',
        activation_function='linear',
        threshold=0.0,
        body=None,
        location=SourceLocation(line=6, column=1, source="neuron input_layer: basic { }")
    )
    
    hidden_neuron = NeuronDeclaration(
        name='hidden_layer',
        neuron_type='perceptron',
        activation_function='sigmoid',
        threshold=0.5,
        body=None,
        location=SourceLocation(line=8, column=1, source="neuron hidden_layer: perceptron { }")
    )
    
    output_neuron = NeuronDeclaration(
        name='output_layer',
        neuron_type='basic',
        activation_function='softmax',
        threshold=0.7,
        body=None,
        location=SourceLocation(line=10, column=1, source="neuron output_layer: basic { }")
    )
    
    # Synapse connections
    synapse1 = SynapseDeclaration(
        source='input_layer',
        target='hidden_layer',
        weight=0.8,
        delay=0.1,
        location=SourceLocation(line=12, column=1, source="synapse input_layer -> hidden_layer { weight: 0.8 }")
    )
    
    synapse2 = SynapseDeclaration(
        source='hidden_layer',
        target='output_layer',
        weight=1.2,
        delay=0.05,
        location=SourceLocation(line=14, column=1, source="synapse hidden_layer -> output_layer { weight: 1.2 }")
    )
    
    return Program(statements=[
        var1, var2, func_decl, input_neuron, hidden_neuron, output_neuron, synapse1, synapse2
    ])


def create_error_program() -> Program:
    """Create a program with intentional errors for testing."""
    
    # Type error: assigning string to int
    var_error = VariableDeclaration(
        name='number',
        type_annotation='int',
        initializer=StringLiteral(value="not a number"),
        location=SourceLocation(line=1, column=1, source='var number: int = "not a number";')
    )
    
    # Undefined symbol error
    undefined_ref = VariableDeclaration(
        name='result',
        type_annotation='int',
        initializer=Identifier(name='undefined_variable'),
        location=SourceLocation(line=3, column=1, source="var result: int = undefined_variable;")
    )
    
    # Invalid neuron type
    invalid_neuron = NeuronDeclaration(
        name='bad_neuron',
        neuron_type='invalid_type',
        activation_function='unknown_function',
        threshold=-1.0,
        body=None,
        location=SourceLocation(line=5, column=1, source="neuron bad_neuron: invalid_type { }")
    )
    
    # Synapse with undefined neurons
    invalid_synapse = SynapseDeclaration(
        source='nonexistent_neuron1',
        target='nonexistent_neuron2',
        weight=1.0,
        location=SourceLocation(line=7, column=1, source="synapse nonexistent_neuron1 -> nonexistent_neuron2;")
    )
    
    return Program(statements=[var_error, undefined_ref, invalid_neuron, invalid_synapse])


def create_complex_neural_network() -> Program:
    """Create a complex neural network for advanced analysis."""
    
    statements = []
    
    # Create multiple layers
    layer_names = ['input', 'hidden1', 'hidden2', 'output']
    neuron_types = ['basic', 'perceptron', 'lstm', 'basic']
    activations = ['linear', 'relu', 'tanh', 'softmax']
    
    # Create neurons
    for i, (name, ntype, activation) in enumerate(zip(layer_names, neuron_types, activations)):
        neuron = NeuronDeclaration(
            name=f'{name}_layer',
            neuron_type=ntype,
            activation_function=activation,
            threshold=0.5,
            body=None,
            location=SourceLocation(line=i+1, column=1)
        )
        statements.append(neuron)
    
    # Create connections between layers
    for i in range(len(layer_names) - 1):
        source = f'{layer_names[i]}_layer'
        target = f'{layer_names[i+1]}_layer'
        weight = 1.0 - (i * 0.1)  # Decreasing weights
        
        synapse = SynapseDeclaration(
            source=source,
            target=target,
            weight=weight,
            delay=0.01 * (i + 1),
            location=SourceLocation(line=len(layer_names) + i + 1, column=1)
        )
        statements.append(synapse)
    
    # Add some recurrent connections
    recurrent_synapse = SynapseDeclaration(
        source='hidden2_layer',
        target='hidden1_layer',
        weight=0.3,
        delay=0.1,
        location=SourceLocation(line=10, column=1)
    )
    statements.append(recurrent_synapse)
    
    return Program(statements=statements)


def demonstrate_basic_analysis():
    """Demonstrate basic semantic analysis."""
    print("=" * 60)
    print("BASIC SEMANTIC ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SemanticAnalyzer()
    
    # Create sample program
    program = create_sample_program()
    
    # Create analysis context
    context = AnalysisContext(
        source_file="demo.amorph",
        module_name="demo",
        options={'verbose': True}
    )
    
    print("Analyzing sample Anamorph program...")
    print()
    
    # Perform analysis
    start_time = time.time()
    result = analyzer.analyze(program, context)
    analysis_time = time.time() - start_time
    
    # Display results
    print(f"Analysis completed in {analysis_time:.3f} seconds")
    print(f"Success: {result.success}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print()
    
    # Display metrics
    print("Analysis Metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print()
    
    # Display symbol table
    print("Symbol Table:")
    if result.symbol_table:
        symbols = result.symbol_table.get_all_symbols()
        for name, symbol in symbols.items():
            print(f"  {symbol.symbol_type.name}: {name} ({symbol.type_info or 'no type'})")
    print()
    
    # Display neural analysis
    if result.neural_analysis:
        print("Neural Network Analysis:")
        for key, value in result.neural_analysis.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {len(value) if isinstance(value, (list, dict)) else value}")
            else:
                print(f"  {key}: {value}")
    print()


def demonstrate_error_detection():
    """Demonstrate error detection capabilities."""
    print("=" * 60)
    print("ERROR DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SemanticAnalyzer()
    
    # Create program with errors
    program = create_error_program()
    
    # Create analysis context
    context = AnalysisContext(
        source_file="error_demo.amorph",
        module_name="error_demo"
    )
    
    print("Analyzing program with intentional errors...")
    print()
    
    # Perform analysis
    result = analyzer.analyze(program, context)
    
    # Display results
    print(f"Analysis Success: {result.success}")
    print(f"Total Errors: {len(result.errors)}")
    print(f"Total Warnings: {len(result.warnings)}")
    print()
    
    # Display errors by type
    error_types = {}
    for error in result.errors:
        error_type = type(error).__name__
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    print("Errors by Type:")
    for error_type, errors in error_types.items():
        print(f"  {error_type}: {len(errors)}")
        for error in errors[:3]:  # Show first 3 errors of each type
            print(f"    - {error.message}")
            if error.location:
                print(f"      at line {error.location.line}, column {error.location.column}")
    print()


def demonstrate_neural_analysis():
    """Demonstrate neural network analysis."""
    print("=" * 60)
    print("NEURAL NETWORK ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SemanticAnalyzer()
    
    # Create complex neural network
    program = create_complex_neural_network()
    
    # Create analysis context
    context = AnalysisContext(
        source_file="neural_demo.amorph",
        module_name="neural_demo"
    )
    
    print("Analyzing complex neural network...")
    print()
    
    # Perform analysis
    result = analyzer.analyze(program, context)
    
    # Display results
    print(f"Analysis Success: {result.success}")
    print(f"Neural Neurons: {result.metrics.get('neural_neurons', 0)}")
    print(f"Neural Synapses: {result.metrics.get('neural_synapses', 0)}")
    print()
    
    # Display neural analysis details
    if result.neural_analysis:
        print("Neural Network Structure:")
        
        if 'flow_paths' in result.neural_analysis:
            paths = result.neural_analysis['flow_paths']
            print(f"  Signal Flow Paths: {len(paths)}")
            for i, path in enumerate(paths[:3]):  # Show first 3 paths
                print(f"    Path {i+1}: {' -> '.join(path)}")
        
        if 'bottlenecks' in result.neural_analysis:
            bottlenecks = result.neural_analysis['bottlenecks']
            print(f"  Bottleneck Neurons: {bottlenecks}")
        
        if 'dead_ends' in result.neural_analysis:
            dead_ends = result.neural_analysis['dead_ends']
            print(f"  Dead End Neurons: {dead_ends}")
        
        if 'connectivity_metrics' in result.neural_analysis:
            metrics = result.neural_analysis['connectivity_metrics']
            print(f"  Network Density: {metrics.get('density', 0):.3f}")
            print(f"  Average Degree: {metrics.get('average_degree', 0):.3f}")
        
        print(f"  Network Depth: {result.neural_analysis.get('network_depth', 0)}")
    print()


def demonstrate_type_system():
    """Demonstrate type system capabilities."""
    print("=" * 60)
    print("TYPE SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create type system
    type_system = TypeSystem()
    
    print("Built-in Types:")
    builtin_types = ['int', 'float', 'string', 'bool', 'neuron']
    for type_name in builtin_types:
        type_obj = type_system.get_type(type_name)
        if type_obj:
            print(f"  {type_name}: {type(type_obj).__name__}")
    print()
    
    print("Complex Types:")
    complex_types = ['int[]', 'Signal<float>', 'Pulse<int>']
    for type_name in complex_types:
        type_obj = type_system.get_type(type_name)
        if type_obj:
            print(f"  {type_name}: {type(type_obj).__name__}")
    print()
    
    print("Type Compatibility:")
    int_type = type_system.get_type('int')
    float_type = type_system.get_type('float')
    string_type = type_system.get_type('string')
    
    compatibility_tests = [
        (int_type, float_type, "int -> float"),
        (float_type, int_type, "float -> int"),
        (int_type, string_type, "int -> string"),
        (string_type, int_type, "string -> int")
    ]
    
    for type1, type2, desc in compatibility_tests:
        if type1 and type2:
            compatible = type_system.are_compatible(type1, type2)
            castable = type_system.can_cast(type1, type2)
            print(f"  {desc}: compatible={compatible}, castable={castable}")
    print()


async def demonstrate_async_analysis():
    """Demonstrate asynchronous analysis."""
    print("=" * 60)
    print("ASYNCHRONOUS ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SemanticAnalyzer()
    
    # Create multiple programs for concurrent analysis
    programs = [
        create_sample_program(),
        create_complex_neural_network(),
        create_error_program()
    ]
    
    contexts = [
        AnalysisContext(source_file=f"async_demo_{i}.amorph", module_name=f"async_demo_{i}")
        for i in range(len(programs))
    ]
    
    print("Running concurrent analysis on multiple programs...")
    print()
    
    # Run analyses concurrently
    start_time = time.time()
    tasks = [
        analyzer.analyze_async(program, context)
        for program, context in zip(programs, contexts)
    ]
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"Concurrent analysis completed in {total_time:.3f} seconds")
    print()
    
    # Display results
    for i, result in enumerate(results):
        print(f"Program {i+1}:")
        print(f"  Success: {result.success}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Analysis Time: {result.analysis_time:.3f}s")
        print(f"  Symbols: {result.metrics.get('symbol_count', 0)}")
    print()


def demonstrate_performance():
    """Demonstrate performance characteristics."""
    print("=" * 60)
    print("PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SemanticAnalyzer()
    
    # Test with programs of different sizes
    sizes = [10, 50, 100, 500]
    
    print("Performance testing with different program sizes:")
    print()
    
    for size in sizes:
        # Create program with many variable declarations
        statements = []
        for i in range(size):
            var_decl = VariableDeclaration(
                name=f'var_{i}',
                type_annotation='int',
                initializer=IntegerLiteral(value=i),
                location=SourceLocation(line=i+1, column=1)
            )
            statements.append(var_decl)
        
        program = Program(statements=statements)
        context = AnalysisContext(
            source_file=f"perf_test_{size}.amorph",
            module_name=f"perf_test_{size}"
        )
        
        # Measure analysis time
        start_time = time.time()
        result = analyzer.analyze(program, context)
        analysis_time = time.time() - start_time
        
        print(f"  {size:3d} statements: {analysis_time:.3f}s "
              f"({size/analysis_time:.0f} stmt/s) - "
              f"{'SUCCESS' if result.success else 'FAILED'}")
    
    print()


def main():
    """Main demonstration function."""
    print("AnamorphX Semantic Analyzer Demonstration")
    print("=========================================")
    print()
    
    try:
        # Run demonstrations
        demonstrate_basic_analysis()
        demonstrate_error_detection()
        demonstrate_neural_analysis()
        demonstrate_type_system()
        
        # Run async demonstration
        print("Running asynchronous analysis demonstration...")
        asyncio.run(demonstrate_async_analysis())
        
        demonstrate_performance()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 