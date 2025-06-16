"""
Anamorph Parser Demonstration.

This script demonstrates the capabilities of the Anamorph parser including:
- AST generation for various language constructs
- Error handling and recovery
- Performance metrics
- Complex program parsing
"""

import asyncio
import time
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.parser import AnamorphParser, parse, parse_async
from src.ast.visitor import ASTDumper, ASTValidator
from src.ast.nodes import *


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demonstrate_basic_parsing():
    """Demonstrate basic parsing functionality."""
    print_section("BASIC PARSING DEMONSTRATION")
    
    # Test cases with different language constructs
    test_cases = [
        ("Variable Declaration", "synap x = 42;"),
        ("Binary Expression", "result = (a + b) * c;"),
        ("Function Call", "process(data, threshold);"),
        ("If Statement", "if (x > 0) { return x; }"),
        ("While Loop", "while (i < 10) { i = i + 1; }"),
        ("Boolean Literals", "synap flag = true;"),
        ("String Literals", 'synap message = "Hello, Anamorph!";'),
        ("Array Access", "synap item = array[index];"),
        ("Member Access", "synap value = object.property;"),
    ]
    
    parser = AnamorphParser()
    
    for name, source in test_cases:
        print_subsection(name)
        print(f"Source: {source}")
        
        result = parser.parse(source)
        
        if result.success:
            print(f"✅ Parsed successfully!")
            print(f"   Tokens: {result.token_count}")
            print(f"   Parse time: {result.parse_time:.4f}s")
            print(f"   AST nodes: {len(result.ast.body)}")
        else:
            print(f"❌ Parse failed with {len(result.errors)} errors")
            for error in result.errors[:2]:  # Show first 2 errors
                print(f"   Error: {error}")


def demonstrate_neural_constructs():
    """Demonstrate parsing of neural-specific constructs."""
    print_section("NEURAL CONSTRUCTS DEMONSTRATION")
    
    neural_examples = [
        ("Neuron Declaration", """
        neuro processSignal(input: Signal, threshold: Number) {
            if (input.strength > threshold) {
                pulse input -> amplifier;
                return input;
            }
            return null;
        }
        """),
        
        ("Pulse Statement", """
        pulse signal -> target;
        pulse complexSignal -> neuralNetwork;
        """),
        
        ("Synapse Declaration", """
        synap connection = createSynapse(neuron1, neuron2);
        synap weight = 0.75;
        """),
        
        ("Complex Neural Network", """
        neuro neuralNetwork(inputs: Array) {
            synap layer1 = [];
            synap layer2 = [];
            
            // Process input layer
            for (synap i = 0; i < inputs.length; i++) {
                synap processed = activate(inputs[i]);
                layer1.push(processed);
                pulse processed -> hiddenLayer;
            }
            
            // Process hidden layer
            for (synap j = 0; j < layer1.length; j++) {
                synap weighted = layer1[j] * weights[j];
                layer2.push(weighted);
            }
            
            synap output = aggregate(layer2);
            pulse output -> outputLayer;
            return output;
        }
        """),
    ]
    
    parser = AnamorphParser()
    
    for name, source in neural_examples:
        print_subsection(name)
        print(f"Source code:")
        print(source.strip())
        
        result = parser.parse(source)
        
        if result.success:
            print(f"\n✅ Neural construct parsed successfully!")
            print(f"   Tokens processed: {result.token_count}")
            print(f"   Parse time: {result.parse_time:.4f}s")
            
            # Show AST structure for neural constructs
            if result.ast.body:
                node = result.ast.body[0]
                if isinstance(node, NeuronDeclaration):
                    print(f"   Neuron name: {node.id.name}")
                    print(f"   Parameters: {len(node.params)}")
                    print(f"   Body statements: {len(node.body.statements)}")
        else:
            print(f"\n❌ Failed to parse neural construct")
            for error in result.errors:
                print(f"   Error: {error}")


def demonstrate_ast_generation():
    """Demonstrate AST generation and visualization."""
    print_section("AST GENERATION DEMONSTRATION")
    
    source = """
    neuro fibonacci(n: Number) {
        if (n <= 1) {
            return n;
        }
        synap a = fibonacci(n - 1);
        synap b = fibonacci(n - 2);
        return a + b;
    }
    """
    
    print("Source code:")
    print(source.strip())
    
    parser = AnamorphParser()
    result = parser.parse(source)
    
    if result.success:
        print(f"\n✅ AST generated successfully!")
        
        # Use AST dumper to show structure
        dumper = ASTDumper(indent="  ")
        ast_dump = dumper.dump(result.ast)
        
        print("\nAST Structure:")
        print(ast_dump)
        
        # Validate AST
        validator = ASTValidator()
        errors, warnings = validator.validate(result.ast)
        
        print(f"\nAST Validation:")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
        
        if warnings:
            print("  Warnings:")
            for warning in warnings:
                print(f"    - {warning}")
    else:
        print(f"\n❌ AST generation failed")


def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    print_section("ERROR HANDLING DEMONSTRATION")
    
    error_examples = [
        ("Missing Semicolon", "synap x = 42"),
        ("Unexpected Token", "synap = 42;"),
        ("Mismatched Parentheses", "func(1, 2;"),
        ("Invalid Neuron", "neuro { return 42; }"),
        ("Invalid Expression", "synap x = + * 42;"),
        ("Multiple Errors", """
        synap x = ;
        neuro {
        synap y = + ;
        """),
    ]
    
    # Test with recovery enabled
    parser_with_recovery = AnamorphParser(enable_recovery=True)
    
    # Test without recovery
    parser_without_recovery = AnamorphParser(enable_recovery=False)
    
    for name, source in error_examples:
        print_subsection(f"{name} - Error Handling")
        print(f"Source: {repr(source)}")
        
        # Parse with recovery
        result_with_recovery = parser_with_recovery.parse(source)
        
        # Parse without recovery
        result_without_recovery = parser_without_recovery.parse(source)
        
        print(f"\nWith Recovery:")
        print(f"  Success: {result_with_recovery.success}")
        print(f"  Errors: {len(result_with_recovery.errors)}")
        print(f"  AST nodes: {len(result_with_recovery.ast.body) if result_with_recovery.ast else 0}")
        
        print(f"\nWithout Recovery:")
        print(f"  Success: {result_without_recovery.success}")
        print(f"  Errors: {len(result_without_recovery.errors)}")
        print(f"  AST nodes: {len(result_without_recovery.ast.body) if result_without_recovery.ast else 0}")
        
        # Show first error details
        if result_with_recovery.errors:
            error = result_with_recovery.errors[0]
            print(f"\nFirst Error Details:")
            print(f"  Code: {error.code.value}")
            print(f"  Message: {error.message}")
            if error.suggestions:
                print(f"  Suggestions:")
                for suggestion in error.suggestions[:2]:
                    print(f"    - {suggestion}")


def demonstrate_performance():
    """Demonstrate parser performance with various code sizes."""
    print_section("PERFORMANCE DEMONSTRATION")
    
    # Generate test programs of different sizes
    test_sizes = [10, 50, 100, 500, 1000]
    
    for size in test_sizes:
        print_subsection(f"Performance Test - {size} Statements")
        
        # Generate program with many variable declarations
        statements = []
        for i in range(size):
            statements.append(f"synap var{i} = {i} + {i * 2};")
        
        source = "\n".join(statements)
        
        # Measure parsing time
        parser = AnamorphParser()
        start_time = time.time()
        result = parser.parse(source)
        end_time = time.time()
        
        parse_time = end_time - start_time
        
        if result.success:
            tokens_per_second = result.token_count / parse_time if parse_time > 0 else 0
            statements_per_second = size / parse_time if parse_time > 0 else 0
            
            print(f"✅ Parsed {size} statements successfully")
            print(f"   Total tokens: {result.token_count:,}")
            print(f"   Parse time: {parse_time:.4f}s")
            print(f"   Tokens/second: {tokens_per_second:,.0f}")
            print(f"   Statements/second: {statements_per_second:,.0f}")
        else:
            print(f"❌ Failed to parse {size} statements")


async def demonstrate_async_parsing():
    """Demonstrate asynchronous parsing capabilities."""
    print_section("ASYNC PARSING DEMONSTRATION")
    
    complex_program = """
    neuro complexNeuralNetwork(inputs: Array, config: Config) {
        synap layers = [];
        synap weights = initializeWeights(config.layers);
        
        // Input layer processing
        for (synap i = 0; i < inputs.length; i++) {
            synap normalized = normalize(inputs[i]);
            layers[0].push(normalized);
        }
        
        // Hidden layers processing
        for (synap layer = 1; layer < config.layers.length; layer++) {
            synap currentLayer = [];
            
            for (synap neuron = 0; neuron < config.layers[layer]; neuron++) {
                synap sum = 0;
                
                for (synap prev = 0; prev < layers[layer-1].length; prev++) {
                    sum += layers[layer-1][prev] * weights[layer][neuron][prev];
                }
                
                synap activated = activate(sum, config.activation);
                currentLayer.push(activated);
                
                if (activated > config.threshold) {
                    pulse activated -> nextLayer;
                }
            }
            
            layers[layer] = currentLayer;
        }
        
        synap output = layers[layers.length - 1];
        pulse output -> outputHandler;
        return output;
    }
    
    neuro trainNetwork(data: TrainingData, epochs: Number) {
        for (synap epoch = 0; epoch < epochs; epoch++) {
            synap totalError = 0;
            
            for (synap sample = 0; sample < data.length; sample++) {
                synap prediction = complexNeuralNetwork(data[sample].input, config);
                synap error = calculateError(prediction, data[sample].expected);
                totalError += error;
                
                pulse error -> backpropagation;
            }
            
            if (epoch % 100 == 0) {
                pulse totalError -> progressMonitor;
            }
        }
    }
    """
    
    print("Parsing complex neural network program asynchronously...")
    print(f"Source length: {len(complex_program)} characters")
    
    start_time = time.time()
    result = await parse_async(complex_program)
    end_time = time.time()
    
    if result.success:
        print(f"\n✅ Async parsing completed successfully!")
        print(f"   Parse time: {end_time - start_time:.4f}s")
        print(f"   Tokens processed: {result.token_count:,}")
        print(f"   AST nodes: {len(result.ast.body)}")
        
        # Show neuron declarations found
        neurons = [node for node in result.ast.body if isinstance(node, NeuronDeclaration)]
        print(f"   Neurons found: {len(neurons)}")
        for neuron in neurons:
            print(f"     - {neuron.id.name}({len(neuron.params)} params)")
    else:
        print(f"\n❌ Async parsing failed with {len(result.errors)} errors")


def demonstrate_real_world_example():
    """Demonstrate parsing a real-world Anamorph program."""
    print_section("REAL-WORLD EXAMPLE DEMONSTRATION")
    
    real_world_program = """
    // Anamorph Neural Signal Processing System
    
    synap globalConfig = {
        threshold: 0.7,
        learningRate: 0.01,
        maxIterations: 1000
    };
    
    neuro signalFilter(signal: Signal, filters: Array) {
        synap filtered = signal;
        
        for (synap i = 0; i < filters.length; i++) {
            filtered = applyFilter(filtered, filters[i]);
            
            if (filtered.noise > globalConfig.threshold) {
                pulse filtered -> noiseReduction;
                filtered = reduceNoise(filtered);
            }
        }
        
        return filtered;
    }
    
    neuro patternRecognizer(signals: Array) {
        synap patterns = [];
        synap confidence = 0;
        
        for (synap i = 0; i < signals.length; i++) {
            synap pattern = extractPattern(signals[i]);
            patterns.push(pattern);
            
            synap match = findBestMatch(pattern, knowledgeBase);
            if (match.confidence > globalConfig.threshold) {
                confidence += match.confidence;
                pulse match -> patternDatabase;
            }
        }
        
        synap result = {
            patterns: patterns,
            confidence: confidence / patterns.length,
            timestamp: getCurrentTime()
        };
        
        pulse result -> outputProcessor;
        return result;
    }
    
    neuro adaptiveLearning(input: TrainingData, target: Expected) {
        synap error = calculateError(input, target);
        synap adjustment = error * globalConfig.learningRate;
        
        if (error > globalConfig.threshold) {
            pulse error -> errorHandler;
            
            // Adjust weights based on error
            for (synap layer = 0; layer < networkLayers.length; layer++) {
                for (synap neuron = 0; neuron < networkLayers[layer].length; neuron++) {
                    networkLayers[layer][neuron].weight += adjustment;
                    pulse networkLayers[layer][neuron] -> weightUpdater;
                }
            }
        }
        
        return adjustment;
    }
    
    neuro mainProcessor() {
        synap inputBuffer = getInputBuffer();
        synap results = [];
        
        while (inputBuffer.length > 0) {
            synap rawSignal = inputBuffer.pop();
            synap filtered = signalFilter(rawSignal, standardFilters);
            synap recognized = patternRecognizer([filtered]);
            
            if (recognized.confidence > globalConfig.threshold) {
                results.push(recognized);
                pulse recognized -> successHandler;
            } else {
                pulse recognized -> reprocessingQueue;
            }
        }
        
        pulse results -> finalOutput;
        return results;
    }
    """
    
    print("Parsing real-world Anamorph neural processing system...")
    print(f"Program size: {len(real_world_program)} characters")
    print(f"Lines of code: {len(real_world_program.splitlines())}")
    
    parser = AnamorphParser()
    result = parser.parse(real_world_program)
    
    if result.success:
        print(f"\n✅ Real-world program parsed successfully!")
        print(f"   Parse time: {result.parse_time:.4f}s")
        print(f"   Tokens processed: {result.token_count:,}")
        print(f"   Statements parsed: {len(result.ast.body)}")
        
        # Analyze the program structure
        neurons = []
        variables = []
        
        for node in result.ast.body:
            if isinstance(node, NeuronDeclaration):
                neurons.append(node)
            elif isinstance(node, VariableDeclaration):
                variables.append(node)
        
        print(f"\nProgram Analysis:")
        print(f"   Global variables: {len(variables)}")
        print(f"   Neuron functions: {len(neurons)}")
        
        if neurons:
            print(f"   Neurons defined:")
            for neuron in neurons:
                param_count = len(neuron.params)
                stmt_count = len(neuron.body.statements) if neuron.body else 0
                print(f"     - {neuron.id.name}({param_count} params, {stmt_count} statements)")
        
        # Performance metrics
        tokens_per_second = result.token_count / result.parse_time if result.parse_time > 0 else 0
        print(f"\nPerformance Metrics:")
        print(f"   Tokens per second: {tokens_per_second:,.0f}")
        print(f"   Characters per second: {len(real_world_program) / result.parse_time:,.0f}")
        
    else:
        print(f"\n❌ Failed to parse real-world program")
        print(f"   Errors: {len(result.errors)}")
        
        # Show detailed error information
        for i, error in enumerate(result.errors[:3], 1):
            print(f"\nError {i}:")
            print(f"   Code: {error.code.value}")
            print(f"   Message: {error.message}")
            if error.suggestions:
                print(f"   Suggestions: {error.suggestions[0]}")


async def main():
    """Main demonstration function."""
    print("🧠 ANAMORPH PARSER DEMONSTRATION")
    print("Advanced Neural Programming Language Parser")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all demonstrations
        demonstrate_basic_parsing()
        demonstrate_neural_constructs()
        demonstrate_ast_generation()
        demonstrate_error_handling()
        demonstrate_performance()
        await demonstrate_async_parsing()
        demonstrate_real_world_example()
        
        print_section("DEMONSTRATION COMPLETE")
        print("✅ All parser demonstrations completed successfully!")
        print("\nThe Anamorph parser demonstrates:")
        print("  • Complete recursive descent parsing")
        print("  • Comprehensive AST generation")
        print("  • Advanced error handling and recovery")
        print("  • High-performance token processing")
        print("  • Neural-specific language constructs")
        print("  • Asynchronous parsing capabilities")
        print("  • Real-world program compatibility")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 