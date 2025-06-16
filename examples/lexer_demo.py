#!/usr/bin/env python3
"""
Anamorph Lexer Demonstration.

This script demonstrates the capabilities of the Anamorph lexer including
neural command recognition, error handling, and performance metrics.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lexer import (
    AnamorphLexer, tokenize, tokenize_async, TokenType,
    LexerError, ErrorCode, get_token_category
)


def print_separator(title: str):
    """Print a formatted separator."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_tokens(tokens, max_tokens=20):
    """Print tokens in a formatted way."""
    print(f"\nTokens ({len(tokens)} total):")
    print("-" * 40)
    
    for i, token in enumerate(tokens[:max_tokens]):
        category = get_token_category(token.type)
        print(f"{i+1:2d}. {token.type.name:15} | {repr(token.value):20} | "
              f"{token.line}:{token.column:2d} | {category}")
    
    if len(tokens) > max_tokens:
        print(f"... and {len(tokens) - max_tokens} more tokens")


def demo_basic_tokenization():
    """Demonstrate basic tokenization."""
    print_separator("BASIC TOKENIZATION")
    
    examples = [
        "neuro webServer",
        "synap port = 8080",
        "pulse request -> handler",
        "resonate listener(port)",
    ]
    
    for example in examples:
        print(f"\nSource: {example}")
        tokens = tokenize(example)
        print_tokens(tokens)


def demo_neural_commands():
    """Demonstrate neural command recognition."""
    print_separator("NEURAL COMMANDS DEMO")
    
    source = """
    # Web server with neural architecture
    neuro webServer {
        synap port = 8080;
        synap routes = {};
        
        # Security layer
        guard requestFilter {
            auth checkToken;
            validate sanitizeInput;
            throttle limitRequests(100);
        }
        
        # Main processing
        resonate httpListener(port) -> requests {
            pulse requests -> requestFilter -> routeHandler;
            echo response <- routeHandler;
        }
        
        # Data processing
        fold userRequests -> processedData {
            filter validRequests;
            merge userData;
            encrypt sensitiveData;
        }
        
        # Machine learning
        train userBehaviorModel(processedData);
        infer predictions <- userBehaviorModel;
        
        # Monitoring
        log serverActivity;
        alert criticalEvents;
        audit securityEvents;
    }
    """
    
    print("Source code:")
    print(source)
    
    tokens = tokenize(source)
    
    # Filter neural command tokens
    neural_tokens = [t for t in tokens if get_token_category(t.type) == 'NEURAL_COMMANDS']
    
    print(f"\nNeural Commands Found ({len(neural_tokens)}):")
    print("-" * 40)
    for i, token in enumerate(neural_tokens, 1):
        print(f"{i:2d}. {token.type.name:12} | {token.value:12} | Line {token.line}")


def demo_case_insensitive():
    """Demonstrate case-insensitive keyword recognition."""
    print_separator("CASE-INSENSITIVE RECOGNITION")
    
    examples = [
        "neuro NEURO Neuro NeUrO",
        "pulse PULSE Pulse PuLsE",
        "if IF If iF",
        "true TRUE True TrUe",
    ]
    
    for example in examples:
        print(f"\nSource: {example}")
        tokens = tokenize(example)
        
        for token in tokens[:-1]:  # Exclude EOF
            print(f"  '{token.value}' -> {token.type.name}")


def demo_literals():
    """Demonstrate literal tokenization."""
    print_separator("LITERALS DEMO")
    
    examples = [
        '42',
        '3.14159',
        '1.23e-4',
        '"Hello, World!"',
        "'Single quotes'",
        '"Escaped \\"quotes\\""',
        'true',
        'false',
    ]
    
    for example in examples:
        print(f"\nSource: {example}")
        tokens = tokenize(example)
        token = tokens[0]
        print(f"  Type: {token.type.name}")
        print(f"  Value: {repr(token.value)}")
        print(f"  Python type: {type(token.value).__name__}")


def demo_operators_and_delimiters():
    """Demonstrate operator and delimiter recognition."""
    print_separator("OPERATORS & DELIMITERS")
    
    source = """
    synap result = (a + b) * c / d;
    if (x == y && z != w) {
        pulse data -> output;
    }
    array[index] = value << 2;
    """
    
    print("Source code:")
    print(source)
    
    tokens = tokenize(source)
    
    # Filter operators and delimiters
    op_del_tokens = [t for t in tokens if get_token_category(t.type) in ['OPERATORS', 'DELIMITERS']]
    
    print(f"\nOperators & Delimiters ({len(op_del_tokens)}):")
    print("-" * 30)
    for token in op_del_tokens:
        category = get_token_category(token.type)
        print(f"  '{token.value}' -> {token.type.name} ({category})")


def demo_error_handling():
    """Demonstrate error handling."""
    print_separator("ERROR HANDLING DEMO")
    
    error_examples = [
        ('"unterminated string', "Unterminated string"),
        ('valid @invalid', "Invalid character"),
        ('123.45.67', "Invalid number format"),
    ]
    
    for source, description in error_examples:
        print(f"\n{description}:")
        print(f"Source: {source}")
        
        lexer = AnamorphLexer()
        try:
            tokens = lexer.tokenize(source)
            print("  No errors (unexpected)")
        except LexerError as e:
            print(f"  Error: {e.code.value} - {e.message}")
            print(f"  Position: Line {e.context.line}, Column {e.context.column}")
            if e.suggestions:
                print("  Suggestions:")
                for suggestion in e.suggestions:
                    print(f"    - {suggestion}")


def demo_performance():
    """Demonstrate performance metrics."""
    print_separator("PERFORMANCE DEMO")
    
    # Generate large source code
    lines = []
    for i in range(1000):
        lines.append(f"neuro func{i} {{ synap var{i} = {i}; pulse var{i} -> output{i}; }}")
    
    large_source = "\n".join(lines)
    
    print(f"Tokenizing large source ({len(large_source)} characters, {len(lines)} lines)")
    
    lexer = AnamorphLexer()
    tokens = lexer.tokenize(large_source)
    metrics = lexer.get_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  Tokens processed: {metrics['tokens_processed']}")
    print(f"  Processing time: {metrics['processing_time']:.4f} seconds")
    print(f"  Tokens per second: {metrics['tokens_per_second']:.0f}")
    print(f"  Errors: {metrics['errors_count']}")


async def demo_async_tokenization():
    """Demonstrate asynchronous tokenization."""
    print_separator("ASYNC TOKENIZATION DEMO")
    
    source = """
    neuro asyncServer {
        synap connections = [];
        
        async resonate connectionHandler(socket) -> messages {
            while (socket.connected) {
                synap message = await socket.receive();
                pulse message -> processMessage;
                await socket.send(response);
            }
        }
        
        async neuro processMessage(msg) {
            # Async processing
            synap result = await database.query(msg.query);
            pulse result -> formatResponse;
            return response;
        }
    }
    """
    
    print("Async tokenization of neural server code...")
    print("Source length:", len(source), "characters")
    
    start_time = asyncio.get_event_loop().time()
    tokens = await tokenize_async(source)
    end_time = asyncio.get_event_loop().time()
    
    print(f"Tokenized {len(tokens)} tokens in {end_time - start_time:.4f} seconds")
    
    # Show some interesting tokens
    neural_tokens = [t for t in tokens if get_token_category(t.type) == 'NEURAL_COMMANDS']
    print(f"Found {len(neural_tokens)} neural commands:")
    for token in neural_tokens:
        print(f"  {token.type.name} at line {token.line}")


def demo_real_world_example():
    """Demonstrate tokenizing a real-world Anamorph program."""
    print_separator("REAL-WORLD EXAMPLE")
    
    source = '''
    # Enterprise Web Server with Neural Architecture
    neuro enterpriseServer {
        # Configuration
        synap config = {
            port: 8080,
            maxConnections: 1000,
            timeout: 30000,
            ssl: true
        };
        
        # Security layer with multiple protections
        guard securityLayer {
            # DDoS protection
            throttle connectionLimiter(config.maxConnections);
            ban suspiciousIPs;
            
            # Authentication
            auth validateToken -> userContext;
            whitelist trustedSources;
            
            # Input validation
            validate sanitizeInput;
            filter maliciousPatterns;
            
            # Encryption
            encrypt sensitiveData;
            audit securityEvents;
        }
        
        # Request processing pipeline
        resonate requestPipeline(request) -> response {
            # Security first
            pulse request -> securityLayer -> validatedRequest;
            
            # Route to appropriate handler
            if (validatedRequest.path.startsWith("/api/")) {
                pulse validatedRequest -> apiHandler -> apiResponse;
            } else if (validatedRequest.path.startsWith("/admin/")) {
                pulse validatedRequest -> adminHandler -> adminResponse;
            } else {
                pulse validatedRequest -> staticHandler -> staticResponse;
            }
            
            # Response processing
            merge [apiResponse, adminResponse, staticResponse] -> finalResponse;
            echo finalResponse -> client;
        }
        
        # API handler with ML capabilities
        neuro apiHandler(request) {
            synap endpoint = request.path;
            synap method = request.method;
            
            # Pattern recognition for API usage
            pattern userBehavior = train(request.userAgent, request.frequency);
            
            # Intelligent routing based on patterns
            if (infer(userBehavior) == "bot") {
                throttle botLimiter(10);  # Limit bot requests
            }
            
            # Database operations with automatic protection
            synap query = buildQuery(endpoint, request.params);
            synap result = database.execute(query);  # Auto SQL injection protection
            
            # Response formatting
            encode result -> jsonResponse;
            return jsonResponse;
        }
        
        # Monitoring and analytics
        neuro monitoringSystem {
            # Real-time metrics
            sense serverMetrics -> currentLoad;
            
            if (currentLoad > 0.8) {
                scaleup serverInstances(2);
                alert "High load detected";
            } else if (currentLoad < 0.2) {
                scaledown serverInstances(1);
            }
            
            # Logging and analytics
            log requestMetrics;
            trace performanceData;
            
            # Backup and recovery
            checkpoint systemState(hourly);
            backup userData(daily);
        }
        
        # Main server loop
        resonate serverLoop {
            listen config.port -> incomingRequests;
            
            fold incomingRequests -> processedRequests {
                pulse request -> requestPipeline -> response;
                notify monitoringSystem(request, response);
            }
        }
    }
    '''
    
    print("Tokenizing enterprise web server example...")
    print(f"Source: {len(source)} characters, {len(source.splitlines())} lines")
    
    try:
        tokens = tokenize(source)
        print(f"✓ Successfully tokenized {len(tokens)} tokens")
        
        # Analyze token distribution
        token_stats = {}
        for token in tokens:
            category = get_token_category(token.type)
            token_stats[category] = token_stats.get(category, 0) + 1
        
        print("\nToken Distribution:")
        for category, count in sorted(token_stats.items()):
            print(f"  {category}: {count}")
        
        # Show neural commands used
        neural_tokens = [t for t in tokens if get_token_category(t.type) == 'NEURAL_COMMANDS']
        neural_commands = {}
        for token in neural_tokens:
            neural_commands[token.type.name] = neural_commands.get(token.type.name, 0) + 1
        
        print(f"\nNeural Commands Used ({len(neural_tokens)} total):")
        for command, count in sorted(neural_commands.items()):
            print(f"  {command}: {count}")
            
    except LexerError as e:
        print(f"✗ Tokenization failed: {e}")


async def main():
    """Run all demonstrations."""
    print("ANAMORPH LEXER DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the capabilities of the Anamorph lexer")
    print("including neural command recognition, error handling, and performance.")
    
    # Run synchronous demos
    demo_basic_tokenization()
    demo_neural_commands()
    demo_case_insensitive()
    demo_literals()
    demo_operators_and_delimiters()
    demo_error_handling()
    demo_performance()
    demo_real_world_example()
    
    # Run async demo
    await demo_async_tokenization()
    
    print_separator("DEMO COMPLETE")
    print("The Anamorph lexer successfully demonstrated:")
    print("✓ Neural command recognition (80 commands)")
    print("✓ Case-insensitive keyword handling")
    print("✓ Comprehensive literal parsing")
    print("✓ Robust error handling with suggestions")
    print("✓ High-performance tokenization")
    print("✓ Asynchronous processing capabilities")
    print("✓ Real-world code tokenization")


if __name__ == "__main__":
    asyncio.run(main()) 