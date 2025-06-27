"""
Unit tests for Anamorph Lexer.

This module contains comprehensive tests for the lexical analyzer including
neural commands, keywords, literals, error handling, and edge cases.
"""

import pytest
import asyncio
from typing import List

from src.lexer import (
    AnamorphLexer, Token, TokenType, LexerError, ErrorCode, ErrorSeverity,
    tokenize, tokenize_async, NEURAL_COMMANDS, KEYWORDS
)


class TestBasicTokenization:
    """Test basic tokenization functionality."""
    
    def test_empty_source(self):
        """Test tokenizing empty source code."""
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_whitespace_only(self):
        """Test tokenizing whitespace-only source."""
        tokens = tokenize("   \t  \n  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_single_identifier(self):
        """Test tokenizing a single identifier."""
        tokens = tokenize("myVariable")
        assert len(tokens) == 2  # identifier + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "myVariable"
        assert tokens[0].line == 1
        assert tokens[0].column == 1
    
    def test_multiple_identifiers(self):
        """Test tokenizing multiple identifiers."""
        tokens = tokenize("var1 var2 var3")
        assert len(tokens) == 4  # 3 identifiers + EOF
        assert all(t.type == TokenType.IDENTIFIER for t in tokens[:-1])
        assert [t.value for t in tokens[:-1]] == ["var1", "var2", "var3"]


class TestNeuralCommands:
    """Test neural command tokenization."""
    
    def test_basic_neural_commands(self):
        """Test basic neural commands."""
        source = "neuro synap pulse resonate"
        tokens = tokenize(source)
        
        expected_types = [TokenType.NEURO, TokenType.SYNAP, TokenType.PULSE, TokenType.RESONATE]
        actual_types = [t.type for t in tokens[:-1]]  # Exclude EOF
        assert actual_types == expected_types
    
    def test_case_insensitive_neural_commands(self):
        """Test case-insensitive neural command recognition."""
        test_cases = [
            ("neuro", TokenType.NEURO),
            ("NEURO", TokenType.NEURO),
            ("Neuro", TokenType.NEURO),
            ("NeUrO", TokenType.NEURO),
            ("pulse", TokenType.PULSE),
            ("PULSE", TokenType.PULSE),
            ("Pulse", TokenType.PULSE),
        ]
        
        for source, expected_type in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == expected_type
            assert tokens[0].value == source  # Original case preserved
    
    def test_all_neural_commands(self):
        """Test all 80 neural commands are recognized."""
        for command, token_type in NEURAL_COMMANDS.items():
            tokens = tokenize(command)
            assert tokens[0].type == token_type
            assert tokens[0].value == command
    
    def test_neural_commands_in_context(self):
        """Test neural commands in realistic code context."""
        source = """
        neuro webServer {
            synap port = 8080;
            pulse request -> handler;
            resonate listener(port);
        }
        """
        tokens = tokenize(source)
        
        # Find neural command tokens
        neural_tokens = [t for t in tokens if t.type in [
            TokenType.NEURO, TokenType.SYNAP, TokenType.PULSE, TokenType.RESONATE
        ]]
        
        assert len(neural_tokens) == 4
        assert neural_tokens[0].type == TokenType.NEURO
        assert neural_tokens[1].type == TokenType.SYNAP
        assert neural_tokens[2].type == TokenType.PULSE
        assert neural_tokens[3].type == TokenType.RESONATE


class TestKeywords:
    """Test keyword tokenization."""
    
    def test_basic_keywords(self):
        """Test basic keyword recognition."""
        source = "if else while for return"
        tokens = tokenize(source)
        
        expected_types = [TokenType.IF, TokenType.ELSE, TokenType.WHILE, TokenType.FOR, TokenType.RETURN]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types
    
    def test_case_insensitive_keywords(self):
        """Test case-insensitive keyword recognition."""
        test_cases = [
            ("if", TokenType.IF),
            ("IF", TokenType.IF),
            ("If", TokenType.IF),
            ("while", TokenType.WHILE),
            ("WHILE", TokenType.WHILE),
            ("While", TokenType.WHILE),
        ]
        
        for source, expected_type in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == expected_type
    
    def test_boolean_literals(self):
        """Test boolean literal recognition."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
        ]
        
        for source, expected_value in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == TokenType.BOOLEAN
            assert tokens[0].value == expected_value


class TestLiterals:
    """Test literal tokenization."""
    
    def test_integer_literals(self):
        """Test integer literal parsing."""
        test_cases = [
            ("0", 0),
            ("123", 123),
            ("999999", 999999),
        ]
        
        for source, expected_value in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == TokenType.INTEGER
            assert tokens[0].value == expected_value
    
    def test_float_literals(self):
        """Test float literal parsing."""
        test_cases = [
            ("0.0", 0.0),
            ("123.456", 123.456),
            ("3.14159", 3.14159),
            ("1.23e4", 1.23e4),
            ("1.23E-4", 1.23E-4),
            ("1.23e+4", 1.23e+4),
        ]
        
        for source, expected_value in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == TokenType.FLOAT
            assert abs(tokens[0].value - expected_value) < 1e-10
    
    def test_string_literals(self):
        """Test string literal parsing."""
        test_cases = [
            ('""', ""),
            ('"hello"', "hello"),
            ('"Hello, World!"', "Hello, World!"),
            ("'single quotes'", "single quotes"),
            ('"mixed \'quotes\'"', "mixed 'quotes'"),
        ]
        
        for source, expected_value in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == TokenType.STRING
            assert tokens[0].value == expected_value
    
    def test_string_escape_sequences(self):
        """Test string escape sequence handling."""
        test_cases = [
            ('"\\n"', "\n"),
            ('"\\t"', "\t"),
            ('"\\r"', "\r"),
            ('"\\\\"', "\\"),
            ('"\\""', '"'),
            ("'\\''", "'"),
            ('"\\0"', "\0"),
            ('"line1\\nline2"', "line1\nline2"),
        ]
        
        for source, expected_value in test_cases:
            tokens = tokenize(source)
            assert tokens[0].type == TokenType.STRING
            assert tokens[0].value == expected_value


class TestOperators:
    """Test operator tokenization."""
    
    def test_arithmetic_operators(self):
        """Test arithmetic operator recognition."""
        source = "+ - * / % **"
        tokens = tokenize(source)
        
        expected_types = [
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY,
            TokenType.DIVIDE, TokenType.MODULO, TokenType.POWER
        ]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types
    
    def test_comparison_operators(self):
        """Test comparison operator recognition."""
        source = "== != < <= > >="
        tokens = tokenize(source)
        
        expected_types = [
            TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.LESS_THAN,
            TokenType.LESS_EQUAL, TokenType.GREATER_THAN, TokenType.GREATER_EQUAL
        ]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types
    
    def test_logical_operators(self):
        """Test logical operator recognition."""
        source = "&& || !"
        tokens = tokenize(source)
        
        expected_types = [TokenType.AND, TokenType.OR, TokenType.NOT]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types
    
    def test_bitwise_operators(self):
        """Test bitwise operator recognition."""
        source = "& | ^ ~ << >>"
        tokens = tokenize(source)
        
        expected_types = [
            TokenType.BIT_AND, TokenType.BIT_OR, TokenType.BIT_XOR,
            TokenType.BIT_NOT, TokenType.LEFT_SHIFT, TokenType.RIGHT_SHIFT
        ]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types


class TestDelimiters:
    """Test delimiter tokenization."""
    
    def test_parentheses_and_braces(self):
        """Test parentheses and brace recognition."""
        source = "( ) { } [ ]"
        tokens = tokenize(source)
        
        expected_types = [
            TokenType.LEFT_PAREN, TokenType.RIGHT_PAREN,
            TokenType.LEFT_BRACE, TokenType.RIGHT_BRACE,
            TokenType.LEFT_BRACKET, TokenType.RIGHT_BRACKET
        ]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types
    
    def test_punctuation(self):
        """Test punctuation recognition."""
        source = "; , . : -> =>"
        tokens = tokenize(source)
        
        expected_types = [
            TokenType.SEMICOLON, TokenType.COMMA, TokenType.DOT,
            TokenType.COLON, TokenType.ARROW, TokenType.DOUBLE_ARROW
        ]
        actual_types = [t.type for t in tokens[:-1]]
        assert actual_types == expected_types


class TestComments:
    """Test comment handling."""
    
    def test_single_line_comment(self):
        """Test single-line comment recognition."""
        source = "# This is a comment"
        lexer = AnamorphLexer()
        lexer.tokenize(source)
        
        # Comments are not included in main token list but are processed
        assert len(lexer.tokens) == 1  # Only EOF
        assert lexer.tokens[0].type == TokenType.EOF
    
    def test_comment_with_code(self):
        """Test comment mixed with code."""
        source = """
        neuro main {  # Main neuron
            synap x = 42;  # Variable declaration
            # Another comment
            pulse x -> output;
        }
        """
        tokens = tokenize(source)
        
        # Should have tokens for code but not comments
        neural_tokens = [t for t in tokens if t.type in [TokenType.NEURO, TokenType.SYNAP, TokenType.PULSE]]
        assert len(neural_tokens) == 3
    
    def test_comment_at_end_of_line(self):
        """Test comment at end of line."""
        source = "neuro test  # comment here"
        tokens = tokenize(source)
        
        assert len(tokens) == 3  # neuro, test, EOF
        assert tokens[0].type == TokenType.NEURO
        assert tokens[1].type == TokenType.IDENTIFIER


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_invalid_character(self):
        """Test handling of invalid characters."""
        lexer = AnamorphLexer()
        
        with pytest.raises(LexerError) as exc_info:
            lexer.tokenize("valid @invalid")
        
        error = exc_info.value
        assert error.code == ErrorCode.INVALID_CHARACTER
        assert "@" in str(error)
    
    def test_unterminated_string(self):
        """Test handling of unterminated strings."""
        lexer = AnamorphLexer()
        
        with pytest.raises(LexerError):
            lexer.tokenize('"unterminated string')
    
    def test_invalid_escape_sequence(self):
        """Test handling of invalid escape sequences."""
        # This should not raise an error but handle gracefully
        tokens = tokenize('"\\x"')  # Invalid escape sequence
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "x"  # Should keep the character after backslash
    
    def test_error_position_tracking(self):
        """Test that errors track position correctly."""
        lexer = AnamorphLexer()
        
        try:
            lexer.tokenize("line1\nline2 @invalid")
        except LexerError as e:
            assert e.line == 2
            assert e.column == 7  # Position of @
    
    def test_multiple_errors(self):
        """Test handling multiple errors."""
        lexer = AnamorphLexer()
        
        try:
            lexer.tokenize("@invalid1 @invalid2")
        except LexerError:
            pass
        
        # Should have collected multiple errors
        assert len(lexer.errors) >= 1


class TestAsyncTokenization:
    """Test asynchronous tokenization."""
    
    def test_async_tokenize_basic(self):
        """Test basic async tokenization."""
        source = "neuro test { synap x = 42; }"

        async def run():
            tokens = await tokenize_async(source)
            assert len(tokens) > 0
            assert tokens[0].type == TokenType.NEURO
            assert tokens[-1].type == TokenType.EOF

        asyncio.run(run())
    
    def test_async_tokenize_large_source(self):
        """Test async tokenization with large source."""
        # Generate large source code
        lines = []
        for i in range(1000):
            lines.append(f"synap var{i} = {i};")
        source = "\n".join(lines)

        async def run():
            tokens = await tokenize_async(source)
            synap_tokens = [t for t in tokens if t.type == TokenType.SYNAP]
            assert len(synap_tokens) == 1000

        asyncio.run(run())


class TestPositionTracking:
    """Test position and location tracking."""
    
    def test_line_column_tracking(self):
        """Test line and column tracking."""
        source = """neuro test {
    synap x = 42;
    pulse x -> output;
}"""
        tokens = tokenize(source)
        
        # Find specific tokens and check their positions
        neuro_token = next(t for t in tokens if t.type == TokenType.NEURO)
        assert neuro_token.line == 1
        assert neuro_token.column == 1
        
        synap_token = next(t for t in tokens if t.type == TokenType.SYNAP)
        assert synap_token.line == 2
        assert synap_token.column == 5  # After 4 spaces
        
        pulse_token = next(t for t in tokens if t.type == TokenType.PULSE)
        assert pulse_token.line == 3
        assert pulse_token.column == 5
    
    def test_position_tracking(self):
        """Test absolute position tracking."""
        source = "abc def ghi"
        tokens = tokenize(source)
        
        assert tokens[0].position == 0  # "abc" starts at 0
        assert tokens[1].position == 4  # "def" starts at 4 (after "abc ")
        assert tokens[2].position == 8  # "ghi" starts at 8 (after "abc def ")
    
    def test_token_length(self):
        """Test token length calculation."""
        source = "neuro identifier 123 'string'"
        tokens = tokenize(source)
        
        assert tokens[0].length == 5   # "neuro"
        assert tokens[1].length == 10  # "identifier"
        assert tokens[2].length == 3   # "123"
        assert tokens[3].length == 8   # "'string'"


class TestComplexScenarios:
    """Test complex tokenization scenarios."""
    
    def test_realistic_anamorph_code(self):
        """Test tokenizing realistic Anamorph code."""
        source = '''
        neuro webServer {
            synap port = 8080;
            synap routes = {};
            
            # Setup routes
            routes["/api/users"] = handleUsers;
            routes["/api/data"] = handleData;
            
            # Start server with security
            resonate httpListener(port) -> requests {
                pulse requests -> securityFilter -> routeHandler;
            }
        }
        
        neuro handleUsers(request) {
            # Automatic SQL injection protection
            synap users = database.query("SELECT * FROM users WHERE id = ?", request.userId);
            
            # Automatic XSS protection
            pulse users -> sanitize -> jsonResponse;
        }
        '''
        
        tokens = tokenize(source)
        
        # Should successfully tokenize without errors
        assert len(tokens) > 0
        assert tokens[-1].type == TokenType.EOF
        
        # Check for expected neural commands
        neural_commands = [t for t in tokens if t.type in [
            TokenType.NEURO, TokenType.SYNAP, TokenType.PULSE, TokenType.RESONATE
        ]]
        assert len(neural_commands) >= 6  # At least 6 neural commands
    
    def test_mixed_case_preservation(self):
        """Test that original case is preserved in token values."""
        source = "NEURO MyFunction { SYNAP MyVar = 42; }"
        tokens = tokenize(source)
        
        # Find tokens and check original case is preserved
        neuro_token = next(t for t in tokens if t.type == TokenType.NEURO)
        assert neuro_token.value == "NEURO"
        
        identifier_token = next(t for t in tokens if t.type == TokenType.IDENTIFIER and t.value == "MyFunction")
        assert identifier_token.value == "MyFunction"
        
        synap_token = next(t for t in tokens if t.type == TokenType.SYNAP)
        assert synap_token.value == "SYNAP"
    
    def test_edge_case_identifiers(self):
        """Test edge cases for identifiers."""
        test_cases = [
            "_underscore",
            "_123",
            "var_with_underscores",
            "CamelCase",
            "snake_case",
            "mixedCase123",
        ]
        
        for identifier in test_cases:
            tokens = tokenize(identifier)
            assert tokens[0].type == TokenType.IDENTIFIER
            assert tokens[0].value == identifier


class TestPerformance:
    """Test lexer performance characteristics."""
    
    def test_lexer_metrics(self):
        """Test lexer performance metrics."""
        source = "neuro test { synap x = 42; pulse x -> output; }"
        lexer = AnamorphLexer()
        tokens = lexer.tokenize(source)
        
        metrics = lexer.get_metrics()
        
        assert metrics['tokens_processed'] > 0
        assert metrics['processing_time'] >= 0
        assert metrics['tokens_per_second'] >= 0
        assert metrics['errors_count'] == 0
    
    def test_large_file_handling(self):
        """Test handling of large source files."""
        # Generate a large source file
        lines = []
        for i in range(100):
            lines.append(f"neuro func{i} {{ synap var{i} = {i}; pulse var{i} -> output; }}")
        
        source = "\n".join(lines)
        tokens = tokenize(source)
        
        # Should handle large files without issues
        assert len(tokens) > 300  # Many tokens
        assert tokens[-1].type == TokenType.EOF


if __name__ == "__main__":
    pytest.main([__file__]) 