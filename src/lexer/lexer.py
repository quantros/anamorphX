"""
Anamorph Neural Programming Language Lexer.

This module implements the lexical analyzer (tokenizer) for Anamorph language
with support for case-insensitive keywords, async processing, and comprehensive
error handling.
"""

import re
import asyncio
from typing import List, Iterator, Optional, Union, TextIO, AsyncIterator
from dataclasses import dataclass
from io import StringIO
import time

from .tokens import (
    Token, TokenType, NEURAL_COMMANDS, KEYWORDS, TYPE_KEYWORDS,
    OPERATORS, DELIMITERS, TOKEN_PATTERNS, RESERVED_WORDS
)
from .errors import LexerError, ErrorContext, ErrorCode, ErrorSeverity


@dataclass
class LexerState:
    """Represents the current state of the lexer."""


@dataclass
class LexerState:
    """Represents the current state of the lexer."""
    
    source: str
    position: int = 0
    line: int = 1
    column: int = 1
    source_file: Optional[str] = None
    
    def current_char(self) -> Optional[str]:
        """Get the current character or None if at end."""
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at a character ahead without advancing position."""
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self) -> Optional[str]:
        """Advance to the next character and return it."""
        if self.position >= len(self.source):
            return None
        
        char = self.source[self.position]
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        return char
    
    def advance_by(self, count: int) -> str:
        """Advance by multiple characters and return the substring."""
        start_pos = self.position
        for _ in range(count):
            if self.advance() is None:
                break
        return self.source[start_pos:self.position]
    
    def match_pattern(self, pattern: re.Pattern) -> Optional[re.Match]:
        """Match a regex pattern at the current position."""
        return pattern.match(self.source, self.position)


class AnamorphLexer:
    """
    Lexical analyzer for Anamorph Neural Programming Language.
    
    Features:
    - Case-insensitive neural commands and keywords
    - Comprehensive error handling with position tracking
    - Async tokenization support for large files
    - String escape sequence processing
    - Comment handling
    - Numeric literal parsing (integers, floats, scientific notation)
    """
    
    def __init__(self, source_file: Optional[str] = None):
        """Initialize the lexer."""
        self.source_file = source_file
        self.tokens: List[Token] = []
        self.errors: List[LexerError] = []
        
        # Performance metrics
        self.start_time: Optional[float] = None
        self.token_count: int = 0
        
        # Case-insensitive lookup tables
        self._neural_commands_lower = {k.lower(): v for k, v in NEURAL_COMMANDS.items()}
        self._keywords_lower = {k.lower(): v for k, v in KEYWORDS.items()}
        self._type_keywords_lower = {k.lower(): v for k, v in TYPE_KEYWORDS.items()}
    
    def tokenize(self, source: str) -> List[Token]:
        """
        Tokenize source code synchronously.
        
        Args:
            source: Source code string to tokenize
            
        Returns:
            List of tokens
            
        Raises:
            LexerError: If tokenization fails
        """
        self.start_time = time.time()
        self.tokens = []
        self.errors = []
        self.token_count = 0
        
        state = LexerState(source=source, source_file=self.source_file)
        
        try:
            while state.current_char() is not None:
                self._tokenize_next(state)
            
            # Add EOF token
            self._add_token(state, TokenType.EOF, None, 0)
            
        except Exception as e:
            context = ErrorContext(
                source_code=source,
                line=state.line,
                column=state.column,
                position=state.position,
                file_path=self.source_file
            )
            error = LexerError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error during tokenization: {e}",
                severity=ErrorSeverity.CRITICAL,
                context=context
            )
            self.errors.append(error)
            raise error
        
        if self.errors:
            raise self.errors[0]  # Raise first error
        
        return self.tokens
    
    async def tokenize_async(self, source: str) -> List[Token]:
        """
        Tokenize source code asynchronously.
        
        This method yields control periodically to allow other coroutines
        to run, making it suitable for large files.
        
        Args:
            source: Source code string to tokenize
            
        Returns:
            List of tokens
        """
        self.start_time = time.time()
        self.tokens = []
        self.errors = []
        self.token_count = 0
        
        state = LexerState(source=source, source_file=self.source_file)
        
        try:
            while state.current_char() is not None:
                self._tokenize_next(state)
                
                # Yield control every 100 tokens for responsiveness
                if self.token_count % 100 == 0:
                    await asyncio.sleep(0)
            
            # Add EOF token
            self._add_token(state, TokenType.EOF, None, 0)
            
        except Exception as e:
            context = ErrorContext(
                source_code=source,
                line=state.line,
                column=state.column,
                position=state.position,
                file_path=self.source_file
            )
            error = LexerError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error during async tokenization: {e}",
                severity=ErrorSeverity.CRITICAL,
                context=context
            )
            self.errors.append(error)
            raise error
        
        if self.errors:
            raise self.errors[0]
        
        return self.tokens
    
    async def tokenize_stream(self, source_stream: AsyncIterator[str]) -> AsyncIterator[Token]:
        """
        Tokenize source code from an async stream.
        
        Args:
            source_stream: Async iterator yielding source code chunks
            
        Yields:
            Tokens as they are parsed
        """
        buffer = ""
        state = LexerState(source="", source_file=self.source_file)
        
        async for chunk in source_stream:
            buffer += chunk
            state.source = buffer
            
            # Process complete tokens
            while state.current_char() is not None:
                start_pos = state.position
                try:
                    token = self._tokenize_next(state)
                    if token:
                        yield token
                except LexerError:
                    # If we can't tokenize, we might need more input
                    state.position = start_pos
                    break
                
                await asyncio.sleep(0)  # Yield control
        
        # Process remaining buffer
        while state.current_char() is not None:
            token = self._tokenize_next(state)
            if token:
                yield token
        
        # Yield EOF token
        yield Token(
            type=TokenType.EOF,
            value=None,
            line=state.line,
            column=state.column,
            position=state.position,
            length=0,
            source_file=self.source_file
        )
    
    def _tokenize_next(self, state: LexerState) -> Optional[Token]:
        """Tokenize the next token from the current position."""
        char = state.current_char()
        if char is None:
            return None
        
        # Skip whitespace (but track it for indentation-sensitive parsing)
        if char in ' \t':
            return self._tokenize_whitespace(state)
        
        # Handle newlines
        if char == '\n':
            return self._tokenize_newline(state)
        
        # Handle comments
        if char == '#':
            return self._tokenize_comment(state)
        
        # Handle string literals
        if char in '"\'':
            return self._tokenize_string(state)
        
        # Handle numeric literals
        if char.isdigit():
            return self._tokenize_number(state)
        
        # Handle identifiers and keywords
        if char.isalpha() or char == '_':
            return self._tokenize_identifier(state)
        
        # Handle multi-character operators
        multi_char_match = state.match_pattern(TOKEN_PATTERNS['MULTI_CHAR_OP'])
        if multi_char_match:
            op = multi_char_match.group()
            # Check if it's in operators first, then delimiters
            if op in OPERATORS:
                token = self._add_token(state, OPERATORS[op], op, len(op))
            elif op in DELIMITERS:
                token = self._add_token(state, DELIMITERS[op], op, len(op))
            else:
                # This shouldn't happen if patterns are correct
                context = ErrorContext(
                    source_code=state.source,
                    line=state.line,
                    column=state.column,
                    position=state.position,
                    file_path=self.source_file
                )
                error = LexerError(
                    code=ErrorCode.INVALID_CHARACTER,
                    message=f"Unknown multi-character operator: '{op}'",
                    severity=ErrorSeverity.ERROR,
                    context=context
                )
                self.errors.append(error)
                token = None
            
            state.advance_by(len(op))
            return token
        
        # Handle single-character tokens
        if char in OPERATORS:
            token = self._add_token(state, OPERATORS[char], char, 1)
            state.advance()
            return token
        
        if char in DELIMITERS:
            token = self._add_token(state, DELIMITERS[char], char, 1)
            state.advance()
            return token
        
        # Unknown character
        context = ErrorContext(
            source_code=state.source,
            line=state.line,
            column=state.column,
            position=state.position,
            file_path=self.source_file
        )
        error = LexerError(
            code=ErrorCode.INVALID_CHARACTER,
            message=f"Unexpected character: '{char}'",
            severity=ErrorSeverity.ERROR,
            context=context
        )
        self.errors.append(error)
        state.advance()  # Skip the problematic character
        return None
    
    def _tokenize_whitespace(self, state: LexerState) -> Optional[Token]:
        """Tokenize whitespace characters."""
        start_pos = state.position
        while state.current_char() is not None and state.current_char() in ' \t':
            state.advance()
        
        whitespace = state.source[start_pos:state.position]
        return self._add_token(state, TokenType.WHITESPACE, whitespace, len(whitespace))
    
    def _tokenize_newline(self, state: LexerState) -> Token:
        """Tokenize newline character."""
        token = self._add_token(state, TokenType.NEWLINE, '\n', 1)
        state.advance()
        return token
    
    def _tokenize_comment(self, state: LexerState) -> Token:
        """Tokenize comment (from # to end of line)."""
        start_pos = state.position
        
        # Consume until newline or EOF
        while state.current_char() is not None and state.current_char() != '\n':
            state.advance()
        
        comment_text = state.source[start_pos:state.position]
        return self._add_token(state, TokenType.COMMENT, comment_text, len(comment_text))
    
    def _tokenize_string(self, state: LexerState) -> Token:
        """Tokenize string literal with escape sequence support."""
        quote_char = state.current_char()
        start_pos = state.position
        state.advance()  # Skip opening quote
        
        value = ""
        while True:
            char = state.current_char()
            
            if char is None:
                context = ErrorContext(
                    source_code=state.source,
                    line=state.line,
                    column=state.column,
                    position=state.position,
                    file_path=self.source_file
                )
                error = LexerError(
                    code=ErrorCode.UNTERMINATED_STRING,
                    message="Unterminated string literal",
                    severity=ErrorSeverity.ERROR,
                    context=context
                )
                self.errors.append(error)
                break
            
            if char == quote_char:
                state.advance()  # Skip closing quote
                break
            
            if char == '\\':
                # Handle escape sequences
                state.advance()
                escape_char = state.current_char()
                if escape_char is None:
                    context = ErrorContext(
                        source_code=state.source,
                        line=state.line,
                        column=state.column,
                        position=state.position,
                        file_path=self.source_file
                    )
                    error = LexerError(
                        code=ErrorCode.INVALID_ESCAPE_SEQUENCE,
                        message="Unterminated escape sequence",
                        severity=ErrorSeverity.ERROR,
                        context=context
                    )
                    self.errors.append(error)
                    break
                
                # Process escape sequences
                if escape_char == 'n':
                    value += '\n'
                elif escape_char == 't':
                    value += '\t'
                elif escape_char == 'r':
                    value += '\r'
                elif escape_char == '\\':
                    value += '\\'
                elif escape_char == quote_char:
                    value += quote_char
                elif escape_char == '0':
                    value += '\0'
                else:
                    # Unknown escape sequence, keep as-is
                    value += escape_char
                
                state.advance()
            else:
                value += char
                state.advance()
        
        length = state.position - start_pos
        return self._add_token(state, TokenType.STRING, value, length)
    
    def _tokenize_number(self, state: LexerState) -> Token:
        """Tokenize numeric literal (integer or float)."""
        start_pos = state.position
        
        # Check for float pattern first (more specific)
        float_match = state.match_pattern(TOKEN_PATTERNS['FLOAT'])
        if float_match:
            value_str = float_match.group()
            state.advance_by(len(value_str))
            try:
                value = float(value_str)
                return self._add_token(state, TokenType.FLOAT, value, len(value_str))
            except ValueError:
                context = ErrorContext(
                    source_code=state.source,
                    line=state.line,
                    column=state.column,
                    position=start_pos,
                    file_path=self.source_file
                )
                error = LexerError(
                    code=ErrorCode.INVALID_FLOAT_FORMAT,
                    message=f"Invalid float literal: {value_str}",
                    severity=ErrorSeverity.ERROR,
                    context=context
                )
                self.errors.append(error)
                return self._add_token(state, TokenType.FLOAT, 0.0, len(value_str))
        
        # Check for integer pattern
        int_match = state.match_pattern(TOKEN_PATTERNS['INTEGER'])
        if int_match:
            value_str = int_match.group()
            state.advance_by(len(value_str))
            try:
                value = int(value_str)
                return self._add_token(state, TokenType.INTEGER, value, len(value_str))
            except ValueError:
                context = ErrorContext(
                    source_code=state.source,
                    line=state.line,
                    column=state.column,
                    position=start_pos,
                    file_path=self.source_file
                )
                error = LexerError(
                    code=ErrorCode.INVALID_NUMBER_FORMAT,
                    message=f"Invalid integer literal: {value_str}",
                    severity=ErrorSeverity.ERROR,
                    context=context
                )
                self.errors.append(error)
                return self._add_token(state, TokenType.INTEGER, 0, len(value_str))
        
        # This shouldn't happen if called correctly
        context = ErrorContext(
            source_code=state.source,
            line=state.line,
            column=state.column,
            position=state.position,
            file_path=self.source_file
        )
        error = LexerError(
            code=ErrorCode.INVALID_NUMBER_FORMAT,
            message=f"Invalid numeric literal starting with: {state.current_char()}",
            severity=ErrorSeverity.ERROR,
            context=context
        )
        self.errors.append(error)
        state.advance()
        return None
    
    def _tokenize_identifier(self, state: LexerState) -> Token:
        """Tokenize identifier or keyword (case-insensitive for keywords)."""
        start_pos = state.position
        
        # Match identifier pattern
        match = state.match_pattern(TOKEN_PATTERNS['IDENTIFIER'])
        if not match:
            context = ErrorContext(
                source_code=state.source,
                line=state.line,
                column=state.column,
                position=state.position,
                file_path=self.source_file
            )
            error = LexerError(
                code=ErrorCode.INVALID_IDENTIFIER,
                message=f"Invalid identifier starting with: {state.current_char()}",
                severity=ErrorSeverity.ERROR,
                context=context
            )
            self.errors.append(error)
            state.advance()
            return None
        
        identifier = match.group()
        state.advance_by(len(identifier))
        
        # Check for neural commands (case-insensitive)
        identifier_lower = identifier.lower()
        if identifier_lower in self._neural_commands_lower:
            token_type = self._neural_commands_lower[identifier_lower]
            return self._add_token(state, token_type, identifier, len(identifier))
        
        # Check for keywords (case-insensitive)
        if identifier_lower in self._keywords_lower:
            token_type = self._keywords_lower[identifier_lower]
            # Special handling for boolean literals
            if token_type == TokenType.BOOLEAN:
                value = identifier_lower == 'true'
                return self._add_token(state, token_type, value, len(identifier))
            return self._add_token(state, token_type, identifier, len(identifier))
        
        # Check for type keywords (case-insensitive)
        if identifier_lower in self._type_keywords_lower:
            token_type = self._type_keywords_lower[identifier_lower]
            return self._add_token(state, token_type, identifier, len(identifier))
        
        # Regular identifier
        return self._add_token(state, TokenType.IDENTIFIER, identifier, len(identifier))
    
    def _add_token(self, state: LexerState, token_type: TokenType, value: any, length: int) -> Token:
        """Create and add a token to the tokens list."""
        token = Token(
            type=token_type,
            value=value,
            line=state.line,
            column=state.column - length,  # Column where token starts
            position=state.position - length,  # Position where token starts
            length=length,
            source_file=self.source_file
        )
        
        # Only add non-whitespace tokens to the main list
        if token_type not in (TokenType.WHITESPACE, TokenType.COMMENT):
            self.tokens.append(token)
            self.token_count += 1
        
        return token
    
    def get_metrics(self) -> dict:
        """Get lexer performance metrics."""
        end_time = time.time()
        duration = end_time - (self.start_time or end_time)
        
        return {
            'tokens_processed': self.token_count,
            'processing_time': duration,
            'tokens_per_second': self.token_count / duration if duration > 0 else 0,
            'errors_count': len(self.errors),
            'source_file': self.source_file
        }
    
    def has_errors(self) -> bool:
        """Check if lexer encountered any errors."""
        return len(self.errors) > 0
    
    def get_errors(self) -> List[LexerError]:
        """Get list of lexer errors."""
        return self.errors.copy()


# Convenience functions for quick tokenization
def tokenize(source: str, source_file: Optional[str] = None) -> List[Token]:
    """Tokenize source code quickly."""
    lexer = AnamorphLexer(source_file)
    return lexer.tokenize(source)


async def tokenize_async(source: str, source_file: Optional[str] = None) -> List[Token]:
    """Tokenize source code asynchronously."""
    lexer = AnamorphLexer(source_file)
    return await lexer.tokenize_async(source)


def tokenize_file(file_path: str) -> List[Token]:
    """Tokenize a source file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    lexer = AnamorphLexer(file_path)
    return lexer.tokenize(source)


async def tokenize_file_async(file_path: str) -> List[Token]:
    """Tokenize a source file asynchronously."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    lexer = AnamorphLexer(file_path)
    return await lexer.tokenize_async(source) 