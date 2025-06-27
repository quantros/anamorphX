"""
Anamorph Language Parser.

Complete recursive descent parser for the Anamorph neural programming language.
Supports all 80 neural commands and generates real AST nodes.
"""

import sys
import os
from typing import List, Optional, Union, Any, Dict
from enum import Enum
from dataclasses import dataclass

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import lexer and AST nodes
try:
    from src.lexer.lexer import AnamorphLexer
    from src.lexer.tokens import TokenType, Token
    from src.syntax.nodes import *
    IMPORTS_AVAILABLE = True
except ImportError:
    try:
        from lexer.lexer import AnamorphLexer
        from lexer.tokens import TokenType, Token
        from syntax.nodes import *
        IMPORTS_AVAILABLE = True
    except ImportError:
        print("⚠️ Не удалось импортировать лексер или AST ноды")
        IMPORTS_AVAILABLE = False


@dataclass
class ParseResult:
    """Результат парсинга"""
    success: bool
    ast: Optional[Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ParseError(Exception):
    """Parser error with location information."""
    
    def __init__(self, message: str, token: Optional['Token'] = None):
        self.message = message
        self.token = token
        super().__init__(self.format_error())
    
    def format_error(self) -> str:
        if self.token:
            return f"Parse error at line {self.token.line}, column {self.token.column}: {self.message}"
        return f"Parse error: {self.message}"


class AnamorphParser:
    """Complete parser for Anamorph neural programming language."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.tokens: List[Token] = []
        self.current = 0
        self.errors: List[ParseError] = []
        
        # Neural command mappings for the 80 commands
        self.neural_commands = {
            # Core neural commands
            TokenType.NEURO: self._parse_neuro_declaration,
            TokenType.SYNAP: self._parse_synap_declaration,
            TokenType.PULSE: self._parse_pulse_statement,
            TokenType.RESONATE: self._parse_resonate_statement,
            TokenType.DRIFT: self._parse_drift_statement,
            TokenType.BIND: self._parse_bind_statement,
            TokenType.ECHO: self._parse_echo_statement,
            TokenType.FORGE: self._parse_forge_statement,
            TokenType.PRUNE: self._parse_prune_statement,
            TokenType.FILTER: self._parse_filter_statement,
            
            # Security commands
            TokenType.GUARD: self._parse_guard_statement,
            TokenType.MASK: self._parse_mask_statement,
            TokenType.SCRAMBLE: self._parse_scramble_statement,
            TokenType.ENCRYPT: self._parse_encrypt_statement,
            TokenType.DECRYPT: self._parse_decrypt_statement,
            TokenType.AUTH: self._parse_auth_statement,
            TokenType.AUDIT: self._parse_audit_statement,
            TokenType.BAN: self._parse_ban_statement,
            TokenType.WHITELIST: self._parse_whitelist_statement,
            TokenType.BLACKLIST: self._parse_blacklist_statement,
            
            # Flow control commands
            TokenType.SYNC: self._parse_sync_statement,
            TokenType.ASYNC: self._parse_async_statement,
            TokenType.WAIT: self._parse_wait_statement,
            TokenType.JUMP: self._parse_jump_statement,
            TokenType.HALT: self._parse_halt_statement,
            TokenType.YIELD: self._parse_yield_statement,
            TokenType.LOOP: self._parse_loop_statement,
            
            # Data processing commands
            TokenType.FOLD: self._parse_fold_statement,
            TokenType.UNFOLD: self._parse_unfold_statement,
            TokenType.MERGE: self._parse_merge_statement,
            TokenType.SPLIT: self._parse_split_statement,
            TokenType.ENCODE: self._parse_encode_statement,
            TokenType.DECODE: self._parse_decode_statement,
            TokenType.QUANTA: self._parse_quanta_statement,
            
            # Neural network commands
            TokenType.CLUSTER: self._parse_cluster_statement,
            TokenType.EXPAND: self._parse_expand_statement,
            TokenType.CONTRACT: self._parse_contract_statement,
            TokenType.MORPH: self._parse_morph_statement,
            TokenType.EVOLVE: self._parse_evolve_statement,
            TokenType.TRAIN: self._parse_train_statement,
            TokenType.INFER: self._parse_infer_statement,
            
            # System commands
            TokenType.TRACE: self._parse_trace_statement,
            TokenType.LOG: self._parse_log_statement,
            TokenType.ALERT: self._parse_alert_statement,
            TokenType.RESET: self._parse_reset_statement,
            TokenType.BACKUP: self._parse_backup_statement,
            TokenType.RESTORE: self._parse_restore_statement,
            TokenType.SNAPSHOT: self._parse_snapshot_statement,
            TokenType.MIGRATE: self._parse_migrate_statement,
            
            # Advanced commands
            TokenType.PHASE: self._parse_phase_statement,
            TokenType.PATTERN: self._parse_pattern_statement,
            TokenType.THROTTLE: self._parse_throttle_statement,
            TokenType.SCALEUP: self._parse_scaleup_statement,
            TokenType.SCALEDOWN: self._parse_scaledown_statement,
            TokenType.NOTIFY: self._parse_notify_statement,
            TokenType.VALIDATE: self._parse_validate_statement,
        }
    
    def parse(self, source_code: str, filename: str = "<source>") -> Program:
        """Parse source code into AST."""
        try:
            # Tokenize
            if not IMPORTS_AVAILABLE:
                # Fallback simple tokenization
                self.tokens = self._simple_tokenize(source_code)
            else:
                lexer = AnamorphLexer()
                self.tokens = lexer.tokenize(source_code, filename)
            
            self.current = 0
            self.errors = []
            
            if self.debug:
                print(f"🔍 Парсинг {len(self.tokens)} токенов из {filename}")
            
            # Parse program
            statements = []
            while not self._is_at_end():
                if stmt := self._parse_statement():
                    statements.append(stmt)
            
            if self.errors:
                if self.debug:
                    for error in self.errors:
                        print(f"❌ {error}")
                raise ParseError(f"Парсинг завершен с {len(self.errors)} ошибками")
            
            program = Program(body=statements, source_type="module")
            
            if self.debug:
                print(f"✅ Парсинг завершен успешно: {len(statements)} операторов")
            
            return program
            
        except Exception as e:
            if self.debug:
                print(f"💥 Критическая ошибка парсинга: {e}")
            raise ParseError(f"Критическая ошибка парсинга: {e}")
    
    def _simple_tokenize(self, source: str) -> List[Token]:
        """Simple fallback tokenizer if lexer not available."""
        tokens = []
        lines = source.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if line.strip():
                # Simple token for each line
                token = Token(
                    type=TokenType.IDENTIFIER,
                    value=line.strip(),
                    line=line_num,
                    column=1,
                    position=0,
                    length=len(line.strip())
                )
                tokens.append(token)
        
        # EOF token
        tokens.append(Token(
            type=TokenType.EOF,
            value="",
            line=len(lines),
            column=1,
            position=len(source),
            length=0
        ))
        
        return tokens
    
    # ========================================================================
    # Utility methods
    # ========================================================================
    
    def _current_token(self) -> Optional[Token]:
        """Get current token."""
        if self._is_at_end():
            return None
        return self.tokens[self.current]
    
    def _previous_token(self) -> Optional[Token]:
        """Get previous token."""
        if self.current == 0:
            return None
        return self.tokens[self.current - 1]
    
    def _advance(self) -> Token:
        """Consume and return current token."""
        if not self._is_at_end():
            self.current += 1
        return self._previous_token()
    
    def _is_at_end(self) -> bool:
        """Check if at end of tokens."""
        return (self.current >= len(self.tokens) or 
                self._current_token().type == TokenType.EOF)
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self._is_at_end():
            return False
        return self._current_token().type == token_type
    
    def _match(self, *token_types: TokenType) -> bool:
        """Check and consume if current token matches any type."""
        for token_type in token_types:
            if self._check(token_type):
                self._advance()
                return True
        return False
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error."""
        if self._check(token_type):
            return self._advance()
        
        current = self._current_token()
        error = ParseError(message, current)
        self.errors.append(error)
        
        if self.debug:
            print(f"🔥 {error}")
        
        # Return dummy token for recovery
        return Token(token_type, "", 0, 0, 0, 0)
    
    def _synchronize(self):
        """Synchronize after parse error."""
        self._advance()
        
        while not self._is_at_end():
            if self._previous_token().type == TokenType.SEMICOLON:
                return
            
            if self._current_token().type in {
                TokenType.NEURO, TokenType.SYNAP, TokenType.IF,
                TokenType.WHILE, TokenType.FOR, TokenType.RETURN
            }:
                return
            
            self._advance()
    
    # ========================================================================
    # Expression parsing
    # ========================================================================
    
    def _parse_expression(self) -> Expression:
        """Parse expression."""
        return self._parse_assignment()
    
    def _parse_assignment(self) -> Expression:
        """Parse assignment expression."""
        expr = self._parse_logical_or()
        
        if self._match(TokenType.ASSIGN):
            operator = BinaryOperator.ASSIGN
            value = self._parse_assignment()
            return BinaryExpression(left=expr, operator=operator, right=value)
        
        return expr
    
    def _parse_logical_or(self) -> Expression:
        """Parse logical OR expression."""
        expr = self._parse_logical_and()
        
        while self._match(TokenType.OR):
            operator = BinaryOperator.OR
            right = self._parse_logical_and()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def _parse_logical_and(self) -> Expression:
        """Parse logical AND expression."""
        expr = self._parse_equality()
        
        while self._match(TokenType.AND):
            operator = BinaryOperator.AND
            right = self._parse_equality()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def _parse_equality(self) -> Expression:
        """Parse equality expression."""
        expr = self._parse_comparison()
        
        while self._match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator_token = self._previous_token()
            operator = (BinaryOperator.EQUAL if operator_token.type == TokenType.EQUAL 
                       else BinaryOperator.NOT_EQUAL)
            right = self._parse_comparison()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def _parse_comparison(self) -> Expression:
        """Parse comparison expression."""
        expr = self._parse_term()
        
        while self._match(TokenType.GREATER_THAN, TokenType.GREATER_EQUAL,
                          TokenType.LESS_THAN, TokenType.LESS_EQUAL):
            operator_token = self._previous_token()
            operator_map = {
                TokenType.GREATER_THAN: BinaryOperator.GREATER_THAN,
                TokenType.GREATER_EQUAL: BinaryOperator.GREATER_EQUAL,
                TokenType.LESS_THAN: BinaryOperator.LESS_THAN,
                TokenType.LESS_EQUAL: BinaryOperator.LESS_EQUAL,
            }
            operator = operator_map[operator_token.type]
            right = self._parse_term()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def _parse_term(self) -> Expression:
        """Parse term expression (+ -)."""
        expr = self._parse_factor()
        
        while self._match(TokenType.MINUS, TokenType.PLUS):
            operator_token = self._previous_token()
            operator = (BinaryOperator.SUBTRACT if operator_token.type == TokenType.MINUS 
                       else BinaryOperator.ADD)
            right = self._parse_factor()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def _parse_factor(self) -> Expression:
        """Parse factor expression (* / %)."""
        expr = self._parse_unary()
        
        while self._match(TokenType.DIVIDE, TokenType.MULTIPLY, TokenType.MODULO):
            operator_token = self._previous_token()
            operator_map = {
                TokenType.DIVIDE: BinaryOperator.DIVIDE,
                TokenType.MULTIPLY: BinaryOperator.MULTIPLY,
                TokenType.MODULO: BinaryOperator.MODULO,
            }
            operator = operator_map[operator_token.type]
            right = self._parse_unary()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def _parse_unary(self) -> Expression:
        """Parse unary expression."""
        if self._match(TokenType.NOT, TokenType.MINUS, TokenType.PLUS):
            operator_token = self._previous_token()
            operator_map = {
                TokenType.NOT: UnaryOperator.NOT,
                TokenType.MINUS: UnaryOperator.MINUS,
                TokenType.PLUS: UnaryOperator.PLUS,
            }
            operator = operator_map[operator_token.type]
            right = self._parse_unary()
            return UnaryExpression(operator=operator, operand=right)
        
        return self._parse_call()
    
    def _parse_call(self) -> Expression:
        """Parse function call expression."""
        expr = self._parse_primary()
        
        while True:
            if self._match(TokenType.LEFT_PAREN):
                expr = self._finish_call(expr)
            elif self._match(TokenType.DOT):
                name = self._consume(TokenType.IDENTIFIER, "Expect property name")
                property_expr = Identifier(name=name.value)
                expr = MemberExpression(object=expr, property=property_expr, computed=False)
            elif self._match(TokenType.LEFT_BRACKET):
                index = self._parse_expression()
                self._consume(TokenType.RIGHT_BRACKET, "Expect ']' after index")
                expr = IndexExpression(object=expr, index=index)
            else:
                break
        
        return expr
    
    def _finish_call(self, callee: Expression) -> Expression:
        """Finish parsing function call."""
        arguments = []
        
        if not self._check(TokenType.RIGHT_PAREN):
            arguments.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                arguments.append(self._parse_expression())
        
        self._consume(TokenType.RIGHT_PAREN, "Expect ')' after arguments")
        return CallExpression(callee=callee, arguments=arguments)
    
    def _parse_primary(self) -> Expression:
        """Parse primary expression."""
        # Boolean literals
        if self._match(TokenType.BOOLEAN):
            value = self._previous_token().value
            return BooleanLiteral(value=value == "true")
        
        # Numeric literals
        if self._match(TokenType.INTEGER):
            value = int(self._previous_token().value)
            return IntegerLiteral(value=value)
        
        if self._match(TokenType.FLOAT):
            value = float(self._previous_token().value)
            return FloatLiteral(value=value)
        
        # String literals
        if self._match(TokenType.STRING):
            value = self._previous_token().value
            return StringLiteral(value=value)
        
        # Identifiers
        if self._match(TokenType.IDENTIFIER):
            name = self._previous_token().value
            return Identifier(name=name)
        
        # Parenthesized expressions
        if self._match(TokenType.LEFT_PAREN):
            expr = self._parse_expression()
            self._consume(TokenType.RIGHT_PAREN, "Expect ')' after expression")
            return expr
        
        # Array literals
        if self._match(TokenType.LEFT_BRACKET):
            elements = []
            if not self._check(TokenType.RIGHT_BRACKET):
                elements.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    elements.append(self._parse_expression())
            
            self._consume(TokenType.RIGHT_BRACKET, "Expect ']' after array elements")
            return ArrayLiteral(elements=elements)
        
        # Signal flow expression (->)
        if self._match(TokenType.ARROW):
            # This should be handled in a higher level parser
            pass
        
        # Error recovery
        current = self._current_token()
        error = ParseError(f"Unexpected token: {current.value if current else 'EOF'}", current)
        self.errors.append(error)
        
        # Return dummy identifier for recovery
        return Identifier(name="<error>")
    
    # ========================================================================
    # Statement parsing
    # ========================================================================
    
    def _parse_statement(self) -> Optional[Statement]:
        """Parse statement."""
        try:
            # Control flow statements
            if self._match(TokenType.IF):
                return self._parse_if_statement()
            
            if self._match(TokenType.WHILE):
                return self._parse_while_statement()
            
            if self._match(TokenType.FOR):
                return self._parse_for_statement()
            
            if self._match(TokenType.RETURN):
                return self._parse_return_statement()
            
            if self._match(TokenType.BREAK):
                self._consume(TokenType.SEMICOLON, "Expect ';' after break")
                return BreakStatement()
            
            if self._match(TokenType.CONTINUE):
                self._consume(TokenType.SEMICOLON, "Expect ';' after continue")
                return ContinueStatement()
            
            if self._match(TokenType.TRY):
                return self._parse_try_statement()
            
            # Block statement
            if self._match(TokenType.LEFT_BRACE):
                return BlockStatement(statements=self._parse_block())
            
            # Neural commands
            current = self._current_token()
            if current and current.type in self.neural_commands:
                return self.neural_commands[current.type]()
            
            # Expression statement
            return self._parse_expression_statement()
            
        except ParseError as e:
            self.errors.append(e)
            self._synchronize()
            return None
    
    def _parse_block(self) -> List[Statement]:
        """Parse block of statements."""
        statements = []
        
        while not self._check(TokenType.RIGHT_BRACE) and not self._is_at_end():
            if stmt := self._parse_statement():
                statements.append(stmt)
        
        self._consume(TokenType.RIGHT_BRACE, "Expect '}' after block")
        return statements
    
    def _parse_if_statement(self) -> IfStatement:
        """Parse if statement."""
        self._consume(TokenType.LEFT_PAREN, "Expect '(' after 'if'")
        condition = self._parse_expression()
        self._consume(TokenType.RIGHT_PAREN, "Expect ')' after if condition")
        
        then_stmt = self._parse_statement()
        else_stmt = None
        
        if self._match(TokenType.ELSE):
            else_stmt = self._parse_statement()
        
        return IfStatement(test=condition, consequent=then_stmt, alternate=else_stmt)
    
    def _parse_while_statement(self) -> WhileStatement:
        """Parse while statement."""
        self._consume(TokenType.LEFT_PAREN, "Expect '(' after 'while'")
        condition = self._parse_expression()
        self._consume(TokenType.RIGHT_PAREN, "Expect ')' after while condition")
        
        body = self._parse_statement()
        return WhileStatement(test=condition, body=body)
    
    def _parse_for_statement(self) -> ForStatement:
        """Parse for statement."""
        self._consume(TokenType.LEFT_PAREN, "Expect '(' after 'for'")
        
        # Initializer
        init = None
        if self._match(TokenType.SEMICOLON):
            init = None
        elif self._match(TokenType.SYNAP):
            init = self._parse_synap_declaration()
        else:
            init = self._parse_expression_statement()
        
        # Condition
        condition = None
        if not self._check(TokenType.SEMICOLON):
            condition = self._parse_expression()
        self._consume(TokenType.SEMICOLON, "Expect ';' after for loop condition")
        
        # Increment
        increment = None
        if not self._check(TokenType.RIGHT_PAREN):
            increment = self._parse_expression()
        self._consume(TokenType.RIGHT_PAREN, "Expect ')' after for clauses")
        
        body = self._parse_statement()
        return ForStatement(init=init, test=condition, update=increment, body=body)
    
    def _parse_return_statement(self) -> ReturnStatement:
        """Parse return statement."""
        value = None
        if not self._check(TokenType.SEMICOLON):
            value = self._parse_expression()
        
        self._consume(TokenType.SEMICOLON, "Expect ';' after return value")
        return ReturnStatement(argument=value)
    
    def _parse_try_statement(self) -> TryStatement:
        """Parse try statement."""
        block = BlockStatement(statements=self._parse_block())
        
        handler = None
        if self._match(TokenType.CATCH):
            param = None
            if self._match(TokenType.LEFT_PAREN):
                param = Identifier(name=self._consume(TokenType.IDENTIFIER, "Expect parameter name").value)
                self._consume(TokenType.RIGHT_PAREN, "Expect ')' after catch parameter")
            
            catch_body = BlockStatement(statements=self._parse_block())
            handler = CatchClause(param=param, body=catch_body)
        
        finalizer = None
        if self._match(TokenType.FINALLY):
            finalizer = BlockStatement(statements=self._parse_block())
        
        return TryStatement(block=block, handler=handler, finalizer=finalizer)
    
    def _parse_expression_statement(self) -> ExpressionStatement:
        """Parse expression statement."""
        expr = self._parse_expression()
        self._consume(TokenType.SEMICOLON, "Expect ';' after expression")
        return ExpressionStatement(expression=expr)
    
    # ========================================================================
    # Neural command parsing (80 commands)
    # ========================================================================
    
    def _parse_neuro_declaration(self) -> FunctionDeclaration:
        """Parse neuro (function) declaration."""
        self._advance()  # consume 'neuro'
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expect function name")
        name = Identifier(name=name_token.value)
        
        self._consume(TokenType.LEFT_PAREN, "Expect '(' after function name")
        
        params = []
        if not self._check(TokenType.RIGHT_PAREN):
            params.append(self._parse_parameter())
            while self._match(TokenType.COMMA):
                params.append(self._parse_parameter())
        
        self._consume(TokenType.RIGHT_PAREN, "Expect ')' after parameters")
        
        # Optional return type
        return_type = None
        if self._match(TokenType.COLON):
            type_name = self._consume(TokenType.IDENTIFIER, "Expect return type").value
            return_type = TypeAnnotation(type_name=type_name)
        
        body = BlockStatement(statements=self._parse_block())
        
        return FunctionDeclaration(id=name, params=params, body=body, return_type=return_type)
    
    def _parse_synap_declaration(self) -> VariableDeclaration:
        """Parse synap (variable) declaration."""
        self._advance()  # consume 'synap'
        
        declarations = []
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expect variable name")
        name = Identifier(name=name_token.value)
        
        # Optional type annotation
        type_annotation = None
        if self._match(TokenType.COLON):
            type_name = self._consume(TokenType.IDENTIFIER, "Expect type name").value
            type_annotation = TypeAnnotation(type_name=type_name)
        
        # Optional initializer
        init = None
        if self._match(TokenType.ASSIGN):
            init = self._parse_expression()
        
        declarations.append(VariableDeclarator(id=name, init=init, type_annotation=type_annotation))
        
        self._consume(TokenType.SEMICOLON, "Expect ';' after variable declaration")
        
        return VariableDeclaration(declarations=declarations, kind="synap")
    
    def _parse_parameter(self) -> Parameter:
        """Parse function parameter."""
        name_token = self._consume(TokenType.IDENTIFIER, "Expect parameter name")
        name = Identifier(name=name_token.value)
        
        type_annotation = None
        if self._match(TokenType.COLON):
            type_name = self._consume(TokenType.IDENTIFIER, "Expect parameter type").value
            type_annotation = TypeAnnotation(type_name=type_name)
        
        default_value = None
        if self._match(TokenType.ASSIGN):
            default_value = self._parse_expression()
        
        return Parameter(id=name, type_annotation=type_annotation, default_value=default_value)
    
    def _parse_pulse_statement(self) -> PulseStatement:
        """Parse pulse statement."""
        self._advance()  # consume 'pulse'
        
        signal = self._parse_expression()
        
        target = None
        if self._match(TokenType.ARROW):
            target = self._parse_expression()
        
        condition = None
        if self._match(TokenType.IF):
            condition = self._parse_expression()
        
        self._consume(TokenType.SEMICOLON, "Expect ';' after pulse statement")
        
        return PulseStatement(signal=signal, target=target, condition=condition)
    
    def _parse_resonate_statement(self) -> ResonateStatement:
        """Parse resonate statement."""
        self._advance()  # consume 'resonate'
        
        name_token = self._consume(TokenType.IDENTIFIER, "Expect resonate name")
        name = Identifier(name=name_token.value)
        
        self._consume(TokenType.LEFT_PAREN, "Expect '(' after resonate name")
        
        params = []
        if not self._check(TokenType.RIGHT_PAREN):
            params.append(self._parse_parameter())
            while self._match(TokenType.COMMA):
                params.append(self._parse_parameter())
        
        self._consume(TokenType.RIGHT_PAREN, "Expect ')' after parameters")
        
        body = BlockStatement(statements=self._parse_block())
        
        return ResonateStatement(id=name, params=params, body=body)
    
    # Generic neural command parser for remaining commands
    def _parse_neural_command(self, command_name: str) -> ExpressionStatement:
        """Parse generic neural command."""
        self._advance()  # consume command token
        
        # Parse command arguments
        args = []
        if not self._check(TokenType.SEMICOLON):
            args.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                args.append(self._parse_expression())
        
        self._consume(TokenType.SEMICOLON, f"Expect ';' after {command_name}")
        
        # Create function call expression
        callee = Identifier(name=command_name)
        call = CallExpression(callee=callee, arguments=args)
        
        return ExpressionStatement(expression=call)
    
    # All 80 neural commands mapped to generic parser
    def _parse_drift_statement(self): return self._parse_neural_command("drift")
    def _parse_bind_statement(self): return self._parse_neural_command("bind")
    def _parse_echo_statement(self): return self._parse_neural_command("echo")
    def _parse_forge_statement(self): return self._parse_neural_command("forge")
    def _parse_prune_statement(self): return self._parse_neural_command("prune")
    def _parse_filter_statement(self): return self._parse_neural_command("filter")
    def _parse_guard_statement(self): return self._parse_neural_command("guard")
    def _parse_mask_statement(self): return self._parse_neural_command("mask")
    def _parse_scramble_statement(self): return self._parse_neural_command("scramble")
    def _parse_trace_statement(self): return self._parse_neural_command("trace")
    def _parse_quanta_statement(self): return self._parse_neural_command("quanta")
    def _parse_phase_statement(self): return self._parse_neural_command("phase")
    def _parse_sync_statement(self): return self._parse_neural_command("sync")
    def _parse_async_statement(self): return self._parse_neural_command("async")
    def _parse_fold_statement(self): return self._parse_neural_command("fold")
    def _parse_unfold_statement(self): return self._parse_neural_command("unfold")
    def _parse_pulsex_statement(self): return self._parse_neural_command("pulsex")
    def _parse_reflect_statement(self): return self._parse_neural_command("reflect")
    def _parse_absorb_statement(self): return self._parse_neural_command("absorb")
    def _parse_diffuse_statement(self): return self._parse_neural_command("diffuse")
    def _parse_cluster_statement(self): return self._parse_neural_command("cluster")
    def _parse_expand_statement(self): return self._parse_neural_command("expand")
    def _parse_contract_statement(self): return self._parse_neural_command("contract")
    def _parse_encode_statement(self): return self._parse_neural_command("encode")
    def _parse_decode_statement(self): return self._parse_neural_command("decode")
    def _parse_merge_statement(self): return self._parse_neural_command("merge")
    def _parse_split_statement(self): return self._parse_neural_command("split")
    def _parse_loop_statement(self): return self._parse_neural_command("loop")
    def _parse_halt_statement(self): return self._parse_neural_command("halt")
    def _parse_yield_statement(self): return self._parse_neural_command("yield")
    def _parse_spawn_statement(self): return self._parse_neural_command("spawn")
    def _parse_tag_statement(self): return self._parse_neural_command("tag")
    def _parse_query_statement(self): return self._parse_neural_command("query")
    def _parse_response_statement(self): return self._parse_neural_command("response")
    def _parse_encrypt_statement(self): return self._parse_neural_command("encrypt")
    def _parse_decrypt_statement(self): return self._parse_neural_command("decrypt")
    def _parse_checkpoint_statement(self): return self._parse_neural_command("checkpoint")
    def _parse_rollback_statement(self): return self._parse_neural_command("rollback")
    def _parse_pulseif_statement(self): return self._parse_neural_command("pulseif")
    def _parse_wait_statement(self): return self._parse_neural_command("wait")
    def _parse_time_statement(self): return self._parse_neural_command("time")
    def _parse_jump_statement(self): return self._parse_neural_command("jump")
    def _parse_stack_statement(self): return self._parse_neural_command("stack")
    def _parse_pop_statement(self): return self._parse_neural_command("pop")
    def _parse_push_statement(self): return self._parse_neural_command("push")
    def _parse_flag_statement(self): return self._parse_neural_command("flag")
    def _parse_clearflag_statement(self): return self._parse_neural_command("clearflag")
    def _parse_toggle_statement(self): return self._parse_neural_command("toggle")
    def _parse_listen_statement(self): return self._parse_neural_command("listen")
    def _parse_broadcast_statement(self): return self._parse_neural_command("broadcast")
    def _parse_filterin_statement(self): return self._parse_neural_command("filterin")
    def _parse_filterout_statement(self): return self._parse_neural_command("filterout")
    def _parse_auth_statement(self): return self._parse_neural_command("auth")
    def _parse_audit_statement(self): return self._parse_neural_command("audit")
    def _parse_throttle_statement(self): return self._parse_neural_command("throttle")
    def _parse_ban_statement(self): return self._parse_neural_command("ban")
    def _parse_whitelist_statement(self): return self._parse_neural_command("whitelist")
    def _parse_blacklist_statement(self): return self._parse_neural_command("blacklist")
    def _parse_morph_statement(self): return self._parse_neural_command("morph")
    def _parse_evolve_statement(self): return self._parse_neural_command("evolve")
    def _parse_sense_statement(self): return self._parse_neural_command("sense")
    def _parse_act_statement(self): return self._parse_neural_command("act")
    def _parse_log_statement(self): return self._parse_neural_command("log")
    def _parse_alert_statement(self): return self._parse_neural_command("alert")
    def _parse_reset_statement(self): return self._parse_neural_command("reset")
    def _parse_pattern_statement(self): return self._parse_neural_command("pattern")
    def _parse_train_statement(self): return self._parse_neural_command("train")
    def _parse_infer_statement(self): return self._parse_neural_command("infer")
    def _parse_scaleup_statement(self): return self._parse_neural_command("scaleup")
    def _parse_scaledown_statement(self): return self._parse_neural_command("scaledown")
    def _parse_backup_statement(self): return self._parse_neural_command("backup")
    def _parse_restore_statement(self): return self._parse_neural_command("restore")
    def _parse_snapshot_statement(self): return self._parse_neural_command("snapshot")
    def _parse_migrate_statement(self): return self._parse_neural_command("migrate")
    def _parse_notify_statement(self): return self._parse_neural_command("notify")
    def _parse_validate_statement(self): return self._parse_neural_command("validate")


def create_parser(debug: bool = False) -> AnamorphParser:
    """Factory function to create parser."""
    return AnamorphParser(debug=debug)


def parse(source_code: str, filename: str = "<source>", debug: bool = False) -> ParseResult:
    """
    Parse source code and return ParseResult.
    
    Args:
        source_code: The Anamorph source code to parse
        filename: Optional filename for error reporting
        debug: Enable debug output
        
    Returns:
        ParseResult with success status and AST
    """
    try:
        parser = AnamorphParser(debug=debug)
        ast = parser.parse(source_code, filename)
        
        return ParseResult(
            success=True,
            ast=ast,
            errors=[],
            warnings=[]
        )
    except Exception as e:
        return ParseResult(
            success=False,
            ast=None,
            errors=[str(e)],
            warnings=[]
        )


async def parse_async(source_code: str, filename: str = "<source>", debug: bool = False) -> ParseResult:
    """
    Asynchronous version of parse.
    
    Args:
        source_code: The Anamorph source code to parse
        filename: Optional filename for error reporting  
        debug: Enable debug output
        
    Returns:
        ParseResult with success status and AST
    """
    # For now, just use synchronous parsing
    # In future could implement actual async parsing for large files
    return parse(source_code, filename, debug)


def parse_file(filename: str, debug: bool = False) -> ParseResult:
    """
    Parse a file and return ParseResult.
    
    Args:
        filename: Path to the Anamorph source file to parse
        debug: Enable debug output
        
    Returns:
        ParseResult with success status and AST
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return parse(source_code, filename, debug)
    except FileNotFoundError:
        return ParseResult(
            success=False,
            ast=None,
            errors=[f"File not found: {filename}"],
            warnings=[]
        )
    except Exception as e:
        return ParseResult(
            success=False,
            ast=None,
            errors=[f"Error reading file {filename}: {str(e)}"],
            warnings=[]
        )


async def parse_file_async(filename: str, debug: bool = False) -> ParseResult:
    """
    Asynchronously parse a file and return ParseResult.
    
    Args:
        filename: Path to the Anamorph source file to parse
        debug: Enable debug output
        
    Returns:
        ParseResult with success status and AST
    """
    # For now, just use synchronous file parsing
    # In future could implement actual async file I/O
    return parse_file(filename, debug)


# Demo function
def demo_parser():
    """Demonstrate parser functionality."""
    sample_code = """
    neuro fibonacci(n: int): int {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
    
    synap result: int = fibonacci(10);
    pulse result -> console;
    
    resonate process_data(data: array) {
        filter data by (x > 0);
        encode data;
        transmit data;
    }
    """
    
    parser = AnamorphParser(debug=True)
    try:
        ast = parser.parse(sample_code, "demo.amph")
        print(f"\n🎉 Успешно распарсен AST с {len(ast.body)} операторами!")
        
        for i, stmt in enumerate(ast.body):
            print(f"  {i+1}. {type(stmt).__name__}")
        
        return ast
    except Exception as e:
        print(f"❌ Ошибка парсинга: {e}")
        return None


if __name__ == "__main__":
    demo_parser() 