# Архитектура системы Anamorph

## Обзор архитектуры

Система Anamorph построена по модульному принципу с четким разделением ответственности между компонентами. Основные принципы:

- **Модульность**: Каждый компонент выполняет четко определенную функцию
- **Расширяемость**: Простое добавление новых команд и функций
- **Безопасность**: Встроенная защита на всех уровнях
- **Производительность**: Оптимизированная обработка сигналов и параллельная обработка

## Компоненты системы

### 1. Лексический анализатор (Lexer)

Первый этап обработки кода - разбиение на токены.

#### Особенности лексера:
- **Case-insensitive ключевые слова**: `neuro`, `NEURO`, `Neuro` - все эквивалентны
- **Case-sensitive идентификаторы**: `myNode` ≠ `MyNode`
- **Автоматическое удаление пробелов и комментариев**
- **Поддержка Unicode-строк**

```python
class AnamorphLexer:
    def __init__(self):
        self.keywords = {
            # Case-insensitive keywords
            'neuro': TokenType.NEURO,
            'synap': TokenType.SYNAP, 
            'pulse': TokenType.PULSE,
            # ... все 80 команд
        }
        self.operators = ['->', '==', '!=', '<=', '>=', '&&', '||']
        self.delimiters = ['(', ')', '[', ']', '{', '}', ',', ';', ':']
    
    def tokenize(self, source_code: str) -> List[Token]:
        """Разбивает исходный код на токены"""
        tokens = []
        position = 0
        line = 1
        column = 1
        
        while position < len(source_code):
            # Пропуск пробелов
            if self.is_whitespace(source_code[position]):
                position, line, column = self.skip_whitespace(source_code, position, line, column)
                continue
            
            # Пропуск комментариев
            if self.is_comment_start(source_code, position):
                position, line, column = self.skip_comment(source_code, position, line, column)
                continue
            
            # Разпознавание токенов
            token = self.next_token(source_code, position, line, column)
            tokens.append(token)
            position = token.end_position
            column = token.end_column
            
        return tokens
    
    def normalize_keyword(self, word: str) -> str:
        """Нормализация ключевых слов к нижнему регистру"""
        return word.lower()
    
    def is_keyword(self, word: str) -> bool:
        """Проверка, является ли слово ключевым"""
        return self.normalize_keyword(word) in self.keywords
```

#### Типы токенов:
```python
class TokenType(Enum):
    # Ключевые слова
    NEURO = "NEURO"
    SYNAP = "SYNAP" 
    PULSE = "PULSE"
    
    # Литералы
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    
    # Идентификаторы
    IDENTIFIER = "IDENTIFIER"
    
    # Операторы
    ARROW = "ARROW"          # ->
    EQUALS = "EQUALS"        # ==
    NOT_EQUALS = "NOT_EQUALS" # !=
    
    # Разделители
    LPAREN = "LPAREN"        # (
    RPAREN = "RPAREN"        # )
    LBRACKET = "LBRACKET"    # [
    RBRACKET = "RBRACKET"    # ]
    
    # Специальные
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    COMMENT = "COMMENT"
```

### 2. Синтаксический анализатор (Parser)

Преобразует поток токенов в абстрактное синтаксическое дерево (AST).

#### Архитектура парсера:
- **Recursive Descent Parser** - простота реализации и отладки
- **Lookahead = 1** - эффективный анализ без backtracking
- **Встроенная обработка ошибок** с recovery стратегиями

```python
class AnamorphParser:
    def __init__(self, lexer: AnamorphLexer):
        self.lexer = lexer
        self.tokens = []
        self.current_token_index = 0
        self.error_recovery_enabled = True
        self.syntax_errors = []
    
    def parse(self, source_code: str) -> ProgramNode:
        """Главная функция парсинга"""
        try:
            self.tokens = self.lexer.tokenize(source_code)
            return self.parse_program()
        except ParseError as e:
            if self.error_recovery_enabled:
                return self.parse_with_recovery()
            else:
                raise e
    
    def parse_program(self) -> ProgramNode:
        """program = { statement }"""
        statements = []
        
        while not self.is_at_end():
            try:
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
            except ParseError as e:
                self.handle_parse_error(e)
                self.synchronize()  # Error recovery
                
        return ProgramNode(statements)
    
    def parse_statement(self) -> StatementNode:
        """Разбор отдельного statement"""
        if self.match(TokenType.NEURO):
            return self.parse_node_declaration()
        elif self.match(TokenType.SYNAP):
            return self.parse_synapse_declaration()
        elif self.is_command_keyword():
            return self.parse_command_statement()
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        # ... другие типы statement
        else:
            raise ParseError(f"Unexpected token: {self.peek().type}")
```

#### Обработка ошибок в парсере:

```python
class ErrorRecoveryStrategy:
    """Стратегии восстановления после ошибок парсинга"""
    
    def panic_mode_recovery(self, parser: AnamorphParser):
        """Panic Mode - пропуск токенов до синхронизирующего символа"""
        sync_tokens = [TokenType.NEWLINE, TokenType.END, TokenType.IF, TokenType.NEURO]
        
        while not parser.is_at_end():
            if parser.peek().type in sync_tokens:
                break
            parser.advance()
    
    def phrase_level_recovery(self, parser: AnamorphParser, error: ParseError):
        """Phrase Level - локальные исправления"""
        current_token = parser.peek()
        
        # Попытка исправления частых ошибок
        if error.error_type == "MISSING_ARROW":
            # Вставляем пропущенную стрелку
            parser.insert_token(Token(TokenType.ARROW, "->"))
            return True
        elif error.error_type == "MISSING_END":
            # Вставляем пропущенный 'end'
            parser.insert_token(Token(TokenType.END, "end"))
            return True
            
        return False
    
    def error_productions(self, parser: AnamorphParser):
        """Error Productions - специальные правила для ошибок"""
        # Правила для частых синтаксических ошибок
        error_rules = {
            "incomplete_node_declaration": self.handle_incomplete_node,
            "missing_synapse_arrow": self.handle_missing_arrow,
            "unterminated_command": self.handle_unterminated_command
        }
        
        for rule_name, handler in error_rules.items():
            if handler(parser):
                return True
        return False
```

### 3. Абстрактное синтаксическое дерево (AST)

Представление программы в виде дерева объектов.

```python
class ASTNode:
    """Базовый класс для всех узлов AST"""
    def __init__(self, position: Position):
        self.position = position
        self.metadata = {}
    
    def accept(self, visitor):
        """Visitor pattern для обхода дерева"""
        raise NotImplementedError

class ProgramNode(ASTNode):
    def __init__(self, statements: List[StatementNode]):
        super().__init__(None)
        self.statements = statements

class NodeDeclarationNode(ASTNode):
    def __init__(self, name: str, node_type: str = "basic"):
        super().__init__(None)
        self.name = name
        self.node_type = node_type
        self.properties = {}

class SynapseDeclarationNode(ASTNode):
    def __init__(self, source: str, target: str, properties: dict = None):
        super().__init__(None)
        self.source = source
        self.target = target
        self.properties = properties or {}

class CommandNode(ASTNode):
    def __init__(self, command: str, arguments: List[ArgumentNode]):
        super().__init__(None)
        self.command = command
        self.arguments = arguments

class SignalNode(ASTNode):
    def __init__(self, signal_type: str, data: dict, metadata: dict, routing: List[dict]):
        super().__init__(None)
        self.signal_type = signal_type
        self.data = data
        self.metadata = metadata
        self.routing = routing
```

### 4. Интерпретатор с расширенной обработкой сигналов

Выполняет код, работая с сигналами и узлами сети.

```python
class AnamorphInterpreter:
    def __init__(self):
        self.neural_network = NeuralNetwork()
        self.signal_processor = SignalProcessor()
        self.error_handler = ErrorHandler()
        self.security_manager = SecurityManager()
        self.performance_monitor = PerformanceMonitor()
    
    def execute(self, ast: ProgramNode, context: ExecutionContext = None):
        """Выполнение программы"""
        context = context or ExecutionContext()
        
        try:
            # Инициализация системы
            self.initialize_system(context)
            
            # Выполнение statements
            for statement in ast.statements:
                self.execute_statement(statement, context)
                
        except Exception as e:
            self.error_handler.handle_execution_error(e, context)
    
    def execute_statement(self, stmt: StatementNode, context: ExecutionContext):
        """Выполнение отдельного statement"""
        if isinstance(stmt, NodeDeclarationNode):
            self.create_node(stmt, context)
        elif isinstance(stmt, SynapseDeclarationNode):
            self.create_synapse(stmt, context)
        elif isinstance(stmt, CommandNode):
            self.execute_command(stmt, context)
        elif isinstance(stmt, SignalNode):
            self.process_signal(stmt, context)
    
    def process_signal(self, signal_node: SignalNode, context: ExecutionContext):
        """Обработка сигнала"""
        signal = Signal(
            data=signal_node.data,
            metadata=signal_node.metadata,
            routing=signal_node.routing
        )
        
        # Валидация сигнала
        self.signal_processor.validate_signal(signal)
        
        # Обработка сигнала согласно маршрутизации
        self.signal_processor.route_and_process(signal, context)
```

### 5. Система сигналов (расширенная)

Центральная часть архитектуры - обработка сигналов между узлами.

```python
class SignalProcessor:
    def __init__(self):
        self.signal_queue = PriorityQueue()
        self.routing_table = RoutingTable()
        self.signal_validators = {}
        self.signal_transformers = {}
    
    def route_and_process(self, signal: Signal, context: ExecutionContext):
        """Маршрутизация и обработка сигнала"""
        # Определение маршрута
        route = self.routing_table.find_route(
            signal.source_node, 
            signal.target_node,
            signal.metadata
        )
        
        if not route:
            raise RoutingError(f"No route found for signal {signal.id}")
        
        # Последовательная передача по маршруту
        for hop in route:
            try:
                self.deliver_to_node(signal, hop, context)
            except DeliveryError as e:
                # Попытка альтернативного маршрута
                alt_route = self.routing_table.find_alternative_route(
                    hop, signal.target_node, signal.metadata
                )
                if alt_route:
                    self.deliver_to_node(signal, alt_route[0], context)
                else:
                    raise e
    
    def deliver_to_node(self, signal: Signal, node_id: str, context: ExecutionContext):
        """Доставка сигнала в конкретный узел"""
        node = context.neural_network.get_node(node_id)
        
        if not node:
            raise NodeNotFoundError(f"Node {node_id} not found")
        
        if not node.is_active():
            # Постановка в очередь для неактивного узла
            self.queue_signal_for_node(signal, node_id)
            return
        
        # Проверка перегрузки узла
        if node.is_overloaded():
            if signal.metadata.get('priority', 5) >= 8:
                # Высокий приоритет - форсированная доставка
                node.force_receive_signal(signal)
            else:
                # Обычный приоритет - троттлинг
                self.throttle_signal(signal, node_id)
                return
        
        # Нормальная доставка
        node.receive_signal(signal)
```

### 6. Система типов (расширенная)

Статическая проверка типов и type inference.

```python
class TypeSystem:
    def __init__(self):
        self.basic_types = {
            'string', 'number', 'boolean', 'array', 'object'
        }
        self.neural_types = {
            'neuro', 'synap', 'signal', 'network'
        }
        self.complex_types = {}  # Generic, Union, Function types
        
    def check_program(self, ast: ProgramNode) -> List[TypeError]:
        """Проверка типов всей программы"""
        type_checker = TypeChecker(self)
        return type_checker.check(ast)
    
    def infer_type(self, expression: ExpressionNode, context: TypeContext) -> Type:
        """Вывод типа выражения"""
        if isinstance(expression, LiteralNode):
            return self.get_literal_type(expression)
        elif isinstance(expression, IdentifierNode):
            return context.get_variable_type(expression.name)
        elif isinstance(expression, FunctionCallNode):
            return self.infer_function_return_type(expression, context)
        # ... другие типы выражений
    
    def register_generic_type(self, name: str, type_parameters: List[str]):
        """Регистрация generic типа"""
        self.complex_types[name] = GenericType(name, type_parameters)
    
    def register_union_type(self, name: str, member_types: List[Type]):
        """Регистрация union типа"""
        self.complex_types[name] = UnionType(name, member_types)

class TypeChecker:
    def __init__(self, type_system: TypeSystem):
        self.type_system = type_system
        self.errors = []
        self.context = TypeContext()
    
    def check_signal_type(self, signal_node: SignalNode) -> List[TypeError]:
        """Проверка типов сигнала"""
        errors = []
        
        # Проверка типа данных
        data_type = self.infer_type(signal_node.data)
        if not self.is_json_serializable(data_type):
            errors.append(TypeError(
                f"Signal data must be JSON-serializable, got {data_type}",
                signal_node.position
            ))
        
        # Проверка метаданных
        metadata_type = self.infer_type(signal_node.metadata)
        if not isinstance(metadata_type, ObjectType):
            errors.append(TypeError(
                f"Signal metadata must be object type, got {metadata_type}",
                signal_node.position
            ))
        
        return errors
```

### 7. Обработчик ошибок (расширенный)

Комплексная система обработки ошибок на всех уровнях.

```python
class ErrorHandler:
    def __init__(self):
        self.error_strategies = {
            'syntax_error': SyntaxErrorStrategy(),
            'type_error': TypeErrorStrategy(),
            'runtime_error': RuntimeErrorStrategy(),
            'signal_error': SignalErrorStrategy()
        }
        self.error_log = []
        self.recovery_enabled = True
    
    def handle_execution_error(self, error: Exception, context: ExecutionContext):
        """Обработка ошибки времени выполнения"""
        error_type = self.classify_error(error)
        strategy = self.error_strategies.get(error_type)
        
        if strategy and self.recovery_enabled:
            try:
                recovery_action = strategy.handle(error, context)
                self.execute_recovery_action(recovery_action, context)
            except RecoveryFailedException:
                self.escalate_error(error, context)
        else:
            self.escalate_error(error, context)
    
    def handle_signal_error(self, signal: Signal, error: Exception, context: ExecutionContext):
        """Специализированная обработка ошибок сигналов"""
        error_metadata = {
            'signal_id': signal.id,
            'source_node': signal.source_node,
            'target_node': signal.target_node,
            'error_type': type(error).__name__,
            'timestamp': time.time()
        }
        
        # Логирование с контекстом
        self.log_error(error, error_metadata)
        
        # Попытка повторной отправки для некритичных ошибок
        if isinstance(error, (NetworkError, TemporaryNodeError)):
            if signal.metadata.get('retry_count', 0) < 3:
                signal.metadata['retry_count'] += 1
                context.signal_processor.retry_signal(signal, delay=5)
                return
        
        # Отправка error signal источнику
        error_signal = Signal(
            data={'error': str(error), 'original_signal_id': signal.id},
            source_node='error_handler',
            target_node=signal.source_node,
            metadata={'signal_type': 'error_notification', 'priority': 8}
        )
        context.signal_processor.send_signal(error_signal)
```

### 8. Система безопасности

Многоуровневая защита системы.

```python
class SecurityManager:
    def __init__(self):
        self.sandbox = Sandbox()
        self.access_control = AccessControlManager()
        self.encryption = EncryptionManager()
        self.audit_log = AuditLogger()
        
    def validate_node_creation(self, node_name: str, context: ExecutionContext) -> bool:
        """Проверка разрешений на создание узла"""
        user = context.get_current_user()
        
        # Проверка квот пользователя
        if not self.check_node_quota(user, node_name):
            raise SecurityError("Node creation quota exceeded")
        
        # Проверка имени узла на безопасность
        if not self.is_safe_node_name(node_name):
            raise SecurityError(f"Unsafe node name: {node_name}")
        
        # Аудит создания узла
        self.audit_log.log_node_creation(user, node_name)
        
        return True
    
    def validate_signal_transmission(self, signal: Signal, context: ExecutionContext) -> bool:
        """Проверка безопасности передачи сигнала"""
        # Проверка разрешений доступа
        if not self.access_control.can_send_signal(
            signal.source_node, 
            signal.target_node, 
            signal.metadata.get('signal_type')
        ):
            raise SecurityError("Signal transmission not authorized")
        
        # Сканирование содержимого на вредоносный код
        if self.contains_malicious_content(signal.data):
            self.audit_log.log_security_threat(signal, "malicious_content_detected")
            raise SecurityError("Malicious content detected in signal")
        
        # Шифрование чувствительных данных
        if self.contains_sensitive_data(signal.data):
            signal.data = self.encryption.encrypt_sensitive_fields(
                signal.data, 
                self.get_encryption_key(signal.target_node)
            )
        
        return True
```

## Интеграция компонентов

### Главный класс интерпретатора:

```python
class AnamorphSystem:
    def __init__(self):
        self.lexer = AnamorphLexer()
        self.parser = AnamorphParser(self.lexer)
        self.interpreter = AnamorphInterpreter()
        self.type_system = TypeSystem()
        self.error_handler = ErrorHandler()
        self.security_manager = SecurityManager()
        
    def execute_file(self, filename: str) -> ExecutionResult:
        """Выполнение файла с кодом Anamorph"""
        try:
            # Чтение файла
            with open(filename, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            return self.execute_code(source_code, filename)
            
        except IOError as e:
            return ExecutionResult(
                success=False,
                error=f"Failed to read file {filename}: {e}"
            )
    
    def execute_code(self, source_code: str, filename: str = "<string>") -> ExecutionResult:
        """Выполнение кода из строки"""
        context = ExecutionContext(filename=filename)
        
        try:
            # 1. Лексический анализ
            tokens = self.lexer.tokenize(source_code)
            
            # 2. Синтаксический анализ
            ast = self.parser.parse_tokens(tokens)
            
            # 3. Проверка типов
            type_errors = self.type_system.check_program(ast)
            if type_errors:
                return ExecutionResult(
                    success=False,
                    type_errors=type_errors
                )
            
            # 4. Проверка безопасности
            security_issues = self.security_manager.analyze_program(ast)
            if security_issues:
                return ExecutionResult(
                    success=False,
                    security_issues=security_issues
                )
            
            # 5. Выполнение
            result = self.interpreter.execute(ast, context)
            
            return ExecutionResult(
                success=True,
                result=result,
                performance_stats=context.performance_monitor.get_stats()
            )
            
        except Exception as e:
            return self.error_handler.handle_system_error(e, context)
```

Данная архитектура обеспечивает надежную, безопасную и производительную систему для выполнения программ на языке Anamorph с учетом всех ваших рекомендаций. 