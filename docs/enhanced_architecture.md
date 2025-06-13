# Расширенная архитектура системы Anamorph

## Принципы архитектуры

### Основные принципы
- **Модульность**: Четкое разделение ответственности между компонентами
- **Расширяемость**: Простое добавление новых функций через интерфейсы
- **Асинхронность**: Неблокирующая обработка сигналов и операций
- **Отказоустойчивость**: Graceful degradation при сбоях компонентов
- **Наблюдаемость**: Полная трассируемость и мониторинг

## Публичные интерфейсы компонентов

### 1. Интерфейс лексера

```python
from abc import ABC, abstractmethod
from typing import List, Iterator, Optional
from dataclasses import dataclass

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int
    start_pos: int
    end_pos: int

class ILexer(ABC):
    """Интерфейс лексического анализатора"""
    
    @abstractmethod
    def tokenize(self, source_code: str) -> List[Token]:
        """Разбивает исходный код на токены"""
        pass
    
    @abstractmethod
    def tokenize_stream(self, source_stream: Iterator[str]) -> Iterator[Token]:
        """Потоковая токенизация для больших файлов"""
        pass
    
    @abstractmethod
    def validate_syntax(self, source_code: str) -> List[LexerError]:
        """Проверяет синтаксис без полной токенизации"""
        pass
    
    @abstractmethod
    def get_keywords(self) -> List[str]:
        """Возвращает список ключевых слов языка"""
        pass

class AnamorphLexer(ILexer):
    """Реализация лексера для Anamorph"""
    
    def __init__(self, config: LexerConfig = None):
        self.config = config or LexerConfig()
        self.keywords = self._load_keywords()
        self.operators = self._load_operators()
        self.delimiters = self._load_delimiters()
    
    def tokenize(self, source_code: str) -> List[Token]:
        """Основная реализация токенизации"""
        tokens = []
        position = 0
        line = 1
        column = 1
        
        while position < len(source_code):
            # Пропуск пробелов и комментариев
            if self._is_whitespace(source_code[position]):
                position, line, column = self._skip_whitespace(source_code, position, line, column)
                continue
            
            if self._is_comment_start(source_code, position):
                position, line, column = self._skip_comment(source_code, position, line, column)
                continue
            
            # Токенизация
            token = self._next_token(source_code, position, line, column)
            tokens.append(token)
            position = token.end_pos
            column = token.column + len(token.value)
            
        return tokens
    
    async def tokenize_async(self, source_code: str) -> List[Token]:
        """Асинхронная токенизация для больших файлов"""
        import asyncio
        
        chunks = self._split_into_chunks(source_code)
        tasks = [self._tokenize_chunk(chunk) for chunk in chunks]
        
        chunk_results = await asyncio.gather(*tasks)
        return self._merge_token_chunks(chunk_results)
```

### 2. Интерфейс парсера

```python
class IASTNode(ABC):
    """Базовый интерфейс для узлов AST"""
    
    @abstractmethod
    def accept(self, visitor: 'IASTVisitor'):
        """Visitor pattern для обхода дерева"""
        pass
    
    @abstractmethod
    def get_position(self) -> Position:
        """Позиция в исходном коде"""
        pass

class IParser(ABC):
    """Интерфейс синтаксического анализатора"""
    
    @abstractmethod
    def parse(self, tokens: List[Token]) -> IASTNode:
        """Парсит токены в AST"""
        pass
    
    @abstractmethod
    def parse_expression(self, tokens: List[Token]) -> IASTNode:
        """Парсит только выражение"""
        pass
    
    @abstractmethod
    def validate_syntax(self, tokens: List[Token]) -> List[ParseError]:
        """Проверяет синтаксис без построения AST"""
        pass

class RecursiveDescentParser(IParser):
    """Реализация recursive descent парсера"""
    
    def __init__(self, lexer: ILexer, error_handler: IErrorHandler = None):
        self.lexer = lexer
        self.error_handler = error_handler or DefaultErrorHandler()
        self.tokens = []
        self.current_token_index = 0
        self.error_recovery_enabled = True
    
    def parse(self, tokens: List[Token]) -> IASTNode:
        """Главная функция парсинга"""
        self.tokens = tokens
        self.current_token_index = 0
        
        try:
            return self._parse_program()
        except ParseError as e:
            if self.error_recovery_enabled:
                return self._parse_with_recovery(e)
            else:
                raise e
    
    async def parse_async(self, tokens: List[Token]) -> IASTNode:
        """Асинхронный парсинг для больших AST"""
        import asyncio
        
        # Разбиваем парсинг на части для больших файлов
        if len(tokens) > 10000:
            return await self._parse_large_file_async(tokens)
        else:
            return self.parse(tokens)
```

### 3. Интерфейс интерпретатора

```python
class IInterpreter(ABC):
    """Интерфейс интерпретатора"""
    
    @abstractmethod
    async def execute(self, ast: IASTNode, context: ExecutionContext) -> ExecutionResult:
        """Выполняет AST асинхронно"""
        pass
    
    @abstractmethod
    def execute_sync(self, ast: IASTNode, context: ExecutionContext) -> ExecutionResult:
        """Синхронное выполнение для простых случаев"""
        pass
    
    @abstractmethod
    def create_context(self, **kwargs) -> ExecutionContext:
        """Создает контекст выполнения"""
        pass

class AsyncAnamorphInterpreter(IInterpreter):
    """Асинхронная реализация интерпретатора"""
    
    def __init__(self):
        self.neural_network = AsyncNeuralNetwork()
        self.signal_processor = AsyncSignalProcessor()
        self.command_handlers = {}
        self.execution_pool = ThreadPoolExecutor(max_workers=10)
    
    async def execute(self, ast: IASTNode, context: ExecutionContext) -> ExecutionResult:
        """Асинхронное выполнение программы"""
        try:
            # Инициализация системы
            await self._initialize_system_async(context)
            
            # Выполнение statements параллельно где возможно
            tasks = []
            for statement in ast.statements:
                if self._can_execute_parallel(statement):
                    task = asyncio.create_task(self._execute_statement_async(statement, context))
                    tasks.append(task)
                else:
                    # Ждем завершения предыдущих задач
                    if tasks:
                        await asyncio.gather(*tasks)
                        tasks = []
                    
                    await self._execute_statement_async(statement, context)
            
            # Ждем завершения оставшихся задач
            if tasks:
                await asyncio.gather(*tasks)
            
            return ExecutionResult(success=True, context=context)
            
        except Exception as e:
            return await self._handle_execution_error_async(e, context)
```

### 4. Интерфейс обработки сигналов

```python
class ISignalProcessor(ABC):
    """Интерфейс обработчика сигналов"""
    
    @abstractmethod
    async def send_signal(self, signal: Signal) -> SignalResult:
        """Отправляет сигнал асинхронно"""
        pass
    
    @abstractmethod
    async def receive_signal(self, signal: Signal) -> None:
        """Получает сигнал для обработки"""
        pass
    
    @abstractmethod
    async def route_signal(self, signal: Signal) -> List[str]:
        """Определяет маршрут сигнала"""
        pass

class AsyncSignalProcessor(ISignalProcessor):
    """Асинхронная реализация обработчика сигналов"""
    
    def __init__(self):
        self.signal_queue = asyncio.Queue(maxsize=10000)
        self.routing_table = AsyncRoutingTable()
        self.signal_handlers = {}
        self.worker_pool = []
        self.metrics = SignalMetrics()
    
    async def start(self, num_workers: int = 5):
        """Запускает пул воркеров для обработки сигналов"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._signal_worker(f"worker-{i}"))
            self.worker_pool.append(worker)
    
    async def send_signal(self, signal: Signal) -> SignalResult:
        """Отправляет сигнал в очередь обработки"""
        try:
            # Валидация сигнала
            validation_result = await self._validate_signal_async(signal)
            if not validation_result.valid:
                return SignalResult(success=False, errors=validation_result.errors)
            
            # Добавление в очередь с приоритетом
            priority = signal.metadata.get('priority', 5)
            await self.signal_queue.put((10 - priority, signal))
            
            # Обновление метрик
            self.metrics.signals_sent.increment()
            
            return SignalResult(success=True, signal_id=signal.id)
            
        except asyncio.QueueFull:
            self.metrics.signals_dropped.increment()
            return SignalResult(success=False, errors=["Signal queue is full"])
    
    async def _signal_worker(self, worker_id: str):
        """Воркер для обработки сигналов из очереди"""
        while True:
            try:
                # Получение сигнала из очереди
                priority, signal = await self.signal_queue.get()
                
                # Обработка сигнала
                start_time = time.time()
                await self._process_signal_async(signal)
                
                # Обновление метрик
                processing_time = time.time() - start_time
                self.metrics.signal_processing_time.observe(processing_time)
                self.metrics.signals_processed.increment()
                
                # Отметка о завершении
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"Signal worker {worker_id} error: {e}")
                self.metrics.signal_errors.increment()
```

## Система безопасности с sandbox

### Интерфейс безопасности

```python
class ISandbox(ABC):
    """Интерфейс песочницы"""
    
    @abstractmethod
    async def execute_in_sandbox(self, code: str, context: SandboxContext) -> SandboxResult:
        """Выполняет код в изолированной среде"""
        pass
    
    @abstractmethod
    def set_resource_limits(self, limits: ResourceLimits):
        """Устанавливает ограничения ресурсов"""
        pass
    
    @abstractmethod
    def get_allowed_operations(self) -> List[str]:
        """Возвращает список разрешенных операций"""
        pass

class DockerSandbox(ISandbox):
    """Реализация песочницы на Docker"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.allowed_operations = [
            'neuro', 'synap', 'pulse', 'log', 'trace'  # Базовые операции
        ]
        self.resource_limits = ResourceLimits(
            memory='512m',
            cpu_quota=50000,  # 50% CPU
            network=False,    # Без сетевого доступа
            filesystem_readonly=True
        )
    
    async def execute_in_sandbox(self, code: str, context: SandboxContext) -> SandboxResult:
        """Выполняет код в Docker контейнере"""
        try:
            # Создание временного файла с кодом
            code_file = self._create_temp_code_file(code)
            
            # Запуск контейнера
            container = self.docker_client.containers.run(
                image='anamorph-sandbox:latest',
                command=f'anamorph-interpreter {code_file}',
                mem_limit=self.resource_limits.memory,
                cpu_quota=self.resource_limits.cpu_quota,
                network_disabled=not self.resource_limits.network,
                read_only=self.resource_limits.filesystem_readonly,
                detach=True,
                remove=True
            )
            
            # Ожидание завершения с таймаутом
            result = container.wait(timeout=context.timeout)
            logs = container.logs().decode('utf-8')
            
            return SandboxResult(
                success=result['StatusCode'] == 0,
                output=logs,
                exit_code=result['StatusCode']
            )
            
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e)
            )
```

## Главный класс системы

```python
class AnamorphSystem:
    """Главный класс системы Anamorph"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # Инициализация компонентов
        self.lexer = self._create_lexer()
        self.parser = self._create_parser()
        self.interpreter = self._create_interpreter()
        self.signal_processor = self._create_signal_processor()
        self.security_manager = self._create_security_manager()
        
        # Состояние системы
        self.is_running = False
        self.startup_tasks = []
        self.shutdown_tasks = []
    
    async def start(self):
        """Запуск системы"""
        if self.is_running:
            return
        
        try:
            # Запуск компонентов
            await self.signal_processor.start()
            
            # Выполнение задач запуска
            for task in self.startup_tasks:
                await task()
            
            self.is_running = True
            logger.info("Anamorph system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise e
    
    async def shutdown(self):
        """Остановка системы"""
        if not self.is_running:
            return
        
        try:
            # Выполнение задач остановки
            for task in self.shutdown_tasks:
                await task()
            
            # Остановка компонентов
            await self.signal_processor.stop()
            
            self.is_running = False
            logger.info("Anamorph system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def execute_code(self, source_code: str, user_context: UserContext = None) -> ExecutionResult:
        """Выполнение кода с полным циклом обработки"""
        try:
            # 1. Лексический анализ
            tokens = await self.lexer.tokenize_async(source_code)
            
            # 2. Синтаксический анализ
            ast = await self.parser.parse_async(tokens)
            
            # 3. Безопасное выполнение
            if user_context and self.config.security_enabled:
                result = await self.security_manager.execute_secure(source_code, user_context)
            else:
                execution_context = self.interpreter.create_context()
                result = await self.interpreter.execute(ast, execution_context)
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _create_lexer(self) -> ILexer:
        """Создает лексер согласно конфигурации"""
        if self.config.lexer_type == 'default':
            return AnamorphLexer(self.config.lexer_config)
        else:
            raise ValueError(f"Unknown lexer type: {self.config.lexer_type}")
    
    def _create_parser(self) -> IParser:
        """Создает парсер согласно конфигурации"""
        if self.config.parser_type == 'recursive_descent':
            return RecursiveDescentParser(self.lexer)
        else:
            raise ValueError(f"Unknown parser type: {self.config.parser_type}")
    
    def _create_interpreter(self) -> IInterpreter:
        """Создает интерпретатор согласно конфигурации"""
        if self.config.async_execution:
            return AsyncAnamorphInterpreter()
        else:
            return SyncAnamorphInterpreter()

# Пример использования
async def main():
    # Создание и настройка системы
    config = SystemConfig(
        async_execution=True,
        security_enabled=True,
        metrics_enabled=True
    )
    
    system = AnamorphSystem(config)
    
    try:
        # Запуск системы
        await system.start()
        
        # Выполнение кода
        code = """
        neuro web_server
        neuro database
        synap web_server -> database
        
        pulse [from: web_server, to: database, data: {query: "SELECT * FROM users"}]
        """
        
        user_context = UserContext(user_id="user123", permissions=["execute_code"])
        result = await system.execute_code(code, user_context)
        
        print(f"Execution result: {result}")
        
    finally:
        # Остановка системы
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Тестирование системы

### Модульные тесты

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

class TestAnamorphLexer:
    """Тесты лексера"""
    
    def setup_method(self):
        self.lexer = AnamorphLexer()
    
    def test_tokenize_simple_command(self):
        """Тест токенизации простой команды"""
        source = "neuro web_server"
        tokens = self.lexer.tokenize(source)
        
        assert len(tokens) == 2
        assert tokens[0].type == "KEYWORD"
        assert tokens[0].value == "neuro"
        assert tokens[1].type == "IDENTIFIER"
        assert tokens[1].value == "web_server"
    
    def test_tokenize_with_comments(self):
        """Тест токенизации с комментариями"""
        source = """
        # Создание узла
        neuro web_server
        """
        tokens = self.lexer.tokenize(source)
        
        # Комментарии должны быть пропущены
        assert len(tokens) == 2
        assert tokens[0].value == "neuro"
        assert tokens[1].value == "web_server"
    
    @pytest.mark.asyncio
    async def test_async_tokenization(self):
        """Тест асинхронной токенизации"""
        source = "neuro web_server\nsynap web_server -> database"
        tokens = await self.lexer.tokenize_async(source)
        
        assert len(tokens) == 6  # neuro, web_server, synap, web_server, ->, database

class TestRecursiveDescentParser:
    """Тесты парсера"""
    
    def setup_method(self):
        self.lexer = Mock(spec=ILexer)
        self.parser = RecursiveDescentParser(self.lexer)
    
    def test_parse_neuro_command(self):
        """Тест парсинга команды neuro"""
        tokens = [
            Token("KEYWORD", "neuro", 1, 1, 0, 5),
            Token("IDENTIFIER", "web_server", 1, 7, 6, 16)
        ]
        
        ast = self.parser.parse(tokens)
        
        assert isinstance(ast, ProgramNode)
        assert len(ast.statements) == 1
        assert isinstance(ast.statements[0], NeuroCommandNode)
        assert ast.statements[0].node_name == "web_server"
    
    def test_parse_with_syntax_error(self):
        """Тест парсинга с синтаксической ошибкой"""
        tokens = [
            Token("KEYWORD", "neuro", 1, 1, 0, 5),
            # Отсутствует имя узла
        ]
        
        with pytest.raises(ParseError):
            self.parser.parse(tokens)

class TestAsyncSignalProcessor:
    """Тесты обработчика сигналов"""
    
    def setup_method(self):
        self.signal_processor = AsyncSignalProcessor()
    
    @pytest.mark.asyncio
    async def test_send_signal(self):
        """Тест отправки сигнала"""
        await self.signal_processor.start(num_workers=1)
        
        signal = Signal(
            id="test-signal-1",
            type="request",
            data={"message": "test"},
            metadata={"priority": 5}
        )
        
        result = await self.signal_processor.send_signal(signal)
        
        assert result.success
        assert result.signal_id == "test-signal-1"
    
    @pytest.mark.asyncio
    async def test_signal_queue_overflow(self):
        """Тест переполнения очереди сигналов"""
        # Создаем процессор с маленькой очередью
        processor = AsyncSignalProcessor()
        processor.signal_queue = asyncio.Queue(maxsize=1)
        
        # Заполняем очередь
        signal1 = Signal(id="1", type="request", data={}, metadata={})
        signal2 = Signal(id="2", type="request", data={}, metadata={})
        
        result1 = await processor.send_signal(signal1)
        result2 = await processor.send_signal(signal2)
        
        assert result1.success
        assert not result2.success
        assert "queue is full" in result2.errors[0].lower()

class TestDockerSandbox:
    """Тесты песочницы"""
    
    def setup_method(self):
        self.sandbox = DockerSandbox()
    
    @pytest.mark.asyncio
    async def test_execute_safe_code(self):
        """Тест выполнения безопасного кода"""
        code = """
        neuro test_node
        log [message: "Hello from sandbox", level: "info"]
        """
        
        context = SandboxContext(
            user_id="test_user",
            timeout=30
        )
        
        result = await self.sandbox.execute_in_sandbox(code, context)
        
        assert result.success
        assert "Hello from sandbox" in result.output
    
    @pytest.mark.asyncio
    async def test_execute_restricted_operation(self):
        """Тест выполнения запрещенной операции"""
        code = """
        # Попытка сетевого доступа (запрещено в sandbox)
        connect [host: "external-api.com", port: 443]
        """
        
        context = SandboxContext(
            user_id="test_user",
            timeout=30
        )
        
        result = await self.sandbox.execute_in_sandbox(code, context)
        
        assert not result.success
        # Должна быть ошибка о запрещенной операции

# Интеграционные тесты
class TestSystemIntegration:
    """Интеграционные тесты системы"""
    
    @pytest.mark.asyncio
    async def test_full_execution_cycle(self):
        """Тест полного цикла выполнения"""
        config = SystemConfig(
            async_execution=True,
            security_enabled=False,  # Отключаем для простоты
            metrics_enabled=False
        )
        
        system = AnamorphSystem(config)
        
        try:
            await system.start()
            
            code = """
            neuro web_server
            neuro database
            synap web_server -> database
            
            pulse [from: web_server, to: database, data: {query: "SELECT 1"}]
            log [message: "Integration test completed", level: "info"]
            """
            
            result = await system.execute_code(code)
            
            assert result.success
            assert result.error is None
            
        finally:
            await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Тест обработки ошибок"""
        config = SystemConfig(async_execution=True)
        system = AnamorphSystem(config)
        
        try:
            await system.start()
            
            # Код с синтаксической ошибкой
            invalid_code = """
            neuro  # Отсутствует имя узла
            """
            
            result = await system.execute_code(invalid_code)
            
            assert not result.success
            assert result.error is not None
            assert result.error_type == "ParseError"
            
        finally:
            await system.shutdown()

# Нагрузочные тесты
class TestPerformance:
    """Тесты производительности"""
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self):
        """Тест параллельной обработки сигналов"""
        processor = AsyncSignalProcessor()
        await processor.start(num_workers=5)
        
        # Создаем много сигналов
        signals = [
            Signal(
                id=f"signal-{i}",
                type="request",
                data={"index": i},
                metadata={"priority": i % 10}
            )
            for i in range(1000)
        ]
        
        # Отправляем все сигналы параллельно
        start_time = time.time()
        tasks = [processor.send_signal(signal) for signal in signals]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Проверяем результаты
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 1000
        
        # Проверяем производительность (должно быть быстрее 10 секунд)
        processing_time = end_time - start_time
        assert processing_time < 10.0
        
        print(f"Processed 1000 signals in {processing_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Тест потребления памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Создаем большую программу
        large_code = """
        """ + "\n".join([f"neuro node_{i}" for i in range(1000)]) + """
        """ + "\n".join([f"synap node_{i} -> node_{(i+1) % 1000}" for i in range(1000)])
        
        system = AnamorphSystem()
        
        try:
            await system.start()
            result = await system.execute_code(large_code)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Проверяем, что увеличение памяти разумно (меньше 100MB)
            assert memory_increase < 100 * 1024 * 1024
            
            print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
            
        finally:
            await system.shutdown()

# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Данная расширенная архитектура обеспечивает высокую модульность, производительность и безопасность системы Anamorph с возможностью легкого расширения и интеграции новых компонентов.

## 6. Интерфейсы и совместимость

### 6.1 Синхронно-асинхронные адаптеры

```python
from abc import ABC, abstractmethod
from typing import Union, Awaitable
import asyncio

class SyncAsyncAdapter:
    """Адаптер для совместимости синхронных и асинхронных интерфейсов"""
    
    @staticmethod
    def sync_to_async(func):
        """Преобразует синхронную функцию в асинхронную"""
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def async_to_sync(coro):
        """Преобразует асинхронную функцию в синхронную"""
        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro(*args, **kwargs))
            finally:
                loop.close()
        return wrapper

class UnifiedLexerInterface(ABC):
    """Унифицированный интерфейс лексера с поддержкой sync/async"""
    
    @abstractmethod
    def tokenize(self, source: str) -> List[Token]:
        """Синхронная токенизация"""
        pass
    
    @abstractmethod
    async def tokenize_async(self, source: str) -> List[Token]:
        """Асинхронная токенизация"""
        pass
    
    def get_sync_interface(self):
        """Получить синхронный интерфейс"""
        return SyncLexerAdapter(self)
    
    def get_async_interface(self):
        """Получить асинхронный интерфейс"""
        return AsyncLexerAdapter(self)
```

### 6.2 Контекстная обработка ошибок

```python
@dataclass
class ErrorContext:
    """Расширенный контекст ошибки"""
    source_code: str
    line: int
    column: int
    file_path: Optional[str]
    function_name: Optional[str]
    call_stack: List[str]
    signal_context: Optional[Dict[str, Any]]
    timestamp: datetime
    severity: ErrorSeverity

class EnhancedErrorHandler:
    """Улучшенный обработчик ошибок с контекстом"""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def handle_error(self, error: Exception, context: ErrorContext) -> RecoveryAction:
        """Обработка ошибки с полным контекстом"""
        # Логирование с контекстом
        self._log_error_with_context(error, context)
        
        # Выбор стратегии восстановления
        strategy = self._select_recovery_strategy(error, context)
        
        # Применение стратегии
        return strategy.apply(error, context)
    
    def _log_error_with_context(self, error: Exception, context: ErrorContext):
        """Детальное логирование ошибки"""
        log_entry = {
            'timestamp': context.timestamp.isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'source_location': f"{context.file_path}:{context.line}:{context.column}",
            'source_snippet': self._get_source_snippet(context),
            'call_stack': context.call_stack,
            'signal_context': context.signal_context,
            'severity': context.severity.value
        }
        logger.error("Anamorph Error", extra=log_entry)
```

## 7. Расширенная система безопасности

### 7.1 Статический анализ безопасности

```python
class SecurityAnalyzer:
    """Статический анализатор безопасности"""
    
    def __init__(self):
        self.security_rules = [
            FileSystemAccessRule(),
            NetworkAccessRule(),
            ProcessExecutionRule(),
            MemoryAccessRule(),
            CryptographicRule()
        ]
    
    def analyze_ast(self, ast: ASTNode) -> SecurityReport:
        """Анализ AST на предмет безопасности"""
        violations = []
        warnings = []
        
        for rule in self.security_rules:
            result = rule.check(ast)
            violations.extend(result.violations)
            warnings.extend(result.warnings)
        
        return SecurityReport(
            violations=violations,
            warnings=warnings,
            risk_level=self._calculate_risk_level(violations, warnings)
        )
    
    def _calculate_risk_level(self, violations, warnings) -> RiskLevel:
        """Расчет уровня риска"""
        critical_count = sum(1 for v in violations if v.severity == Severity.CRITICAL)
        high_count = sum(1 for v in violations if v.severity == Severity.HIGH)
        
        if critical_count > 0:
            return RiskLevel.CRITICAL
        elif high_count > 2:
            return RiskLevel.HIGH
        elif len(violations) > 5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

class FileSystemAccessRule(SecurityRule):
    """Правило проверки доступа к файловой системе"""
    
    def check(self, ast: ASTNode) -> RuleResult:
        violations = []
        
        # Поиск операций с файлами
        file_operations = self._find_file_operations(ast)
        
        for op in file_operations:
            if self._is_dangerous_path(op.path):
                violations.append(SecurityViolation(
                    rule="file_access",
                    severity=Severity.HIGH,
                    message=f"Dangerous file access: {op.path}",
                    location=op.location
                ))
        
        return RuleResult(violations=violations)
```

### 7.2 Расширенная песочница

```python
class EnhancedSandbox:
    """Расширенная песочница с многоуровневой защитой"""
    
    def __init__(self):
        self.docker_manager = DockerSandboxManager()
        self.resource_limiter = ResourceLimiter()
        self.network_filter = NetworkFilter()
        self.audit_logger = AuditLogger()
    
    async def execute_secure(self, code: str, context: ExecutionContext) -> ExecutionResult:
        """Безопасное выполнение кода"""
        # Статический анализ
        security_report = SecurityAnalyzer().analyze_ast(context.ast)
        if security_report.risk_level == RiskLevel.CRITICAL:
            raise SecurityException("Code rejected by static analysis")
        
        # Подготовка контейнера
        container_config = self._prepare_container_config(context)
        
        # Выполнение в песочнице
        try:
            result = await self.docker_manager.execute(
                code=code,
                config=container_config,
                timeout=context.timeout
            )
            
            # Аудит
            self.audit_logger.log_execution(context, result)
            
            return result
            
        except Exception as e:
            self.audit_logger.log_security_incident(context, e)
            raise
```

## 8. Мониторинг и метрики

### 8.1 Система метрик

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

class MetricsCollector:
    """Сборщик метрик для Anamorph"""
    
    def __init__(self):
        # Метрики лексера
        self.lexer_tokens_total = Counter('anamorph_lexer_tokens_total', 'Total tokens processed')
        self.lexer_errors_total = Counter('anamorph_lexer_errors_total', 'Total lexer errors')
        self.lexer_duration = Histogram('anamorph_lexer_duration_seconds', 'Lexer processing time')
        
        # Метрики парсера
        self.parser_ast_nodes = Counter('anamorph_parser_ast_nodes_total', 'Total AST nodes created')
        self.parser_errors_total = Counter('anamorph_parser_errors_total', 'Total parser errors')
        self.parser_duration = Histogram('anamorph_parser_duration_seconds', 'Parser processing time')
        
        # Метрики сигналов
        self.signals_processed = Counter('anamorph_signals_processed_total', 'Total signals processed', ['type', 'status'])
        self.signal_queue_size = Gauge('anamorph_signal_queue_size', 'Current signal queue size')
        self.signal_processing_duration = Histogram('anamorph_signal_processing_duration_seconds', 'Signal processing time')
        
        # Метрики безопасности
        self.security_violations = Counter('anamorph_security_violations_total', 'Security violations', ['type', 'severity'])
        self.sandbox_executions = Counter('anamorph_sandbox_executions_total', 'Sandbox executions', ['status'])
    
    def record_lexer_processing(self, tokens_count: int, duration: float, errors: int = 0):
        """Запись метрик лексера"""
        self.lexer_tokens_total.inc(tokens_count)
        self.lexer_duration.observe(duration)
        if errors > 0:
            self.lexer_errors_total.inc(errors)
    
    def record_signal_processing(self, signal_type: str, status: str, duration: float):
        """Запись метрик обработки сигналов"""
        self.signals_processed.labels(type=signal_type, status=status).inc()
        self.signal_processing_duration.observe(duration)

class HealthChecker:
    """Проверка состояния системы"""
    
    def __init__(self):
        self.components = {
            'lexer': LexerHealthCheck(),
            'parser': ParserHealthCheck(),
            'interpreter': InterpreterHealthCheck(),
            'signal_processor': SignalProcessorHealthCheck(),
            'sandbox': SandboxHealthCheck()
        }
    
    async def check_health(self) -> HealthReport:
        """Проверка состояния всех компонентов"""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, checker in self.components.items():
            try:
                result = await checker.check()
                results[name] = result
                
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}"
                )
                overall_status = HealthStatus.UNHEALTHY
        
        return HealthReport(
            overall_status=overall_status,
            components=results,
            timestamp=datetime.utcnow()
        )
```

### 8.2 Централизованное логирование

```python
import structlog
from pythonjsonlogger import jsonlogger

class CentralizedLogger:
    """Централизованная система логирования"""
    
    def __init__(self):
        # Настройка структурированного логирования
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
    
    def log_signal_processing(self, signal_id: str, signal_type: str, status: str, duration: float, metadata: Dict):
        """Логирование обработки сигнала"""
        self.logger.info(
            "signal_processed",
            signal_id=signal_id,
            signal_type=signal_type,
            status=status,
            duration_ms=duration * 1000,
            metadata=metadata
        )
    
    def log_security_event(self, event_type: str, severity: str, details: Dict):
        """Логирование событий безопасности"""
        self.logger.warning(
            "security_event",
            event_type=event_type,
            severity=severity,
            details=details
        )
```

## 9. Расширенное тестирование

### 9.1 Fuzz-тестирование

```python
import hypothesis
from hypothesis import strategies as st
import string

class FuzzTester:
    """Fuzz-тестирование для лексера и парсера"""
    
    @hypothesis.given(st.text(alphabet=string.printable, min_size=1, max_size=1000))
    def test_lexer_robustness(self, random_input: str):
        """Тест устойчивости лексера к случайному вводу"""
        lexer = AnamorphLexer()
        
        try:
            tokens = lexer.tokenize(random_input)
            # Лексер не должен падать, даже на некорректном вводе
            assert isinstance(tokens, list)
        except LexerException as e:
            # Ожидаемые исключения допустимы
            assert e.error_code in EXPECTED_LEXER_ERRORS
        except Exception as e:
            # Неожиданные исключения - это баг
            pytest.fail(f"Unexpected exception in lexer: {e}")
    
    @hypothesis.given(
        st.lists(
            st.one_of(
                st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),  # identifiers
                st.sampled_from(['neuro', 'synap', 'pulse', 'resonate']),  # keywords
                st.text(alphabet=string.digits, min_size=1, max_size=10),  # numbers
                st.sampled_from(['{', '}', '(', ')', ';', ','])  # delimiters
            ),
            min_size=1,
            max_size=100
        )
    )
    def test_parser_robustness(self, token_sequence: List[str]):
        """Тест устойчивости парсера к случайным последовательностям токенов"""
        source = ' '.join(token_sequence)
        lexer = AnamorphLexer()
        parser = AnamorphParser()
        
        try:
            tokens = lexer.tokenize(source)
            ast = parser.parse(tokens)
            # Парсер должен либо создать AST, либо выбросить корректное исключение
            if ast is not None:
                assert isinstance(ast, ASTNode)
        except (LexerException, ParserException) as e:
            # Ожидаемые исключения допустимы
            assert e.error_code in EXPECTED_PARSER_ERRORS
        except Exception as e:
            pytest.fail(f"Unexpected exception in parser: {e}")

class LoadTester:
    """Нагрузочное тестирование"""
    
    async def test_signal_processing_load(self):
        """Тест нагрузки на обработку сигналов"""
        processor = SignalProcessor()
        
        # Создание большого количества сигналов
        signals = [
            Signal(
                id=f"test_signal_{i}",
                type=SignalType.ASYNC,
                payload={"data": f"test_data_{i}"},
                priority=random.randint(1, 10)
            )
            for i in range(10000)
        ]
        
        start_time = time.time()
        
        # Параллельная отправка сигналов
        tasks = [processor.process_signal(signal) for signal in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Проверка результатов
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        assert successful > 0.95 * len(signals), f"Too many failures: {failed}/{len(signals)}"
        assert duration < 60, f"Processing took too long: {duration}s"
        
        # Проверка метрик
        metrics = processor.get_metrics()
        assert metrics.throughput > 100, f"Low throughput: {metrics.throughput} signals/sec"
```

## 10. Система плагинов и расширений

### 10.1 Plugin API

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class PluginInterface(ABC):
    """Базовый интерфейс для плагинов"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Имя плагина"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Версия плагина"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Инициализация плагина"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Очистка ресурсов плагина"""
        pass

class CommandPlugin(PluginInterface):
    """Плагин для добавления новых команд"""
    
    @abstractmethod
    def get_commands(self) -> Dict[str, CommandDefinition]:
        """Получить определения команд"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: str, args: List[Any], context: ExecutionContext) -> Any:
        """Выполнить команду"""
        pass

class TypePlugin(PluginInterface):
    """Плагин для добавления новых типов данных"""
    
    @abstractmethod
    def get_types(self) -> Dict[str, TypeDefinition]:
        """Получить определения типов"""
        pass
    
    @abstractmethod
    def validate_value(self, type_name: str, value: Any) -> bool:
        """Валидация значения типа"""
        pass

class SignalPlugin(PluginInterface):
    """Плагин для добавления новых типов сигналов"""
    
    @abstractmethod
    def get_signal_types(self) -> Dict[str, SignalTypeDefinition]:
        """Получить определения типов сигналов"""
        pass
    
    @abstractmethod
    async def process_signal(self, signal: Signal) -> SignalResult:
        """Обработать сигнал"""
        pass

class PluginManager:
    """Менеджер плагинов"""
    
    def __init__(self):
        self.plugins: Dict[str, PluginInterface] = {}
        self.command_plugins: Dict[str, CommandPlugin] = {}
        self.type_plugins: Dict[str, TypePlugin] = {}
        self.signal_plugins: Dict[str, SignalPlugin] = {}
    
    def load_plugin(self, plugin_path: str) -> None:
        """Загрузка плагина"""
        plugin = self._import_plugin(plugin_path)
        
        # Валидация плагина
        if not self._validate_plugin(plugin):
            raise PluginValidationError(f"Invalid plugin: {plugin_path}")
        
        # Регистрация плагина
        self.plugins[plugin.name] = plugin
        
        if isinstance(plugin, CommandPlugin):
            self.command_plugins[plugin.name] = plugin
        elif isinstance(plugin, TypePlugin):
            self.type_plugins[plugin.name] = plugin
        elif isinstance(plugin, SignalPlugin):
            self.signal_plugins[plugin.name] = plugin
        
        # Инициализация
        plugin.initialize({})
    
    def get_available_commands(self) -> Dict[str, CommandDefinition]:
        """Получить все доступные команды (встроенные + плагины)"""
        commands = BUILTIN_COMMANDS.copy()
        
        for plugin in self.command_plugins.values():
            plugin_commands = plugin.get_commands()
            commands.update(plugin_commands)
        
        return commands
    
    async def execute_plugin_command(self, command: str, args: List[Any], context: ExecutionContext) -> Any:
        """Выполнить команду плагина"""
        for plugin in self.command_plugins.values():
            if command in plugin.get_commands():
                return await plugin.execute_command(command, args, context)
        
        raise CommandNotFoundError(f"Plugin command not found: {command}")
```

### 10.2 REST API для управления

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Anamorph Management API", version="1.0.0")
security = HTTPBearer()

class ExecuteCodeRequest(BaseModel):
    code: str
    context: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = 30

class SignalRequest(BaseModel):
    type: str
    payload: Dict[str, Any]
    priority: Optional[int] = 5
    target: Optional[str] = None

class PluginInstallRequest(BaseModel):
    plugin_url: str
    config: Optional[Dict[str, Any]] = None

@app.post("/api/v1/execute")
async def execute_code(request: ExecuteCodeRequest, token: str = Depends(security)):
    """Выполнение Anamorph кода"""
    try:
        interpreter = get_interpreter()
        result = await interpreter.execute_async(
            code=request.code,
            context=request.context or {},
            timeout=request.timeout
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/signals/send")
async def send_signal(request: SignalRequest, token: str = Depends(security)):
    """Отправка сигнала"""
    try:
        processor = get_signal_processor()
        signal = Signal(
            type=request.type,
            payload=request.payload,
            priority=request.priority,
            target=request.target
        )
        result = await processor.process_signal(signal)
        return {"status": "success", "signal_id": result.signal_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Проверка состояния системы"""
    health_checker = get_health_checker()
    report = await health_checker.check_health()
    
    status_code = 200 if report.overall_status == HealthStatus.HEALTHY else 503
    return Response(
        content=report.to_json(),
        status_code=status_code,
        media_type="application/json"
    )

@app.get("/api/v1/metrics")
async def get_metrics():
    """Получение метрик системы"""
    collector = get_metrics_collector()
    return collector.get_all_metrics()

@app.post("/api/v1/plugins/install")
async def install_plugin(request: PluginInstallRequest, token: str = Depends(security)):
    """Установка плагина"""
    try:
        plugin_manager = get_plugin_manager()
        plugin_manager.install_plugin(request.plugin_url, request.config)
        return {"status": "success", "message": "Plugin installed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/plugins")
async def list_plugins():
    """Список установленных плагинов"""
    plugin_manager = get_plugin_manager()
    plugins = plugin_manager.get_installed_plugins()
    return {"plugins": [{"name": p.name, "version": p.version} for p in plugins]}
```

## 11. Очереди повторных попыток для сигналов

```python
class SignalRetryQueue:
    """Очередь повторных попыток для сигналов"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_queue: asyncio.Queue = asyncio.Queue()
        self.dead_letter_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """Запуск обработчика очереди повторов"""
        self.running = True
        asyncio.create_task(self._process_retry_queue())
    
    async def stop(self):
        """Остановка обработчика"""
        self.running = False
    
    async def add_failed_signal(self, signal: Signal, error: Exception):
        """Добавление неудачного сигнала в очередь повторов"""
        retry_info = RetryInfo(
            signal=signal,
            error=error,
            attempt_count=getattr(signal, 'retry_count', 0) + 1,
            next_retry_time=time.time() + self.retry_delay
        )
        
        if retry_info.attempt_count <= self.max_retries:
            await self.retry_queue.put(retry_info)
        else:
            await self.dead_letter_queue.put(retry_info)
    
    async def _process_retry_queue(self):
        """Обработка очереди повторов"""
        while self.running:
            try:
                retry_info = await asyncio.wait_for(
                    self.retry_queue.get(), 
                    timeout=1.0
                )
                
                # Ждем время следующей попытки
                wait_time = retry_info.next_retry_time - time.time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Повторная попытка обработки
                signal = retry_info.signal
                signal.retry_count = retry_info.attempt_count
                
                processor = get_signal_processor()
                await processor.process_signal(signal)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                # Если повтор тоже неудачен, добавляем обратно в очередь
                await self.add_failed_signal(retry_info.signal, e)

@dataclass
class RetryInfo:
    signal: Signal
    error: Exception
    attempt_count: int
    next_retry_time: float
```
