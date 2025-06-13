# Модель сигнальной обработки Anamorph

## Концепция сигналов

Сигналы в Anamorph - это основные единицы коммуникации между нейронами. Они представляют собой структурированные сообщения, которые передаются по синапсам и обрабатываются узлами.

## Структура сигнала

### Базовая структура
```anamorph
signal RequestSignal {
    data: {
        method: "GET",
        url: "/api/users",
        headers: {"Authorization": "Bearer token123"}
    },
    metadata: {
        priority: 5,
        timestamp: 1704067200,
        source_node: "web_gateway",
        correlation_id: "req-12345"
    },
    routing: [
        {from: "web_gateway", to: "auth_filter", condition: "always"},
        {from: "auth_filter", to: "rate_limiter", condition: "auth_success == true"},
        {from: "rate_limiter", to: "request_processor", condition: "rate_limit_ok == true"}
    ]
}
```

### Компоненты сигнала

#### 1. Data (Данные)
- **Назначение**: Полезная нагрузка сигнала
- **Тип**: Любой JSON-совместимый объект
- **Ограничения**: Максимальный размер 10 МБ
- **Валидация**: Согласно схеме сигнала

#### 2. Metadata (Метаданные)
- **priority**: Приоритет сигнала (0-10, где 10 - критический)
- **timestamp**: Unix timestamp создания сигнала
- **source_node**: Идентификатор узла-отправителя
- **correlation_id**: Уникальный ID для трассировки
- **ttl**: Время жизни сигнала в секундах
- **retry_count**: Количество попыток доставки

#### 3. Routing (Маршрутизация)
- **from/to**: Узлы источника и назначения
- **condition**: Условие для передачи сигнала
- **transform**: Преобразование данных при передаче
- **timeout**: Таймаут для передачи

## Типы сигналов

### 1. Синхронные сигналы (Request-Response)
```anamorph
# Отправка запроса и ожидание ответа
neuro client
neuro server

synap client -> server

# Синхронная отправка
pulse [from: client, to: server, data: request, wait_response: true, timeout: 30]

# Автоматический ответ
on "request_received" from client
    # Обработка запроса
    process_request(signal.data)
    
    # Отправка ответа
    response [to: signal.source_node, data: result]
end
```

### 2. Асинхронные сигналы (Fire-and-Forget)
```anamorph
# Асинхронная отправка без ожидания ответа
pulse [from: logger, to: log_aggregator, data: log_entry, async: true]

# Широковещательная рассылка
broadcast [signal: notification_signal, exclude: ["maintenance_nodes"]]
```

### 3. Приоритетные сигналы
```anamorph
# Критический сигнал с высоким приоритетом
signal AlertSignal {
    data: {
        level: "CRITICAL",
        message: "System overload detected",
        affected_nodes: ["web_server_1", "web_server_2"]
    },
    metadata: {
        priority: 10,        # Максимальный приоритет
        source_node: "monitor",
        immediate: true      # Прерывание текущих операций
    }
}

pulse [signal: AlertSignal, priority_override: true]
```

### 4. Потоковые сигналы
```anamorph
# Создание потока данных
signal DataStream {
    data: {
        stream_id: "metrics_stream_001",
        chunk_size: 1024,
        total_chunks: 100
    },
    metadata: {
        stream_type: "chunked",
        ordering: "sequential"
    }
}

# Обработка потока
on "data_chunk" from data_source
    if signal.metadata.chunk_index == last_chunk then
        # Завершение потока
        finalize_stream(signal.data.stream_id)
    end
end
```

## Жизненный цикл сигнала

### 1. Создание сигнала
```python
class Signal:
    def __init__(self, data, source_node, target_node, **metadata):
        self.id = generate_uuid()
        self.data = data
        self.source_node = source_node
        self.target_node = target_node
        self.metadata = {
            'timestamp': time.time(),
            'priority': metadata.get('priority', 5),
            'ttl': metadata.get('ttl', 300),  # 5 минут по умолчанию
            'retry_count': 0,
            **metadata
        }
        self.routing_history = []
        self.state = SignalState.CREATED
```

### 2. Валидация сигнала
```python
def validate_signal(signal: Signal) -> bool:
    # Проверка размера
    if get_signal_size(signal) > MAX_SIGNAL_SIZE:
        raise SignalValidationError("Signal too large")
    
    # Проверка TTL
    if signal.is_expired():
        raise SignalValidationError("Signal expired")
    
    # Валидация схемы данных
    if not validate_schema(signal.data, signal.schema):
        raise SignalValidationError("Invalid signal schema")
    
    return True
```

### 3. Маршрутизация
```python
class SignalRouter:
    def route_signal(self, signal: Signal) -> List[str]:
        """Определяет маршрут сигнала"""
        path = []
        current_node = signal.source_node
        
        while current_node != signal.target_node:
            next_node = self.find_next_hop(current_node, signal)
            if not next_node:
                raise RoutingError(f"No route from {current_node} to {signal.target_node}")
            
            # Проверка условий маршрутизации
            if not self.check_routing_condition(signal, current_node, next_node):
                raise RoutingError(f"Routing condition failed: {current_node} -> {next_node}")
            
            path.append(next_node)
            current_node = next_node
        
        return path
```

### 4. Доставка и обработка
```python
class SignalDelivery:
    def deliver_signal(self, signal: Signal, target_node: NeuroNode):
        """Доставка сигнала в узел"""
        try:
            # Проверка доступности узла
            if not target_node.is_active():
                self.queue_signal(signal, target_node)
                return
            
            # Проверка перегрузки
            if target_node.is_overloaded():
                if signal.metadata.get('priority', 5) < 8:
                    self.throttle_signal(signal)
                    return
            
            # Доставка сигнала
            target_node.receive_signal(signal)
            
        except Exception as e:
            self.handle_delivery_error(signal, e)
```

## Обработка сигналов в узлах

### 1. Базовая обработка
```anamorph
neuro message_processor

# Обработчик для всех входящих сигналов
on "*" from *
    log [message: "Received signal", data: signal.metadata]
    
    # Валидация сигнала
    validate [data: signal.data, schema: "message_schema"]
    
    # Обработка по типу
    switch signal.metadata.type
        case "user_message":
            process_user_message(signal.data)
        case "system_notification":
            process_system_notification(signal.data)
        default:
            log [message: "Unknown signal type", level: "warning"]
    end
end
```

### 2. Фильтрация сигналов
```anamorph
neuro security_filter

# Фильтрация по источнику
on "*" from *
    if signal.source_node in blacklist then
        ban [target: signal.source_node, duration: 3600]
        halt # Блокировать сигнал
    end
    
    # Фильтрация по содержимому
    if contains_malicious_content(signal.data) then
        alert [level: "HIGH", message: "Malicious signal detected"]
        quarantine_signal(signal)
        halt
    end
    
    # Передача валидного сигнала дальше
    pulse [from: security_filter, to: next_processor, data: signal.data]
end
```

### 3. Преобразование сигналов
```anamorph
neuro data_transformer

on "raw_data" from data_source
    # Преобразование данных
    transformed_data = transform_format(signal.data, "json_to_xml")
    
    # Обогащение метаданными
    enriched_data = enrich_with_metadata(transformed_data, {
        "transformation_timestamp": current_time(),
        "original_format": "json",
        "target_format": "xml"
    })
    
    # Отправка преобразованных данных
    pulse [from: data_transformer, to: xml_processor, data: enriched_data]
end
```

## Буферизация и очереди сигналов

### 1. Буфер сигналов
```python
class SignalBuffer:
    def __init__(self, max_size=1000, prioritized=True):
        self.max_size = max_size
        self.buffer = PriorityQueue() if prioritized else Queue()
        self.overflow_strategy = "drop_oldest"  # или "drop_lowest_priority"
    
    def add_signal(self, signal: Signal):
        if self.buffer.qsize() >= self.max_size:
            self.handle_overflow(signal)
        else:
            priority = signal.metadata.get('priority', 5)
            self.buffer.put((10 - priority, signal))  # Инверсия для PriorityQueue
    
    def handle_overflow(self, new_signal: Signal):
        if self.overflow_strategy == "drop_oldest":
            self.buffer.get()  # Удаляем старый
            self.buffer.put((10 - new_signal.metadata['priority'], new_signal))
        elif self.overflow_strategy == "drop_lowest_priority":
            # Сравниваем приоритеты и оставляем более важный сигнал
            pass
```

### 2. Стратегии обработки перегрузки
```anamorph
neuro load_balancer

on "high_load_detected" from monitor
    # Активация режима защиты от перегрузки
    throttle [rate: 100, window: 60]  # Максимум 100 сигналов в минуту
    
    # Перенаправление части нагрузки
    diffuse [
        signal: incoming_signals, 
        channels: ["backup_processor_1", "backup_processor_2"],
        strategy: "round_robin"
    ]
    
    # Временное снижение приоритета некритичных сигналов
    filter [
        condition: "signal.metadata.priority < 7", 
        action: "delay",
        delay_seconds: 30
    ]
end
```

## Мониторинг и отладка сигналов

### 1. Трассировка сигналов
```anamorph
# Включение детальной трассировки
trace [events: ["signal_created", "signal_routed", "signal_delivered"], level: "debug"]

# Создание контрольных точек
checkpoint [name: "before_critical_processing"]

on "signal_processing_error" from *
    # Логирование ошибки с полным контекстом
    log [
        message: "Signal processing failed",
        level: "error",
        context: {
            "signal_id": signal.id,
            "source_node": signal.source_node,
            "target_node": signal.target_node,
            "error_details": error.message,
            "routing_history": signal.routing_history
        }
    ]
    
    # Откат к контрольной точке при критической ошибке
    if error.severity == "critical" then
        rollback [checkpoint: "before_critical_processing"]
    end
end
```

### 2. Метрики производительности
```python
SIGNAL_METRICS = {
    "signals_per_second": RateCounter(),
    "signal_latency": HistogramMetric(),
    "signal_queue_size": GaugeMetric(),
    "failed_signals": CounterMetric(),
    "signal_retries": CounterMetric()
}

def track_signal_metrics(signal: Signal, operation: str):
    if operation == "delivered":
        latency = time.time() - signal.metadata['timestamp']
        SIGNAL_METRICS["signal_latency"].observe(latency)
        SIGNAL_METRICS["signals_per_second"].increment()
    elif operation == "failed":
        SIGNAL_METRICS["failed_signals"].increment()
```

## Безопасность сигналов

### 1. Шифрование сигналов
```anamorph
# Шифрование чувствительных данных в сигнале
if contains_sensitive_data(signal.data) then
    encrypt [
        data: signal.data.sensitive_fields,
        algorithm: "AES-256-GCM",
        key: get_node_encryption_key(target_node)
    ]
end

# Подпись сигнала для проверки целостности
sign [
    data: signal,
    algorithm: "Ed25519",
    private_key: source_node.private_key
]
```

### 2. Контроль доступа к сигналам
```anamorph
# Проверка разрешений перед обработкой сигнала
auth [method: "signal_access_token", credentials: signal.metadata.access_token]

if not has_permission(current_node, "receive_signal", signal.metadata.signal_type) then
    log [message: "Unauthorized signal access attempt", level: "warning"]
    ban [target: signal.source_node, duration: 300]
    halt
end
```

## Примеры использования

### 1. Обработка HTTP-запросов
```anamorph
# Определение сети для веб-сервера
network http_processing
    neuro gateway
    neuro auth_service
    neuro rate_limiter
    neuro request_processor
    neuro response_formatter
end

# Создание синапсов
synap gateway -> auth_service
synap auth_service -> rate_limiter  
synap rate_limiter -> request_processor
synap request_processor -> response_formatter

# Обработка HTTP-запроса как сигнала
signal HttpRequest {
    data: {
        method: "POST",
        path: "/api/users",
        headers: {...},
        body: {...}
    },
    metadata: {
        client_ip: "192.168.1.100",
        user_agent: "Mozilla/5.0...",
        request_id: "req-12345"
    }
}

# Последовательная обработка
on "http_request" from gateway
    # Аутентификация
    pulse [from: gateway, to: auth_service, data: signal.data]
    wait [for: "auth_result", timeout: 5]
    
    if auth_result.success then
        # Проверка лимитов
        pulse [from: auth_service, to: rate_limiter, data: signal.data]
        wait [for: "rate_limit_check", timeout: 1]
        
        if rate_limit_ok then
            # Обработка запроса
            pulse [from: rate_limiter, to: request_processor, data: signal.data]
        else
            response [to: gateway, data: {status: 429, message: "Rate limit exceeded"}]
        end
    else
        response [to: gateway, data: {status: 401, message: "Unauthorized"}]
    end
end
```

### 2. Система уведомлений
```anamorph
signal NotificationSignal {
    data: {
        type: "user_notification",
        user_id: "user123",
        message: "Your order has been shipped",
        channels: ["email", "push", "sms"]
    },
    metadata: {
        priority: 6,
        delivery_preference: "immediate"
    }
}

# Многоканальная доставка уведомлений
on "notification_request" from notification_service
    # Определение каналов доставки
    channels = signal.data.channels
    
    # Параллельная отправка по всем каналам
    parallel
        section email_delivery
            if "email" in channels then
                pulse [from: notification_service, to: email_sender, data: signal.data]
            end
        end
        
        section push_delivery
            if "push" in channels then
                pulse [from: notification_service, to: push_sender, data: signal.data]
            end
        end
        
        section sms_delivery
            if "sms" in channels then
                pulse [from: notification_service, to: sms_sender, data: signal.data]
            end
        end
    end
end
```

Данная модель сигналов является уникальной особенностью языка Anamorph и обеспечивает мощную и гибкую систему коммуникации между узлами нейросети. 