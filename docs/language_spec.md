# Спецификация языка Anamorph v1.0

## Общие принципы

Anamorph - язык программирования, основанный на метафоре нейронных сетей. Основные сущности:
- **Нейроны (neuro)** - вычислительные узлы
- **Синапсы (synap)** - каналы связи между узлами
- **Сигналы (pulse)** - единицы передачи информации
- **Резонансы (resonate)** - активация узлов

## Лексические правила

### Чувствительность к регистру
**ВАЖНО**: Язык Anamorph использует **case-insensitive** подход для ключевых слов:
- `neuro`, `NEURO`, `Neuro` - все варианты эквивалентны
- `pulse`, `PULSE`, `Pulse` - все варианты эквивалентны
- Это сделано для удобства пользователей и снижения количества синтаксических ошибок

**Исключения** (case-sensitive):
- **Пользовательские идентификаторы**: `myNode` ≠ `mynode` ≠ `MyNode`
- **Строковые литералы**: `"Hello"` ≠ `"hello"`
- **Имена файлов и путей**: чувствительны к регистру согласно ОС

### Примеры эквивалентности:
```anamorph
# Все эти варианты корректны и эквивалентны:
neuro sensor_node
NEURO sensor_node  
Neuro sensor_node

# Но имена узлов остаются case-sensitive:
neuro MyNode
neuro mynode   # Это ДРУГОЙ узел!
```

## Полный словарь команд (80 команд)

### Категории команд

Все команды разделены на 7 основных категорий:
- **Структурные** (10 команд) - создание и управление узлами
- **Управление потоком** (15 команд) - передача сигналов и управление
- **Безопасность** (12 команд) - защита и контроль доступа
- **Машинное обучение** (8 команд) - ML операции
- **Сетевые операции** (10 команд) - сетевое взаимодействие
- **Мониторинг и отладка** (15 команд) - диагностика и логирование
- **Системные операции** (10 команд) - управление ресурсами

---

## СТРУКТУРНЫЕ КОМАНДЫ (10 команд)

### 1. `neuro` - Объявление узла
**Категория**: Структурные  
**Синтаксис**: `neuro <node_name> [type: <node_type>]`  
**Описание**: Создает новый нейронный узел в сети  
**Возвращаемое значение**: `void`  
**Ошибки**:
- `NodeAlreadyExistsError` - если узел с таким именем уже существует
- `InvalidNodeNameError` - если имя содержит недопустимые символы
- `NodeQuotaExceededError` - если превышена квота узлов для пользователя

**Пример**:
```anamorph
neuro web_server
neuro database_node type: "persistent"
neuro ml_processor type: "gpu_accelerated"
```

**Примечания**: 
- Имена узлов case-sensitive
- Максимум 1000 узлов на сеть в sandbox режиме
- GPU-узлы требуют специальных разрешений

### 2. `synap` - Создание синапса
**Категория**: Структурные  
**Синтаксис**: `synap <source> -> <target> [weight: <value>] [bidirectional: <bool>]`  
**Описание**: Создает направленную связь между двумя узлами  
**Возвращаемое значение**: `synapse_id: string`  
**Ошибки**:
- `NodeNotFoundError` - если один из узлов не существует
- `SynapseAlreadyExistsError` - если синапс уже создан
- `CircularDependencyError` - если создается циклическая зависимость

**Пример**:
```anamorph
synap web_server -> auth_service
synap data_input -> ml_processor weight: 0.8
synap node_a -> node_b bidirectional: true
```

**Примечания**: 
- Вес синапса влияет на приоритет маршрутизации
- Двунаправленные синапсы создают два отдельных канала

### 3. `bind` - Привязка данных к узлу
**Категория**: Структурные  
**Синтаксис**: `bind [node: <node_name>, data: <data_source>, schema: <schema_name>]`  
**Описание**: Привязывает источник данных к узлу с валидацией схемы  
**Возвращаемое значение**: `binding_id: string`  
**Ошибки**:
- `NodeNotFoundError` - узел не существует
- `SchemaValidationError` - данные не соответствуют схеме
- `DataSourceUnavailableError` - источник данных недоступен

**Пример**:
```anamorph
bind [node: user_service, data: "users_database", schema: "user_schema"]
bind [node: logger, data: "/var/log/app.log", schema: "log_entry"]
```

**Примечания**: 
- Схемы валидируются в реальном времени
- Поддерживаются JSON Schema, Avro, Protocol Buffers

### 4. `cluster` - Создание кластера узлов
**Категория**: Структурные  
**Синтаксис**: `cluster <cluster_name> [nodes: [<node_list>], strategy: <strategy>]`  
**Описание**: Группирует узлы в кластер для совместной обработки  
**Возвращаемое значение**: `cluster_id: string`  
**Ошибки**:
- `InsufficientNodesError` - недостаточно узлов для кластера
- `NodeAlreadyClusteredError` - узел уже в другом кластере
- `InvalidStrategyError` - неподдерживаемая стратегия кластеризации

**Пример**:
```anamorph
cluster web_cluster [nodes: ["web1", "web2", "web3"], strategy: "load_balance"]
cluster ml_cluster [nodes: ["gpu1", "gpu2"], strategy: "parallel_processing"]
```

**Примечания**: 
- Стратегии: `load_balance`, `failover`, `parallel_processing`, `consensus`
- Максимум 50 узлов в кластере

### 5. `expand` - Расширение кластера
**Категория**: Структурные  
**Синтаксис**: `expand [cluster: <cluster_name>, nodes: [<new_nodes>]]`  
**Описание**: Добавляет новые узлы в существующий кластер  
**Возвращаемое значение**: `void`  
**Ошибки**:
- `ClusterNotFoundError` - кластер не существует
- `ClusterCapacityExceededError` - превышена максимальная емкость
- `NodeCompatibilityError` - узел несовместим с кластером

**Пример**:
```anamorph
expand [cluster: web_cluster, nodes: ["web4", "web5"]]
```

### 6. `contract` - Сжатие кластера
**Категория**: Структурные  
**Синтаксис**: `contract [cluster: <cluster_name>, nodes: [<nodes_to_remove>]]`  
**Описание**: Удаляет узлы из кластера  
**Возвращаемое значение**: `void`  
**Ошибки**:
- `ClusterNotFoundError` - кластер не существует
- `MinimumNodesError` - нельзя удалить критически важные узлы
- `ActiveProcessingError` - узел обрабатывает активные задачи

**Пример**:
```anamorph
contract [cluster: web_cluster, nodes: ["web5"]]
```

### 7. `morph` - Трансформация узла
**Категория**: Структурные  
**Синтаксис**: `morph [node: <node_name>, new_type: <type>, preserve_state: <bool>]`  
**Описание**: Изменяет тип или конфигурацию узла  
**Возвращаемое значение**: `void`  
**Ошибки**:
- `NodeNotFoundError` - узел не существует
- `IncompatibleTypeError` - невозможно преобразовать к новому типу
- `StatePreservationError` - не удается сохранить состояние

**Пример**:
```anamorph
morph [node: basic_processor, new_type: "gpu_accelerated", preserve_state: true]
```

### 8. `evolve` - Эволюция узла
**Категория**: Структурные  
**Синтаксис**: `evolve [node: <node_name>, adaptation: <adaptation_type>]`  
**Описание**: Автоматическая адаптация узла к изменяющимся условиям  
**Возвращаемое значение**: `evolution_report: object`  
**Ошибки**:
- `NodeNotFoundError` - узел не существует
- `EvolutionNotSupportedError` - узел не поддерживает эволюцию
- `AdaptationFailedError` - процесс адаптации завершился неудачно

**Пример**:
```anamorph
evolve [node: ml_model, adaptation: "performance_optimization"]
evolve [node: web_server, adaptation: "load_adaptation"]
```

### 9. `prune` - Удаление неактивных элементов
**Категория**: Структурные  
**Синтаксис**: `prune [target: <target>, criteria: <criteria>]`  
**Описание**: Удаляет неиспользуемые узлы, синапсы или данные  
**Возвращаемое значение**: `pruned_count: number`  
**Ошибки**:
- `InvalidCriteriaError` - неверные критерии отбора
- `ProtectedResourceError` - попытка удалить защищенный ресурс

**Пример**:
```anamorph
prune [target: "inactive_nodes", criteria: "unused_for_24h"]
prune [target: "weak_synapses", criteria: "weight < 0.1"]
```

### 10. `forge` - Создание составного узла
**Категория**: Структурные  
**Синтаксис**: `forge [name: <composite_name>, components: [<node_list>], interface: <interface_spec>]`  
**Описание**: Создает составной узел из нескольких простых узлов  
**Возвращаемое значение**: `composite_node_id: string`  
**Ошибки**:
- `ComponentNotFoundError` - один из компонентов не существует
- `InterfaceConflictError` - конфликт интерфейсов компонентов
- `CircularCompositionError` - циклическая композиция

**Пример**:
```anamorph
forge [
    name: "auth_system", 
    components: ["auth_validator", "token_manager", "user_store"],
    interface: "authentication_interface"
]
```

---

## УПРАВЛЕНИЕ ПОТОКОМ (15 команд)

### 11. `pulse` - Базовая передача сигнала
**Категория**: Управление потоком  
**Синтаксис**: `pulse [from: <source>, to: <target>, data: <data>, priority: <level>]`  
**Описание**: Отправляет сигнал с данными от одного узла к другому  
**Возвращаемое значение**: `signal_id: string`  
**Ошибки**:
- `NodeNotFoundError` - источник или получатель не существует
- `SynapseNotFoundError` - нет синапса между узлами
- `SignalValidationError` - данные не прошли валидацию
- `NodeOverloadedError` - узел-получатель перегружен

**Пример**:
```anamorph
pulse [from: web_gateway, to: auth_service, data: login_request]
pulse [from: sensor, to: processor, data: temperature_data, priority: 8]
```

**Примечания**: 
- Приоритет от 0 (низкий) до 10 (критический)
- Проходит через фильтры `filterIn` и `filterOut`
- Не поддерживает мультиточечную отправку

### 12. `pulseX` - Расширенная передача сигнала
**Категория**: Управление потоком  
**Синтаксис**: `pulseX [from: <source>, to: [<targets>], data: <data>, mode: <mode>]`  
**Описание**: Отправляет сигнал множественным получателям с различными режимами  
**Возвращаемое значение**: `signal_batch_id: string`  
**Ошибки**:
- `EmptyTargetListError` - список получателей пуст
- `PartialDeliveryError` - не все получатели доступны
- `InvalidModeError` - неподдерживаемый режим отправки

**Пример**:
```anamorph
pulseX [from: notification_center, to: ["email", "sms", "push"], data: alert, mode: "parallel"]
pulseX [from: load_balancer, to: ["server1", "server2"], data: request, mode: "round_robin"]
```

**Примечания**: 
- Режимы: `parallel`, `sequential`, `round_robin`, `failover`
- Максимум 100 получателей за раз

### 13. `pulseIf` - Условная передача сигнала
**Категория**: Управление потоком  
**Синтаксис**: `pulseIf [condition: <expression>, from: <source>, to: <target>, data: <data>]`  
**Описание**: Отправляет сигнал только при выполнении условия  
**Возвращаемое значение**: `signal_id: string | null`  
**Ошибки**:
- `ConditionEvaluationError` - ошибка при вычислении условия
- `InvalidExpressionError` - неверное выражение условия

**Пример**:
```anamorph
pulseIf [condition: "temperature > 80", from: sensor, to: cooling_system, data: temp_data]
pulseIf [condition: "user.role == 'admin'", from: request_handler, to: admin_panel, data: request]
```

### 14. `resonate` - Активация узла
**Категория**: Управление потоком  
**Синтаксис**: `resonate [node: <node_name>, frequency: <freq>, duration: <time>]`  
**Описание**: Активирует узел с определенной частотой обработки  
**Возвращаемое значение**: `activation_id: string`  
**Ошибки**:
- `NodeNotFoundError` - узел не существует
- `InvalidFrequencyError` - недопустимая частота
- `NodeAlreadyActiveError` - узел уже активен

**Пример**:
```anamorph
resonate [node: heartbeat_monitor, frequency: "1Hz", duration: "continuous"]
resonate [node: batch_processor, frequency: "0.1Hz", duration: "1h"]
```

### 15. `drift` - Плавное изменение параметров
**Категория**: Управление потоком  
**Синтаксис**: `drift [target: <target>, parameter: <param>, from: <start>, to: <end>, duration: <time>]`  
**Описание**: Плавно изменяет параметр узла или системы  
**Возвращаемое значение**: `drift_id: string`  
**Ошибки**:
- `ParameterNotFoundError` - параметр не существует
- `InvalidRangeError` - недопустимый диапазон значений
- `DriftInProgressError` - уже выполняется изменение этого параметра

**Пример**:
```anamorph
drift [target: load_balancer, parameter: "weight", from: 0.5, to: 0.8, duration: "30s"]
drift [target: ml_model, parameter: "learning_rate", from: 0.01, to: 0.001, duration: "100epochs"]
```

### 16. `sync` - Синхронизация узлов
**Категория**: Управление потоком  
**Синтаксис**: `sync [nodes: [<node_list>], barrier: <barrier_name>, timeout: <time>]`  
**Описание**: Синхронизирует выполнение нескольких узлов  
**Возвращаемое значение**: `sync_id: string`  
**Ошибки**:
- `SyncTimeoutError` - превышено время ожидания синхронизации
- `NodeNotRespondingError` - один из узлов не отвечает
- `BarrierAlreadyExistsError` - барьер с таким именем уже существует

**Пример**:
```anamorph
sync [nodes: ["worker1", "worker2", "worker3"], barrier: "batch_complete", timeout: "60s"]
```

### 17. `async` - Асинхронное выполнение
**Категория**: Управление потоком  
**Синтаксис**: `async [task: <task_definition>, callback: <callback_node>]`  
**Описание**: Запускает задачу асинхронно с обратным вызовом  
**Возвращаемое значение**: `task_id: string`  
**Ошибки**:
- `TaskDefinitionError` - неверное определение задачи
- `CallbackNodeError` - узел обратного вызова недоступен

**Пример**:
```anamorph
async [task: "process_large_dataset", callback: result_handler]
async [task: "send_email_batch", callback: delivery_tracker]
```

### 18. `fold` - Агрегация данных
**Категория**: Управление потоком  
**Синтаксис**: `fold [inputs: [<input_list>], operation: <op>, initial: <value>]`  
**Описание**: Агрегирует данные из множественных источников  
**Возвращаемое значение**: `aggregated_result: any`  
**Ошибки**:
- `EmptyInputError` - список входов пуст
- `OperationError` - ошибка при выполнении операции
- `TypeMismatchError` - несовместимые типы данных

**Пример**:
```anamorph
fold [inputs: ["sensor1", "sensor2", "sensor3"], operation: "average", initial: 0]
fold [inputs: ["log1", "log2"], operation: "concat", initial: ""]
```

### 19. `unfold` - Распределение данных
**Категория**: Управление потоком  
**Синтаксис**: `unfold [source: <source>, targets: [<target_list>], strategy: <strategy>]`  
**Описание**: Распределяет данные от одного источника к множественным получателям  
**Возвращаемое значение**: `distribution_id: string`  
**Ошибки**:
- `SourceNotFoundError` - источник не существует
- `DistributionError` - ошибка при распределении данных

**Пример**:
```anamorph
unfold [source: data_stream, targets: ["processor1", "processor2"], strategy: "round_robin"]
unfold [source: broadcast_message, targets: ["all_clients"], strategy: "multicast"]
```

### 20. `reflect` - Отражение сигнала
**Категория**: Управление потоком  
**Синтаксис**: `reflect [signal: <signal_id>, back_to: <source>, transform: <transform_func>]`  
**Описание**: Отправляет сигнал обратно к источнику с возможной трансформацией  
**Возвращаемое значение**: `reflected_signal_id: string`  
**Ошибки**:
- `SignalNotFoundError` - сигнал не найден
- `TransformationError` - ошибка при трансформации
- `ReflectionLoopError` - обнаружена петля отражений

**Пример**:
```anamorph
reflect [signal: request_123, back_to: client, transform: "add_response_headers"]
reflect [signal: ping_signal, back_to: sender, transform: "echo"]
```

### 21. `absorb` - Поглощение сигнала
**Категория**: Управление потоком  
**Синтаксис**: `absorb [signal: <signal_id>, reason: <reason>]`  
**Описание**: Поглощает сигнал, прекращая его дальнейшую передачу  
**Возвращаемое значение**: `void`  
**Ошибки**:
- `SignalNotFoundError` - сигнал не найден
- `SignalAlreadyProcessedError` - сигнал уже обработан

**Пример**:
```anamorph
absorb [signal: malicious_request, reason: "security_violation"]
absorb [signal: duplicate_message, reason: "deduplication"]
```

### 22. `diffuse` - Диффузия сигнала
**Категория**: Управление потоком  
**Синтаксис**: `diffuse [signal: <signal>, radius: <radius>, decay: <decay_rate>]`  
**Описание**: Распространяет сигнал по сети с затуханием  
**Возвращаемое значение**: `diffusion_id: string`  
**Ошибки**:
- `InvalidRadiusError` - недопустимый радиус распространения
- `NetworkTopologyError` - ошибка топологии сети

**Пример**:
```anamorph
diffuse [signal: alert_signal, radius: 3, decay: 0.5]
diffuse [signal: update_notification, radius: "unlimited", decay: 0.1]
```

### 23. `merge` - Слияние сигналов
**Категория**: Управление потоком  
**Синтаксис**: `merge [signals: [<signal_list>], strategy: <merge_strategy>]`  
**Описание**: Объединяет несколько сигналов в один  
**Возвращаемое значение**: `merged_signal_id: string`  
**Ошибки**:
- `IncompatibleSignalsError` - сигналы несовместимы для слияния
- `MergeStrategyError` - неподдерживаемая стратегия слияния

**Пример**:
```anamorph
merge [signals: ["data_part1", "data_part2"], strategy: "concatenate"]
merge [signals: ["vote1", "vote2", "vote3"], strategy: "majority"]
```

### 24. `split` - Разделение сигнала
**Категория**: Управление потоком  
**Синтаксис**: `split [signal: <signal>, criteria: <split_criteria>]`  
**Описание**: Разделяет один сигнал на несколько частей  
**Возвращаемое значение**: `split_signals: [string]`  
**Ошибки**:
- `SignalNotSplittableError` - сигнал нельзя разделить
- `InvalidCriteriaError` - неверные критерии разделения

**Пример**:
```anamorph
split [signal: batch_data, criteria: "by_user_id"]
split [signal: large_file, criteria: "chunk_size:1MB"]
```

### 25. `halt` - Остановка выполнения
**Категория**: Управление потоком  
**Синтаксис**: `halt [scope: <scope>, reason: <reason>, graceful: <bool>]`  
**Описание**: Останавливает выполнение в указанной области  
**Возвращаемое значение**: `void`  
**Ошибки**:
- `InvalidScopeError` - неверная область остановки
- `HaltNotAllowedError` - остановка не разрешена в данном контексте

**Пример**:
```anamorph
halt [scope: "current_node", reason: "critical_error", graceful: true]
halt [scope: "entire_network", reason: "emergency_shutdown", graceful: false]
```

---

## БЕЗОПАСНОСТЬ (12 команд)

### 26. `guard` - Защитный барьер
**Категория**: Безопасность  
**Синтаксис**: `guard [condition: <condition>, action: <action>, severity: <level>]`  
**Описание**: Создает защитный барьер с условием активации  
**Возвращаемое значение**: `guard_id: string`  
**Ошибки**:
- `InvalidConditionError` - неверное условие
- `UnauthorizedActionError` - недостаточно прав для действия

**Пример**:
```anamorph
guard [condition: "request_rate > 1000/min", action: "throttle", severity: "high"]
guard [condition: "suspicious_pattern_detected", action: "block", severity: "critical"]
```

**Примечания**: 
- Может блокировать выполнение команд `pulse`, `pulseX`
- Уровни серьезности: `low`, `medium`, `high`, `critical`

### 27. `filter` - Фильтрация данных
**Категория**: Безопасность  
**Синтаксис**: `filter [data: <data>, rules: [<rule_list>], action: <action>]`  
**Описание**: Фильтрует данные согласно правилам безопасности  
**Возвращаемое значение**: `filtered_data: any`  
**Ошибки**:
- `FilterRuleError` - ошибка в правиле фильтрации
- `DataCorruptionError` - данные повреждены при фильтрации

**Пример**:
```anamorph
filter [data: user_input, rules: ["xss_protection", "sql_injection"], action: "sanitize"]
filter [data: file_upload, rules: ["virus_scan", "file_type_check"], action: "quarantine"]
```

### 28. `mask` - Маскирование данных
**Категория**: Безопасность  
**Синтаксис**: `mask [data: <data>, fields: [<field_list>], method: <mask_method>]`  
**Описание**: Маскирует чувствительные данные  
**Возвращаемое значение**: `masked_data: any`  
**Ошибки**:
- `FieldNotFoundError` - поле для маскирования не найдено
- `MaskingMethodError` - неподдерживаемый метод маскирования

**Пример**:
```anamorph
mask [data: user_record, fields: ["ssn", "credit_card"], method: "partial_hide"]
mask [data: log_entry, fields: ["password", "token"], method: "hash"]
```

### 29. `scramble` - Шифрование данных
**Категория**: Безопасность  
**Синтаксис**: `scramble [data: <data>, algorithm: <algorithm>, key: <key_ref>]`  
**Описание**: Шифрует данные указанным алгоритмом  
**Возвращаемое значение**: `encrypted_data: string`  
**Ошибки**:
- `EncryptionError` - ошибка при шифровании
- `KeyNotFoundError` - ключ шифрования не найден
- `AlgorithmNotSupportedError` - алгоритм не поддерживается

**Пример**:
```anamorph
scramble [data: sensitive_payload, algorithm: "AES-256-GCM", key: "user_key_123"]
scramble [data: password, algorithm: "bcrypt", key: "salt_key"]
```

### 30. `auth` - Аутентификация
**Категория**: Безопасность  
**Синтаксис**: `auth [method: <auth_method>, credentials: <creds>, context: <context>]`  
**Описание**: Выполняет аутентификацию пользователя или узла  
**Возвращаемое значение**: `auth_result: {success: bool, token: string, expires: timestamp}`  
**Ошибки**:
- `AuthenticationFailedError` - аутентификация не удалась
- `InvalidCredentialsError` - неверные учетные данные
- `AuthMethodNotSupportedError` - метод аутентификации не поддерживается

**Пример**:
```anamorph
auth [method: "jwt", credentials: bearer_token, context: "api_access"]
auth [method: "oauth2", credentials: oauth_code, context: "user_login"]
```

### 31. `audit` - Аудит действий
**Категория**: Безопасность  
**Синтаксис**: `audit [event: <event_type>, actor: <actor>, resource: <resource>, details: <details>]`  
**Описание**: Записывает событие в журнал аудита  
**Возвращаемое значение**: `audit_id: string`  
**Ошибки**:
- `AuditLogFullError` - журнал аудита переполнен
- `InvalidEventTypeError` - неверный тип события

**Пример**:
```anamorph
audit [event: "data_access", actor: "user_123", resource: "sensitive_db", details: query_info]
audit [event: "privilege_escalation", actor: "admin_user", resource: "system_config", details: changes]
```

### 32. `throttle` - Ограничение скорости
**Категория**: Безопасность  
**Синтаксис**: `throttle [target: <target>, rate: <rate>, window: <time_window>, action: <action>]`  
**Описание**: Ограничивает скорость операций для защиты от перегрузки  
**Возвращаемое значение**: `throttle_id: string`  
**Ошибки**:
- `InvalidRateError` - неверная скорость ограничения
- `ThrottleConflictError` - конфликт с существующим ограничением

**Пример**:
```anamorph
throttle [target: "api_endpoint", rate: "100/min", window: "1min", action: "delay"]
throttle [target: "login_attempts", rate: "5/min", window: "5min", action: "block"]
```

### 33. `ban` - Блокировка доступа
**Категория**: Безопасность  
**Синтаксис**: `ban [target: <target>, duration: <duration>, reason: <reason>]`  
**Описание**: Блокирует доступ для указанной цели  
**Возвращаемое значение**: `ban_id: string`  
**Ошибки**:
- `TargetNotFoundError` - цель блокировки не найдена
- `InvalidDurationError` - неверная продолжительность блокировки

**Пример**:
```anamorph
ban [target: "192.168.1.100", duration: "1h", reason: "suspicious_activity"]
ban [target: "user_malicious", duration: "permanent", reason: "policy_violation"]
```

### 34. `whitelist` - Белый список
**Категория**: Безопасность  
**Синтаксис**: `whitelist [items: [<item_list>], category: <category>, expiry: <expiry>]`  
**Описание**: Добавляет элементы в белый список  
**Возвращаемое значение**: `whitelist_id: string`  
**Ошибки**:
- `WhitelistFullError` - белый список переполнен
- `InvalidCategoryError` - неверная категория

**Пример**:
```anamorph
whitelist [items: ["trusted_api.com", "partner_service.net"], category: "domains", expiry: "30d"]
whitelist [items: ["admin_user", "system_service"], category: "users", expiry: "never"]
```

### 35. `blacklist` - Черный список
**Категория**: Безопасность  
**Синтаксис**: `blacklist [items: [<item_list>], category: <category>, auto_expire: <bool>]`  
**Описание**: Добавляет элементы в черный список  
**Возвращаемое значение**: `blacklist_id: string`  
**Ошибки**:
- `BlacklistFullError` - черный список переполнен
- `ItemAlreadyBlacklistedError` - элемент уже в черном списке

**Пример**:
```anamorph
blacklist [items: ["malware_domain.com"], category: "domains", auto_expire: false]
blacklist [items: ["192.168.1.50"], category: "ips", auto_expire: true]
```

### 36. `encrypt` - Шифрование
**Категория**: Безопасность  
**Синтаксис**: `encrypt [data: <data>, algorithm: <algorithm>, key_id: <key_id>]`  
**Описание**: Шифрует данные с использованием указанного алгоритма  
**Возвращаемое значение**: `encrypted_result: {data: string, iv: string, tag: string}`  
**Ошибки**:
- `EncryptionFailedError` - ошибка шифрования
- `KeyNotAvailableError` - ключ недоступен

**Пример**:
```anamorph
encrypt [data: personal_info, algorithm: "AES-256-GCM", key_id: "user_key_001"]
```

### 37. `decrypt` - Расшифровка
**Категория**: Безопасность  
**Синтаксис**: `decrypt [encrypted_data: <data>, key_id: <key_id>, verify: <bool>]`  
**Описание**: Расшифровывает данные  
**Возвращаемое значение**: `decrypted_data: any`  
**Ошибки**:
- `DecryptionFailedError` - ошибка расшифровки
- `IntegrityCheckFailedError` - проверка целостности не пройдена

**Пример**:
```anamorph
decrypt [encrypted_data: cipher_text, key_id: "user_key_001", verify: true]
```

---

*[Продолжение следует с остальными категориями команд...]*

## Матрица совместимости команд

### Приоритет реализации
1. **Высокий приоритет** (Основа языка): neuro, synap, pulse, resonate, filter, guard
2. **Средний приоритет** (Расширенная функциональность): остальные структурные и управляющие команды
3. **Низкий приоритет** (Специализированные функции): ML команды, облачные операции

### Ограничения совместимости
- Команды ML-категории требуют предварительной инициализации ML-подсистемы
- Сетевые команды требуют активного сетевого контекста
- Команды безопасности могут блокировать выполнение других команд при нарушении политик

## Семантические правила
1. Все узлы должны быть объявлены перед использованием
2. Синапсы создаются только между существующими узлами
3. Сигналы передаются только по существующим синапсам
4. Команды безопасности имеют приоритет над обычными командами
5. Ресурсы освобождаются автоматически при завершении контекста 