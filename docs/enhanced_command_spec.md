# Расширенная спецификация команд Anamorph

## Категории команд

Все 80 команд разделены на 7 основных категорий:
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
- Может быть заблокирован командой `guard`

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
- В sandbox режиме ограничено 10 получателями

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
- `GuardConflictError` - конфликт с существующим guard

**Пример**:
```anamorph
guard [condition: "request_rate > 1000/min", action: "throttle", severity: "high"]
guard [condition: "suspicious_pattern_detected", action: "block", severity: "critical"]
```

**Примечания**: 
- Может блокировать выполнение команд `pulse`, `pulseX`
- Уровни серьезности: `low`, `medium`, `high`, `critical`
- В критическом режиме может остановить всю сеть

### 27. `filter` - Фильтрация данных
**Категория**: Безопасность  
**Синтаксис**: `filter [data: <data>, rules: [<rule_list>], action: <action>]`  
**Описание**: Фильтрует данные согласно правилам безопасности  
**Возвращаемое значение**: `filtered_data: any`  
**Ошибки**:
- `FilterRuleError` - ошибка в правиле фильтрации
- `DataCorruptionError` - данные повреждены при фильтрации
- `UnsupportedDataTypeError` - неподдерживаемый тип данных

**Пример**:
```anamorph
filter [data: user_input, rules: ["xss_protection", "sql_injection"], action: "sanitize"]
filter [data: file_upload, rules: ["virus_scan", "file_type_check"], action: "quarantine"]
```

**Примечания**:
- Встроенные правила: `xss_protection`, `sql_injection`, `csrf_protection`
- Действия: `sanitize`, `block`, `quarantine`, `log`
- Поддерживает пользовательские правила через regex

---

## МАШИННОЕ ОБУЧЕНИЕ (8 команд)

### 66. `train` - Обучение модели
**Категория**: Машинное обучение  
**Синтаксис**: `train [model: <model_name>, data: <training_data>, epochs: <count>, validation: <validation_data>]`  
**Описание**: Запускает процесс обучения ML-модели  
**Возвращаемое значение**: `training_job_id: string`  
**Ошибки**:
- `ModelNotFoundError` - модель не существует
- `InsufficientDataError` - недостаточно данных для обучения
- `ResourceUnavailableError` - недостаточно вычислительных ресурсов
- `ValidationFailedError` - ошибка валидации данных

**Пример**:
```anamorph
train [model: "fraud_detector", data: transaction_history, epochs: 100, validation: test_set]
train [model: "recommendation_engine", data: user_behavior, epochs: 50]
```

**Примечания**:
- Требует GPU-ресурсы для больших моделей
- Автоматическое сохранение checkpoint каждые 10 эпох
- Поддерживает distributed training для кластеров

### 67. `infer` - Инференс модели
**Категория**: Машинное обучение  
**Синтаксис**: `infer [model: <model_name>, input: <input_data>, confidence_threshold: <threshold>]`  
**Описание**: Выполняет предсказание с использованием обученной модели  
**Возвращаемое значение**: `prediction_result: {prediction: any, confidence: number, metadata: object}`  
**Ошибки**:
- `ModelNotTrainedError` - модель не обучена
- `InputValidationError` - неверный формат входных данных
- `ConfidenceThresholdError` - результат ниже порога уверенности

**Пример**:
```anamorph
infer [model: "fraud_detector", input: transaction_data, confidence_threshold: 0.8]
infer [model: "image_classifier", input: image_bytes]
```

**Примечания**:
- Кэширует результаты для повторных запросов
- Поддерживает batch inference для множественных входов
- Автоматическое A/B тестирование моделей

---

## МОНИТОРИНГ И ОТЛАДКА (15 команд)

### 71. `trace` - Трассировка выполнения
**Категория**: Мониторинг и отладка  
**Синтаксис**: `trace [events: [<event_list>], level: <level>, output: <output_target>]`  
**Описание**: Включает детальную трассировку указанных событий  
**Возвращаемое значение**: `trace_session_id: string`  
**Ошибки**:
- `InvalidEventTypeError` - неподдерживаемый тип события
- `TraceLevelError` - неверный уровень трассировки
- `OutputTargetError` - недоступная цель вывода

**Пример**:
```anamorph
trace [events: ["signal_created", "signal_routed", "signal_delivered"], level: "debug"]
trace [events: ["node_activation", "error_occurred"], level: "info", output: "trace_log.json"]
```

**Примечания**:
- Уровни: `debug`, `info`, `warn`, `error`
- Может значительно снизить производительность на уровне `debug`
- Автоматическая ротация логов при превышении размера

### 72. `log` - Логирование
**Категория**: Мониторинг и отладка  
**Синтаксис**: `log [message: <message>, level: <level>, context: <context_data>]`  
**Описание**: Записывает сообщение в журнал системы  
**Возвращаемое значение**: `log_entry_id: string`  
**Ошибки**:
- `LogBufferFullError` - буфер логов переполнен
- `InvalidLogLevelError` - неверный уровень логирования

**Пример**:
```anamorph
log [message: "User authentication successful", level: "info", context: user_data]
log [message: "Critical system error", level: "error", context: error_details]
```

**Примечания**:
- Структурированное логирование в JSON формате
- Автоматическое добавление timestamp и node_id
- Интеграция с внешними системами мониторинга

---

## СИСТЕМНЫЕ ОПЕРАЦИИ (10 команд)

### 76. `checkpoint` - Создание контрольной точки
**Категория**: Системные операции  
**Синтаксис**: `checkpoint [name: <checkpoint_name>, scope: <scope>, metadata: <metadata>]`  
**Описание**: Создает контрольную точку состояния системы  
**Возвращаемое значение**: `checkpoint_id: string`  
**Ошибки**:
- `CheckpointExistsError` - контрольная точка с таким именем уже существует
- `InsufficientStorageError` - недостаточно места для сохранения
- `InvalidScopeError` - неверная область сохранения

**Пример**:
```anamorph
checkpoint [name: "before_critical_update", scope: "entire_network"]
checkpoint [name: "model_trained", scope: "ml_nodes", metadata: training_stats]
```

**Примечания**:
- Поддерживает инкрементальные checkpoint для экономии места
- Автоматическое сжатие старых checkpoint
- Максимум 100 checkpoint на систему

### 77. `rollback` - Откат к контрольной точке
**Категория**: Системные операции  
**Синтаксис**: `rollback [checkpoint: <checkpoint_name>, confirm: <bool>, preserve_logs: <bool>]`  
**Описание**: Восстанавливает состояние системы из контрольной точки  
**Возвращаемое значение**: `rollback_result: {success: bool, restored_state: object, warnings: [string]}`  
**Ошибки**:
- `CheckpointNotFoundError` - контрольная точка не найдена
- `RollbackFailedError` - ошибка при восстановлении
- `ConfirmationRequiredError` - требуется подтверждение для критических операций

**Пример**:
```anamorph
rollback [checkpoint: "before_critical_update", confirm: true]
rollback [checkpoint: "stable_state", preserve_logs: true]
```

**Примечания**:
- Критические rollback требуют двойного подтверждения
- Автоматическое создание checkpoint перед rollback
- Невозможно откатить операции с внешними системами

---

## Матрица совместимости команд

### Команды, которые могут блокировать другие:

| Блокирующая команда | Блокируемые команды | Условие блокировки |
|-------------------|-------------------|-------------------|
| `guard` | `pulse`, `pulseX`, `pulseIf` | При срабатывании условия |
| `throttle` | `pulse`, `pulseX` | При превышении лимита |
| `ban` | Все команды от заблокированного источника | Активная блокировка |
| `halt` | Все команды в указанной области | Остановка выполнения |
| `filter` | `pulse` с отфильтрованными данными | Данные не прошли фильтр |

### Зависимости команд:

| Команда | Требует предварительного выполнения | Примечание |
|---------|-----------------------------------|------------|
| `synap` | `neuro` (для обоих узлов) | Узлы должны существовать |
| `pulse` | `synap` (между узлами) | Синапс должен существовать |
| `infer` | `train` (для модели) | Модель должна быть обучена |
| `rollback` | `checkpoint` | Контрольная точка должна существовать |
| `decrypt` | `encrypt` | Данные должны быть зашифрованы |

### Команды, разрешенные в sandbox режиме:

**Разрешенные** (с ограничениями):
- Все структурные команды (лимит узлов: 1000)
- Базовые команды потока (`pulse`, `pulseIf` - лимит получателей: 10)
- Команды мониторинга (`log`, `trace`)
- Базовые ML команды (лимит моделей: 5)

**Запрещенные**:
- Системные команды (`halt`, `reset`)
- Сетевые команды с внешним доступом
- Команды безопасности высокого уровня (`ban`, `blacklist`)
- Команды масштабирования (`scaleUp`, `scaleDown`)

---

## Модель угроз и защита

### Основные угрозы:

1. **XSS атаки** - защита через `filter` с правилом `xss_protection`
2. **SQL инъекции** - защита через `filter` с правилом `sql_injection`
3. **DDoS атаки** - защита через `throttle` и `guard`
4. **Несанкционированный доступ** - защита через `auth` и `whitelist`
5. **Утечка данных** - защита через `encrypt` и `mask`

### Рекомендуемые политики безопасности:

```anamorph
# Базовая защита веб-приложения
guard [condition: "request_rate > 100/min", action: "throttle", severity: "medium"]
filter [data: "all_user_input", rules: ["xss_protection", "sql_injection"], action: "sanitize"]
auth [method: "jwt", credentials: "bearer_token", context: "api_access"]

# Защита чувствительных данных
mask [data: "user_records", fields: ["ssn", "credit_card"], method: "partial_hide"]
encrypt [data: "personal_info", algorithm: "AES-256-GCM", key_id: "master_key"]

# Аудит критических операций
audit [event: "admin_access", actor: "current_user", resource: "admin_panel"]
```

---

## Управление жизненным циклом

### Жизненный цикл узла:
1. **Создание** (`neuro`) → **Неактивный**
2. **Активация** (`resonate`) → **Активный**
3. **Обработка** (получение сигналов) → **Занят**
4. **Ожидание** → **Активный**
5. **Деактивация** → **Неактивный**
6. **Удаление** (`prune`) → **Удален**

### Жизненный цикл сигнала:
1. **Создание** (`pulse`) → **Создан**
2. **Валидация** → **Валидный/Невалидный**
3. **Маршрутизация** → **В пути**
4. **Доставка** → **Доставлен**
5. **Обработка** → **Обработан**
6. **Завершение** → **Завершен**

### Управление ресурсами:
- **Память**: Автоматическая сборка мусора для неиспользуемых узлов
- **CPU**: Приоритетная обработка сигналов по уровню важности
- **Сеть**: Throttling для предотвращения перегрузки
- **Хранилище**: Автоматическая архивация старых checkpoint и логов

---

## Примеры сценариев использования

### Сценарий 1: Настройка безопасного канала передачи
```anamorph
# Создание узлов
neuro secure_gateway
neuro data_processor
neuro secure_storage

# Создание защищенных соединений
synap secure_gateway -> data_processor
synap data_processor -> secure_storage

# Настройка безопасности
guard [condition: "source_ip not in whitelist", action: "block", severity: "high"]
filter [data: "incoming_data", rules: ["malware_scan", "data_validation"], action: "sanitize"]
encrypt [data: "sensitive_payload", algorithm: "AES-256-GCM", key_id: "channel_key"]

# Передача данных
pulse [from: secure_gateway, to: data_processor, data: encrypted_payload]
```

### Сценарий 2: Кластерное машинное обучение
```anamorph
# Создание ML кластера
neuro ml_coordinator
neuro ml_worker_1 type: "gpu_accelerated"
neuro ml_worker_2 type: "gpu_accelerated"
neuro ml_worker_3 type: "gpu_accelerated"

cluster ml_cluster [nodes: ["ml_worker_1", "ml_worker_2", "ml_worker_3"], strategy: "parallel_processing"]

# Распределенное обучение
train [model: "large_language_model", data: training_dataset, epochs: 1000]
sync [nodes: ["ml_worker_1", "ml_worker_2", "ml_worker_3"], barrier: "epoch_complete"]

# Мониторинг прогресса
trace [events: ["training_progress", "model_convergence"], level: "info"]
log [message: "Training epoch completed", level: "info", context: epoch_stats]
```

### Сценарий 3: Реакция на критическое событие
```anamorph
# Мониторинг системы
neuro system_monitor
neuro alert_manager
neuro emergency_response

# Настройка реакции на критические события
guard [condition: "cpu_usage > 95%", action: "alert", severity: "critical"]
guard [condition: "memory_usage > 90%", action: "scale_up", severity: "high"]

# Обработка критического события
on "critical_alert" from system_monitor
    # Немедленное уведомление
    pulseX [from: alert_manager, to: ["admin_team", "ops_team"], data: alert_details, mode: "parallel"]
    
    # Создание checkpoint перед восстановительными действиями
    checkpoint [name: "before_emergency_response", scope: "affected_nodes"]
    
    # Автоматическое масштабирование
    scaleUp [target: "overloaded_services", factor: 2]
    
    # Аудит инцидента
    audit [event: "critical_incident", actor: "system", resource: "infrastructure", details: incident_data]
end
```

Данная расширенная спецификация обеспечивает полное понимание каждой команды, её ограничений, взаимодействий и практического применения в реальных сценариях. 