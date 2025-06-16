Обзор
Система Anamorph использует формальные схемы для валидации структуры сигналов на этапе компиляции и выполнения. Это обеспечивает типобезопасность, совместимость и надежность передачи данных между узлами.

Базовая структура схемы сигнала
JSON Schema для сигналов
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://anamorph.dev/schemas/signal/v1.0.0",
  "title": "Anamorph Signal Schema",
  "type": "object",
  "required": ["schema_version", "signal_type", "data", "metadata"],
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "description": "Версия схемы в формате semver"
    },
    "signal_type": {
      "type": "string",
      "enum": ["request", "response", "notification", "stream", "error"],
      "description": "Тип сигнала"
    },
    "data": {
      "type": "object",
      "description": "Полезная нагрузка сигнала"
    },
    "metadata": {
      "type": "object",
      "required": ["timestamp", "source_node", "signal_id"],
      "properties": {
        "signal_id": {
          "type": "string",
          "format": "uuid",
          "description": "Уникальный идентификатор сигнала"
        },
        "timestamp": {
          "type": "integer",
          "minimum": 0,
          "description": "Unix timestamp создания сигнала"
        },
        "source_node": {
          "type": "string",
          "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$",
          "description": "Идентификатор узла-отправителя"
        },
        "target_node": {
          "type": "string",
          "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$",
          "description": "Идентификатор узла-получателя"
        },
        "priority": {
          "type": "integer",
          "minimum": 0,
          "maximum": 10,
          "default": 5,
          "description": "Приоритет сигнала (0-10)"
        },
        "ttl": {
          "type": "integer",
          "minimum": 1,
          "default": 300,
          "description": "Время жизни сигнала в секундах"
        },
        "correlation_id": {
          "type": "string",
          "description": "ID для связывания связанных сигналов"
        },
        "retry_count": {
          "type": "integer",
          "minimum": 0,
          "default": 0,
          "description": "Количество попыток доставки"
        }
      }
    },
    "routing": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["from", "to"],
        "properties": {
          "from": {
            "type": "string",
            "description": "Узел-источник для этого участка маршрута"
          },
          "to": {
            "type": "string", 
            "description": "Узел-назначение для этого участка маршрута"
          },
          "condition": {
            "type": "string",
            "description": "Условие для передачи по этому маршруту"
          },
          "transform": {
            "type": "string",
            "description": "Функция трансформации данных"
          },
          "timeout": {
            "type": "integer",
            "minimum": 1,
            "description": "Таймаут для этого участка в секундах"
          },
          "weight": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Вес маршрута для балансировки нагрузки"
          }
        }
      }
    }
  }
}
Специализированные схемы сигналов
1. HTTP Request Signal
{
  "$schema": "https://anamorph.dev/schemas/signal/http-request/v1.0.0",
  "allOf": [
    {"$ref": "https://anamorph.dev/schemas/signal/v1.0.0"},
    {
      "properties": {
        "signal_type": {"const": "request"},
        "data": {
          "type": "object",
          "required": ["method", "path"],
          "properties": {
            "method": {
              "type": "string",
              "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
            },
            "path": {
              "type": "string",
              "pattern": "^/.*"
            },
            "headers": {
              "type": "object",
              "patternProperties": {
                "^[a-zA-Z][a-zA-Z0-9-]*$": {"type": "string"}
              }
            },
            "query_params": {
              "type": "object",
              "patternProperties": {
                "^[a-zA-Z][a-zA-Z0-9_]*$": {
                  "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}}
                  ]
                }
              }
            },
            "body": {
              "description": "Тело запроса (любой JSON-совместимый тип)"
            }
          }
        }
      }
    }
  ]
}
2. ML Training Signal
{
  "$schema": "https://anamorph.dev/schemas/signal/ml-training/v1.0.0",
  "allOf": [
    {"$ref": "https://anamorph.dev/schemas/signal/v1.0.0"},
    {
      "properties": {
        "signal_type": {"const": "request"},
        "data": {
          "type": "object",
          "required": ["model_id", "dataset", "training_config"],
          "properties": {
            "model_id": {
              "type": "string",
              "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$"
            },
            "dataset": {
              "type": "object",
              "required": ["source", "format"],
              "properties": {
                "source": {"type": "string"},
                "format": {
                  "type": "string",
                  "enum": ["csv", "json", "parquet", "tfrecord"]
                },
                "size": {"type": "integer", "minimum": 1}
              }
            },
            "training_config": {
              "type": "object",
              "required": ["epochs", "batch_size"],
              "properties": {
                "epochs": {"type": "integer", "minimum": 1, "maximum": 10000},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "learning_rate": {"type": "number", "minimum": 0.0001, "maximum": 1.0}
              }
            }
          }
        }
      }
    }
  ]
}
Версионирование схем
Семантическое версионирование
Схемы сигналов используют семантическое версионирование (semver):

MAJOR - несовместимые изменения API
MINOR - добавление новой функциональности с обратной совместимостью
PATCH - исправления ошибок с обратной совместимостью
Правила совместимости
Обратно совместимые изменения (MINOR/PATCH):
Добавление новых необязательных полей
Расширение enum значений
Ослабление ограничений
Добавление новых схем сигналов
Несовместимые изменения (MAJOR):
Удаление обязательных полей
Изменение типов существующих полей
Ужесточение ограничений
Удаление enum значений
Миграция схем
# Автоматическая миграция при получении сигнала старой версии
on "signal_received" from *
    if signal.schema_version != current_schema_version then
        migrated_signal = migrate_signal_schema(signal, current_schema_version)
        if migrated_signal.success then
            process_signal(migrated_signal.data)
        else
            log [message: "Schema migration failed", level: "error"]
            reject_signal(signal, "incompatible_schema")
        end
    else
        process_signal(signal)
    end
end
Валидация схем в runtime
Встроенная валидация
class SignalValidator:
    def __init__(self):
        self.schema_registry = SchemaRegistry()
        self.validators = {}
    
    def validate_signal(self, signal: Signal) -> ValidationResult:
        """Валидирует сигнал согласно его схеме"""
        try:
            schema_id = self.get_schema_id(signal)
            schema = self.schema_registry.get_schema(schema_id)
            
            if schema_id not in self.validators:
                self.validators[schema_id] = jsonschema.Draft202012Validator(schema)
            
            validator = self.validators[schema_id]
            errors = list(validator.iter_errors(signal.to_dict()))
            
            if errors:
                return ValidationResult(
                    valid=False,
                    errors=[self.format_error(error) for error in errors]
                )
            
            return ValidationResult(valid=True)
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
Компиляционная проверка схем
# Объявление схемы сигнала в коде
signal_schema UserRegistration {
    version: "1.0.0",
    type: "request",
    data: {
        username: string,
        email: string,
        password: string,
        profile: {
            first_name: string,
            last_name: string,
            age?: number
        }
    }
}

# Использование типизированного сигнала
neuro registration_handler

on UserRegistration from web_gateway
    # Компилятор автоматически проверяет соответствие схеме
    validate_email(signal.data.email)
    hash_password(signal.data.password)
    
    create_user_record(signal.data)
    
    response UserRegistrationResponse [
        to: signal.source_node,
        data: {success: true, user_id: new_user.id}
    ]
end
Расширенные возможности
Пользовательские валидаторы
# Определение пользовательского валидатора
validator email_validator(value: string) -> bool {
    return regex_match(value, "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
}

# Использование в схеме сигнала
signal_schema UserRegistration {
    data: {
        email: string @email_validator,
        password: string @strong_password
    }
}
Производительность и оптимизация
class OptimizedSignalValidator:
    def __init__(self):
        self.validator_cache = LRUCache(maxsize=1000)
        self.schema_cache = LRUCache(maxsize=100)
    
    @lru_cache(maxsize=1000)
    def get_compiled_validator(self, schema_id: str):
        """Кэширует скомпилированные валидаторы"""
        schema = self.schema_cache[schema_id]
        return jsonschema.Draft202012Validator(schema)
    
    def validate_signal_fast(self, signal: Signal) -> bool:
        """Быстрая валидация с кэшированием"""
        schema_id = self.get_schema_id(signal)
        validator = self.get_compiled_validator(schema_id)
        return validator.is_valid(signal.to_dict())
