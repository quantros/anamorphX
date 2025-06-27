# 📁 Структура проекта AnamorphX

## 🎯 Корневой каталог
```
anamorphX/
├── README.md           # Главный README проекта
├── LICENSE             # Лицензия MIT
├── requirements.txt    # Зависимости Python
├── .gitignore         # Файлы для игнорирования Git
└── STRUCTURE.md       # Этот файл
```

## 📂 Основные директории

### `src/` - Исходный код
```
src/
├── apps/              # Готовые приложения
│   ├── enhanced_ide_with_visualization.py
│   ├── complete_ml_ide.py
│   ├── full_ml_interpreter_ide.py
│   └── run_full_ml_interpreter_ide.py
├── interpreter/       # Интерпретатор языка
├── parser/           # Парсер
├── lexer/            # Лексический анализатор
├── neural_backend/   # Neural backend
├── semantic/         # Семантический анализ
├── types/            # Система типов
├── tools/            # IDE инструменты
└── ide/              # Компоненты IDE
```

### `docs/` - Документация
```
docs/
├── specs/            # Спецификации
│   ├── PROJECT_ROADMAP.md
│   ├── PARSER_README.md
│   └── schema_v0.01.md
├── reports/          # Отчеты о прогрессе
│   ├── ENHANCED_TYPE_SYSTEM_REPORT.md
│   ├── PRIORITY1_COMPLETION_REPORT.md
│   └── [другие отчеты]
├── STATUS.md         # Текущий статус
├── REPOSITORY_INFO.md
├── QUICK_START_GUIDE.md
└── [другая документация]
```

### `tests/` - Тесты
```
tests/
├── unit/             # Юнит-тесты
├── integration/      # Интеграционные тесты
├── test_enhanced_system.py
├── test_commands_system.py
└── [другие тесты]
```

### `demos/` - Демонстрации
```
demos/
├── demo_enhanced_types_simple.py
├── demo_enhanced_types.py
├── demo_all_commands.py
└── [другие демо]
```

### `scripts/` - Служебные скрипты
```
scripts/
├── fix_command_imports.py
├── visual_command_integration.py
└── [другие скрипты]
```

### `examples/` - Примеры кода
```
examples/
├── neural_program.amph
├── lexer_demo.py
└── [другие примеры]
```

### `tools/` - Инструменты разработки
```
tools/
└── scripts/
    ├── add_methods.py
    ├── enhanced_file_operations.py
    └── [другие инструменты]
```

## 🎯 Принципы организации

### ✅ Что ДОЛЖНО быть в корне:
- `README.md` - главная документация
- `LICENSE` - лицензия
- `requirements.txt` - зависимости
- `.gitignore` - игнорируемые файлы
- Конфигурационные файлы (.env, config.yaml, etc.)

### ❌ Что НЕ должно быть в корне:
- Демо-файлы (`demo_*.py`)
- Тестовые файлы (`test_*.py`)
- Отчеты и документация (`.md файлы`)
- Скрипты (`fix_*.py`, `visual_*.py`)
- Большие приложения (`*_ide.py`)

## 🚀 Новые правила

1. **Все демо** → `demos/`
2. **Все тесты** → `tests/`
3. **Все отчеты** → `docs/reports/`
4. **Все скрипты** → `scripts/`
5. **Все приложения** → `src/apps/`
6. **Вся документация** → `docs/`

## 📝 Результат

Теперь корень проекта чистый и профессиональный! 🎉

**До**: 60+ файлов в корне  
**После**: 5 файлов в корне  
**Улучшение**: 91% очистка 🚀
