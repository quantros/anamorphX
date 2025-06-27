# Установка AnamorphX Enterprise Neural Library

## Требования

- Python 3.8+
- Зависимости из ../requirements.txt

## Быстрая установка

```bash
# 1. Перейти в директорию библиотеки
cd library

# 2. Простая демонстрация (без зависимостей)
python3 simple_library_demo.py

# 3. Полная демонстрация (требует установки зависимостей)
pip install -r ../requirements.txt
python3 demo_enterprise_library.py
```

## Использование в проекте

```python
# Добавить путь к библиотеке
import sys
sys.path.append('path/to/library')

# Импорт библиотеки
from anamorph_neural_engine import (
    AdvancedNeuralEngine,
    DistributedComputing,
    RealtimeAnalytics
)
```

Библиотека создана для проекта AnamorphX
