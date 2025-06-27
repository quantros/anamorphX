# 🚀 AnamorphX Quick Start Guide

## Быстрый старт с AnamorphX

### 1. Запуск IDE

```bash
# Основная IDE
python3 run_full_ml_interpreter_ide.py

# Улучшенная IDE с визуализацией
python3 enhanced_ide_with_visualization.py
```

### 2. Основные команды

#### 🧠 Создание нейронов:
```anamorph
neuro "input_node" {
    activation: "linear"
    units: 784
}

neuro "hidden_node" {
    activation: "relu"
    units: 128
}
```

#### 🔗 Создание связей:
```anamorph
synap "input_node" -> "hidden_node" {
    weight: 0.5
    learning_rate: 0.01
}
```

#### ⚡ Отправка сигналов:
```anamorph
pulse {
    target: "input_node"
    intensity: 1.0
    signal: "training_data"
}
```

### 3. Визуализация

- **Двойной клик** по файлу → загрузка в редактор
- **Кнопка "Run+Viz"** → выполнение с визуализацией
- **Клик по узлу** → информация об узле
- **Правый клик** → контекстное меню

### 4. Палитра команд

- **Ctrl+Shift+P** или кнопка "🎯 Commands"
- Выберите команду и нажмите "Insert"

### 5. Примеры файлов

IDE содержит готовые примеры:
- `main.anamorph` - базовая сеть
- `neural_classifier.anamorph` - классификатор
- `deep_network.anamorph` - глубокая сеть

**Готово! Начинайте создавать нейронные сети! 🎉**
