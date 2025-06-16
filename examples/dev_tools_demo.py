"""
Демонстрация инструментов разработки AnamorphX

Показывает работу:
- Подсветки синтаксиса
- IDE компонентов  
- Отладчика
- Профайлера производительности
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tools.syntax_highlighter import highlight_anamorph_code, AnamorphSyntaxHighlighter, THEMES
from src.tools.debugger import create_debugger, DebugState
from src.tools.profiler import start_profiling, stop_profiling, profile, profile_neural
from src.tools.ide_components import launch_ide
import time
import json


def demo_syntax_highlighting():
    """Демонстрация подсветки синтаксиса"""
    print("🎨 === ДЕМОНСТРАЦИЯ ПОДСВЕТКИ СИНТАКСИСА ===")
    
    # Тестовый код Anamorph
    test_code = '''
// Пример нейронной сети на языке Anamorph
network simple_mlp {
    layers: [784, 128, 64, 10]
    activation: "relu"
    optimizer: "adam"
}

neuron input_neuron {
    activation: "linear"
    size: 784
    input_shape: [28, 28, 1]
}

synapse hidden_connection {
    from: input_neuron
    to: hidden_layer
    weight: random_normal(0.0, 0.1)
    bias: zeros()
}

signal training_data {
    batch_size: 32
    data: load_mnist("train")
    preprocessing: normalize()
}

def train_network(epochs=100, lr=0.001):
    """Обучение нейронной сети"""
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in training_data:
            # Прямое распространение
            predictions = simple_mlp.forward(batch.inputs)
            
            # Вычисление ошибки
            loss = cross_entropy_loss(predictions, batch.targets)
            total_loss += loss
            
            # Обратное распространение
            gradients = simple_mlp.backward(loss)
            
            # Обновление весов
            simple_mlp.update_weights(gradients, lr)
        
        # Логирование
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Валидация каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            accuracy = validate_model(simple_mlp)
            print(f"Validation Accuracy: {accuracy:.2f}%")

def validate_model(model):
    """Валидация модели"""
    correct = 0
    total = 0
    
    for batch in validation_data:
        predictions = model.predict(batch.inputs)
        correct += sum(predictions.argmax(axis=1) == batch.targets)
        total += len(batch.targets)
    
    return (correct / total) * 100

# Создание и обучение сети
network = simple_mlp
train_network(epochs=50, lr=0.001)
'''
    
    print("\n📄 Исходный код:")
    print("-" * 50)
    print(test_code[:300] + "...\n")
    
    # Тестирование разных тем
    themes = ['light', 'dark', 'vs_code_dark']
    
    for theme_name in themes:
        print(f"\n🎭 Тема: {theme_name}")
        print("-" * 30)
        
        # HTML подсветка
        html_result = highlight_anamorph_code(test_code, theme=theme_name, format='html')
        html_filename = f"syntax_highlight_{theme_name}.html"
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            # Добавляем CSS стили
            highlighter = AnamorphSyntaxHighlighter(THEMES[theme_name])
            css_styles = highlighter.generate_css()
            
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anamorph Syntax Highlighting - {theme_name}</title>
    <style>
        body {{ background: {'#1E1E1E' if 'dark' in theme_name else '#FFFFFF'}; }}
        {css_styles}
    </style>
</head>
<body>
    <h1>🎨 Подсветка синтаксиса Anamorph - {theme_name}</h1>
    {html_result}
</body>
</html>
            """)
        
        print(f"  ✅ HTML файл создан: {html_filename}")
        
        # JSON экспорт
        json_result = highlight_anamorph_code(test_code, theme=theme_name, format='json')
        json_filename = f"syntax_tokens_{theme_name}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            f.write(json_result)
        
        print(f"  ✅ JSON файл создан: {json_filename}")
    
    # VS Code тема
    highlighter = AnamorphSyntaxHighlighter(THEMES['vs_code_dark'])
    vs_code_theme = highlighter.export_vs_code_theme()
    
    with open('anamorph_vscode_theme.json', 'w', encoding='utf-8') as f:
        json.dump(vs_code_theme, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎯 VS Code тема создана: anamorph_vscode_theme.json")
    print("   Установка: скопируйте в папку .vscode/extensions")


def demo_debugger():
    """Демонстрация отладчика"""
    print("\n🐛 === ДЕМОНСТРАЦИЯ ОТЛАДЧИКА ===")
    
    # Тестовый код для отладки
    debug_code = '''
def factorial(n):
    print(f"Вычисление факториала {n}")
    if n <= 1:
        return 1
    else:
        result = n * factorial(n - 1)
        print(f"Факториал {n} = {result}")
        return result

def main():
    numbers = [3, 5, 7]
    results = []
    
    for num in numbers:
        fact = factorial(num)
        results.append(fact)
        print(f"Результат для {num}: {fact}")
    
    print(f"Все результаты: {results}")
    return results

# Выполнение
main()
'''
    
    print("\n📝 Код для отладки:")
    print("-" * 40)
    print(debug_code)
    
    # Создание отладчика
    debugger = create_debugger()
    
    # Добавление точек останова
    bp1 = debugger.add_line_breakpoint("debug_test.py", 3, condition="n > 1")
    bp2 = debugger.add_function_breakpoint("factorial")
    bp3 = debugger.add_line_breakpoint("debug_test.py", 15)
    
    print(f"\n🎯 Добавлены точки останова:")
    for bp in debugger.list_breakpoints():
        print(f"  • {bp['type']} в {bp.get('file_path', 'функция')}:{bp.get('line', bp.get('function_name'))}")
    
    # Отслеживание переменных
    debugger.add_watch("n")
    debugger.add_watch("result")
    debugger.add_watch("numbers")
    
    print(f"\n👁️ Отслеживаемые переменные: {list(debugger.watched_variables)}")
    
    # Запуск отладки (в симуляции)
    print(f"\n▶️ Запуск отладки...")
    debugger.start_debugging(debug_code, "debug_test.py")
    
    # Ждем завершения
    time.sleep(2)
    
    # Получение данных сессии
    session_data = debugger.export_debug_session()
    
    with open('debug_session.json', 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Сессия отладки:")
    print(f"  • События: {len(session_data['event_history'])}")
    print(f"  • Кадры стека: {len(session_data['call_stack'])}")
    print(f"  • Точки останова: {len(session_data['breakpoints'])}")
    print(f"  ✅ Данные сохранены: debug_session.json")
    
    debugger.stop()


@profile()
def fibonacci_test(n):
    """Тестовая функция для профилирования"""
    if n <= 1:
        return n
    return fibonacci_test(n-1) + fibonacci_test(n-2)


@profile()
def matrix_operations():
    """Матричные операции для тестирования"""
    import random
    
    # Создание матриц
    size = 100
    matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
    
    # Умножение матриц
    result = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result


@profile_neural("activation", "test_neuron")
def neural_computation(inputs):
    """Нейронные вычисления"""
    import math
    
    # Эмуляция нейронной активации
    weights = [0.5, -0.3, 0.8, 0.2, -0.1]
    bias = 0.1
    
    # Взвешенная сумма
    weighted_sum = sum(w * x for w, x in zip(weights, inputs)) + bias
    
    # Активация (sigmoid)  
    activation = 1 / (1 + math.exp(-weighted_sum))
    
    # Эмуляция задержки
    time.sleep(0.001)
    
    return activation


@profile_neural("forward_pass", "neural_network")
def neural_network_forward(network_inputs):
    """Прямой проход нейронной сети"""
    layer1 = [neural_computation(network_inputs) for _ in range(10)]
    layer2 = [neural_computation(layer1) for _ in range(5)]
    output = neural_computation(layer2)
    return output


def demo_profiler():
    """Демонстрация профайлера"""
    print("\n📊 === ДЕМОНСТРАЦИЯ ПРОФАЙЛЕРА ===")
    
    # Запуск профилирования
    print("🎯 Начало профилирования...")
    start_profiling("demo_session")
    
    # Тестовые вычисления
    print("  🔢 Вычисление чисел Фибоначчи...")
    for i in range(1, 15):
        result = fibonacci_test(i)
    
    print("  📊 Матричные операции...")
    matrix_operations()
    
    print("  🧠 Нейронные вычисления...")
    for i in range(50):
        inputs = [0.1 * j for j in range(5)]
        neural_network_forward(inputs)
    
    # Остановка профилирования
    print("⏱️ Завершение профилирования...")
    report = stop_profiling("demo_session")
    
    if report:
        print(f"\n📈 Отчет о производительности:")
        print(f"  • Общее время: {report['summary']['total_execution_time']:.3f}s")
        print(f"  • Вызовов функций: {report['summary']['total_function_calls']}")
        print(f"  • Нейронных операций: {report['summary']['total_neural_operations']}")
        print(f"  • Пиковая память: {report['summary']['peak_memory_usage']:.1f}MB")
        
        print(f"\n🔥 Топ-5 функций по времени:")
        for i, func in enumerate(report['top_functions'][:5], 1):
            print(f"  {i}. {func['name']}: {func['total_time']:.3f}s ({func['percentage']:.1f}%)")
        
        print(f"\n🧠 Нейронная производительность:")
        for i, neural in enumerate(report['neural_performance'][:3], 1):
            print(f"  {i}. {neural['operation']}: {neural['avg_time']*1000:.2f}ms")
        
        print(f"\n💡 Рекомендации:")
        for rec in report['performance_analysis']['recommendations']:
            print(f"  • {rec}")
        
        # Сохранение отчетов
        with open('performance_report_demo.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # HTML отчет
        from src.tools.profiler import _global_profiler
        analyzer = _global_profiler.active_sessions["demo_session"]
        analyzer.export_report("performance_report_demo.html", "html")
        
        print(f"\n💾 Отчеты сохранены:")
        print(f"  ✅ performance_report_demo.json")
        print(f"  ✅ performance_report_demo.html")


def demo_ide():
    """Демонстрация IDE"""
    print("\n🖥️ === ДЕМОНСТРАЦИЯ IDE ===")
    print("🚀 Запуск AnamorphX IDE...")
    print("   (Закройте окно IDE для продолжения демо)")
    
    try:
        # Запуск IDE (блокирующий вызов)
        launch_ide()
    except Exception as e:
        print(f"⚠️ Ошибка запуска IDE: {e}")
        print("   (Возможно, отсутствует поддержка GUI)")


def create_summary_report():
    """Создание итогового отчета"""
    print("\n📋 === СОЗДАНИЕ ИТОГОВОГО ОТЧЕТА ===")
    
    summary = {
        "title": "AnamorphX Development Tools Demo Report",
        "timestamp": time.time(),
        "tools_demonstrated": [
            {
                "name": "Syntax Highlighter",
                "description": "Подсветка синтаксиса для языка Anamorph",
                "features": [
                    "Поддержка 4 тем (light, dark, vs_code_light, vs_code_dark)",
                    "Экспорт в HTML и JSON",
                    "Генерация VS Code темы",
                    "Распознавание нейронных конструкций"
                ],
                "files_created": [
                    "syntax_highlight_*.html",
                    "syntax_tokens_*.json", 
                    "anamorph_vscode_theme.json"
                ]
            },
            {
                "name": "Debugger",
                "description": "Отладчик с поддержкой точек останова и инспекции",
                "features": [
                    "Точки останова (по строкам и функциям)",
                    "Пошаговое выполнение",
                    "Отслеживание переменных",
                    "Стек вызовов",
                    "Экспорт сессии отладки"
                ],
                "files_created": ["debug_session.json"]
            },
            {
                "name": "Performance Profiler", 
                "description": "Профайлер производительности с анализом нейронных операций",
                "features": [
                    "Профилирование функций",
                    "Анализ нейронных операций",
                    "Мониторинг памяти",
                    "Рекомендации по оптимизации",
                    "HTML и JSON отчеты"
                ],
                "files_created": [
                    "performance_report_demo.json",
                    "performance_report_demo.html"
                ]
            },
            {
                "name": "IDE Components",
                "description": "Компоненты интегрированной среды разработки",
                "features": [
                    "Редактор кода с подсветкой",
                    "Файловый менеджер",
                    "Встроенный терминал",
                    "Панель отладки",
                    "Настраиваемые темы"
                ],
                "files_created": []
            }
        ],
        "statistics": {
            "total_files_created": 0,
            "total_lines_of_code": 0,
            "supported_languages": ["Python", "JavaScript", "HTML", "JSON"],
            "ide_features": 15,
            "themes_supported": 4
        },
        "next_steps": [
            "Интеграция всех инструментов в единую IDE",
            "Добавление плагинной системы",
            "Расширение поддержки языков",
            "Создание пакетов для популярных редакторов",
            "Добавление юнит-тестов для всех компонентов"
        ]
    }
    
    # Подсчет созданных файлов
    created_files = []
    for tool in summary["tools_demonstrated"]:
        created_files.extend(tool["files_created"])
    
    # Фильтрация существующих файлов
    existing_files = [f for f in created_files if os.path.exists(f) or '*' in f]
    summary["statistics"]["total_files_created"] = len(existing_files)
    
    # Сохранение отчета
    with open('dev_tools_demo_report.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("📊 Итоговый отчет:")
    print(f"  • Инструментов продемонстрировано: {len(summary['tools_demonstrated'])}")
    print(f"  • Файлов создано: {summary['statistics']['total_files_created']}")
    print(f"  • Поддерживаемых языков: {len(summary['statistics']['supported_languages'])}")
    print(f"  ✅ Отчет сохранен: dev_tools_demo_report.json")


def main():
    """Главная функция демонстрации"""
    print("🎯 ДЕМОНСТРАЦИЯ ИНСТРУМЕНТОВ РАЗРАБОТКИ ANAMORPHX")
    print("=" * 60)
    
    # Демонстрация всех инструментов
    demo_syntax_highlighting()
    demo_debugger()  
    demo_profiler()
    
    # IDE демо (опционально)
    ide_demo = input("\n🖥️ Запустить демо IDE? (y/N): ").lower().strip()
    if ide_demo == 'y':
        demo_ide()
    else:
        print("⏭️ Пропуск демо IDE")
    
    # Итоговый отчет
    create_summary_report()
    
    print("\n✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("📁 Созданные файлы:")
    
    # Список созданных файлов
    demo_files = [
        "syntax_highlight_light.html",
        "syntax_highlight_dark.html", 
        "syntax_highlight_vs_code_dark.html",
        "syntax_tokens_light.json",
        "syntax_tokens_dark.json",
        "syntax_tokens_vs_code_dark.json",
        "anamorph_vscode_theme.json",
        "debug_session.json",
        "performance_report_demo.json",
        "performance_report_demo.html",
        "dev_tools_demo_report.json"
    ]
    
    for filename in demo_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✅ {filename} ({size} bytes)")
        else:
            print(f"  ❌ {filename} (не создан)")
    
    print(f"\n🎉 Все инструменты разработки AnamorphX готовы к использованию!")
    print(f"📚 Документация доступна в созданных HTML файлах")


if __name__ == "__main__":
    main() 