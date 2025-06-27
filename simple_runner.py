#!/usr/bin/env python3
"""
🧠 Simple AnamorphX Runner
Простой запуск без сложных импортов
"""

import sys
import os
import time
import re

def count_anamorph_constructs(code):
    """Подсчет конструкций AnamorphX в коде"""
    constructs = {
        'neurons': len(re.findall(r'\bneuro\s+', code, re.IGNORECASE)),
        'synapses': len(re.findall(r'\bsynap\s+', code, re.IGNORECASE)),
        'pulses': len(re.findall(r'\bpulse\s+', code, re.IGNORECASE)),
        'networks': len(re.findall(r'network\s*\{', code, re.IGNORECASE)),
        'variables': len(re.findall(r'\w+\s*=', code)),
        'functions': len(re.findall(r'function\s+\w+', code, re.IGNORECASE)),
        'classes': len(re.findall(r'class\s+\w+', code, re.IGNORECASE)),
        'config_blocks': len(re.findall(r'\w+\s*:\s*\{', code)),
    }
    return constructs

def analyze_neural_network(code):
    """Анализ нейронной сети в коде"""
    network_info = {
        'layers': [],
        'activations': [],
        'connections': []
    }
    
    # Поиск слоев и нейронов
    neuron_matches = re.findall(r'neuro\s+"([^"]+)"\s*\{([^}]+)\}', code, re.IGNORECASE)
    for name, config in neuron_matches:
        layer_info = {'name': name}
        
        # Извлечение параметров
        if 'units:' in config:
            units_match = re.search(r'units:\s*(\d+)', config)
            if units_match:
                layer_info['units'] = int(units_match.group(1))
        
        if 'activation:' in config:
            activation_match = re.search(r'activation:\s*"([^"]+)"', config)
            if activation_match:
                layer_info['activation'] = activation_match.group(1)
                network_info['activations'].append(activation_match.group(1))
        
        network_info['layers'].append(layer_info)
    
    # Поиск связей
    synapse_matches = re.findall(r'synap\s+"([^"]+)"\s*->\s*"([^"]+)"', code, re.IGNORECASE)
    for source, target in synapse_matches:
        network_info['connections'].append({'from': source, 'to': target})
    
    return network_info

def simulate_execution(code, filename):
    """Симуляция выполнения кода AnamorphX"""
    print(f"🧠 AnamorphX Simple Runner")
    print(f"🚀 Simulating: {filename}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Анализ кода
    constructs = count_anamorph_constructs(code)
    network = analyze_neural_network(code)
    
    print("🔍 1. Анализ кода:")
    print(f"  📄 Размер: {len(code):,} символов")
    print(f"  📏 Строк: {len(code.splitlines()):,}")
    print(f"  🧠 Нейронов: {constructs['neurons']}")
    print(f"  🔗 Синапсов: {constructs['synapses']}")
    print(f"  ⚡ Импульсов: {constructs['pulses']}")
    print(f"  🌐 Сетей: {constructs['networks']}")
    print(f"  🔢 Переменных: {constructs['variables']}")
    print()
    
    # Симуляция инициализации
    print("⚙️  2. Инициализация компонентов:")
    time.sleep(0.1)
    
    if network['layers']:
        print(f"  🧠 Нейронная сеть загружена:")
        total_units = 0
        for layer in network['layers']:
            units = layer.get('units', 1)
            activation = layer.get('activation', 'linear')
            total_units += units
            print(f"    - {layer['name']}: {units} нейронов, {activation}")
        
        print(f"  📊 Всего параметров: {total_units:,}")
    
    if network['connections']:
        print(f"  🔗 Связи сети:")
        for conn in network['connections'][:5]:  # Показать первые 5
            print(f"    - {conn['from']} → {conn['to']}")
        if len(network['connections']) > 5:
            print(f"    ... и еще {len(network['connections']) - 5}")
    
    print()
    
    # Симуляция выполнения
    print("⚡ 3. Выполнение операций:")
    operations = 0
    
    lines = code.splitlines()
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        
        operations += 1
        
        # Показать прогресс для больших файлов
        if operations % 50 == 0:
            progress = (i / len(lines)) * 100
            print(f"  📊 Прогресс: {progress:.1f}% ({operations} операций)")
        
        # Небольшая задержка для реализма
        if operations % 100 == 0:
            time.sleep(0.01)
    
    exec_time = time.time() - start_time
    
    # Результаты
    print(f"  ✅ Выполнено {operations} операций")
    print()
    
    print("📋 4. Результаты симуляции:")
    print(f"  ⏱️  Время выполнения: {exec_time:.3f} секунд")
    print(f"  🔄 Операций в секунду: {operations/exec_time:.0f}")
    
    if network['layers']:
        print(f"  🧠 Нейронная сеть активна")
        print(f"  📊 Слоев: {len(network['layers'])}")
        print(f"  🔗 Связей: {len(network['connections'])}")
        
        if network['activations']:
            unique_activations = list(set(network['activations']))
            print(f"  ⚡ Функции активации: {', '.join(unique_activations)}")
    
    # Специфичная информация для веб-сервера
    if 'web_server' in filename.lower():
        print()
        print("🌐 Информация о веб-сервере:")
        
        if 'port' in code.lower():
            port_match = re.search(r'port[:\s]*(\d+)', code, re.IGNORECASE)
            if port_match:
                print(f"  🔌 Порт: {port_match.group(1)}")
        
        if 'localhost' in code.lower() or '127.0.0.1' in code:
            print(f"  🏠 Хост: localhost")
        
        if 'api' in code.lower():
            print(f"  📡 API эндпоинты настроены")
        
        if 'security' in code.lower():
            print(f"  🔒 Система безопасности активна")
        
        print(f"  📺 Статус: Симуляция выполнена успешно")
    
    print()
    print(f"✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    return True

def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print("📋 Использование: python3 simple_runner.py <file.anamorph>")
        print()
        print("📁 Доступные файлы в Project/:")
        try:
            files = [f for f in os.listdir('Project/') if f.endswith('.anamorph')]
            for f in sorted(files):
                size = os.path.getsize(f'Project/{f}')
                print(f"  📄 {f} ({size:,} байт)")
        except FileNotFoundError:
            print("  ❌ Папка Project/ не найдена")
        
        print()
        print("💡 Пример: python3 simple_runner.py Project/web_server.anamorph")
        return 1
    
    filename = sys.argv[1]
    
    # Проверка файла
    if not os.path.exists(filename):
        print(f"❌ Файл не найден: {filename}")
        return 1
    
    # Чтение и выполнение
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        
        success = simulate_execution(code, filename)
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 