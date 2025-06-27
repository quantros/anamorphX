#!/usr/bin/env python3
"""
🧠 AnamorphX Neural Code Runner
РЕАЛЬНЫЙ запуск AnamorphX файлов с парсером и интерпретатором
"""

import sys
import os
import time
import traceback

# Добавляем пути для импортов
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

def run_anamorph_file(filename):
    """Реальный запуск файла AnamorphX"""
    print(f"🧠 AnamorphX Neural Runner")
    print(f"🚀 Running: {filename}")
    print("=" * 60)
    
    # Проверка файла
    if not os.path.exists(filename):
        print(f"❌ Файл не найден: {filename}")
        return False
    
    # Чтение кода
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        print(f"📝 Размер файла: {len(source_code)} символов")
        print(f"📏 Строк: {len(source_code.splitlines())}")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False
    
    start_time = time.time()
    
    try:
        # 1. ПАРСИНГ
        print("🔍 1. Парсинг кода...")
        from parser.parser import AnamorphParser
        
        parser = AnamorphParser(debug=True)
        ast = parser.parse(source_code, filename)
        
        parse_time = time.time() - start_time
        print(f"✅ Парсинг завершен за {parse_time:.3f}с")
        print(f"📊 AST узлов: {len(ast.body) if hasattr(ast, 'body') and ast.body else 0}")
        print()
        
        # 2. ИНТЕРПРЕТАЦИЯ  
        print("⚡ 2. Выполнение кода...")
        from interpreter.ast_interpreter import ASTInterpreter
        
        interpreter = ASTInterpreter()
        result = interpreter.interpret(ast)
        
        exec_time = time.time() - start_time - parse_time
        total_time = time.time() - start_time
        
        print(f"✅ Выполнение завершено за {exec_time:.3f}с")
        print()
        
        # 3. РЕЗУЛЬТАТЫ
        print("📋 3. Результаты выполнения:")
        print(f"⏱️  Общее время: {total_time:.3f}с")
        print(f"🔍 Время парсинга: {parse_time:.3f}с")
        print(f"⚡ Время выполнения: {exec_time:.3f}с")
        
        # Статистика интерпретатора
        if hasattr(interpreter, 'get_execution_summary'):
            summary = interpreter.get_execution_summary()
            print(f"📊 Узлов выполнено: {summary.get('nodes_executed', 0)}")
            print(f"🧠 Нейронов создано: {len(interpreter.state.neurons)}")
            print(f"🔗 Синапсов создано: {len(interpreter.state.synapses)}")
            print(f"🔢 Переменных: {len(interpreter.state.variables)}")
        
        # Переменные
        if hasattr(interpreter, 'state') and interpreter.state.variables:
            print("\n🔢 Созданные переменные:")
            for name, value in list(interpreter.state.variables.items())[:10]:
                print(f"  {name} = {value}")
            if len(interpreter.state.variables) > 10:
                print(f"  ... и еще {len(interpreter.state.variables) - 10}")
        
        # Нейроны
        if hasattr(interpreter, 'state') and interpreter.state.neurons:
            print(f"\n🧠 Нейроны ({len(interpreter.state.neurons)}):")
            for name, neuron in list(interpreter.state.neurons.items())[:5]:
                print(f"  {name}: {neuron}")
            if len(interpreter.state.neurons) > 5:
                print(f"  ... и еще {len(interpreter.state.neurons) - 5}")
        
        print(f"\n✅ УСПЕШНО! Результат: {result}")
        return True
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"\n❌ ОШИБКА за {error_time:.3f}с:")
        print(f"💥 {type(e).__name__}: {e}")
        
        if "Import" in str(e) or "module" in str(e).lower():
            print("\n🔧 Возможные решения:")
            print("  - Проверьте структуру проекта")
            print("  - Убедитесь что все модули на месте")
            print("  - Попробуйте запустить из корня проекта")
        
        if "--debug" in sys.argv:
            print("\n📋 Полная трассировка:")
            traceback.print_exc()
        
        return False

def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print("📋 Использование: python3 anamorph_runner.py <file.anamorph> [--debug]")
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
        print("💡 Примеры:")
        print("  python3 anamorph_runner.py Project/web_server.anamorph")
        print("  python3 anamorph_runner.py Project/config.anamorph --debug")
        return 1
    
    filename = sys.argv[1]
    success = run_anamorph_file(filename)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 