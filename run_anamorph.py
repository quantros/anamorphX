#!/usr/bin/env python3
"""
🧠 AnamorphX Console Runner
Простой консольный запуск AnamorphX файлов
"""

import sys
import os
import time

# Добавляем путь к src
sys.path.insert(0, 'src')

def load_interpreter():
    """Загрузка интерпретатора AnamorphX"""
    try:
        from apps.full_ml_interpreter_ide import AnamorphXInterpreter
        print("✅ AnamorphX Interpreter loaded")
        return AnamorphXInterpreter()
    except Exception as e:
        print(f"❌ Failed to load interpreter: {e}")
        print("🔄 Trying alternative method...")
        
        # Альтернативный метод загрузки
        try:
            from interpreter.ast_interpreter import ASTInterpreter
            print("✅ AST Interpreter loaded")
            return ASTInterpreter()
        except Exception as e2:
            print(f"❌ Alternative method failed: {e2}")
            return None

def run_file(filepath):
    """Запуск файла AnamorphX"""
    print(f"🚀 Running AnamorphX file: {filepath}")
    print("=" * 60)
    
    # Проверка существования файла
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Загрузка интерпретатора
    interpreter = load_interpreter()
    if not interpreter:
        print("❌ No interpreter available")
        return False
    
    # Чтение файла
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        print(f"📝 File size: {len(code)} characters")
        print(f"📏 Lines: {len(code.splitlines())}")
        print()
        
        # Выполнение
        start_time = time.time()
        
        if hasattr(interpreter, 'execute_code'):
            result = interpreter.execute_code(code)
        elif hasattr(interpreter, 'interpret'):
            result = interpreter.interpret(code)
        else:
            print("❌ Interpreter method not found")
            return False
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Результат
        print()
        print("=" * 60)
        print(f"⏱️  Execution time: {execution_time:.3f} seconds")
        
        if isinstance(result, dict):
            if result.get('success', False):
                print("✅ Execution completed successfully")
                if 'variables' in result and result['variables']:
                    print(f"🔢 Variables created: {len(result['variables'])}")
                    for name, value in list(result['variables'].items())[:5]:  # Показать первые 5
                        print(f"  {name} = {value}")
                    if len(result['variables']) > 5:
                        print(f"  ... and {len(result['variables']) - 5} more")
                
                if 'output' in result and result['output']:
                    print(f"📤 Output: {result['output']}")
            else:
                print(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"✅ Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return False

def main():
    """Главная функция"""
    if len(sys.argv) != 2:
        print("📋 Usage: python3 run_anamorph.py <file.anamorph>")
        print()
        print("📁 Available files in Project/:")
        try:
            files = [f for f in os.listdir('Project/') if f.endswith('.anamorph')]
            for f in files:
                size = os.path.getsize(f'Project/{f}')
                print(f"  📄 {f} ({size} bytes)")
        except:
            print("  ❌ Project/ directory not found")
        
        print()
        print("💡 Example: python3 run_anamorph.py Project/web_server.anamorph")
        return 1
    
    filepath = sys.argv[1]
    success = run_file(filepath)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 