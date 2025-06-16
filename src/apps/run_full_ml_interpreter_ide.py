#!/usr/bin/env python3
"""
Скрипт запуска AnamorphX IDE - ПОЛНАЯ интеграция ML + Интерпретатор
Запускает полнофункциональную IDE с ВСЕМИ возможностями
"""

import sys
import os
import traceback

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Основные зависимости
    try:
        import tkinter
        print("✅ tkinter - OK")
    except ImportError:
        missing_deps.append("tkinter")
        print("❌ tkinter - MISSING")
    
    # ML зависимости
    ml_deps = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    ml_available = 0
    for dep, name in ml_deps.items():
        try:
            __import__(dep)
            print(f"✅ {name} - OK")
            ml_available += 1
        except ImportError:
            print(f"⚠️ {name} - MISSING (ML features limited)")
    
    print(f"📊 ML Libraries: {ml_available}/{len(ml_deps)} available")
    
    if missing_deps:
        print(f"\n❌ Critical dependencies missing: {', '.join(missing_deps)}")
        print("Please install missing dependencies and try again.")
        return False
    
    return True

def check_interpreter_components():
    """Проверка компонентов интерпретатора"""
    print("\n🤖 Checking interpreter components...")
    
    # Добавляем путь к src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    components = {
        'ExecutionEngine': 'interpreter.execution_engine',
        'ASTInterpreter': 'interpreter.ast_interpreter', 
        'TypeSystem': 'interpreter.type_system',
        'ErrorHandler': 'interpreter.error_handler',
        'MemoryManager': 'interpreter.enhanced_memory_manager',
        'Commands': 'interpreter.commands'
    }
    
    available_components = 0
    
    for comp_name, module_path in components.items():
        try:
            module = __import__(module_path, fromlist=[comp_name])
            getattr(module, comp_name)
            print(f"✅ {comp_name} - OK")
            available_components += 1
        except Exception as e:
            print(f"⚠️ {comp_name} - MISSING ({str(e)[:50]}...)")
    
    print(f"🔧 Interpreter Components: {available_components}/{len(components)} available")
    
    if available_components >= 3:
        print("✅ Interpreter ready (sufficient components)")
        return True
    else:
        print("⚠️ Interpreter partial (will use simulation mode)")
        return False

def display_capability_matrix():
    """Отображение матрицы возможностей"""
    print("\n" + "="*60)
    print("🚀 AnamorphX IDE - FULL ML + Interpreter Edition")
    print("="*60)
    
    # GUI Framework
    try:
        import tkinter
        gui_status = "✅ READY"
    except:
        gui_status = "❌ MISSING"
    
    # ML Integration
    ml_libs = ['torch', 'numpy', 'matplotlib', 'sklearn']
    ml_count = sum(1 for lib in ml_libs if check_import(lib))
    
    if ml_count == 4:
        ml_status = "✅ FULL"
    elif ml_count >= 2:
        ml_status = "⚠️ PARTIAL"
    else:
        ml_status = "❌ LIMITED"
    
    # Interpreter
    interpreter_ready = check_interpreter_components()
    interp_status = "✅ READY" if interpreter_ready else "⚠️ PARTIAL"
    
    print(f"GUI Framework:     {gui_status}")
    print(f"ML Integration:    {ml_status} ({ml_count}/4 libraries)")
    print(f"Interpreter:       {interp_status}")
    print(f"Overall Status:    {'✅ READY TO LAUNCH' if gui_status.startswith('✅') else '❌ CANNOT LAUNCH'}")
    print("="*60)
    
    # Возможности
    print("\n🎯 Available Features:")
    print("• 📝 Advanced Code Editor with Syntax Highlighting")
    print("• 🤖 Real-time ML Code Analysis")
    print("• 🧠 Neural Network Visualization")
    print("• 📈 Training Progress Monitoring")
    print("• 💡 ML-powered Auto-completion")
    print("• 🔍 Intelligent Code Suggestions")
    print("• ⚡ AnamorphX Code Execution")
    print("• 🐛 Advanced Debugging with ML Insights")
    print("• 📊 Performance Profiling")
    print("• 🎨 Professional UI/UX")
    
    return gui_status.startswith('✅')

def check_import(module_name):
    """Проверка импорта модуля"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    """Главная функция запуска"""
    print("🚀 Starting AnamorphX IDE - Full ML + Interpreter Edition")
    print("=" * 60)
    
    try:
        # Импорт и запуск IDE
        from full_ml_interpreter_ide import UnifiedMLIDE
        
        print("✅ IDE module loaded successfully")
        
        # Создание и запуск IDE
        ide = UnifiedMLIDE()
        print("✅ IDE instance created")
        
        print("🎉 IDE launched successfully!")
        print("💡 Tip: Use F5 to run code, Ctrl+M for ML analysis")
        
        # Запуск главного цикла
        ide.root.mainloop()
        
        print("👋 IDE closed successfully")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error launching IDE: {str(e)}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 