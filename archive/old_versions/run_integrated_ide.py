#!/usr/bin/env python3
"""
Запуск AnamorphX IDE - Integrated Interpreter Edition
Демонстрация интеграции интерпретатора с IDE
"""

import os
import sys
import time

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "tkinter": False,
        "torch": False,
        "numpy": False,
        "matplotlib": False,
        "sklearn": False
    }
    
    # Проверка tkinter
    try:
        import tkinter
        dependencies["tkinter"] = True
        print("✅ tkinter - GUI framework available")
    except ImportError:
        print("❌ tkinter - GUI framework not available")
    
    # Проверка ML библиотек
    try:
        import torch
        dependencies["torch"] = True
        print("✅ PyTorch - Deep learning framework available")
    except ImportError:
        print("⚠️ PyTorch - Deep learning framework not available (will use simulation)")
    
    try:
        import numpy
        dependencies["numpy"] = True
        print("✅ NumPy - Numerical computing available")
    except ImportError:
        print("⚠️ NumPy - Numerical computing not available")
    
    try:
        import matplotlib
        dependencies["matplotlib"] = True
        print("✅ Matplotlib - Plotting library available")
    except ImportError:
        print("⚠️ Matplotlib - Plotting library not available")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        dependencies["sklearn"] = True
        print("✅ scikit-learn - Machine learning library available")
    except ImportError:
        print("⚠️ scikit-learn - Machine learning library not available")
    
    return dependencies

def check_interpreter_components():
    """Проверка компонентов интерпретатора"""
    print("\n🧠 Checking interpreter components...")
    
    # Настройка путей
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    components = {}
    
    # Проверка компонентов
    try:
        from interpreter.execution_engine import ExecutionEngine
        components["ExecutionEngine"] = True
        print("✅ Execution Engine - Core execution system")
    except Exception as e:
        components["ExecutionEngine"] = False
        print(f"⚠️ Execution Engine - {e}")
    
    try:
        from interpreter.ast_interpreter import ASTInterpreter
        components["ASTInterpreter"] = True
        print("✅ AST Interpreter - Syntax tree processor")
    except Exception as e:
        components["ASTInterpreter"] = False
        print(f"⚠️ AST Interpreter - {e}")
    
    try:
        from interpreter.type_system import TypeSystem
        components["TypeSystem"] = True
        print("✅ Type System - Type checking and inference")
    except Exception as e:
        components["TypeSystem"] = False
        print(f"⚠️ Type System - {e}")
    
    try:
        from interpreter.error_handler import ErrorHandler
        components["ErrorHandler"] = True
        print("✅ Error Handler - Error management system")
    except Exception as e:
        components["ErrorHandler"] = False
        print(f"⚠️ Error Handler - {e}")
    
    try:
        from interpreter.enhanced_memory_manager import EnhancedMemoryManager
        components["MemoryManager"] = True
        print("✅ Memory Manager - Advanced memory management")
    except Exception as e:
        components["MemoryManager"] = False
        print(f"⚠️ Memory Manager - {e}")
    
    try:
        from interpreter.commands import CommandRegistry
        components["Commands"] = True
        print("✅ Command Registry - Command execution system")
    except Exception as e:
        components["Commands"] = False
        print(f"⚠️ Command Registry - {e}")
    
    return components

def print_system_status(dependencies, components):
    """Вывод статуса системы"""
    print("\n" + "="*60)
    print("📊 SYSTEM STATUS REPORT")
    print("="*60)
    
    # GUI статус
    gui_ready = dependencies.get("tkinter", False)
    print(f"🖥️  GUI Framework: {'✅ READY' if gui_ready else '❌ NOT AVAILABLE'}")
    
    # ML статус
    ml_components = ["torch", "numpy", "matplotlib", "sklearn"]
    ml_available = sum(dependencies.get(comp, False) for comp in ml_components)
    ml_status = "✅ FULL" if ml_available >= 3 else "⚠️ PARTIAL" if ml_available > 0 else "❌ NONE"
    print(f"🤖 ML Integration: {ml_status} ({ml_available}/{len(ml_components)} libraries)")
    
    # Интерпретатор статус
    interpreter_available = sum(components.values())
    interpreter_total = len(components)
    interpreter_status = "✅ READY" if interpreter_available >= 3 else "⚠️ PARTIAL" if interpreter_available > 0 else "❌ NOT AVAILABLE"
    print(f"🧠 Interpreter: {interpreter_status} ({interpreter_available}/{interpreter_total} components)")
    
    # Общий статус
    overall_ready = gui_ready and interpreter_available >= 3
    print(f"🚀 Overall Status: {'✅ READY TO LAUNCH' if overall_ready else '⚠️ PARTIAL FUNCTIONALITY'}")
    
    return overall_ready

def show_feature_matrix():
    """Показать матрицу возможностей"""
    print("\n📋 FEATURE MATRIX")
    print("-" * 40)
    
    features = [
        ("📝 Code Editor", "✅ Available"),
        ("🎨 Syntax Highlighting", "✅ Available"),
        ("📁 File Operations", "✅ Available"),
        ("▶️ Code Execution", "✅ Available"),
        ("🐛 Basic Debugging", "✅ Available"),
        ("📊 Variable Inspector", "✅ Available"),
        ("🤖 ML Code Analysis", "⚠️ Depends on ML libraries"),
        ("🧠 Neural Visualization", "⚠️ Depends on ML libraries"),
        ("🔍 Real-time Analysis", "⚠️ Depends on interpreter components"),
        ("💾 Project Management", "✅ Available"),
    ]
    
    for feature, status in features:
        print(f"  {feature:<25} {status}")

def launch_ide():
    """Запуск IDE"""
    print("\n🚀 LAUNCHING ANAMORPHX IDE...")
    print("-" * 40)
    
    try:
        # Импорт и запуск IDE
        from integrated_ide_interpreter import IntegratedMLIDE
        
        print("✅ IDE module loaded successfully")
        print("🎯 Initializing integrated interpreter...")
        
        ide = IntegratedMLIDE()
        
        print("🎨 Starting GUI...")
        ide.run()
        
    except ImportError as e:
        print(f"❌ Failed to import IDE: {e}")
        print("💡 Make sure integrated_ide_interpreter.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Failed to launch IDE: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Главная функция"""
    print("🚀 AnamorphX IDE - Integrated Interpreter Edition")
    print("=" * 60)
    print("Combining ML-powered IDE with native AnamorphX interpreter")
    print("=" * 60)
    
    # Проверка зависимостей
    dependencies = check_dependencies()
    
    # Проверка компонентов интерпретатора
    components = check_interpreter_components()
    
    # Статус системы
    ready = print_system_status(dependencies, components)
    
    # Матрица возможностей
    show_feature_matrix()
    
    if not dependencies.get("tkinter", False):
        print("\n❌ CRITICAL ERROR: tkinter not available")
        print("💡 Install tkinter to run the GUI")
        return
    
    if not ready:
        print("\n⚠️ WARNING: System not fully ready")
        print("💡 Some features may not work properly")
        
        response = input("\n❓ Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("👋 Exiting...")
            return
    
    print("\n" + "="*60)
    print("🎯 LAUNCHING IDE IN 3 SECONDS...")
    print("="*60)
    
    for i in range(3, 0, -1):
        print(f"⏰ {i}...")
        time.sleep(1)
    
    # Запуск IDE
    success = launch_ide()
    
    if success:
        print("\n✅ IDE session completed successfully")
    else:
        print("\n❌ IDE session failed")
    
    print("👋 Thank you for using AnamorphX IDE!")

if __name__ == "__main__":
    main() 