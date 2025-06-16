#!/usr/bin/env python3
"""
Запуск единой AnamorphX IDE с полностью интегрированным ML
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Импорт основного класса IDE
    from unified_ml_ide import UnifiedMLIDE, HAS_FULL_ML
    
    print("🚀 Запуск AnamorphX IDE - Unified ML Edition")
    print("=" * 50)
    print(f"🤖 ML Status: {'✅ Full PyTorch Integration' if HAS_FULL_ML else '⚠️ Simulated Mode'}")
    print("🧠 Neural Network Visualization: Enabled")
    print("📈 Real-time Training Monitor: Enabled") 
    print("🔍 ML Code Analysis: Enabled")
    print("💡 Smart Autocomplete: Enabled")
    print("=" * 50)
    
    # Создание и запуск IDE
    ide = UnifiedMLIDE()
    
    print("✨ IDE initialized successfully!")
    print("🎯 Loading sample AnamorphX neural network code...")
    print("🔄 Starting real-time ML analysis...")
    print("\n👋 Enjoy coding with ML superpowers!")
    
    # Запуск главного цикла
    ide.root.mainloop()
    
except KeyboardInterrupt:
    print("\n👋 AnamorphX IDE закрыта пользователем")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("💡 Убедитесь, что файл unified_ml_ide.py находится в той же папке")
except Exception as e:
    print(f"❌ Ошибка IDE: {e}")
    import traceback
    traceback.print_exc()
    
print("\n🔚 Завершение работы AnamorphX IDE") 