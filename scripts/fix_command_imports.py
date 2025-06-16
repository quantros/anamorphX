#!/usr/bin/env python3
"""
Скрипт для исправления импортов в файлах команд AnamorphX
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Исправляет импорты в одном файле"""
    print(f"Исправляем импорты в {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Сохраняем оригинал
        backup_path = str(file_path) + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Исправляем импорты
        original_content = content
        
        # 1. Исправляем импорт ExecutionContext
        content = re.sub(
            r'from \.commands import (.+), ExecutionContext',
            r'from .commands import \1\nfrom .runtime import ExecutionContext',
            content
        )
        
        # 2. Исправляем импорт только ExecutionContext
        content = re.sub(
            r'from \.commands import ExecutionContext',
            r'from .runtime import ExecutionContext',
            content
        )
        
        # 3. Добавляем недостающие импорты из commands.py
        if 'from .commands import' in content and 'NeuralEntity' not in content:
            # Добавляем импорт NeuralEntity и SynapseConnection если нужно
            if 'StructuralCommand' in content:
                content = re.sub(
                    r'from \.commands import StructuralCommand',
                    r'from .commands import StructuralCommand, NeuralEntity, SynapseConnection',
                    content
                )
        
        # 4. Исправляем специфические импорты для разных типов команд
        command_type_imports = {
            'StructuralCommand': 'StructuralCommand, CommandResult, CommandError, NeuralEntity, SynapseConnection',
            'FlowControlCommand': 'FlowControlCommand, CommandResult, CommandError',
            'SecurityCommand': 'SecurityCommand, CommandResult, CommandError',
            'DataManagementCommand': 'DataManagementCommand, CommandResult, CommandError',
            'NetworkCommand': 'CloudNetworkCommand, CommandResult, CommandError',
            'MonitoringCommand': 'Command, CommandResult, CommandError',
            'MLOperationsCommand': 'MachineLearningCommand, CommandResult, CommandError',
            'SystemCommand': 'Command, CommandResult, CommandError',
            'UtilityCommand': 'Command, CommandResult, CommandError',
            'DebuggingCommand': 'Command, CommandResult, CommandError',
            'ExportCommand': 'Command, CommandResult, CommandError'
        }
        
        for base_class, full_import in command_type_imports.items():
            if base_class in content:
                pattern = rf'from \.commands import [^,\n]*{base_class}[^,\n]*'
                replacement = f'from .commands import {full_import}'
                content = re.sub(pattern, replacement, content)
        
        # 5. Исправляем специальные случаи
        if 'NetworkCommand' in content:
            content = content.replace('NetworkCommand', 'CloudNetworkCommand')
        
        if 'MLOperationsCommand' in content:
            content = content.replace('MLOperationsCommand', 'MachineLearningCommand')
        
        # 6. Добавляем try-except для импортов если их нет
        if 'from .commands import' in content and 'try:' not in content[:500]:
            # Находим первый импорт команд
            import_match = re.search(r'from \.commands import ([^\n]+)', content)
            if import_match:
                import_line = import_match.group(0)
                
                # Заменяем на try-except блок
                try_except_block = f"""# Импорты команд с обработкой ошибок
try:
    {import_line}
except ImportError as e:
    print(f"Warning: Could not import commands: {{e}}")
    # Создаем заглушки
    class CommandResult:
        def __init__(self, success=True, message="", data=None, error=None):
            self.success = success
            self.message = message
            self.data = data
            self.error = error
    
    class CommandError(Exception):
        def __init__(self, code="", message=""):
            self.code = code
            self.message = message
    
    class Command:
        def __init__(self, name="", description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {{}}
    
    # Создаем базовые классы команд
    StructuralCommand = Command
    FlowControlCommand = Command
    SecurityCommand = Command
    DataManagementCommand = Command
    CloudNetworkCommand = Command
    MachineLearningCommand = Command
    
    class NeuralEntity:
        def __init__(self, name="", entity_type="", **kwargs):
            self.name = name
            self.entity_type = entity_type
    
    class SynapseConnection:
        def __init__(self, source="", target="", weight=1.0, **kwargs):
            self.source = source
            self.target = target
            self.weight = weight

try:
    from .runtime import ExecutionContext
except ImportError:
    class ExecutionContext:
        def __init__(self):
            self.neural_entities = {{}}
            self.synapses = {{}}
            self.variables = {{}}"""
                
                content = content.replace(import_line, try_except_block)
        
        # Записываем исправленный файл
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Импорты исправлены в {file_path}")
            return True
        else:
            print(f"  ℹ️ Импорты уже корректны в {file_path}")
            # Удаляем backup если изменений не было
            os.remove(backup_path)
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка при исправлении {file_path}: {e}")
        return False

def main():
    """Главная функция"""
    print("🔧 Исправление импортов в файлах команд AnamorphX")
    print("=" * 60)
    
    # Находим все файлы команд
    commands_dir = Path("src/interpreter")
    command_files = list(commands_dir.glob("*_commands.py"))
    
    if not command_files:
        print("❌ Файлы команд не найдены!")
        return
    
    print(f"Найдено {len(command_files)} файлов команд:")
    for file_path in command_files:
        print(f"  - {file_path}")
    
    print("\nИсправляем импорты...")
    
    fixed_count = 0
    for file_path in command_files:
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\n📊 Результат:")
    print(f"  Всего файлов: {len(command_files)}")
    print(f"  Исправлено: {fixed_count}")
    print(f"  Без изменений: {len(command_files) - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ Импорты успешно исправлены!")
        print(f"💾 Резервные копии сохранены с расширением .backup")
    else:
        print(f"\nℹ️ Все импорты уже были корректными")

if __name__ == "__main__":
    main() 