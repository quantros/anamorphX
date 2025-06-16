"""
Центральный реестр всех команд AnamorphX

Объединяет все 100 команд из 10 файлов по логическим группам.
"""

from typing import Dict, List, Any
from .commands import Command

# Импортируем команды из всех файлов
from .structural_commands import STRUCTURAL_COMMANDS
from .flow_control_commands import FLOW_CONTROL_COMMANDS
from .security_commands import SECURITY_COMMANDS
from .data_management_commands import DATA_MANAGEMENT_COMMANDS  
from .network_commands import NETWORK_COMMANDS
from .monitoring_commands import MONITORING_COMMANDS
from .ml_operations_commands import ML_COMMANDS
from .system_commands import SYSTEM_COMMANDS
from .utility_commands import UTILITY_COMMANDS
from .debugging_commands import DEBUGGING_COMMANDS
from .export_commands import EXPORT_COMMANDS

# Дополнительные команды (для совместимости)
try:
    from .advanced_flow_commands import ADVANCED_FLOW_COMMANDS
except ImportError:
    ADVANCED_FLOW_COMMANDS = []

try:
    from .advanced_security_commands import ADVANCED_SECURITY_COMMANDS
except ImportError:
    ADVANCED_SECURITY_COMMANDS = []


class CommandRegistry:
    """Центральный реестр всех команд"""
    
    def __init__(self):
        self.commands: Dict[str, Command] = {}
        self.command_groups: Dict[str, List[Command]] = {}
        self._register_all_commands()
    
    def _register_all_commands(self):
        """Регистрирует все команды из всех групп"""
        
        # Группа 1: Структурные команды (10 команд)
        self.command_groups["structural"] = STRUCTURAL_COMMANDS
        for cmd in STRUCTURAL_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 2: Управление потоком (10 команд)
        self.command_groups["flow_control"] = FLOW_CONTROL_COMMANDS
        for cmd in FLOW_CONTROL_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 3: Безопасность (10 команд)
        self.command_groups["security"] = SECURITY_COMMANDS
        for cmd in SECURITY_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 4: Управление данными (10 команд)
        self.command_groups["data_management"] = DATA_MANAGEMENT_COMMANDS
        for cmd in DATA_MANAGEMENT_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 5: Сетевые команды (10 команд)
        self.command_groups["network"] = NETWORK_COMMANDS
        for cmd in NETWORK_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 6: Мониторинг (10 команд)
        self.command_groups["monitoring"] = MONITORING_COMMANDS
        for cmd in MONITORING_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 7: ML операции (9 команд)
        self.command_groups["ml_operations"] = ML_COMMANDS
        for cmd in ML_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 8: Системные команды (10 команд)
        self.command_groups["system"] = SYSTEM_COMMANDS
        for cmd in SYSTEM_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 9: Утилитарные команды (10 команд)
        self.command_groups["utility"] = UTILITY_COMMANDS
        for cmd in UTILITY_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 10: Отладка (10 команд)
        self.command_groups["debugging"] = DEBUGGING_COMMANDS
        for cmd in DEBUGGING_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Группа 11: Экспорт (10 команд)
        self.command_groups["export"] = EXPORT_COMMANDS
        for cmd in EXPORT_COMMANDS:
            self.commands[cmd.name] = cmd
        
        # Дополнительные команды для совместимости
        if ADVANCED_FLOW_COMMANDS:
            self.command_groups["advanced_flow"] = ADVANCED_FLOW_COMMANDS
            for cmd in ADVANCED_FLOW_COMMANDS:
                if cmd.name not in self.commands:  # Избегаем дублирования
                    self.commands[cmd.name] = cmd
        
        if ADVANCED_SECURITY_COMMANDS:
            self.command_groups["advanced_security"] = ADVANCED_SECURITY_COMMANDS
            for cmd in ADVANCED_SECURITY_COMMANDS:
                if cmd.name not in self.commands:  # Избегаем дублирования
                    self.commands[cmd.name] = cmd
    
    def get_command(self, name: str) -> Command:
        """Получить команду по имени"""
        return self.commands.get(name)
    
    def get_commands_by_group(self, group: str) -> List[Command]:
        """Получить команды по группе"""
        return self.command_groups.get(group, [])
    
    def get_all_commands(self) -> Dict[str, Command]:
        """Получить все команды"""
        return self.commands.copy()
    
    def get_command_count(self) -> int:
        """Получить общее количество команд"""
        return len(self.commands)
    
    def get_group_stats(self) -> Dict[str, int]:
        """Получить статистику по группам"""
        return {group: len(commands) for group, commands in self.command_groups.items()}
    
    def list_commands(self, group: str = None) -> List[str]:
        """Список имен команд"""
        if group:
            return [cmd.name for cmd in self.command_groups.get(group, [])]
        return list(self.commands.keys())
    
    def search_commands(self, query: str) -> List[Command]:
        """Поиск команд по описанию или имени"""
        results = []
        query_lower = query.lower()
        
        for cmd in self.commands.values():
            if (query_lower in cmd.name.lower() or 
                query_lower in cmd.description.lower()):
                results.append(cmd)
        
        return results
    
    def get_commands_by_category(self) -> Dict[str, List[str]]:
        """Получить команды, сгруппированные по категориям"""
        categories = {
            "Core Operations": ["structural", "flow_control", "data_management"],
            "System & Infrastructure": ["system", "network", "monitoring"],
            "Development & Analysis": ["debugging", "utility", "ml_operations"],
            "Security & Export": ["security", "export"]
        }
        
        result = {}
        for category, groups in categories.items():
            result[category] = []
            for group in groups:
                if group in self.command_groups:
                    result[category].extend([cmd.name for cmd in self.command_groups[group]])
        
        return result


# Создаем глобальный экземпляр реестра
command_registry = CommandRegistry()


def get_command_registry() -> CommandRegistry:
    """Получить глобальный реестр команд"""
    return command_registry


def get_all_commands() -> Dict[str, Command]:
    """Получить все зарегистрированные команды"""
    return command_registry.get_all_commands()


def get_command_by_name(name: str) -> Command:
    """Получить команду по имени"""
    return command_registry.get_command(name)


def get_commands_summary() -> Dict[str, Any]:
    """Получить сводку по всем командам"""
    registry = command_registry
    
    return {
        "total_commands": registry.get_command_count(),
        "groups": registry.get_group_stats(),
        "command_list": registry.list_commands(),
        "groups_detail": {
            group: [cmd.name for cmd in commands]
            for group, commands in registry.command_groups.items()
        },
        "categories": registry.get_commands_by_category()
    }


def get_commands_report() -> str:
    """Получить детальный отчет о командах"""
    registry = command_registry
    stats = registry.get_group_stats()
    
    report = "# AnamorphX Commands Registry Report\n\n"
    report += f"**Total Commands:** {registry.get_command_count()}\n\n"
    
    report += "## Commands by Group:\n\n"
    for group, count in stats.items():
        report += f"- **{group.replace('_', ' ').title()}:** {count} commands\n"
    
    report += "\n## Detailed Command List:\n\n"
    for group, commands in registry.command_groups.items():
        report += f"### {group.replace('_', ' ').title()} ({len(commands)} commands)\n\n"
        for cmd in commands:
            report += f"- `{cmd.name}`: {cmd.description}\n"
        report += "\n"
    
    return report


# Экспортируем основные функции
__all__ = [
    'CommandRegistry',
    'command_registry', 
    'get_command_registry',
    'get_all_commands',
    'get_command_by_name',
    'get_commands_summary',
    'get_commands_report'
] 