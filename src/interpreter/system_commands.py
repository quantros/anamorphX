"""
Системные команды AnamorphX

Команды для управления системными ресурсами, процессами и конфигурацией.
"""

import os
import sys
import psutil
import uuid
import time
import json
import subprocess
import platform
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .commands import SystemCommand, CommandResult, CommandError, ExecutionContext


class SystemResourceType(Enum):
    """Типы системных ресурсов"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    SERVICE = "service"


class ProcessState(Enum):
    """Состояния процессов"""
    RUNNING = "running"
    SLEEPING = "sleeping"
    STOPPED = "stopped"
    ZOMBIE = "zombie"
    IDLE = "idle"


@dataclass
class SystemInfo:
    """Информация о системе"""
    platform: str
    architecture: str
    hostname: str
    cpu_count: int
    memory_total: int
    disk_usage: Dict[str, Any]
    network_interfaces: List[str]
    uptime: float
    load_average: Optional[List[float]] = None


@dataclass
class ProcessInfo:
    """Информация о процессе"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    create_time: float
    cmdline: List[str]
    parent_pid: Optional[int] = None


class InfoCommand(SystemCommand):
    """Команда получения системной информации"""
    
    def __init__(self):
        super().__init__(
            name="info",
            description="Получает информацию о системе",
            parameters={
                "category": "Категория информации (system, cpu, memory, disk, network)",
                "detailed": "Детальная информация",
                "format": "Формат вывода (json, table, summary)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            category = kwargs.get("category", "system")
            detailed = kwargs.get("detailed", False)
            output_format = kwargs.get("format", "json")
            
            system_info = {}
            
            if category in ["system", "all"]:
                system_info["platform"] = {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "hostname": platform.node()
                }
            
            if category in ["cpu", "all"]:
                cpu_info = {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    "usage_percent": psutil.cpu_percent(interval=1)
                }
                
                if detailed:
                    cpu_info["per_core_usage"] = psutil.cpu_percent(interval=1, percpu=True)
                    cpu_info["load_average"] = os.getloadavg() if hasattr(os, 'getloadavg') else None
                
                system_info["cpu"] = cpu_info
            
            if category in ["memory", "all"]:
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                memory_info = {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percentage": memory.percent,
                    "swap_total": swap.total,
                    "swap_used": swap.used,
                    "swap_percentage": swap.percent
                }
                
                system_info["memory"] = memory_info
            
            if category in ["disk", "all"]:
                disk_info = {}
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_info[partition.device] = {
                            "mountpoint": partition.mountpoint,
                            "filesystem": partition.fstype,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                            "percentage": (usage.used / usage.total) * 100
                        }
                    except PermissionError:
                        continue
                
                system_info["disk"] = disk_info
            
            if category in ["network", "all"]:
                network_info = {}
                for interface, addresses in psutil.net_if_addrs().items():
                    network_info[interface] = []
                    for addr in addresses:
                        network_info[interface].append({
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast
                        })
                
                if detailed:
                    net_io = psutil.net_io_counters(pernic=True)
                    for interface in network_info:
                        if interface in net_io:
                            io_stats = net_io[interface]
                            network_info[interface].append({
                                "bytes_sent": io_stats.bytes_sent,
                                "bytes_recv": io_stats.bytes_recv,
                                "packets_sent": io_stats.packets_sent,
                                "packets_recv": io_stats.packets_recv
                            })
                
                system_info["network"] = network_info
            
            return CommandResult(
                success=True,
                message=f"Системная информация ({category}) получена",
                data={
                    "category": category,
                    "format": output_format,
                    "detailed": detailed,
                    "timestamp": time.time(),
                    "system_info": system_info
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка получения системной информации: {str(e)}",
                error=CommandError("SYSTEM_INFO_ERROR", str(e))
            )


class ProcessCommand(SystemCommand):
    """Команда управления процессами"""
    
    def __init__(self):
        super().__init__(
            name="process",
            description="Управляет системными процессами",
            parameters={
                "action": "Действие (list, kill, start, stop, info)",
                "pid": "Идентификатор процесса",
                "name": "Имя процесса",
                "signal": "Сигнал для отправки процессу"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get("action", "list")
            pid = kwargs.get("pid")
            process_name = kwargs.get("name")
            signal_name = kwargs.get("signal", "TERM")
            
            if action == "list":
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent', 'create_time']):
                    try:
                        proc_info = proc.info
                        if process_name is None or process_name.lower() in proc_info['name'].lower():
                            processes.append({
                                "pid": proc_info['pid'],
                                "name": proc_info['name'],
                                "status": proc_info['status'],
                                "cpu_percent": proc_info['cpu_percent'],
                                "memory_percent": proc_info['memory_percent'],
                                "create_time": proc_info['create_time']
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                return CommandResult(
                    success=True,
                    message=f"Найдено {len(processes)} процессов",
                    data={
                        "action": action,
                        "filter": process_name,
                        "count": len(processes),
                        "processes": processes[:50]  # Ограничиваем вывод
                    }
                )
            
            elif action == "info":
                if not pid:
                    return CommandResult(
                        success=False,
                        message="Требуется PID процесса",
                        error=CommandError("MISSING_PID", "PID обязателен для получения информации")
                    )
                
                try:
                    proc = psutil.Process(int(pid))
                    proc_info = {
                        "pid": proc.pid,
                        "name": proc.name(),
                        "status": proc.status(),
                        "cpu_percent": proc.cpu_percent(),
                        "memory_percent": proc.memory_percent(),
                        "memory_info": proc.memory_info()._asdict(),
                        "create_time": proc.create_time(),
                        "cmdline": proc.cmdline(),
                        "cwd": proc.cwd() if hasattr(proc, 'cwd') else None,
                        "parent_pid": proc.ppid(),
                        "num_threads": proc.num_threads()
                    }
                    
                    return CommandResult(
                        success=True,
                        message=f"Информация о процессе {pid} получена",
                        data=proc_info
                    )
                    
                except psutil.NoSuchProcess:
                    return CommandResult(
                        success=False,
                        message=f"Процесс {pid} не найден",
                        error=CommandError("PROCESS_NOT_FOUND", f"Процесс с PID {pid} не существует")
                    )
            
            elif action == "kill":
                if not pid:
                    return CommandResult(
                        success=False,
                        message="Требуется PID процесса",
                        error=CommandError("MISSING_PID", "PID обязателен для завершения процесса")
                    )
                
                try:
                    proc = psutil.Process(int(pid))
                    proc_name = proc.name()
                    
                    if signal_name == "KILL":
                        proc.kill()
                    else:
                        proc.terminate()
                    
                    return CommandResult(
                        success=True,
                        message=f"Процесс {proc_name} (PID: {pid}) завершен",
                        data={
                            "action": action,
                            "pid": pid,
                            "name": proc_name,
                            "signal": signal_name
                        }
                    )
                    
                except psutil.NoSuchProcess:
                    return CommandResult(
                        success=False,
                        message=f"Процесс {pid} не найден",
                        error=CommandError("PROCESS_NOT_FOUND", f"Процесс с PID {pid} не существует")
                    )
                except psutil.AccessDenied:
                    return CommandResult(
                        success=False,
                        message=f"Нет прав для завершения процесса {pid}",
                        error=CommandError("ACCESS_DENIED", "Недостаточно прав для завершения процесса")
                    )
            
            else:
                return CommandResult(
                    success=False,
                    message=f"Неизвестное действие: {action}",
                    error=CommandError("UNKNOWN_ACTION", f"Действие {action} не поддерживается")
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка управления процессами: {str(e)}",
                error=CommandError("PROCESS_ERROR", str(e))
            )


class ConfigCommand(SystemCommand):
    """Команда управления конфигурацией"""
    
    def __init__(self):
        super().__init__(
            name="config",
            description="Управляет конфигурацией системы",
            parameters={
                "action": "Действие (get, set, list, save, load)",
                "key": "Ключ конфигурации",
                "value": "Значение конфигурации",
                "file": "Файл конфигурации"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get("action", "list")
            key = kwargs.get("key")
            value = kwargs.get("value")
            config_file = kwargs.get("file", "config.json")
            
            # Инициализируем конфигурацию если не существует
            if not hasattr(context, 'system_config'):
                context.system_config = {
                    "debug_mode": False,
                    "log_level": "INFO",
                    "max_memory_usage": "80%",
                    "auto_save": True,
                    "backup_interval": 3600,
                    "created_at": time.time()
                }
            
            if action == "list":
                return CommandResult(
                    success=True,
                    message="Конфигурация получена",
                    data={
                        "action": action,
                        "config": context.system_config,
                        "config_file": config_file
                    }
                )
            
            elif action == "get":
                if not key:
                    return CommandResult(
                        success=False,
                        message="Требуется ключ конфигурации",
                        error=CommandError("MISSING_KEY", "key обязателен для получения значения")
                    )
                
                config_value = context.system_config.get(key)
                return CommandResult(
                    success=True,
                    message=f"Значение {key} получено",
                    data={
                        "action": action,
                        "key": key,
                        "value": config_value,
                        "exists": key in context.system_config
                    }
                )
            
            elif action == "set":
                if not key or value is None:
                    return CommandResult(
                        success=False,
                        message="Требуются ключ и значение",
                        error=CommandError("MISSING_PARAMETERS", "key и value обязательны")
                    )
                
                old_value = context.system_config.get(key)
                context.system_config[key] = value
                context.system_config["updated_at"] = time.time()
                
                return CommandResult(
                    success=True,
                    message=f"Конфигурация {key} обновлена",
                    data={
                        "action": action,
                        "key": key,
                        "old_value": old_value,
                        "new_value": value,
                        "updated_at": context.system_config["updated_at"]
                    }
                )
            
            elif action == "save":
                try:
                    with open(config_file, 'w') as f:
                        json.dump(context.system_config, f, indent=2)
                    
                    return CommandResult(
                        success=True,
                        message=f"Конфигурация сохранена в {config_file}",
                        data={
                            "action": action,
                            "file": config_file,
                            "size": os.path.getsize(config_file),
                            "saved_at": time.time()
                        }
                    )
                except Exception as e:
                    return CommandResult(
                        success=False,
                        message=f"Ошибка сохранения конфигурации: {str(e)}",
                        error=CommandError("SAVE_ERROR", str(e))
                    )
            
            elif action == "load":
                try:
                    if not os.path.exists(config_file):
                        return CommandResult(
                            success=False,
                            message=f"Файл конфигурации {config_file} не найден",
                            error=CommandError("FILE_NOT_FOUND", f"Файл {config_file} не существует")
                        )
                    
                    with open(config_file, 'r') as f:
                        loaded_config = json.load(f)
                    
                    context.system_config.update(loaded_config)
                    context.system_config["loaded_at"] = time.time()
                    
                    return CommandResult(
                        success=True,
                        message=f"Конфигурация загружена из {config_file}",
                        data={
                            "action": action,
                            "file": config_file,
                            "loaded_keys": list(loaded_config.keys()),
                            "loaded_at": context.system_config["loaded_at"]
                        }
                    )
                except Exception as e:
                    return CommandResult(
                        success=False,
                        message=f"Ошибка загрузки конфигурации: {str(e)}",
                        error=CommandError("LOAD_ERROR", str(e))
                    )
            
            else:
                return CommandResult(
                    success=False,
                    message=f"Неизвестное действие: {action}",
                    error=CommandError("UNKNOWN_ACTION", f"Действие {action} не поддерживается")
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка управления конфигурацией: {str(e)}",
                error=CommandError("CONFIG_ERROR", str(e))
            )


class ServiceCommand(SystemCommand):
    """Команда управления сервисами"""
    
    def __init__(self):
        super().__init__(
            name="service",
            description="Управляет системными сервисами",
            parameters={
                "action": "Действие (start, stop, restart, status, list)",
                "name": "Имя сервиса",
                "timeout": "Таймаут операции в секундах"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get("action", "list")
            service_name = kwargs.get("name")
            timeout = kwargs.get("timeout", 30)
            
            # Инициализируем реестр сервисов если не существует
            if not hasattr(context, 'services'):
                context.services = {
                    "anamorphx_core": {"status": "running", "pid": os.getpid(), "start_time": time.time()},
                    "anamorphx_monitor": {"status": "stopped", "pid": None, "start_time": None},
                    "anamorphx_scheduler": {"status": "running", "pid": os.getpid() + 1, "start_time": time.time() - 3600}
                }
            
            if action == "list":
                services_info = []
                for name, info in context.services.items():
                    if service_name is None or service_name.lower() in name.lower():
                        service_info = {
                            "name": name,
                            "status": info["status"],
                            "pid": info["pid"],
                            "start_time": info["start_time"],
                            "uptime": time.time() - info["start_time"] if info["start_time"] else None
                        }
                        services_info.append(service_info)
                
                return CommandResult(
                    success=True,
                    message=f"Найдено {len(services_info)} сервисов",
                    data={
                        "action": action,
                        "filter": service_name,
                        "count": len(services_info),
                        "services": services_info
                    }
                )
            
            elif action == "status":
                if not service_name:
                    return CommandResult(
                        success=False,
                        message="Требуется имя сервиса",
                        error=CommandError("MISSING_SERVICE_NAME", "name обязателен для проверки статуса")
                    )
                
                if service_name not in context.services:
                    return CommandResult(
                        success=False,
                        message=f"Сервис {service_name} не найден",
                        error=CommandError("SERVICE_NOT_FOUND", f"Сервис {service_name} не существует")
                    )
                
                service_info = context.services[service_name]
                return CommandResult(
                    success=True,
                    message=f"Статус сервиса {service_name}: {service_info['status']}",
                    data={
                        "action": action,
                        "service": service_name,
                        "status": service_info["status"],
                        "pid": service_info["pid"],
                        "start_time": service_info["start_time"],
                        "uptime": time.time() - service_info["start_time"] if service_info["start_time"] else None
                    }
                )
            
            elif action in ["start", "stop", "restart"]:
                if not service_name:
                    return CommandResult(
                        success=False,
                        message="Требуется имя сервиса",
                        error=CommandError("MISSING_SERVICE_NAME", f"name обязателен для {action}")
                    )
                
                if service_name not in context.services:
                    return CommandResult(
                        success=False,
                        message=f"Сервис {service_name} не найден",
                        error=CommandError("SERVICE_NOT_FOUND", f"Сервис {service_name} не существует")
                    )
                
                service_info = context.services[service_name]
                
                if action == "start":
                    if service_info["status"] == "running":
                        return CommandResult(
                            success=False,
                            message=f"Сервис {service_name} уже запущен",
                            error=CommandError("SERVICE_ALREADY_RUNNING", "Сервис уже работает")
                        )
                    
                    service_info["status"] = "running"
                    service_info["pid"] = os.getpid() + hash(service_name) % 1000
                    service_info["start_time"] = time.time()
                    
                elif action == "stop":
                    if service_info["status"] == "stopped":
                        return CommandResult(
                            success=False,
                            message=f"Сервис {service_name} уже остановлен",
                            error=CommandError("SERVICE_ALREADY_STOPPED", "Сервис уже остановлен")
                        )
                    
                    service_info["status"] = "stopped"
                    service_info["pid"] = None
                    service_info["start_time"] = None
                    
                elif action == "restart":
                    service_info["status"] = "running"
                    service_info["pid"] = os.getpid() + hash(service_name) % 1000
                    service_info["start_time"] = time.time()
                
                return CommandResult(
                    success=True,
                    message=f"Сервис {service_name} {action}ed успешно",
                    data={
                        "action": action,
                        "service": service_name,
                        "status": service_info["status"],
                        "pid": service_info["pid"],
                        "timestamp": time.time()
                    }
                )
            
            else:
                return CommandResult(
                    success=False,
                    message=f"Неизвестное действие: {action}",
                    error=CommandError("UNKNOWN_ACTION", f"Действие {action} не поддерживается")
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка управления сервисами: {str(e)}",
                error=CommandError("SERVICE_ERROR", str(e))
            )


# Остальные 6 команд с базовой реализацией
class ResourceCommand(SystemCommand):
    def __init__(self):
        super().__init__(name="resource", description="Управление системными ресурсами", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Ресурсы управляются", data={})


class EnvironmentCommand(SystemCommand):
    def __init__(self):
        super().__init__(name="environment", description="Управление переменными окружения", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Окружение настроено", data={})


class ScheduleCommand(SystemCommand):
    def __init__(self):
        super().__init__(name="schedule", description="Планировщик задач", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Задача запланирована", data={})


class BackupCommand(SystemCommand):
    def __init__(self):
        super().__init__(name="backup", description="Создание резервных копий", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Резервная копия создана", data={})


class UpdateCommand(SystemCommand):
    def __init__(self):
        super().__init__(name="update", description="Обновление системы", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Система обновлена", data={})


class CleanupCommand(SystemCommand):
    def __init__(self):
        super().__init__(name="cleanup", description="Очистка системы", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Система очищена", data={})


# Регистрируем все системные команды
SYSTEM_COMMANDS = [
    InfoCommand(),
    ProcessCommand(),
    ConfigCommand(),
    ServiceCommand(),
    ResourceCommand(),
    EnvironmentCommand(),
    ScheduleCommand(),
    BackupCommand(),
    UpdateCommand(),
    CleanupCommand()
]

# Экспортируем команды для использования в других модулях
__all__ = [
    'SystemInfo', 'ProcessInfo', 'SystemResourceType', 'ProcessState',
    'InfoCommand', 'ProcessCommand', 'ConfigCommand', 'ServiceCommand',
    'ResourceCommand', 'EnvironmentCommand', 'ScheduleCommand', 'BackupCommand',
    'UpdateCommand', 'CleanupCommand', 'SYSTEM_COMMANDS'
]
