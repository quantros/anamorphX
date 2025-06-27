"""
Расширенные команды безопасности для AnamorphX

Реализация полного набора команд безопасности и защиты данных.
"""

import hashlib
import hmac
import secrets
import time
import json
import re
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .commands import SecurityCommand, CommandResult, CommandError, ExecutionContext


class SecurityLevel(Enum):
    """Уровни безопасности"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthMethod(Enum):
    """Методы аутентификации"""
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"


@dataclass
class SecurityPolicy:
    """Политика безопасности"""
    level: SecurityLevel
    encryption_required: bool = True
    audit_required: bool = True
    access_control: bool = True
    rate_limiting: bool = True
    whitelist_enabled: bool = False
    blacklist_enabled: bool = True
    max_attempts: int = 3
    lockout_duration: int = 300  # 5 минут
    session_timeout: int = 3600  # 1 час


@dataclass
class AccessRecord:
    """Запись о доступе"""
    user_id: str
    resource: str
    action: str
    timestamp: float
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ScrambleCommand(SecurityCommand):
    """Команда шифрования данных"""
    
    def __init__(self):
        super().__init__(
            name="scramble",
            description="Шифрует данные с использованием различных алгоритмов",
            parameters={
                "data": "Данные для шифрования",
                "algorithm": "Алгоритм шифрования (aes, fernet, rsa)",
                "key": "Ключ шифрования (опционально)",
                "mode": "Режим шифрования",
                "output_format": "Формат вывода (base64, hex, binary)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data", "")
            algorithm = kwargs.get("algorithm", "fernet")
            key = kwargs.get("key")
            mode = kwargs.get("mode", "default")
            output_format = kwargs.get("output_format", "base64")
            
            if not data:
                return CommandResult(
                    success=False,
                    message="Не указаны данные для шифрования",
                    error=CommandError("MISSING_DATA", "Data parameter required")
                )
            
            # Выполняем шифрование
            encryption_result = self._encrypt_data(data, algorithm, key, mode, output_format)
            
            # Сохраняем ключ в безопасном хранилище
            if not hasattr(context, 'encryption_keys'):
                context.encryption_keys = {}
            
            key_id = f"key_{len(context.encryption_keys)}"
            context.encryption_keys[key_id] = {
                "algorithm": algorithm,
                "key": encryption_result["key"],
                "created_at": time.time()
            }
            
            return CommandResult(
                success=True,
                message="Данные успешно зашифрованы",
                data={
                    "encrypted_data": encryption_result["encrypted_data"],
                    "key_id": key_id,
                    "algorithm": algorithm,
                    "output_format": output_format,
                    "data_size": len(data),
                    "encrypted_size": len(encryption_result["encrypted_data"])
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка шифрования: {str(e)}",
                error=CommandError("ENCRYPTION_ERROR", str(e))
            )
    
    def _encrypt_data(self, data: str, algorithm: str, key: Optional[str], 
                     mode: str, output_format: str) -> Dict[str, Any]:
        """Зашифровать данные"""
        if algorithm == "fernet":
            return self._encrypt_fernet(data, key, output_format)
        elif algorithm == "aes":
            return self._encrypt_aes(data, key, mode, output_format)
        else:
            raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
    
    def _encrypt_fernet(self, data: str, key: Optional[str], output_format: str) -> Dict[str, Any]:
        """Шифрование Fernet"""
        if key:
            # Используем предоставленный ключ
            fernet_key = key.encode() if isinstance(key, str) else key
        else:
            # Генерируем новый ключ
            fernet_key = Fernet.generate_key()
        
        fernet = Fernet(fernet_key)
        encrypted_data = fernet.encrypt(data.encode())
        
        if output_format == "base64":
            encrypted_str = base64.b64encode(encrypted_data).decode()
        elif output_format == "hex":
            encrypted_str = encrypted_data.hex()
        else:
            encrypted_str = encrypted_data.decode('latin-1')
        
        return {
            "encrypted_data": encrypted_str,
            "key": fernet_key.decode() if isinstance(fernet_key, bytes) else fernet_key,
            "algorithm": "fernet"
        }
    
    def _encrypt_aes(self, data: str, key: Optional[str], mode: str, output_format: str) -> Dict[str, Any]:
        """Шифрование AES (упрощенная реализация)"""
        # Для демонстрации используем простое XOR шифрование
        if not key:
            key = secrets.token_hex(16)
        
        key_bytes = key.encode()[:16].ljust(16, b'\0')  # Обрезаем или дополняем до 16 байт
        data_bytes = data.encode()
        
        encrypted_bytes = bytes(a ^ b for a, b in zip(data_bytes, key_bytes * (len(data_bytes) // 16 + 1)))
        
        if output_format == "base64":
            encrypted_str = base64.b64encode(encrypted_bytes).decode()
        elif output_format == "hex":
            encrypted_str = encrypted_bytes.hex()
        else:
            encrypted_str = encrypted_bytes.decode('latin-1')
        
        return {
            "encrypted_data": encrypted_str,
            "key": key,
            "algorithm": "aes_xor"
        }


# Регистрируем все команды
ADVANCED_SECURITY_COMMANDS = [
    ScrambleCommand(),
    # # AuditCommand()  # Дублируется с security_commands.py - удалено  # Дублируется с security_commands.py - удалено
] 