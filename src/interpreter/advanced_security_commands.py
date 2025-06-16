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


class FilterCommand(SecurityCommand):
    """Команда фильтрации данных"""
    
    def __init__(self):
        super().__init__(
            name="filter",
            description="Фильтрует данные по заданным критериям",
            parameters={
                "data": "Данные для фильтрации",
                "rules": "Правила фильтрации",
                "mode": "Режим фильтрации (allow, deny, sanitize)",
                "patterns": "Паттерны для фильтрации",
                "case_sensitive": "Учитывать регистр (true/false)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data", "")
            rules = kwargs.get("rules", "").split(",")
            mode = kwargs.get("mode", "sanitize")
            patterns = kwargs.get("patterns", "").split(",")
            case_sensitive = kwargs.get("case_sensitive", "false").lower() == "true"
            
            # Выполняем фильтрацию
            filter_result = self._apply_filter(data, rules, mode, patterns, case_sensitive)
            
            return CommandResult(
                success=True,
                message=f"Фильтрация выполнена в режиме {mode}",
                data=filter_result
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка фильтрации: {str(e)}",
                error=CommandError("FILTER_ERROR", str(e))
            )
    
    def _apply_filter(self, data: str, rules: List[str], mode: str, 
                     patterns: List[str], case_sensitive: bool) -> Dict[str, Any]:
        """Применить фильтр к данным"""
        original_size = len(data)
        filtered_data = data
        applied_rules = []
        blocked_patterns = []
        
        # Компилируем паттерны
        compiled_patterns = []
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    compiled_patterns.append(re.compile(pattern, flags))
                except re.error:
                    # Если паттерн не является регулярным выражением, используем как строку
                    compiled_patterns.append(pattern)
        
        # Применяем правила фильтрации
        for rule in rules:
            rule = rule.strip()
            if not rule:
                continue
                
            if rule == "remove_html":
                filtered_data = self._remove_html_tags(filtered_data)
                applied_rules.append("remove_html")
            elif rule == "remove_scripts":
                filtered_data = self._remove_scripts(filtered_data)
                applied_rules.append("remove_scripts")
            elif rule == "sanitize_sql":
                filtered_data = self._sanitize_sql(filtered_data)
                applied_rules.append("sanitize_sql")
            elif rule == "remove_special_chars":
                filtered_data = self._remove_special_chars(filtered_data)
                applied_rules.append("remove_special_chars")
        
        # Применяем паттерны
        for pattern in compiled_patterns:
            if isinstance(pattern, re.Pattern):
                matches = pattern.findall(filtered_data)
                if matches:
                    blocked_patterns.extend(matches)
                    if mode == "deny":
                        return {
                            "filtered_data": "",
                            "blocked": True,
                            "reason": f"Blocked by pattern: {pattern.pattern}",
                            "original_size": original_size,
                            "filtered_size": 0
                        }
                    elif mode == "sanitize":
                        filtered_data = pattern.sub("[FILTERED]", filtered_data)
            else:
                # Простой поиск строки
                if pattern in filtered_data:
                    blocked_patterns.append(pattern)
                    if mode == "deny":
                        return {
                            "filtered_data": "",
                            "blocked": True,
                            "reason": f"Blocked by pattern: {pattern}",
                            "original_size": original_size,
                            "filtered_size": 0
                        }
                    elif mode == "sanitize":
                        filtered_data = filtered_data.replace(pattern, "[FILTERED]")
        
        return {
            "filtered_data": filtered_data,
            "blocked": False,
            "applied_rules": applied_rules,
            "blocked_patterns": blocked_patterns,
            "original_size": original_size,
            "filtered_size": len(filtered_data),
            "reduction_percent": ((original_size - len(filtered_data)) / original_size * 100) if original_size > 0 else 0
        }
    
    def _remove_html_tags(self, data: str) -> str:
        """Удалить HTML теги"""
        return re.sub(r'<[^>]+>', '', data)
    
    def _remove_scripts(self, data: str) -> str:
        """Удалить скрипты"""
        return re.sub(r'<script[^>]*>.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
    
    def _sanitize_sql(self, data: str) -> str:
        """Санитизация SQL"""
        dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', r'\bUPDATE\b',
            r'\bINSERT\b', r'\bEXEC\b', r'\bUNION\b', r'--', r'/\*', r'\*/'
        ]
        
        for pattern in dangerous_patterns:
            data = re.sub(pattern, '[SQL_FILTERED]', data, flags=re.IGNORECASE)
        
        return data
    
    def _remove_special_chars(self, data: str) -> str:
        """Удалить специальные символы"""
        return re.sub(r'[<>"\';()&+]', '', data)


class AuthCommand(SecurityCommand):
    """Команда аутентификации"""
    
    def __init__(self):
        super().__init__(
            name="auth",
            description="Выполняет аутентификацию пользователей",
            parameters={
                "user": "Имя пользователя",
                "credentials": "Учетные данные",
                "method": "Метод аутентификации",
                "session_timeout": "Время жизни сессии (секунды)",
                "require_2fa": "Требовать двухфакторную аутентификацию"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            user = kwargs.get("user")
            credentials = kwargs.get("credentials")
            method = kwargs.get("method", "password")
            session_timeout = int(kwargs.get("session_timeout", 3600))
            require_2fa = kwargs.get("require_2fa", "false").lower() == "true"
            
            if not user or not credentials:
                return CommandResult(
                    success=False,
                    message="Не указаны учетные данные",
                    error=CommandError("MISSING_CREDENTIALS", "User and credentials required")
                )
            
            # Выполняем аутентификацию
            auth_result = self._authenticate_user(
                context, user, credentials, method, session_timeout, require_2fa
            )
            
            if auth_result["success"]:
                # Создаем сессию
                session_id = self._create_session(context, user, session_timeout)
                auth_result["session_id"] = session_id
                
                # Логируем успешную аутентификацию
                self._log_access(context, user, "authentication", True)
                
                return CommandResult(
                    success=True,
                    message=f"Пользователь {user} успешно аутентифицирован",
                    data=auth_result
                )
            else:
                # Логируем неудачную попытку
                self._log_access(context, user, "authentication", False)
                
                return CommandResult(
                    success=False,
                    message="Аутентификация не удалась",
                    error=CommandError("AUTH_FAILED", auth_result.get("reason", "Invalid credentials"))
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка аутентификации: {str(e)}",
                error=CommandError("AUTH_ERROR", str(e))
            )
    
    def _authenticate_user(self, context: ExecutionContext, user: str, credentials: str,
                          method: str, session_timeout: int, require_2fa: bool) -> Dict[str, Any]:
        """Аутентифицировать пользователя"""
        # Инициализируем хранилище пользователей
        if not hasattr(context, 'users'):
            context.users = self._init_default_users()
        
        if not hasattr(context, 'failed_attempts'):
            context.failed_attempts = {}
        
        # Проверяем блокировку пользователя
        if user in context.failed_attempts:
            attempts_info = context.failed_attempts[user]
            if attempts_info["count"] >= 3 and time.time() - attempts_info["last_attempt"] < 300:
                return {
                    "success": False,
                    "reason": "Account temporarily locked due to multiple failed attempts"
                }
        
        # Проверяем существование пользователя
        if user not in context.users:
            return {
                "success": False,
                "reason": "User not found"
            }
        
        user_info = context.users[user]
        
        # Проверяем учетные данные
        if method == "password":
            if not self._verify_password(credentials, user_info.get("password_hash")):
                self._record_failed_attempt(context, user)
                return {
                    "success": False,
                    "reason": "Invalid password"
                }
        elif method == "token":
            if not self._verify_token(credentials, user_info.get("token")):
                self._record_failed_attempt(context, user)
                return {
                    "success": False,
                    "reason": "Invalid token"
                }
        
        # Проверяем 2FA если требуется
        if require_2fa and not user_info.get("2fa_verified", False):
            return {
                "success": False,
                "reason": "Two-factor authentication required"
            }
        
        # Сбрасываем счетчик неудачных попыток
        if user in context.failed_attempts:
            del context.failed_attempts[user]
        
        return {
            "success": True,
            "user_id": user,
            "method": method,
            "session_timeout": session_timeout,
            "permissions": user_info.get("permissions", []),
            "last_login": user_info.get("last_login"),
            "login_count": user_info.get("login_count", 0) + 1
        }
    
    def _init_default_users(self) -> Dict[str, Dict[str, Any]]:
        """Инициализировать пользователей по умолчанию"""
        return {
            "admin": {
                "password_hash": self._hash_password("admin123"),
                "permissions": ["all"],
                "created_at": time.time(),
                "active": True
            },
            "user": {
                "password_hash": self._hash_password("user123"),
                "permissions": ["read", "execute"],
                "created_at": time.time(),
                "active": True
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Хешировать пароль"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Проверить пароль"""
        try:
            salt, hash_hex = password_hash.split(':')
            expected_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_hex == expected_hash.hex()
        except:
            return False
    
    def _verify_token(self, token: str, expected_token: str) -> bool:
        """Проверить токен"""
        return token == expected_token
    
    def _record_failed_attempt(self, context: ExecutionContext, user: str):
        """Записать неудачную попытку"""
        if user not in context.failed_attempts:
            context.failed_attempts[user] = {"count": 0, "last_attempt": 0}
        
        context.failed_attempts[user]["count"] += 1
        context.failed_attempts[user]["last_attempt"] = time.time()
    
    def _create_session(self, context: ExecutionContext, user: str, timeout: int) -> str:
        """Создать сессию"""
        if not hasattr(context, 'sessions'):
            context.sessions = {}
        
        session_id = secrets.token_urlsafe(32)
        context.sessions[session_id] = {
            "user": user,
            "created_at": time.time(),
            "expires_at": time.time() + timeout,
            "active": True
        }
        
        return session_id
    
    def _log_access(self, context: ExecutionContext, user: str, action: str, success: bool):
        """Логировать доступ"""
        if not hasattr(context, 'access_log'):
            context.access_log = []
        
        record = AccessRecord(
            user_id=user,
            resource="authentication",
            action=action,
            timestamp=time.time(),
            success=success
        )
        
        context.access_log.append(record)


class AuditCommand(SecurityCommand):
    """Команда аудита безопасности"""
    
    def __init__(self):
        super().__init__(
            name="audit",
            description="Выполняет аудит безопасности системы",
            parameters={
                "scope": "Область аудита (all, users, permissions, sessions, logs)",
                "period": "Период аудита (last_hour, last_day, last_week)",
                "format": "Формат отчета (json, csv, html)",
                "include_details": "Включать детали (true/false)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            scope = kwargs.get("scope", "all")
            period = kwargs.get("period", "last_day")
            format_type = kwargs.get("format", "json")
            include_details = kwargs.get("include_details", "true").lower() == "true"
            
            # Выполняем аудит
            audit_result = self._perform_audit(context, scope, period, include_details)
            
            # Форматируем отчет
            formatted_report = self._format_audit_report(audit_result, format_type)
            
            return CommandResult(
                success=True,
                message=f"Аудит выполнен для области {scope}",
                data={
                    "audit_result": audit_result,
                    "formatted_report": formatted_report,
                    "scope": scope,
                    "period": period,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка аудита: {str(e)}",
                error=CommandError("AUDIT_ERROR", str(e))
            )
    
    def _perform_audit(self, context: ExecutionContext, scope: str, 
                      period: str, include_details: bool) -> Dict[str, Any]:
        """Выполнить аудит безопасности"""
        audit_result = {
            "scope": scope,
            "period": period,
            "timestamp": time.time(),
            "summary": {},
            "findings": [],
            "recommendations": []
        }
        
        # Определяем временной диапазон
        time_threshold = self._get_time_threshold(period)
        
        if scope in ["all", "users"]:
            audit_result["users"] = self._audit_users(context, time_threshold, include_details)
        
        if scope in ["all", "sessions"]:
            audit_result["sessions"] = self._audit_sessions(context, time_threshold, include_details)
        
        if scope in ["all", "logs"]:
            audit_result["logs"] = self._audit_logs(context, time_threshold, include_details)
        
        if scope in ["all", "permissions"]:
            audit_result["permissions"] = self._audit_permissions(context, include_details)
        
        # Генерируем сводку
        audit_result["summary"] = self._generate_audit_summary(audit_result)
        
        # Генерируем рекомендации
        audit_result["recommendations"] = self._generate_recommendations(audit_result)
        
        return audit_result
    
    def _get_time_threshold(self, period: str) -> float:
        """Получить временной порог для периода"""
        now = time.time()
        if period == "last_hour":
            return now - 3600
        elif period == "last_day":
            return now - 86400
        elif period == "last_week":
            return now - 604800
        else:
            return now - 86400
    
    def _audit_users(self, context: ExecutionContext, time_threshold: float, 
                    include_details: bool) -> Dict[str, Any]:
        """Аудит пользователей"""
        users_audit = {
            "total_users": 0,
            "active_users": 0,
            "inactive_users": 0,
            "admin_users": 0,
            "recent_logins": 0,
            "failed_attempts": 0
        }
        
        if hasattr(context, 'users'):
            users_audit["total_users"] = len(context.users)
            
            for user_id, user_info in context.users.items():
                if user_info.get("active", True):
                    users_audit["active_users"] += 1
                else:
                    users_audit["inactive_users"] += 1
                
                if "all" in user_info.get("permissions", []):
                    users_audit["admin_users"] += 1
                
                if user_info.get("last_login", 0) > time_threshold:
                    users_audit["recent_logins"] += 1
        
        if hasattr(context, 'failed_attempts'):
            users_audit["failed_attempts"] = len(context.failed_attempts)
        
        return users_audit
    
    def _audit_sessions(self, context: ExecutionContext, time_threshold: float,
                       include_details: bool) -> Dict[str, Any]:
        """Аудит сессий"""
        sessions_audit = {
            "total_sessions": 0,
            "active_sessions": 0,
            "expired_sessions": 0,
            "recent_sessions": 0
        }
        
        if hasattr(context, 'sessions'):
            sessions_audit["total_sessions"] = len(context.sessions)
            
            current_time = time.time()
            for session_id, session_info in context.sessions.items():
                if session_info.get("active", True) and session_info.get("expires_at", 0) > current_time:
                    sessions_audit["active_sessions"] += 1
                else:
                    sessions_audit["expired_sessions"] += 1
                
                if session_info.get("created_at", 0) > time_threshold:
                    sessions_audit["recent_sessions"] += 1
        
        return sessions_audit
    
    def _audit_logs(self, context: ExecutionContext, time_threshold: float,
                   include_details: bool) -> Dict[str, Any]:
        """Аудит логов"""
        logs_audit = {
            "total_records": 0,
            "recent_records": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "unique_users": set(),
            "top_actions": {}
        }
        
        if hasattr(context, 'access_log'):
            logs_audit["total_records"] = len(context.access_log)
            
            for record in context.access_log:
                if record.timestamp > time_threshold:
                    logs_audit["recent_records"] += 1
                
                if record.success:
                    logs_audit["successful_actions"] += 1
                else:
                    logs_audit["failed_actions"] += 1
                
                logs_audit["unique_users"].add(record.user_id)
                
                # Подсчитываем популярные действия
                action = record.action
                logs_audit["top_actions"][action] = logs_audit["top_actions"].get(action, 0) + 1
        
        logs_audit["unique_users"] = len(logs_audit["unique_users"])
        
        return logs_audit
    
    def _audit_permissions(self, context: ExecutionContext, include_details: bool) -> Dict[str, Any]:
        """Аудит разрешений"""
        permissions_audit = {
            "users_with_admin": 0,
            "users_with_limited_access": 0,
            "permission_distribution": {}
        }
        
        if hasattr(context, 'users'):
            for user_id, user_info in context.users.items():
                permissions = user_info.get("permissions", [])
                
                if "all" in permissions:
                    permissions_audit["users_with_admin"] += 1
                else:
                    permissions_audit["users_with_limited_access"] += 1
                
                for perm in permissions:
                    permissions_audit["permission_distribution"][perm] = \
                        permissions_audit["permission_distribution"].get(perm, 0) + 1
        
        return permissions_audit
    
    def _generate_audit_summary(self, audit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Генерировать сводку аудита"""
        summary = {
            "overall_status": "healthy",
            "risk_level": "low",
            "issues_found": 0,
            "recommendations_count": 0
        }
        
        # Анализируем результаты и определяем статус
        if "users" in audit_result:
            users_data = audit_result["users"]
            if users_data.get("failed_attempts", 0) > 10:
                summary["risk_level"] = "medium"
                summary["issues_found"] += 1
        
        if "sessions" in audit_result:
            sessions_data = audit_result["sessions"]
            if sessions_data.get("expired_sessions", 0) > sessions_data.get("active_sessions", 0):
                summary["issues_found"] += 1
        
        return summary
    
    def _generate_recommendations(self, audit_result: Dict[str, Any]) -> List[str]:
        """Генерировать рекомендации"""
        recommendations = []
        
        if "users" in audit_result:
            users_data = audit_result["users"]
            if users_data.get("admin_users", 0) > 1:
                recommendations.append("Рассмотрите ограничение количества администраторов")
            
            if users_data.get("failed_attempts", 0) > 5:
                recommendations.append("Усильте политику паролей и рассмотрите блокировку IP")
        
        if "sessions" in audit_result:
            sessions_data = audit_result["sessions"]
            if sessions_data.get("expired_sessions", 0) > 0:
                recommendations.append("Настройте автоматическую очистку истекших сессий")
        
        return recommendations
    
    def _format_audit_report(self, audit_result: Dict[str, Any], format_type: str) -> str:
        """Форматировать отчет аудита"""
        if format_type == "json":
            return json.dumps(audit_result, indent=2, default=str)
        elif format_type == "csv":
            return self._format_as_csv(audit_result)
        elif format_type == "html":
            return self._format_as_html(audit_result)
        else:
            return str(audit_result)
    
    def _format_as_csv(self, audit_result: Dict[str, Any]) -> str:
        """Форматировать как CSV"""
        lines = ["Category,Metric,Value"]
        
        for category, data in audit_result.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    lines.append(f"{category},{metric},{value}")
        
        return "\n".join(lines)
    
    def _format_as_html(self, audit_result: Dict[str, Any]) -> str:
        """Форматировать как HTML"""
        html = ["<html><head><title>Security Audit Report</title></head><body>"]
        html.append("<h1>Security Audit Report</h1>")
        html.append(f"<p>Generated at: {time.ctime(audit_result.get('timestamp', time.time()))}</p>")
        
        for category, data in audit_result.items():
            if isinstance(data, dict):
                html.append(f"<h2>{category.title()}</h2>")
                html.append("<table border='1'>")
                html.append("<tr><th>Metric</th><th>Value</th></tr>")
                
                for metric, value in data.items():
                    html.append(f"<tr><td>{metric}</td><td>{value}</td></tr>")
                
                html.append("</table>")
        
        html.append("</body></html>")
        return "\n".join(html)


# Регистрируем все команды
ADVANCED_SECURITY_COMMANDS = [
    ScrambleCommand(),
    FilterCommand(),
    AuthCommand(),
    AuditCommand()
] 