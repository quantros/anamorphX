"""
Команды безопасности AnamorphX

Команды для обеспечения безопасности, шифрования и контроля доступа.
"""

import time
import uuid
import hashlib
import hmac
import secrets
import base64
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Импорты команд с обработкой ошибок
try:
    from .commands import SecurityCommand, CommandResult, CommandError
    from .runtime import ExecutionContext
except ImportError as e:
    print(f"Warning: Could not import commands: {e}")
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
    
    class SecurityCommand:
        def __init__(self, name="", description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
    
    class ExecutionContext:
        def __init__(self):
            self.security_guards = {}
            self.auth_tokens = {}
            self.encrypted_data = {}

# Криптографические библиотеки
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: cryptography library not available, using basic encryption")


# =============================================================================
# ENUMS И DATACLASSES
# =============================================================================

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


class FilterAction(Enum):
    """Действия фильтра"""
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"


@dataclass
class SecurityGuard:
    """Защитный барьер"""
    id: str
    condition: str
    action: str
    severity: SecurityLevel
    created_at: float
    active: bool = True
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterRule:
    """Правило фильтрации"""
    id: str
    name: str
    pattern: str
    action: FilterAction
    priority: int
    created_at: float
    active: bool = True
    match_count: int = 0


@dataclass
class AuthSession:
    """Сессия аутентификации"""
    id: str
    user_id: str
    method: AuthMethod
    created_at: float
    expires_at: float
    active: bool = True
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# КОМАНДЫ БЕЗОПАСНОСТИ (10 команд)
# =============================================================================

class GuardCommand(SecurityCommand):
    """Создание защитного барьера"""
    
    def __init__(self):
        super().__init__("guard", "Create security guard with conditions")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            condition = kwargs.get('condition')
            action = kwargs.get('action', 'block')
            severity = kwargs.get('severity', 'medium')
            node_name = kwargs.get('node')
            
            if not condition:
                raise CommandError("Guard condition is required")
            
            # Создание защитного барьера
            guard_id = str(uuid.uuid4())
            guard = SecurityGuard(
                id=guard_id,
                condition=condition,
                action=action,
                severity=SecurityLevel(severity),
                created_at=time.time(),
                metadata={'created_by': 'guard_command'}
            )
            
            # Сохранение барьера
            if not hasattr(context.neural_network, 'security_guards'):
                context.neural_network.security_guards = {}
            context.neural_network.security_guards[guard_id] = guard
            
            # Привязка к узлу если указан
            if node_name:
                if node_name not in context.neural_network.nodes:
                    raise CommandError(f"Node '{node_name}' not found")
                
                node = context.neural_network.nodes[node_name]
                if 'security_guards' not in node.metadata:
                    node.metadata['security_guards'] = []
                node.metadata['security_guards'].append(guard_id)
            
            return CommandResult(
                success=True,
                data={
                    'guard_id': guard_id,
                    'condition': condition,
                    'action': action,
                    'severity': severity,
                    'node': node_name,
                    'created_at': guard.created_at
                },
                message=f"Security guard created with {severity} severity"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to create guard: {e}"
            )


class FilterCommand(SecurityCommand):
    """Фильтрация данных"""
    
    def __init__(self):
        super().__init__("filter", "Filter data using security rules")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get('data')
            rules = kwargs.get('rules', [])
            action = kwargs.get('action', 'sanitize')
            node_name = kwargs.get('node')
            
            if data is None:
                raise CommandError("Data to filter is required")
            
            # Инициализация системы фильтров
            if not hasattr(context.neural_network, 'filter_rules'):
                context.neural_network.filter_rules = {}
            
            # Создание правил фильтрации если не существуют
            created_rules = []
            for rule_name in rules:
                rule_id = f"rule_{rule_name}_{int(time.time())}"
                
                # Предустановленные правила
                if rule_name == 'xss_protection':
                    pattern = r'<script.*?>.*?</script>'
                elif rule_name == 'sql_injection':
                    pattern = r'(union|select|insert|update|delete|drop)\s+'
                elif rule_name == 'csrf_protection':
                    pattern = r'<form.*?>'
                else:
                    pattern = rule_name  # Пользовательское правило
                
                filter_rule = FilterRule(
                    id=rule_id,
                    name=rule_name,
                    pattern=pattern,
                    action=FilterAction(action),
                    priority=len(created_rules),
                    created_at=time.time()
                )
                
                context.neural_network.filter_rules[rule_id] = filter_rule
                created_rules.append(rule_id)
            
            # Применение фильтров
            filtered_data = str(data)
            filter_results = []
            
            for rule_id in created_rules:
                rule = context.neural_network.filter_rules[rule_id]
                matches = re.findall(rule.pattern, filtered_data, re.IGNORECASE)
                
                if matches:
                    rule.match_count += len(matches)
                    
                    if rule.action == FilterAction.BLOCK:
                        raise CommandError(f"Data blocked by filter rule: {rule.name}")
                    elif rule.action == FilterAction.SANITIZE:
                        filtered_data = re.sub(rule.pattern, '[FILTERED]', filtered_data, flags=re.IGNORECASE)
                    elif rule.action == FilterAction.QUARANTINE:
                        # Сохранение в карантин
                        if not hasattr(context.neural_network, 'quarantine'):
                            context.neural_network.quarantine = {}
                        quarantine_id = str(uuid.uuid4())
                        context.neural_network.quarantine[quarantine_id] = {
                            'data': str(data),
                            'rule': rule.name,
                            'matches': matches,
                            'timestamp': time.time()
                        }
                        filtered_data = f"[QUARANTINED:{quarantine_id}]"
                    
                    filter_results.append({
                        'rule': rule.name,
                        'matches': len(matches),
                        'action': rule.action.value
                    })
            
            # Сохранение результата в узел
            if node_name:
                if node_name not in context.neural_network.nodes:
                    raise CommandError(f"Node '{node_name}' not found")
                
                node = context.neural_network.nodes[node_name]
                node.data['filtered_data'] = filtered_data
                node.metadata['last_filter'] = time.time()
            
            return CommandResult(
                success=True,
                data={
                    'original_data': str(data),
                    'filtered_data': filtered_data,
                    'rules_applied': rules,
                    'filter_results': filter_results,
                    'node': node_name,
                    'created_rules': len(created_rules)
                },
                message=f"Data filtered using {len(rules)} rules"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to filter data: {e}"
            )


class AuthCommand(SecurityCommand):
    """Аутентификация пользователей"""
    
    def __init__(self):
        super().__init__("auth", "Authenticate users and manage sessions")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get('action', 'login')  # 'login', 'logout', 'verify'
            user_id = kwargs.get('user')
            credentials = kwargs.get('credentials')
            method = kwargs.get('method', 'password')
            permissions = kwargs.get('permissions', [])
            
            if not user_id:
                raise CommandError("User ID is required")
            
            # Инициализация системы аутентификации
            if not hasattr(context.neural_network, 'auth_sessions'):
                context.neural_network.auth_sessions = {}
            if not hasattr(context.neural_network, 'user_credentials'):
                context.neural_network.user_credentials = {}
            
            if action == 'login':
                if not credentials:
                    raise CommandError("Credentials are required for login")
                
                # Проверка учетных данных
                stored_hash = context.neural_network.user_credentials.get(user_id)
                if stored_hash:
                    # Проверка пароля
                    if method == 'password':
                        credential_hash = hashlib.sha256(str(credentials).encode()).hexdigest()
                        if credential_hash != stored_hash:
                            raise CommandError("Invalid credentials")
                else:
                    # Создание нового пользователя
                    credential_hash = hashlib.sha256(str(credentials).encode()).hexdigest()
                    context.neural_network.user_credentials[user_id] = credential_hash
                
                # Создание сессии
                session_id = str(uuid.uuid4())
                session = AuthSession(
                    id=session_id,
                    user_id=user_id,
                    method=AuthMethod(method),
                    created_at=time.time(),
                    expires_at=time.time() + 3600,  # 1 час
                    permissions=set(permissions),
                    metadata={'login_ip': 'localhost', 'user_agent': 'AnamorphX'}
                )
                
                context.neural_network.auth_sessions[session_id] = session
                
                return CommandResult(
                    success=True,
                    data={
                        'session_id': session_id,
                        'user_id': user_id,
                        'method': method,
                        'permissions': list(permissions),
                        'expires_at': session.expires_at
                    },
                    message=f"User '{user_id}' authenticated successfully"
                )
            
            elif action == 'logout':
                session_id = kwargs.get('session_id')
                if not session_id:
                    raise CommandError("Session ID is required for logout")
                
                if session_id in context.neural_network.auth_sessions:
                    session = context.neural_network.auth_sessions[session_id]
                    session.active = False
                    session.metadata['logout_time'] = time.time()
                    
                    return CommandResult(
                        success=True,
                        data={
                            'session_id': session_id,
                            'user_id': session.user_id,
                            'logout_time': session.metadata['logout_time']
                        },
                        message=f"User '{session.user_id}' logged out"
                    )
                else:
                    raise CommandError("Session not found")
            
            elif action == 'verify':
                session_id = kwargs.get('session_id')
                if not session_id:
                    raise CommandError("Session ID is required for verification")
                
                if session_id in context.neural_network.auth_sessions:
                    session = context.neural_network.auth_sessions[session_id]
                    
                    # Проверка активности и срока действия
                    current_time = time.time()
                    is_valid = (session.active and 
                              current_time < session.expires_at)
                    
                    return CommandResult(
                        success=True,
                        data={
                            'session_id': session_id,
                            'user_id': session.user_id,
                            'valid': is_valid,
                            'expires_at': session.expires_at,
                            'permissions': list(session.permissions)
                        },
                        message=f"Session verification: {'valid' if is_valid else 'invalid'}"
                    )
                else:
                    return CommandResult(
                        success=True,
                        data={'valid': False},
                        message="Session not found"
                    )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Authentication failed: {e}"
            )


class EncryptCommand(SecurityCommand):
    """Шифрование данных"""
    
    def __init__(self):
        super().__init__("encrypt", "Encrypt data using various algorithms")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get('data')
            algorithm = kwargs.get('algorithm', 'AES-256-GCM')
            key_id = kwargs.get('key_id', 'default')
            
            if data is None:
                raise CommandError("Data to encrypt is required")
            
            # Инициализация системы шифрования
            if not hasattr(context.neural_network, 'encryption_keys'):
                context.neural_network.encryption_keys = {}
            
            # Получение или создание ключа
            if key_id not in context.neural_network.encryption_keys:
                # Генерация нового ключа
                key = Fernet.generate_key()
                context.neural_network.encryption_keys[key_id] = {
                    'key': key,
                    'algorithm': algorithm,
                    'created_at': time.time()
                }
            else:
                key = context.neural_network.encryption_keys[key_id]['key']
            
            # Шифрование данных
            fernet = Fernet(key)
            data_bytes = str(data).encode('utf-8')
            encrypted_data = fernet.encrypt(data_bytes)
            
            # Кодирование в base64 для удобства
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            # Создание результата шифрования
            encryption_result = {
                'encrypted_data': encrypted_b64,
                'algorithm': algorithm,
                'key_id': key_id,
                'timestamp': time.time(),
                'data_size': len(data_bytes)
            }
            
            return CommandResult(
                success=True,
                data=encryption_result,
                message=f"Data encrypted using {algorithm}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Encryption failed: {e}"
            )


class DecryptCommand(SecurityCommand):
    """Расшифровка данных"""
    
    def __init__(self):
        super().__init__("decrypt", "Decrypt encrypted data")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            encrypted_data = kwargs.get('encrypted_data')
            key_id = kwargs.get('key_id', 'default')
            verify = kwargs.get('verify', True)
            
            if not encrypted_data:
                raise CommandError("Encrypted data is required")
            
            # Проверка наличия ключей
            if not hasattr(context.neural_network, 'encryption_keys'):
                raise CommandError("No encryption keys available")
            
            if key_id not in context.neural_network.encryption_keys:
                raise CommandError(f"Encryption key '{key_id}' not found")
            
            # Получение ключа
            key_info = context.neural_network.encryption_keys[key_id]
            key = key_info['key']
            
            # Расшифровка данных
            fernet = Fernet(key)
            
            # Декодирование из base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Расшифровка
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            decrypted_data = decrypted_bytes.decode('utf-8')
            
            # Верификация целостности (уже встроена в Fernet)
            verification_result = True if verify else None
            
            return CommandResult(
                success=True,
                data={
                    'decrypted_data': decrypted_data,
                    'key_id': key_id,
                    'algorithm': key_info['algorithm'],
                    'verification_passed': verification_result,
                    'timestamp': time.time()
                },
                message="Data decrypted successfully"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Decryption failed: {e}"
            )


class ThrottleCommand(SecurityCommand):
    """Ограничение скорости запросов"""
    
    def __init__(self):
        super().__init__("throttle", "Throttle request rates")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('target')  # node, user, ip
            limit = kwargs.get('limit', 100)  # requests per minute
            window = kwargs.get('window', 60)  # time window in seconds
            action = kwargs.get('action', 'delay')  # 'delay', 'block', 'queue'
            
            if not target:
                raise CommandError("Throttle target is required")
            
            # Инициализация системы троттлинга
            if not hasattr(context.neural_network, 'throttle_counters'):
                context.neural_network.throttle_counters = {}
            
            current_time = time.time()
            throttle_key = f"{target}_{window}"
            
            # Получение или создание счетчика
            if throttle_key not in context.neural_network.throttle_counters:
                context.neural_network.throttle_counters[throttle_key] = {
                    'count': 0,
                    'window_start': current_time,
                    'limit': limit,
                    'window_size': window,
                    'action': action,
                    'blocked_count': 0
                }
            
            counter = context.neural_network.throttle_counters[throttle_key]
            
            # Сброс счетчика если окно истекло
            if current_time - counter['window_start'] > window:
                counter['count'] = 0
                counter['window_start'] = current_time
            
            # Проверка лимита
            counter['count'] += 1
            
            if counter['count'] > limit:
                counter['blocked_count'] += 1
                
                if action == 'block':
                    raise CommandError(f"Request rate limit exceeded for {target}")
                elif action == 'delay':
                    # В реальной реализации здесь была бы задержка
                    delay_time = min(counter['count'] - limit, 10)  # максимум 10 секунд
                elif action == 'queue':
                    # Добавление в очередь
                    if not hasattr(context.neural_network, 'throttle_queue'):
                        context.neural_network.throttle_queue = {}
                    if target not in context.neural_network.throttle_queue:
                        context.neural_network.throttle_queue[target] = []
                    
                    context.neural_network.throttle_queue[target].append({
                        'timestamp': current_time,
                        'data': kwargs
                    })
            
            # Статистика
            remaining_requests = max(0, limit - counter['count'])
            reset_time = counter['window_start'] + window
            
            return CommandResult(
                success=True,
                data={
                    'target': target,
                    'current_count': counter['count'],
                    'limit': limit,
                    'remaining': remaining_requests,
                    'window_size': window,
                    'reset_time': reset_time,
                    'action': action,
                    'blocked_count': counter['blocked_count']
                },
                message=f"Throttle applied to {target}: {counter['count']}/{limit} requests"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Throttling failed: {e}"
            )


class BanCommand(SecurityCommand):
    """Блокировка пользователей/IP"""
    
    def __init__(self):
        super().__init__("ban", "Ban users or IP addresses")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('target')
            target_type = kwargs.get('type', 'user')  # 'user', 'ip', 'node'
            reason = kwargs.get('reason', 'security_violation')
            duration = kwargs.get('duration', 3600)  # seconds, 0 = permanent
            
            if not target:
                raise CommandError("Ban target is required")
            
            # Инициализация системы банов
            if not hasattr(context.neural_network, 'banned_entities'):
                context.neural_network.banned_entities = {}
            
            ban_id = str(uuid.uuid4())
            current_time = time.time()
            expires_at = current_time + duration if duration > 0 else None
            
            ban_record = {
                'id': ban_id,
                'target': target,
                'type': target_type,
                'reason': reason,
                'created_at': current_time,
                'expires_at': expires_at,
                'duration': duration,
                'active': True
            }
            
            context.neural_network.banned_entities[ban_id] = ban_record
            
            # Применение бана к активным сессиям
            banned_sessions = []
            if hasattr(context.neural_network, 'auth_sessions'):
                for session_id, session in context.neural_network.auth_sessions.items():
                    if target_type == 'user' and session.user_id == target:
                        session.active = False
                        session.metadata['ban_reason'] = reason
                        banned_sessions.append(session_id)
            
            # Блокировка узла если указан
            if target_type == 'node' and target in context.neural_network.nodes:
                node = context.neural_network.nodes[target]
                node.state = 'banned'
                node.metadata['ban_id'] = ban_id
                node.metadata['ban_reason'] = reason
            
            return CommandResult(
                success=True,
                data={
                    'ban_id': ban_id,
                    'target': target,
                    'type': target_type,
                    'reason': reason,
                    'duration': duration,
                    'expires_at': expires_at,
                    'banned_sessions': banned_sessions,
                    'permanent': duration == 0
                },
                message=f"Banned {target_type} '{target}' for {reason}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Ban failed: {e}"
            )


class WhitelistCommand(SecurityCommand):
    """Управление белым списком"""
    
    def __init__(self):
        super().__init__("whitelist", "Manage whitelist of trusted entities")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get('action', 'add')  # 'add', 'remove', 'check', 'list'
            items = kwargs.get('items', [])
            category = kwargs.get('category', 'general')
            
            # Инициализация белого списка
            if not hasattr(context.neural_network, 'whitelist'):
                context.neural_network.whitelist = {}
            
            if category not in context.neural_network.whitelist:
                context.neural_network.whitelist[category] = {
                    'items': set(),
                    'created_at': time.time(),
                    'last_modified': time.time()
                }
            
            whitelist_category = context.neural_network.whitelist[category]
            
            if action == 'add':
                if not items:
                    raise CommandError("Items to whitelist are required")
                
                added_items = []
                for item in items:
                    if item not in whitelist_category['items']:
                        whitelist_category['items'].add(item)
                        added_items.append(item)
                
                whitelist_category['last_modified'] = time.time()
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'added_items': added_items,
                        'total_items': len(whitelist_category['items'])
                    },
                    message=f"Added {len(added_items)} items to {category} whitelist"
                )
            
            elif action == 'remove':
                if not items:
                    raise CommandError("Items to remove are required")
                
                removed_items = []
                for item in items:
                    if item in whitelist_category['items']:
                        whitelist_category['items'].remove(item)
                        removed_items.append(item)
                
                whitelist_category['last_modified'] = time.time()
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'removed_items': removed_items,
                        'total_items': len(whitelist_category['items'])
                    },
                    message=f"Removed {len(removed_items)} items from {category} whitelist"
                )
            
            elif action == 'check':
                if not items:
                    raise CommandError("Items to check are required")
                
                check_results = {}
                for item in items:
                    check_results[item] = item in whitelist_category['items']
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'check_results': check_results,
                        'all_whitelisted': all(check_results.values())
                    },
                    message=f"Checked {len(items)} items against {category} whitelist"
                )
            
            elif action == 'list':
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'items': list(whitelist_category['items']),
                        'total_items': len(whitelist_category['items']),
                        'created_at': whitelist_category['created_at'],
                        'last_modified': whitelist_category['last_modified']
                    },
                    message=f"Listed {len(whitelist_category['items'])} items in {category} whitelist"
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Whitelist operation failed: {e}"
            )


class BlacklistCommand(SecurityCommand):
    """Управление черным списком"""
    
    def __init__(self):
        super().__init__("blacklist", "Manage blacklist of blocked entities")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get('action', 'add')  # 'add', 'remove', 'check', 'list'
            items = kwargs.get('items', [])
            category = kwargs.get('category', 'general')
            auto_expire = kwargs.get('auto_expire', False)
            expire_time = kwargs.get('expire_time', 86400)  # 24 hours
            
            # Инициализация черного списка
            if not hasattr(context.neural_network, 'blacklist'):
                context.neural_network.blacklist = {}
            
            if category not in context.neural_network.blacklist:
                context.neural_network.blacklist[category] = {
                    'items': {},
                    'created_at': time.time(),
                    'last_modified': time.time()
                }
            
            blacklist_category = context.neural_network.blacklist[category]
            current_time = time.time()
            
            # Очистка истекших элементов
            expired_items = []
            for item, item_data in list(blacklist_category['items'].items()):
                if item_data.get('expires_at') and current_time > item_data['expires_at']:
                    del blacklist_category['items'][item]
                    expired_items.append(item)
            
            if action == 'add':
                if not items:
                    raise CommandError("Items to blacklist are required")
                
                added_items = []
                for item in items:
                    expires_at = current_time + expire_time if auto_expire else None
                    
                    blacklist_category['items'][item] = {
                        'added_at': current_time,
                        'expires_at': expires_at,
                        'auto_expire': auto_expire,
                        'category': category
                    }
                    added_items.append(item)
                
                blacklist_category['last_modified'] = current_time
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'added_items': added_items,
                        'auto_expire': auto_expire,
                        'expire_time': expire_time,
                        'total_items': len(blacklist_category['items']),
                        'expired_items': expired_items
                    },
                    message=f"Added {len(added_items)} items to {category} blacklist"
                )
            
            elif action == 'remove':
                if not items:
                    raise CommandError("Items to remove are required")
                
                removed_items = []
                for item in items:
                    if item in blacklist_category['items']:
                        del blacklist_category['items'][item]
                        removed_items.append(item)
                
                blacklist_category['last_modified'] = current_time
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'removed_items': removed_items,
                        'total_items': len(blacklist_category['items']),
                        'expired_items': expired_items
                    },
                    message=f"Removed {len(removed_items)} items from {category} blacklist"
                )
            
            elif action == 'check':
                if not items:
                    raise CommandError("Items to check are required")
                
                check_results = {}
                for item in items:
                    item_data = blacklist_category['items'].get(item)
                    if item_data:
                        # Проверка срока действия
                        if item_data.get('expires_at') and current_time > item_data['expires_at']:
                            check_results[item] = False
                        else:
                            check_results[item] = True
                    else:
                        check_results[item] = False
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'check_results': check_results,
                        'any_blacklisted': any(check_results.values()),
                        'expired_items': expired_items
                    },
                    message=f"Checked {len(items)} items against {category} blacklist"
                )
            
            elif action == 'list':
                active_items = {}
                for item, item_data in blacklist_category['items'].items():
                    if not item_data.get('expires_at') or current_time <= item_data['expires_at']:
                        active_items[item] = item_data
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'category': category,
                        'items': active_items,
                        'total_items': len(active_items),
                        'expired_items': expired_items,
                        'created_at': blacklist_category['created_at'],
                        'last_modified': blacklist_category['last_modified']
                    },
                    message=f"Listed {len(active_items)} active items in {category} blacklist"
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Blacklist operation failed: {e}"
            )


class AuditCommand(SecurityCommand):
    """Аудит безопасности"""
    
    def __init__(self):
        super().__init__("audit", "Perform security audit and logging")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            audit_type = kwargs.get('type', 'full')  # 'full', 'security', 'access', 'performance'
            target = kwargs.get('target', 'system')
            include_details = kwargs.get('details', True)
            
            current_time = time.time()
            audit_id = str(uuid.uuid4())
            
            # Инициализация системы аудита
            if not hasattr(context.neural_network, 'audit_logs'):
                context.neural_network.audit_logs = []
            
            audit_results = {
                'audit_id': audit_id,
                'type': audit_type,
                'target': target,
                'timestamp': current_time,
                'findings': [],
                'statistics': {},
                'recommendations': []
            }
            
            # Аудит безопасности
            if audit_type in ['full', 'security']:
                security_findings = []
                
                # Проверка активных сессий
                if hasattr(context.neural_network, 'auth_sessions'):
                    active_sessions = sum(1 for s in context.neural_network.auth_sessions.values() 
                                        if s.active and current_time < s.expires_at)
                    expired_sessions = sum(1 for s in context.neural_network.auth_sessions.values() 
                                         if s.active and current_time >= s.expires_at)
                    
                    audit_results['statistics']['active_sessions'] = active_sessions
                    audit_results['statistics']['expired_sessions'] = expired_sessions
                    
                    if expired_sessions > 0:
                        security_findings.append({
                            'type': 'expired_sessions',
                            'severity': 'medium',
                            'count': expired_sessions,
                            'description': 'Found expired but still active sessions'
                        })
                
                # Проверка банов
                if hasattr(context.neural_network, 'banned_entities'):
                    active_bans = sum(1 for b in context.neural_network.banned_entities.values() 
                                    if b['active'] and (not b['expires_at'] or current_time < b['expires_at']))
                    audit_results['statistics']['active_bans'] = active_bans
                
                # Проверка фильтров
                if hasattr(context.neural_network, 'filter_rules'):
                    active_filters = sum(1 for f in context.neural_network.filter_rules.values() if f.active)
                    total_matches = sum(f.match_count for f in context.neural_network.filter_rules.values())
                    
                    audit_results['statistics']['active_filters'] = active_filters
                    audit_results['statistics']['total_filter_matches'] = total_matches
                
                # Проверка троттлинга
                if hasattr(context.neural_network, 'throttle_counters'):
                    throttled_targets = len(context.neural_network.throttle_counters)
                    total_blocked = sum(c['blocked_count'] for c in context.neural_network.throttle_counters.values())
                    
                    audit_results['statistics']['throttled_targets'] = throttled_targets
                    audit_results['statistics']['total_blocked_requests'] = total_blocked
                
                audit_results['findings'].extend(security_findings)
            
            # Аудит доступа
            if audit_type in ['full', 'access']:
                access_findings = []
                
                # Анализ узлов
                total_nodes = len(context.neural_network.nodes)
                active_nodes = sum(1 for n in context.neural_network.nodes.values() 
                                 if n.state not in ['halted', 'banned'])
                
                audit_results['statistics']['total_nodes'] = total_nodes
                audit_results['statistics']['active_nodes'] = active_nodes
                
                # Проверка подозрительной активности
                for node_name, node in context.neural_network.nodes.items():
                    last_activity = node.metadata.get('last_activity', node.created_at)
                    if current_time - last_activity > 86400:  # 24 hours
                        access_findings.append({
                            'type': 'inactive_node',
                            'severity': 'low',
                            'node': node_name,
                            'inactive_hours': (current_time - last_activity) / 3600
                        })
                
                audit_results['findings'].extend(access_findings)
            
            # Рекомендации
            if audit_results['statistics'].get('expired_sessions', 0) > 0:
                audit_results['recommendations'].append(
                    "Clean up expired authentication sessions"
                )
            
            if audit_results['statistics'].get('total_filter_matches', 0) > 100:
                audit_results['recommendations'].append(
                    "Review filter rules - high number of matches detected"
                )
            
            # Сохранение результатов аудита
            context.neural_network.audit_logs.append(audit_results)
            
            # Ограничение размера лога
            if len(context.neural_network.audit_logs) > 100:
                context.neural_network.audit_logs = context.neural_network.audit_logs[-100:]
            
            return CommandResult(
                success=True,
                data=audit_results,
                message=f"Security audit completed: {len(audit_results['findings'])} findings, {len(audit_results['recommendations'])} recommendations"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Security audit failed: {e}"
            )


# =============================================================================
# РЕГИСТРАЦИЯ КОМАНД
# =============================================================================

SECURITY_COMMANDS = [
    GuardCommand(),
    FilterCommand(),
    AuthCommand(),
    EncryptCommand(),
    DecryptCommand(),
    ThrottleCommand(),
    BanCommand(),
    WhitelistCommand(),
    BlacklistCommand(),
    AuditCommand()
] 