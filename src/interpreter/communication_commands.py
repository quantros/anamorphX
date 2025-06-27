"""
Команды коммуникации AnamorphX

Команды для обеспечения коммуникации, шифрования и миграции данных.
"""

import time
import uuid
import hashlib
import hmac
import secrets
import base64
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Импорты команд с обработкой ошибок
try:
    from .commands import FlowControlCommand, CommandResult, CommandError
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
    
    class FlowControlCommand:
        def __init__(self, name="", description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
    
    class ExecutionContext:
        def __init__(self):
            self.neural_entities = {}
            self.encrypted_data = {}
            self.notifications = []
            self.migration_states = {}

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

class EncryptionType(Enum):
    """Типы шифрования"""
    FERNET = "fernet"
    AES = "aes"
    BASIC = "basic"
    HASH = "hash"


class NotificationType(Enum):
    """Типы уведомлений"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    SYSTEM = "system"


class MigrationType(Enum):
    """Типы миграции"""
    NODE_DATA = "node_data"
    NETWORK_STATE = "network_state"
    VARIABLES = "variables"
    CONNECTIONS = "connections"
    FULL_SYSTEM = "full_system"


@dataclass
class EncryptedData:
    """Зашифрованные данные"""
    id: str
    algorithm: str
    encrypted_content: str
    key_hash: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """Уведомление"""
    id: str
    type: NotificationType
    title: str
    message: str
    source: str
    target: Optional[str]
    created_at: float
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationTask:
    """Задача миграции"""
    id: str
    type: MigrationType
    source: str
    target: str
    data: Any
    status: str
    progress: float
    created_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# КОМАНДЫ КОММУНИКАЦИИ (4 команды)
# =============================================================================

class EncryptCommand(FlowControlCommand):
    """Шифрование данных"""
    
    def __init__(self):
        super().__init__("encrypt", "Encrypt data using specified algorithm")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get('data')
            algorithm = kwargs.get('algorithm', 'basic')
            key = kwargs.get('key')
            output_format = kwargs.get('format', 'base64')
            
            if data is None:
                raise CommandError("Data is required for encryption")
            
            # Инициализация хранилища зашифрованных данных
            if not hasattr(context, 'encrypted_data'):
                context.encrypted_data = {}
            
            # Преобразование данных в строку
            if isinstance(data, dict):
                data_str = json.dumps(data)
            elif isinstance(data, (list, tuple)):
                data_str = json.dumps(list(data))
            else:
                data_str = str(data)
            
            data_bytes = data_str.encode('utf-8')
            encryption_id = str(uuid.uuid4())
            
            if algorithm == 'fernet' and CRYPTO_AVAILABLE:
                # Fernet шифрование
                if not key:
                    key = Fernet.generate_key()
                elif isinstance(key, str):
                    # Генерация ключа из пароля
                    password = key.encode('utf-8')
                    salt = secrets.token_bytes(16)
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=salt,
                        iterations=100000,
                    )
                    key = base64.urlsafe_b64encode(kdf.derive(password))
                
                fernet = Fernet(key)
                encrypted_bytes = fernet.encrypt(data_bytes)
                
                if output_format == 'base64':
                    encrypted_content = base64.b64encode(encrypted_bytes).decode('utf-8')
                else:
                    encrypted_content = encrypted_bytes.hex()
                
                key_hash = hashlib.sha256(key).hexdigest()
                
            elif algorithm == 'basic':
                # Простое XOR шифрование
                if not key:
                    key = secrets.token_hex(16)
                
                key_bytes = key.encode('utf-8')
                encrypted_bytes = bytearray()
                
                for i, byte in enumerate(data_bytes):
                    encrypted_bytes.append(byte ^ key_bytes[i % len(key_bytes)])
                
                if output_format == 'base64':
                    encrypted_content = base64.b64encode(encrypted_bytes).decode('utf-8')
                else:
                    encrypted_content = encrypted_bytes.hex()
                
                key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()
                
            elif algorithm == 'hash':
                # Хеширование (необратимое)
                if key:
                    # HMAC с ключом
                    key_bytes = key.encode('utf-8') if isinstance(key, str) else key
                    hash_obj = hmac.new(key_bytes, data_bytes, hashlib.sha256)
                    encrypted_content = hash_obj.hexdigest()
                else:
                    # Простое хеширование
                    hash_obj = hashlib.sha256(data_bytes)
                    encrypted_content = hash_obj.hexdigest()
                
                key_hash = hashlib.sha256(str(key).encode('utf-8')).hexdigest() if key else "no_key"
                
            else:
                raise CommandError(f"Unknown encryption algorithm: {algorithm}")
            
            # Создание записи зашифрованных данных
            encrypted_data = EncryptedData(
                id=encryption_id,
                algorithm=algorithm,
                encrypted_content=encrypted_content,
                key_hash=key_hash,
                created_at=time.time(),
                metadata={
                    'original_type': type(data).__name__,
                    'data_size': len(data_bytes),
                    'output_format': output_format,
                    'reversible': algorithm != 'hash'
                }
            )
            
            context.encrypted_data[encryption_id] = encrypted_data
            
            return CommandResult(
                success=True,
                data={
                    'encryption_id': encryption_id,
                    'algorithm': algorithm,
                    'encrypted_content': encrypted_content,
                    'key_hash': key_hash,
                    'reversible': algorithm != 'hash',
                    'size': len(encrypted_content),
                    'format': output_format
                },
                message=f"Data encrypted using {algorithm} algorithm"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to encrypt data: {e}"
            )


class DecryptCommand(FlowControlCommand):
    """Дешифрование данных"""
    
    def __init__(self):
        super().__init__("decrypt", "Decrypt previously encrypted data")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            encryption_id = kwargs.get('id')
            encrypted_content = kwargs.get('content')
            algorithm = kwargs.get('algorithm')
            key = kwargs.get('key')
            
            if not encryption_id and not encrypted_content:
                raise CommandError("Either encryption ID or encrypted content is required")
            
            # Получение зашифрованных данных
            if encryption_id:
                if not hasattr(context, 'encrypted_data') or encryption_id not in context.encrypted_data:
                    raise CommandError(f"Encrypted data with ID '{encryption_id}' not found")
                
                encrypted_data = context.encrypted_data[encryption_id]
                encrypted_content = encrypted_data.encrypted_content
                algorithm = encrypted_data.algorithm
                stored_key_hash = encrypted_data.key_hash
                
                # Проверка ключа
                if key:
                    provided_key_hash = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
                    if provided_key_hash != stored_key_hash:
                        raise CommandError("Invalid decryption key")
                else:
                    raise CommandError("Decryption key is required")
            
            if not key:
                raise CommandError("Decryption key is required")
            
            if algorithm == 'hash':
                raise CommandError("Hash algorithm is not reversible")
            
            try:
                # Декодирование содержимого
                try:
                    encrypted_bytes = base64.b64decode(encrypted_content)
                except:
                    encrypted_bytes = bytes.fromhex(encrypted_content)
                
                if algorithm == 'fernet' and CRYPTO_AVAILABLE:
                    # Fernet дешифрование
                    if isinstance(key, str):
                        # Генерация ключа из пароля (нужна соль из оригинальной операции)
                        password = key.encode('utf-8')
                        salt = secrets.token_bytes(16)  # В реальной реализации соль должна храниться
                        kdf = PBKDF2HMAC(
                            algorithm=hashes.SHA256(),
                            length=32,
                            salt=salt,
                            iterations=100000,
                        )
                        key = base64.urlsafe_b64encode(kdf.derive(password))
                    
                    fernet = Fernet(key)
                    decrypted_bytes = fernet.decrypt(encrypted_bytes)
                    
                elif algorithm == 'basic':
                    # Простое XOR дешифрование
                    key_bytes = key.encode('utf-8')
                    decrypted_bytes = bytearray()
                    
                    for i, byte in enumerate(encrypted_bytes):
                        decrypted_bytes.append(byte ^ key_bytes[i % len(key_bytes)])
                    
                    decrypted_bytes = bytes(decrypted_bytes)
                
                else:
                    raise CommandError(f"Unknown decryption algorithm: {algorithm}")
                
                # Декодирование результата
                decrypted_str = decrypted_bytes.decode('utf-8')
                
                # Попытка преобразовать обратно в оригинальный тип
                try:
                    decrypted_data = json.loads(decrypted_str)
                except:
                    decrypted_data = decrypted_str
                
                return CommandResult(
                    success=True,
                    data={
                        'encryption_id': encryption_id,
                        'algorithm': algorithm,
                        'decrypted_data': decrypted_data,
                        'data_type': type(decrypted_data).__name__,
                        'size': len(decrypted_str)
                    },
                    message=f"Data successfully decrypted using {algorithm}"
                )
                
            except Exception as decrypt_error:
                raise CommandError(f"Decryption failed: {decrypt_error}")
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to decrypt data: {e}"
            )


class NotifyCommand(FlowControlCommand):
    """Отправка уведомлений"""
    
    def __init__(self):
        super().__init__("notify", "Send notifications to nodes or users")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            title = kwargs.get('title', 'Notification')
            message = kwargs.get('message', '')
            notification_type = kwargs.get('type', 'info')
            target = kwargs.get('target')
            source = kwargs.get('source', 'system')
            priority = kwargs.get('priority', 5)
            metadata = kwargs.get('metadata', {})
            
            if not message:
                raise CommandError("Notification message is required")
            
            # Инициализация системы уведомлений
            if not hasattr(context, 'notifications'):
                context.notifications = []
            
            notification_id = str(uuid.uuid4())
            
            # Создание уведомления
            notification = Notification(
                id=notification_id,
                type=NotificationType(notification_type),
                title=title,
                message=message,
                source=source,
                target=target,
                created_at=time.time(),
                metadata={
                    'priority': priority,
                    **metadata
                }
            )
            
            # Добавление уведомления
            context.notifications.append(notification)
            
            # Обработка целевого узла
            target_processed = False
            if target and hasattr(context, 'neural_network'):
                if target in context.neural_network.nodes:
                    target_node = context.neural_network.nodes[target]
                    
                    # Добавление уведомления к узлу
                    if 'notifications' not in target_node.data:
                        target_node.data['notifications'] = []
                    
                    target_node.data['notifications'].append({
                        'id': notification_id,
                        'title': title,
                        'message': message,
                        'type': notification_type,
                        'timestamp': notification.created_at,
                        'read': False
                    })
                    
                    target_node.metadata['last_notification'] = time.time()
                    target_node.metadata['notification_count'] = target_node.metadata.get('notification_count', 0) + 1
                    target_processed = True
            
            # Логирование уведомления
            notification_log = {
                'id': notification_id,
                'timestamp': notification.created_at,
                'level': notification_type.upper(),
                'source': source,
                'target': target,
                'title': title,
                'message': message
            }
            
            if not hasattr(context, 'notification_log'):
                context.notification_log = []
            context.notification_log.append(notification_log)
            
            return CommandResult(
                success=True,
                data={
                    'notification_id': notification_id,
                    'title': title,
                    'message': message,
                    'type': notification_type,
                    'source': source,
                    'target': target,
                    'target_processed': target_processed,
                    'priority': priority,
                    'created_at': notification.created_at
                },
                message=f"Notification sent: {title}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to send notification: {e}"
            )


class MigrateCommand(FlowControlCommand):
    """Миграция данных между узлами или системами"""
    
    def __init__(self):
        super().__init__("migrate", "Migrate data between nodes or systems")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get('from')
            target = kwargs.get('to')
            migration_type = kwargs.get('type', 'node_data')
            data_filter = kwargs.get('filter')
            preserve_source = kwargs.get('preserve_source', True)
            batch_size = kwargs.get('batch_size', 100)
            validate_integrity = kwargs.get('validate', True)
            
            if not source or not target:
                raise CommandError("Source and target are required for migration")
            
            # Инициализация системы миграции
            if not hasattr(context, 'migration_states'):
                context.migration_states = {}
            
            migration_id = str(uuid.uuid4())
            migration_start = time.time()
            
            migration_task = MigrationTask(
                id=migration_id,
                type=MigrationType(migration_type),
                source=source,
                target=target,
                data=None,
                status='started',
                progress=0.0,
                created_at=migration_start
            )
            
            context.migration_states[migration_id] = migration_task
            
            migrated_data = None
            migration_stats = {
                'items_processed': 0,
                'items_migrated': 0,
                'items_skipped': 0,
                'errors': []
            }
            
            try:
                if migration_type == 'node_data':
                    # Миграция данных узла
                    if source not in context.neural_network.nodes:
                        raise CommandError(f"Source node '{source}' not found")
                    
                    source_node = context.neural_network.nodes[source]
                    
                    # Создание или получение целевого узла
                    if target not in context.neural_network.nodes:
                        # Создание нового узла
                        from .structural_commands import NeuralNode
                        target_node = NeuralNode(
                            id=str(uuid.uuid4()),
                            name=target,
                            node_type='migrated',
                            created_at=time.time(),
                            metadata={'migrated_from': source}
                        )
                        context.neural_network.nodes[target] = target_node
                        context.neural_network.node_by_id[target_node.id] = target_node
                    else:
                        target_node = context.neural_network.nodes[target]
                    
                    # Копирование данных
                    source_data = source_node.data.copy()
                    
                    # Применение фильтра
                    if data_filter:
                        filtered_data = {}
                        if isinstance(data_filter, list):
                            for key in data_filter:
                                if key in source_data:
                                    filtered_data[key] = source_data[key]
                                    migration_stats['items_processed'] += 1
                                    migration_stats['items_migrated'] += 1
                                else:
                                    migration_stats['items_skipped'] += 1
                        else:
                            for key, value in source_data.items():
                                if str(data_filter).lower() in key.lower():
                                    filtered_data[key] = value
                                    migration_stats['items_processed'] += 1
                                    migration_stats['items_migrated'] += 1
                                else:
                                    migration_stats['items_skipped'] += 1
                        migrated_data = filtered_data
                    else:
                        migrated_data = source_data
                        migration_stats['items_processed'] = len(source_data)
                        migration_stats['items_migrated'] = len(source_data)
                    
                    # Перенос данных
                    target_node.data.update(migrated_data)
                    target_node.metadata['migration_source'] = source
                    target_node.metadata['migration_time'] = time.time()
                    target_node.metadata['migration_id'] = migration_id
                    
                    # Удаление из источника если не сохраняем
                    if not preserve_source:
                        if data_filter:
                            for key in migrated_data.keys():
                                source_node.data.pop(key, None)
                        else:
                            source_node.data.clear()
                        source_node.metadata['data_migrated_to'] = target
                
                elif migration_type == 'variables':
                    # Миграция переменных
                    if hasattr(context, 'variables'):
                        migrated_data = context.variables.copy()
                        migration_stats['items_processed'] = len(migrated_data)
                        migration_stats['items_migrated'] = len(migrated_data)
                        
                        # В реальной реализации здесь была бы передача в другой контекст
                        if not hasattr(context, 'migrated_variables'):
                            context.migrated_variables = {}
                        context.migrated_variables[target] = migrated_data
                
                elif migration_type == 'network_state':
                    # Миграция состояния сети
                    network_state = {
                        'nodes': {name: {
                            'id': node.id,
                            'type': node.node_type,
                            'state': node.state,
                            'data_keys': list(node.data.keys()),
                            'metadata': node.metadata
                        } for name, node in context.neural_network.nodes.items()},
                        'node_count': len(context.neural_network.nodes),
                        'timestamp': time.time()
                    }
                    
                    migrated_data = network_state
                    migration_stats['items_processed'] = len(network_state['nodes'])
                    migration_stats['items_migrated'] = len(network_state['nodes'])
                
                # Валидация целостности
                if validate_integrity and migrated_data:
                    # Простая проверка целостности
                    if isinstance(migrated_data, dict):
                        for key, value in migrated_data.items():
                            if value is None:
                                migration_stats['errors'].append(f"Null value for key: {key}")
                
                # Завершение миграции
                migration_task.status = 'completed'
                migration_task.progress = 100.0
                migration_task.completed_at = time.time()
                migration_task.data = migration_stats
                
                total_time = migration_task.completed_at - migration_start
                
                return CommandResult(
                    success=True,
                    data={
                        'migration_id': migration_id,
                        'type': migration_type,
                        'source': source,
                        'target': target,
                        'stats': migration_stats,
                        'total_time': total_time,
                        'preserve_source': preserve_source,
                        'validate_integrity': validate_integrity,
                        'data_size': len(str(migrated_data)) if migrated_data else 0
                    },
                    message=f"Migration completed: {migration_stats['items_migrated']} items in {total_time:.2f}s"
                )
                
            except Exception as migration_error:
                migration_task.status = 'failed'
                migration_task.error = str(migration_error)
                migration_task.completed_at = time.time()
                raise migration_error
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to execute migration: {e}"
            )


# =============================================================================
# РЕГИСТРАЦИЯ КОМАНД
# =============================================================================

COMMUNICATION_COMMANDS = [
    NotifyCommand(),
    MigrateCommand()
] 