"""
Команды управления данными AnamorphX

Полная реализация 10 команд для управления данными и хранением.
"""

import time
import uuid
import json
import gzip
import hashlib
import pickle
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .commands import DataManagementCommand, CommandResult, CommandError, ExecutionContext


# =============================================================================
# ENUMS И DATACLASSES
# =============================================================================

class StorageType(Enum):
    """Типы хранилищ"""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    CLOUD = "cloud"


class CompressionType(Enum):
    """Типы сжатия"""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    BZIP2 = "bzip2"


@dataclass
class DataRecord:
    """Запись данных"""
    id: str
    key: str
    data: Any
    storage_type: StorageType
    created_at: float
    accessed_at: float
    size: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Запись кэша"""
    key: str
    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    size: int = 0


# =============================================================================
# КОМАНДЫ УПРАВЛЕНИЯ ДАННЫМИ (10 команд)
# =============================================================================

class StoreCommand(DataManagementCommand):
    """Сохранение данных"""
    
    def __init__(self):
        super().__init__("store", "Store data in various storage systems")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            key = kwargs.get('key')
            data = kwargs.get('data')
            storage_type = kwargs.get('type', 'memory')
            node_name = kwargs.get('node')
            
            if not key or data is None:
                raise CommandError("Key and data are required")
            
            # Инициализация хранилища
            if not hasattr(context.neural_network, 'data_storage'):
                context.neural_network.data_storage = {}
            
            # Создание записи данных
            record_id = str(uuid.uuid4())
            data_str = json.dumps(data) if not isinstance(data, str) else data
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
            
            record = DataRecord(
                id=record_id,
                key=key,
                data=data,
                storage_type=StorageType(storage_type),
                created_at=time.time(),
                accessed_at=time.time(),
                size=len(data_str),
                checksum=checksum,
                metadata={'created_by': 'store_command'}
            )
            
            context.neural_network.data_storage[key] = record
            
            # Сохранение в узел если указан
            if node_name:
                if node_name not in context.neural_network.nodes:
                    raise CommandError(f"Node '{node_name}' not found")
                
                node = context.neural_network.nodes[node_name]
                node.data[key] = data
                node.metadata['last_store'] = time.time()
            
            return CommandResult(
                success=True,
                data={
                    'record_id': record_id,
                    'key': key,
                    'storage_type': storage_type,
                    'size': record.size,
                    'checksum': checksum,
                    'node': node_name
                },
                message=f"Data stored with key '{key}' ({record.size} bytes)"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to store data: {e}"
            )


class LoadCommand(DataManagementCommand):
    """Загрузка данных"""
    
    def __init__(self):
        super().__init__("load", "Load data from storage")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            key = kwargs.get('key')
            node_name = kwargs.get('node')
            verify_checksum = kwargs.get('verify', True)
            
            if not key:
                raise CommandError("Data key is required")
            
            # Поиск данных в хранилище
            if hasattr(context.neural_network, 'data_storage'):
                if key in context.neural_network.data_storage:
                    record = context.neural_network.data_storage[key]
                    
                    # Обновление времени доступа
                    record.accessed_at = time.time()
                    
                    # Верификация контрольной суммы
                    if verify_checksum:
                        data_str = json.dumps(record.data) if not isinstance(record.data, str) else record.data
                        current_checksum = hashlib.sha256(data_str.encode()).hexdigest()
                        if current_checksum != record.checksum:
                            raise CommandError("Data integrity check failed")
                    
                    # Загрузка в узел если указан
                    if node_name:
                        if node_name not in context.neural_network.nodes:
                            raise CommandError(f"Node '{node_name}' not found")
                        
                        node = context.neural_network.nodes[node_name]
                        node.data[key] = record.data
                        node.metadata['last_load'] = time.time()
                    
                    return CommandResult(
                        success=True,
                        data={
                            'key': key,
                            'data': record.data,
                            'record_id': record.id,
                            'storage_type': record.storage_type.value,
                            'size': record.size,
                            'created_at': record.created_at,
                            'checksum_verified': verify_checksum,
                            'node': node_name
                        },
                        message=f"Data loaded with key '{key}'"
                    )
                else:
                    raise CommandError(f"Data with key '{key}' not found")
            else:
                raise CommandError("No data storage available")
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to load data: {e}"
            )


class CacheCommand(DataManagementCommand):
    """Кэширование данных"""
    
    def __init__(self):
        super().__init__("cache", "Cache data for fast access")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get('action', 'set')  # 'set', 'get', 'clear', 'stats'
            key = kwargs.get('key')
            data = kwargs.get('data')
            ttl = kwargs.get('ttl', 3600)  # 1 hour default
            
            # Инициализация кэша
            if not hasattr(context.neural_network, 'cache'):
                context.neural_network.cache = {}
            
            current_time = time.time()
            
            # Очистка истекших записей
            expired_keys = []
            for cache_key, entry in context.neural_network.cache.items():
                if entry.ttl and current_time > entry.created_at + entry.ttl:
                    expired_keys.append(cache_key)
            
            for expired_key in expired_keys:
                del context.neural_network.cache[expired_key]
            
            if action == 'set':
                if not key or data is None:
                    raise CommandError("Key and data are required for cache set")
                
                data_size = len(str(data))
                entry = CacheEntry(
                    key=key,
                    data=data,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=1,
                    ttl=ttl,
                    size=data_size
                )
                
                context.neural_network.cache[key] = entry
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'key': key,
                        'ttl': ttl,
                        'size': data_size,
                        'expires_at': current_time + ttl
                    },
                    message=f"Data cached with key '{key}'"
                )
            
            elif action == 'get':
                if not key:
                    raise CommandError("Key is required for cache get")
                
                if key in context.neural_network.cache:
                    entry = context.neural_network.cache[key]
                    
                    # Проверка TTL
                    if entry.ttl and current_time > entry.created_at + entry.ttl:
                        del context.neural_network.cache[key]
                        raise CommandError(f"Cache entry '{key}' has expired")
                    
                    # Обновление статистики
                    entry.accessed_at = current_time
                    entry.access_count += 1
                    
                    return CommandResult(
                        success=True,
                        data={
                            'action': action,
                            'key': key,
                            'data': entry.data,
                            'access_count': entry.access_count,
                            'created_at': entry.created_at,
                            'size': entry.size
                        },
                        message=f"Cache hit for key '{key}'"
                    )
                else:
                    return CommandResult(
                        success=False,
                        error=f"Cache miss for key '{key}'",
                        message=f"Key '{key}' not found in cache"
                    )
            
            elif action == 'clear':
                if key:
                    # Очистка конкретного ключа
                    if key in context.neural_network.cache:
                        del context.neural_network.cache[key]
                        cleared_count = 1
                    else:
                        cleared_count = 0
                else:
                    # Очистка всего кэша
                    cleared_count = len(context.neural_network.cache)
                    context.neural_network.cache.clear()
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'key': key,
                        'cleared_count': cleared_count
                    },
                    message=f"Cleared {cleared_count} cache entries"
                )
            
            elif action == 'stats':
                total_entries = len(context.neural_network.cache)
                total_size = sum(entry.size for entry in context.neural_network.cache.values())
                total_accesses = sum(entry.access_count for entry in context.neural_network.cache.values())
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'total_entries': total_entries,
                        'total_size': total_size,
                        'total_accesses': total_accesses,
                        'expired_cleaned': len(expired_keys)
                    },
                    message=f"Cache stats: {total_entries} entries, {total_size} bytes"
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Cache operation failed: {e}"
            )


class BackupCommand(DataManagementCommand):
    """Резервное копирование"""
    
    def __init__(self):
        super().__init__("backup", "Create backup of data")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('target', 'all')  # 'all', 'nodes', 'storage', specific node
            backup_name = kwargs.get('name', f"backup_{int(time.time())}")
            compress = kwargs.get('compress', True)
            
            # Инициализация системы резервного копирования
            if not hasattr(context.neural_network, 'backups'):
                context.neural_network.backups = {}
            
            backup_data = {}
            backup_id = str(uuid.uuid4())
            
            if target == 'all' or target == 'nodes':
                # Резервное копирование узлов
                nodes_backup = {}
                for node_name, node in context.neural_network.nodes.items():
                    nodes_backup[node_name] = {
                        'id': node.id,
                        'name': node.name,
                        'type': node.node_type.value,
                        'state': node.state,
                        'data': node.data,
                        'metadata': node.metadata,
                        'created_at': node.created_at
                    }
                backup_data['nodes'] = nodes_backup
            
            if target == 'all' or target == 'storage':
                # Резервное копирование хранилища данных
                if hasattr(context.neural_network, 'data_storage'):
                    storage_backup = {}
                    for key, record in context.neural_network.data_storage.items():
                        storage_backup[key] = {
                            'id': record.id,
                            'key': record.key,
                            'data': record.data,
                            'storage_type': record.storage_type.value,
                            'created_at': record.created_at,
                            'size': record.size,
                            'checksum': record.checksum,
                            'metadata': record.metadata
                        }
                    backup_data['storage'] = storage_backup
            
            if target not in ['all', 'nodes', 'storage']:
                # Резервное копирование конкретного узла
                if target in context.neural_network.nodes:
                    node = context.neural_network.nodes[target]
                    backup_data['node'] = {
                        'id': node.id,
                        'name': node.name,
                        'type': node.node_type.value,
                        'state': node.state,
                        'data': node.data,
                        'metadata': node.metadata,
                        'created_at': node.created_at
                    }
                else:
                    raise CommandError(f"Target '{target}' not found")
            
            # Сжатие данных если требуется
            if compress:
                backup_json = json.dumps(backup_data)
                compressed_data = gzip.compress(backup_json.encode())
                backup_size = len(compressed_data)
                backup_data_final = compressed_data
            else:
                backup_data_final = backup_data
                backup_size = len(json.dumps(backup_data))
            
            # Создание записи резервной копии
            backup_record = {
                'id': backup_id,
                'name': backup_name,
                'target': target,
                'created_at': time.time(),
                'size': backup_size,
                'compressed': compress,
                'data': backup_data_final,
                'checksum': hashlib.sha256(str(backup_data_final).encode()).hexdigest()
            }
            
            context.neural_network.backups[backup_id] = backup_record
            
            return CommandResult(
                success=True,
                data={
                    'backup_id': backup_id,
                    'name': backup_name,
                    'target': target,
                    'size': backup_size,
                    'compressed': compress,
                    'created_at': backup_record['created_at']
                },
                message=f"Backup '{backup_name}' created ({backup_size} bytes)"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Backup failed: {e}"
            )


class RestoreCommand(DataManagementCommand):
    """Восстановление из резервной копии"""
    
    def __init__(self):
        super().__init__("restore", "Restore data from backup")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            backup_id = kwargs.get('backup_id')
            backup_name = kwargs.get('name')
            overwrite = kwargs.get('overwrite', False)
            
            if not backup_id and not backup_name:
                raise CommandError("Backup ID or name is required")
            
            # Поиск резервной копии
            if not hasattr(context.neural_network, 'backups'):
                raise CommandError("No backups available")
            
            backup_record = None
            if backup_id:
                backup_record = context.neural_network.backups.get(backup_id)
            else:
                for record in context.neural_network.backups.values():
                    if record['name'] == backup_name:
                        backup_record = record
                        break
            
            if not backup_record:
                raise CommandError("Backup not found")
            
            # Восстановление данных
            backup_data = backup_record['data']
            
            # Распаковка если данные сжаты
            if backup_record['compressed']:
                if isinstance(backup_data, bytes):
                    decompressed_json = gzip.decompress(backup_data).decode()
                    backup_data = json.loads(decompressed_json)
            
            restored_items = []
            
            # Восстановление узлов
            if 'nodes' in backup_data:
                for node_name, node_data in backup_data['nodes'].items():
                    if node_name in context.neural_network.nodes and not overwrite:
                        continue
                    
                    # Создание узла из резервной копии
                    from .structural_commands import NeuralNode, NodeType
                    
                    node = NeuralNode(
                        id=node_data['id'],
                        name=node_data['name'],
                        node_type=NodeType(node_data['type']),
                        created_at=node_data['created_at'],
                        data=node_data['data'],
                        state=node_data['state'],
                        metadata=node_data['metadata']
                    )
                    
                    context.neural_network.nodes[node_name] = node
                    context.neural_network.node_by_id[node.id] = node
                    restored_items.append(f"node:{node_name}")
            
            # Восстановление хранилища
            if 'storage' in backup_data:
                if not hasattr(context.neural_network, 'data_storage'):
                    context.neural_network.data_storage = {}
                
                for key, record_data in backup_data['storage'].items():
                    if key in context.neural_network.data_storage and not overwrite:
                        continue
                    
                    record = DataRecord(
                        id=record_data['id'],
                        key=record_data['key'],
                        data=record_data['data'],
                        storage_type=StorageType(record_data['storage_type']),
                        created_at=record_data['created_at'],
                        accessed_at=time.time(),
                        size=record_data['size'],
                        checksum=record_data['checksum'],
                        metadata=record_data['metadata']
                    )
                    
                    context.neural_network.data_storage[key] = record
                    restored_items.append(f"storage:{key}")
            
            # Восстановление отдельного узла
            if 'node' in backup_data:
                node_data = backup_data['node']
                node_name = node_data['name']
                
                if node_name in context.neural_network.nodes and not overwrite:
                    raise CommandError(f"Node '{node_name}' already exists (use overwrite=true)")
                
                from .structural_commands import NeuralNode, NodeType
                
                node = NeuralNode(
                    id=node_data['id'],
                    name=node_data['name'],
                    node_type=NodeType(node_data['type']),
                    created_at=node_data['created_at'],
                    data=node_data['data'],
                    state=node_data['state'],
                    metadata=node_data['metadata']
                )
                
                context.neural_network.nodes[node_name] = node
                context.neural_network.node_by_id[node.id] = node
                restored_items.append(f"node:{node_name}")
            
            return CommandResult(
                success=True,
                data={
                    'backup_id': backup_record['id'],
                    'backup_name': backup_record['name'],
                    'restored_items': restored_items,
                    'restore_count': len(restored_items),
                    'overwrite': overwrite
                },
                message=f"Restored {len(restored_items)} items from backup '{backup_record['name']}'"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Restore failed: {e}"
            )


class CompressCommand(DataManagementCommand):
    """Сжатие данных"""
    
    def __init__(self):
        super().__init__("compress", "Compress data using various algorithms")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get('data')
            algorithm = kwargs.get('algorithm', 'gzip')
            level = kwargs.get('level', 6)
            
            if data is None:
                raise CommandError("Data to compress is required")
            
            # Преобразование данных в строку
            if not isinstance(data, (str, bytes)):
                data_str = json.dumps(data)
            else:
                data_str = data
            
            if isinstance(data_str, str):
                data_bytes = data_str.encode('utf-8')
            else:
                data_bytes = data_str
            
            original_size = len(data_bytes)
            
            # Сжатие данных
            if algorithm == 'gzip':
                compressed_data = gzip.compress(data_bytes, compresslevel=level)
            else:
                raise CommandError(f"Unsupported compression algorithm: {algorithm}")
            
            compressed_size = len(compressed_data)
            compression_ratio = (original_size - compressed_size) / original_size * 100
            
            return CommandResult(
                success=True,
                data={
                    'compressed_data': compressed_data,
                    'algorithm': algorithm,
                    'level': level,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'savings': original_size - compressed_size
                },
                message=f"Data compressed: {original_size} -> {compressed_size} bytes ({compression_ratio:.1f}% reduction)"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Compression failed: {e}"
            )


class DecompressCommand(DataManagementCommand):
    """Распаковка данных"""
    
    def __init__(self):
        super().__init__("decompress", "Decompress compressed data")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            compressed_data = kwargs.get('data')
            algorithm = kwargs.get('algorithm', 'gzip')
            output_format = kwargs.get('format', 'auto')  # 'auto', 'string', 'json'
            
            if compressed_data is None:
                raise CommandError("Compressed data is required")
            
            if not isinstance(compressed_data, bytes):
                raise CommandError("Compressed data must be bytes")
            
            # Распаковка данных
            if algorithm == 'gzip':
                decompressed_bytes = gzip.decompress(compressed_data)
            else:
                raise CommandError(f"Unsupported decompression algorithm: {algorithm}")
            
            # Преобразование в нужный формат
            decompressed_str = decompressed_bytes.decode('utf-8')
            
            if output_format == 'auto':
                # Попытка определить формат автоматически
                try:
                    decompressed_data = json.loads(decompressed_str)
                    detected_format = 'json'
                except:
                    decompressed_data = decompressed_str
                    detected_format = 'string'
            elif output_format == 'json':
                decompressed_data = json.loads(decompressed_str)
                detected_format = 'json'
            else:
                decompressed_data = decompressed_str
                detected_format = 'string'
            
            return CommandResult(
                success=True,
                data={
                    'decompressed_data': decompressed_data,
                    'algorithm': algorithm,
                    'format': detected_format,
                    'compressed_size': len(compressed_data),
                    'decompressed_size': len(decompressed_bytes)
                },
                message=f"Data decompressed: {len(compressed_data)} -> {len(decompressed_bytes)} bytes"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Decompression failed: {e}"
            )


class HashCommand(DataManagementCommand):
    """Хеширование данных"""
    
    def __init__(self):
        super().__init__("hash", "Generate hash of data")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get('data')
            algorithm = kwargs.get('algorithm', 'sha256')
            encoding = kwargs.get('encoding', 'utf-8')
            
            if data is None:
                raise CommandError("Data to hash is required")
            
            # Преобразование данных в байты
            if isinstance(data, str):
                data_bytes = data.encode(encoding)
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_str = json.dumps(data)
                data_bytes = data_str.encode(encoding)
            
            # Генерация хеша
            if algorithm == 'md5':
                hash_obj = hashlib.md5(data_bytes)
            elif algorithm == 'sha1':
                hash_obj = hashlib.sha1(data_bytes)
            elif algorithm == 'sha256':
                hash_obj = hashlib.sha256(data_bytes)
            elif algorithm == 'sha512':
                hash_obj = hashlib.sha512(data_bytes)
            else:
                raise CommandError(f"Unsupported hash algorithm: {algorithm}")
            
            hash_hex = hash_obj.hexdigest()
            
            return CommandResult(
                success=True,
                data={
                    'hash': hash_hex,
                    'algorithm': algorithm,
                    'encoding': encoding,
                    'data_size': len(data_bytes),
                    'hash_size': len(hash_hex)
                },
                message=f"Hash generated using {algorithm}: {hash_hex[:16]}..."
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Hashing failed: {e}"
            )


class VerifyCommand(DataManagementCommand):
    """Верификация целостности данных"""
    
    def __init__(self):
        super().__init__("verify", "Verify data integrity")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get('data')
            expected_hash = kwargs.get('hash')
            algorithm = kwargs.get('algorithm', 'sha256')
            key = kwargs.get('key')  # для проверки данных из хранилища
            
            if not data and not key:
                raise CommandError("Data or storage key is required")
            
            if not expected_hash and not key:
                raise CommandError("Expected hash is required for verification")
            
            # Получение данных из хранилища если указан ключ
            if key:
                if hasattr(context.neural_network, 'data_storage'):
                    if key in context.neural_network.data_storage:
                        record = context.neural_network.data_storage[key]
                        data = record.data
                        expected_hash = record.checksum
                        algorithm = 'sha256'  # предполагаем SHA256 для хранилища
                    else:
                        raise CommandError(f"Data with key '{key}' not found in storage")
                else:
                    raise CommandError("No data storage available")
            
            # Генерация хеша данных
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_str = json.dumps(data)
                data_bytes = data_str.encode('utf-8')
            
            if algorithm == 'md5':
                hash_obj = hashlib.md5(data_bytes)
            elif algorithm == 'sha1':
                hash_obj = hashlib.sha1(data_bytes)
            elif algorithm == 'sha256':
                hash_obj = hashlib.sha256(data_bytes)
            elif algorithm == 'sha512':
                hash_obj = hashlib.sha512(data_bytes)
            else:
                raise CommandError(f"Unsupported hash algorithm: {algorithm}")
            
            actual_hash = hash_obj.hexdigest()
            is_valid = actual_hash == expected_hash
            
            return CommandResult(
                success=True,
                data={
                    'valid': is_valid,
                    'expected_hash': expected_hash,
                    'actual_hash': actual_hash,
                    'algorithm': algorithm,
                    'data_size': len(data_bytes),
                    'key': key
                },
                message=f"Data integrity: {'VALID' if is_valid else 'INVALID'}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Verification failed: {e}"
            )


class IndexCommand(DataManagementCommand):
    """Индексирование данных"""
    
    def __init__(self):
        super().__init__("index", "Create and manage data indexes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            action = kwargs.get('action', 'create')  # 'create', 'search', 'update', 'delete'
            index_name = kwargs.get('name', 'default')
            field = kwargs.get('field')
            value = kwargs.get('value')
            data_key = kwargs.get('key')
            
            # Инициализация системы индексов
            if not hasattr(context.neural_network, 'indexes'):
                context.neural_network.indexes = {}
            
            if action == 'create':
                if not field:
                    raise CommandError("Field name is required for index creation")
                
                # Создание нового индекса
                if index_name not in context.neural_network.indexes:
                    context.neural_network.indexes[index_name] = {
                        'field': field,
                        'created_at': time.time(),
                        'entries': {},
                        'stats': {'total_entries': 0, 'last_updated': time.time()}
                    }
                
                index = context.neural_network.indexes[index_name]
                
                # Индексирование существующих данных
                indexed_count = 0
                if hasattr(context.neural_network, 'data_storage'):
                    for key, record in context.neural_network.data_storage.items():
                        if isinstance(record.data, dict) and field in record.data:
                            field_value = str(record.data[field])
                            if field_value not in index['entries']:
                                index['entries'][field_value] = []
                            index['entries'][field_value].append(key)
                            indexed_count += 1
                
                index['stats']['total_entries'] = indexed_count
                index['stats']['last_updated'] = time.time()
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'index_name': index_name,
                        'field': field,
                        'indexed_count': indexed_count,
                        'unique_values': len(index['entries'])
                    },
                    message=f"Index '{index_name}' created for field '{field}' ({indexed_count} entries)"
                )
            
            elif action == 'search':
                if not value:
                    raise CommandError("Search value is required")
                
                if index_name not in context.neural_network.indexes:
                    raise CommandError(f"Index '{index_name}' not found")
                
                index = context.neural_network.indexes[index_name]
                search_value = str(value)
                
                # Поиск по индексу
                if search_value in index['entries']:
                    matching_keys = index['entries'][search_value]
                    
                    # Получение данных по найденным ключам
                    results = []
                    if hasattr(context.neural_network, 'data_storage'):
                        for key in matching_keys:
                            if key in context.neural_network.data_storage:
                                record = context.neural_network.data_storage[key]
                                results.append({
                                    'key': key,
                                    'data': record.data,
                                    'created_at': record.created_at
                                })
                    
                    return CommandResult(
                        success=True,
                        data={
                            'action': action,
                            'index_name': index_name,
                            'search_value': search_value,
                            'matching_keys': matching_keys,
                            'results': results,
                            'result_count': len(results)
                        },
                        message=f"Found {len(results)} results for '{search_value}'"
                    )
                else:
                    return CommandResult(
                        success=True,
                        data={
                            'action': action,
                            'index_name': index_name,
                            'search_value': search_value,
                            'matching_keys': [],
                            'results': [],
                            'result_count': 0
                        },
                        message=f"No results found for '{search_value}'"
                    )
            
            elif action == 'update':
                if not data_key:
                    raise CommandError("Data key is required for index update")
                
                # Обновление всех индексов для данного ключа
                updated_indexes = []
                if hasattr(context.neural_network, 'data_storage'):
                    if data_key in context.neural_network.data_storage:
                        record = context.neural_network.data_storage[data_key]
                        
                        for idx_name, index in context.neural_network.indexes.items():
                            field_name = index['field']
                            
                            # Удаление старых записей
                            for field_val, keys in index['entries'].items():
                                if data_key in keys:
                                    keys.remove(data_key)
                                    if not keys:
                                        del index['entries'][field_val]
                            
                            # Добавление новой записи
                            if isinstance(record.data, dict) and field_name in record.data:
                                field_value = str(record.data[field_name])
                                if field_value not in index['entries']:
                                    index['entries'][field_value] = []
                                index['entries'][field_value].append(data_key)
                                updated_indexes.append(idx_name)
                            
                            index['stats']['last_updated'] = time.time()
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'data_key': data_key,
                        'updated_indexes': updated_indexes,
                        'update_count': len(updated_indexes)
                    },
                    message=f"Updated {len(updated_indexes)} indexes for key '{data_key}'"
                )
            
            elif action == 'delete':
                if index_name not in context.neural_network.indexes:
                    raise CommandError(f"Index '{index_name}' not found")
                
                del context.neural_network.indexes[index_name]
                
                return CommandResult(
                    success=True,
                    data={
                        'action': action,
                        'index_name': index_name
                    },
                    message=f"Index '{index_name}' deleted"
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Index operation failed: {e}"
            )


# =============================================================================
# РЕГИСТРАЦИЯ КОМАНД
# =============================================================================

DATA_MANAGEMENT_COMMANDS = [
    StoreCommand(),
    LoadCommand(),
    CacheCommand(),
    BackupCommand(),
    RestoreCommand(),
    CompressCommand(),
    DecompressCommand(),
    HashCommand(),
    VerifyCommand(),
    IndexCommand()
] 