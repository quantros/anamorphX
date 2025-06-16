"""
Сетевые команды AnamorphX

Команды для работы с сетевыми соединениями и распределенными вычислениями.
"""

import time
import uuid
import socket
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .commands import CloudNetworkCommand, CommandResult, CommandError, ExecutionContext


class ConnectionState(Enum):
    """Состояния соединения"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    TIMEOUT = "timeout"


class MessageType(Enum):
    """Типы сообщений"""
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"


@dataclass
class NetworkConnection:
    """Сетевое соединение"""
    id: str
    host: str
    port: int
    state: ConnectionState
    created_at: float
    last_activity: float
    bytes_sent: int = 0
    bytes_received: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class NetworkMessage:
    """Сетевое сообщение"""
    id: str
    type: MessageType
    source: str
    destination: str
    data: Any
    timestamp: float
    size: int


class ConnectCommand(CloudNetworkCommand):
    """Команда подключения к сети"""
    
    def __init__(self):
        super().__init__(
            name="connect",
            description="Устанавливает соединение с удаленным узлом",
            parameters={
                "host": "Адрес хоста для подключения",
                "port": "Порт для подключения",
                "protocol": "Протокол соединения (tcp, udp, websocket)",
                "timeout": "Таймаут соединения в секундах",
                "retry_count": "Количество попыток переподключения"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            host = kwargs.get("host", "localhost")
            port = int(kwargs.get("port", 8080))
            protocol = kwargs.get("protocol", "tcp")
            timeout = int(kwargs.get("timeout", 30))
            retry_count = int(kwargs.get("retry_count", 3))
            
            connection_id = f"conn_{uuid.uuid4().hex[:8]}"
            
            # Симулируем установку соединения
            for attempt in range(retry_count):
                try:
                    # В реальной реализации здесь был бы код подключения
                    connection = NetworkConnection(
                        id=connection_id,
                        host=host,
                        port=port,
                        state=ConnectionState.CONNECTED,
                        created_at=time.time(),
                        last_activity=time.time(),
                        metadata={
                            "protocol": protocol,
                            "timeout": timeout,
                            "attempt": attempt + 1
                        }
                    )
                    
                    # Сохраняем соединение в контексте
                    if not hasattr(context, 'network_connections'):
                        context.network_connections = {}
                    context.network_connections[connection_id] = connection
                    
                    return CommandResult(
                        success=True,
                        message=f"Соединение установлено с {host}:{port}",
                        data={
                            "connection_id": connection_id,
                            "host": host,
                            "port": port,
                            "protocol": protocol,
                            "state": connection.state.value,
                            "attempt": attempt + 1,
                            "established_at": connection.created_at
                        }
                    )
                    
                except Exception as e:
                    if attempt == retry_count - 1:
                        raise e
                    time.sleep(1)  # Пауза между попытками
            
            raise ConnectionError(f"Не удалось подключиться к {host}:{port} за {retry_count} попыток")
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка подключения: {str(e)}",
                error=CommandError("CONNECTION_ERROR", str(e))
            )


class DisconnectCommand(CloudNetworkCommand):
    """Команда отключения от сети"""
    
    def __init__(self):
        super().__init__(
            name="disconnect",
            description="Закрывает сетевое соединение",
            parameters={
                "connection": "ID соединения для закрытия",
                "graceful": "Корректное закрытие соединения (true/false)",
                "timeout": "Таймаут для корректного закрытия"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            connection_id = kwargs.get("connection")
            graceful = kwargs.get("graceful", "true").lower() == "true"
            timeout = int(kwargs.get("timeout", 10))
            
            if not connection_id:
                return CommandResult(
                    success=False,
                    message="Не указан ID соединения",
                    error=CommandError("NO_CONNECTION_ID", "Connection ID required")
                )
            
            if not hasattr(context, 'network_connections'):
                return CommandResult(
                    success=False,
                    message="Нет активных соединений",
                    error=CommandError("NO_CONNECTIONS", "No active connections")
                )
            
            if connection_id not in context.network_connections:
                return CommandResult(
                    success=False,
                    message=f"Соединение {connection_id} не найдено",
                    error=CommandError("CONNECTION_NOT_FOUND", f"Connection {connection_id} not found")
                )
            
            connection = context.network_connections[connection_id]
            
            # Обновляем состояние соединения
            connection.state = ConnectionState.DISCONNECTED
            connection.last_activity = time.time()
            
            disconnect_info = {
                "connection_id": connection_id,
                "host": connection.host,
                "port": connection.port,
                "graceful": graceful,
                "timeout": timeout,
                "disconnected_at": time.time(),
                "duration": time.time() - connection.created_at,
                "bytes_sent": connection.bytes_sent,
                "bytes_received": connection.bytes_received
            }
            
            # Удаляем соединение из активных
            del context.network_connections[connection_id]
            
            return CommandResult(
                success=True,
                message=f"Соединение {connection_id} закрыто",
                data=disconnect_info
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка отключения: {str(e)}",
                error=CommandError("DISCONNECT_ERROR", str(e))
            )


class SendCommand(CloudNetworkCommand):
    """Команда отправки данных"""
    
    def __init__(self):
        super().__init__(
            name="send",
            description="Отправляет данные через сетевое соединение",
            parameters={
                "connection": "ID соединения для отправки",
                "data": "Данные для отправки",
                "format": "Формат данных (json, binary, text)",
                "priority": "Приоритет сообщения (low, normal, high)",
                "timeout": "Таймаут отправки в секундах"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            connection_id = kwargs.get("connection")
            data = kwargs.get("data")
            data_format = kwargs.get("format", "json")
            priority = kwargs.get("priority", "normal")
            timeout = int(kwargs.get("timeout", 30))
            
            if not connection_id or data is None:
                return CommandResult(
                    success=False,
                    message="Не указано соединение или данные",
                    error=CommandError("MISSING_PARAMS", "Connection and data required")
                )
            
            if not hasattr(context, 'network_connections'):
                return CommandResult(
                    success=False,
                    message="Нет активных соединений",
                    error=CommandError("NO_CONNECTIONS", "No active connections")
                )
            
            if connection_id not in context.network_connections:
                return CommandResult(
                    success=False,
                    message=f"Соединение {connection_id} не найдено",
                    error=CommandError("CONNECTION_NOT_FOUND", f"Connection {connection_id} not found")
                )
            
            connection = context.network_connections[connection_id]
            
            if connection.state != ConnectionState.CONNECTED:
                return CommandResult(
                    success=False,
                    message=f"Соединение {connection_id} не активно",
                    error=CommandError("CONNECTION_INACTIVE", f"Connection {connection_id} not active")
                )
            
            # Сериализуем данные
            if data_format == "json":
                import json
                serialized_data = json.dumps(data)
            elif data_format == "binary":
                import pickle
                serialized_data = pickle.dumps(data)
            else:
                serialized_data = str(data)
            
            data_size = len(serialized_data.encode() if isinstance(serialized_data, str) else serialized_data)
            
            # Создаем сообщение
            message = NetworkMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                type=MessageType.DATA,
                source="local",
                destination=f"{connection.host}:{connection.port}",
                data=serialized_data,
                timestamp=time.time(),
                size=data_size
            )
            
            # Обновляем статистику соединения
            connection.bytes_sent += data_size
            connection.last_activity = time.time()
            
            # Сохраняем сообщение в истории
            if not hasattr(context, 'sent_messages'):
                context.sent_messages = []
            context.sent_messages.append(message)
            
            return CommandResult(
                success=True,
                message=f"Данные отправлены через соединение {connection_id}",
                data={
                    "message_id": message.id,
                    "connection_id": connection_id,
                    "data_size": data_size,
                    "format": data_format,
                    "priority": priority,
                    "sent_at": message.timestamp,
                    "destination": message.destination
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка отправки: {str(e)}",
                error=CommandError("SEND_ERROR", str(e))
            )


class ReceiveCommand(CloudNetworkCommand):
    """Команда получения данных"""
    
    def __init__(self):
        super().__init__(
            name="receive",
            description="Получает данные через сетевое соединение",
            parameters={
                "connection": "ID соединения для получения",
                "timeout": "Таймаут получения в секундах",
                "buffer_size": "Размер буфера для получения",
                "format": "Ожидаемый формат данных"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            connection_id = kwargs.get("connection")
            timeout = int(kwargs.get("timeout", 30))
            buffer_size = int(kwargs.get("buffer_size", 4096))
            data_format = kwargs.get("format", "auto")
            
            if not connection_id:
                return CommandResult(
                    success=False,
                    message="Не указан ID соединения",
                    error=CommandError("NO_CONNECTION_ID", "Connection ID required")
                )
            
            if not hasattr(context, 'network_connections'):
                return CommandResult(
                    success=False,
                    message="Нет активных соединений",
                    error=CommandError("NO_CONNECTIONS", "No active connections")
                )
            
            if connection_id not in context.network_connections:
                return CommandResult(
                    success=False,
                    message=f"Соединение {connection_id} не найдено",
                    error=CommandError("CONNECTION_NOT_FOUND", f"Connection {connection_id} not found")
                )
            
            connection = context.network_connections[connection_id]
            
            if connection.state != ConnectionState.CONNECTED:
                return CommandResult(
                    success=False,
                    message=f"Соединение {connection_id} не активно",
                    error=CommandError("CONNECTION_INACTIVE", f"Connection {connection_id} not active")
                )
            
            # Симулируем получение данных
            received_data = f"Симулированные данные от {connection.host}:{connection.port}"
            data_size = len(received_data.encode())
            
            # Создаем сообщение
            message = NetworkMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                type=MessageType.DATA,
                source=f"{connection.host}:{connection.port}",
                destination="local",
                data=received_data,
                timestamp=time.time(),
                size=data_size
            )
            
            # Обновляем статистику соединения
            connection.bytes_received += data_size
            connection.last_activity = time.time()
            
            # Сохраняем сообщение в истории
            if not hasattr(context, 'received_messages'):
                context.received_messages = []
            context.received_messages.append(message)
            
            return CommandResult(
                success=True,
                message=f"Данные получены через соединение {connection_id}",
                data={
                    "message_id": message.id,
                    "connection_id": connection_id,
                    "data": received_data,
                    "data_size": data_size,
                    "format": data_format,
                    "received_at": message.timestamp,
                    "source": message.source
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка получения: {str(e)}",
                error=CommandError("RECEIVE_ERROR", str(e))
            )


class BroadcastCommand(CloudNetworkCommand):
    """Команда широковещательной рассылки"""
    
    def __init__(self):
        super().__init__(
            name="broadcast",
            description="Отправляет данные всем подключенным узлам",
            parameters={
                "data": "Данные для рассылки",
                "exclude": "Исключить соединения (список ID)",
                "format": "Формат данных",
                "priority": "Приоритет сообщения"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            exclude_str = kwargs.get("exclude", "")
            data_format = kwargs.get("format", "json")
            priority = kwargs.get("priority", "normal")
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Не указаны данные для рассылки",
                    error=CommandError("NO_DATA", "Data parameter required")
                )
            
            exclude_list = [conn.strip() for conn in exclude_str.split(",") if conn.strip()]
            
            if not hasattr(context, 'network_connections'):
                context.network_connections = {}
            
            # Получаем список активных соединений
            active_connections = [
                conn_id for conn_id, conn in context.network_connections.items()
                if conn.state == ConnectionState.CONNECTED and conn_id not in exclude_list
            ]
            
            # Сериализуем данные
            if data_format == "json":
                import json
                serialized_data = json.dumps(data)
            else:
                serialized_data = str(data)
            
            data_size = len(serialized_data.encode())
            broadcast_id = f"broadcast_{uuid.uuid4().hex[:8]}"
            
            # Отправляем всем активным соединениям
            sent_to = []
            for conn_id in active_connections:
                connection = context.network_connections[conn_id]
                
                # Создаем сообщение
                message = NetworkMessage(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    type=MessageType.BROADCAST,
                    source="local",
                    destination=f"{connection.host}:{connection.port}",
                    data=serialized_data,
                    timestamp=time.time(),
                    size=data_size
                )
                
                # Обновляем статистику
                connection.bytes_sent += data_size
                connection.last_activity = time.time()
                
                sent_to.append({
                    "connection_id": conn_id,
                    "destination": f"{connection.host}:{connection.port}",
                    "message_id": message.id
                })
            
            return CommandResult(
                success=True,
                message=f"Данные разосланы {len(sent_to)} получателям",
                data={
                    "broadcast_id": broadcast_id,
                    "data_size": data_size,
                    "format": data_format,
                    "priority": priority,
                    "recipients_count": len(sent_to),
                    "excluded_count": len(exclude_list),
                    "sent_to": sent_to,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка рассылки: {str(e)}",
                error=CommandError("BROADCAST_ERROR", str(e))
            )


class MulticastCommand(CloudNetworkCommand):
    """Команда многоадресной рассылки"""
    
    def __init__(self):
        super().__init__(
            name="multicast",
            description="Отправляет данные группе узлов",
            parameters={
                "data": "Данные для рассылки",
                "group": "Группа получателей",
                "connections": "Список соединений для рассылки",
                "format": "Формат данных"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            group = kwargs.get("group", "default")
            connections_str = kwargs.get("connections", "")
            data_format = kwargs.get("format", "json")
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Не указаны данные для рассылки",
                    error=CommandError("NO_DATA", "Data parameter required")
                )
            
            connection_list = [conn.strip() for conn in connections_str.split(",") if conn.strip()]
            
            if not hasattr(context, 'network_connections'):
                context.network_connections = {}
            
            # Фильтруем соединения
            target_connections = []
            if connection_list:
                # Используем указанный список соединений
                target_connections = [
                    conn_id for conn_id in connection_list
                    if conn_id in context.network_connections and
                    context.network_connections[conn_id].state == ConnectionState.CONNECTED
                ]
            else:
                # Используем все соединения группы (симуляция)
                target_connections = [
                    conn_id for conn_id, conn in context.network_connections.items()
                    if conn.state == ConnectionState.CONNECTED
                ]
            
            # Сериализуем данные
            if data_format == "json":
                import json
                serialized_data = json.dumps(data)
            else:
                serialized_data = str(data)
            
            data_size = len(serialized_data.encode())
            multicast_id = f"multicast_{uuid.uuid4().hex[:8]}"
            
            # Отправляем целевым соединениям
            sent_to = []
            for conn_id in target_connections:
                connection = context.network_connections[conn_id]
                
                # Создаем сообщение
                message = NetworkMessage(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    type=MessageType.MULTICAST,
                    source="local",
                    destination=f"{connection.host}:{connection.port}",
                    data=serialized_data,
                    timestamp=time.time(),
                    size=data_size
                )
                
                # Обновляем статистику
                connection.bytes_sent += data_size
                connection.last_activity = time.time()
                
                sent_to.append({
                    "connection_id": conn_id,
                    "destination": f"{connection.host}:{connection.port}",
                    "message_id": message.id
                })
            
            return CommandResult(
                success=True,
                message=f"Данные разосланы группе {group}",
                data={
                    "multicast_id": multicast_id,
                    "group": group,
                    "data_size": data_size,
                    "format": data_format,
                    "recipients_count": len(sent_to),
                    "sent_to": sent_to,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка многоадресной рассылки: {str(e)}",
                error=CommandError("MULTICAST_ERROR", str(e))
            )


class TunnelCommand(CloudNetworkCommand):
    """Команда создания туннеля"""
    
    def __init__(self):
        super().__init__(
            name="tunnel",
            description="Создает защищенный туннель между узлами",
            parameters={
                "source": "Исходный узел",
                "destination": "Целевой узел",
                "encryption": "Тип шифрования (none, aes, rsa)",
                "compression": "Сжимать данные в туннеле (true/false)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get("source", "local")
            destination = kwargs.get("destination")
            encryption = kwargs.get("encryption", "aes")
            compression = kwargs.get("compression", "false").lower() == "true"
            
            if not destination:
                return CommandResult(
                    success=False,
                    message="Не указан целевой узел",
                    error=CommandError("NO_DESTINATION", "Destination parameter required")
                )
            
            tunnel_id = f"tunnel_{uuid.uuid4().hex[:8]}"
            
            tunnel_info = {
                "tunnel_id": tunnel_id,
                "source": source,
                "destination": destination,
                "encryption": encryption,
                "compression": compression,
                "created_at": time.time(),
                "status": "active",
                "bytes_transferred": 0,
                "packets_sent": 0,
                "packets_received": 0
            }
            
            # Сохраняем туннель в контексте
            if not hasattr(context, 'network_tunnels'):
                context.network_tunnels = {}
            context.network_tunnels[tunnel_id] = tunnel_info
            
            return CommandResult(
                success=True,
                message=f"Туннель создан между {source} и {destination}",
                data=tunnel_info
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка создания туннеля: {str(e)}",
                error=CommandError("TUNNEL_ERROR", str(e))
            )


class ProxyCommand(CloudNetworkCommand):
    """Команда создания прокси соединения"""
    
    def __init__(self):
        super().__init__(
            name="proxy",
            description="Создает прокси соединение для перенаправления трафика",
            parameters={
                "listen_port": "Порт для прослушивания",
                "target_host": "Целевой хост",
                "target_port": "Целевой порт",
                "proxy_type": "Тип прокси (http, socks, transparent)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            listen_port = int(kwargs.get("listen_port", 8080))
            target_host = kwargs.get("target_host", "localhost")
            target_port = int(kwargs.get("target_port", 80))
            proxy_type = kwargs.get("proxy_type", "http")
            
            proxy_id = f"proxy_{uuid.uuid4().hex[:8]}"
            
            proxy_info = {
                "proxy_id": proxy_id,
                "listen_port": listen_port,
                "target_host": target_host,
                "target_port": target_port,
                "proxy_type": proxy_type,
                "created_at": time.time(),
                "status": "active",
                "connections_count": 0,
                "bytes_proxied": 0
            }
            
            # Сохраняем прокси в контексте
            if not hasattr(context, 'network_proxies'):
                context.network_proxies = {}
            context.network_proxies[proxy_id] = proxy_info
            
            return CommandResult(
                success=True,
                message=f"Прокси создан на порту {listen_port}",
                data=proxy_info
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка создания прокси: {str(e)}",
                error=CommandError("PROXY_ERROR", str(e))
            )


class BalanceCommand(CloudNetworkCommand):
    """Команда балансировки нагрузки"""
    
    def __init__(self):
        super().__init__(
            name="balance",
            description="Настраивает балансировку нагрузки между узлами",
            parameters={
                "nodes": "Список узлов для балансировки",
                "algorithm": "Алгоритм балансировки (round_robin, least_connections, weighted)",
                "weights": "Веса узлов (для weighted алгоритма)",
                "health_check": "Проверка здоровья узлов (true/false)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            nodes_str = kwargs.get("nodes", "")
            algorithm = kwargs.get("algorithm", "round_robin")
            weights_str = kwargs.get("weights", "")
            health_check = kwargs.get("health_check", "true").lower() == "true"
            
            nodes = [node.strip() for node in nodes_str.split(",") if node.strip()]
            
            if not nodes:
                return CommandResult(
                    success=False,
                    message="Не указаны узлы для балансировки",
                    error=CommandError("NO_NODES", "Nodes parameter required")
                )
            
            # Парсим веса
            weights = []
            if weights_str:
                try:
                    weights = [float(w.strip()) for w in weights_str.split(",")]
                except ValueError:
                    weights = [1.0] * len(nodes)
            else:
                weights = [1.0] * len(nodes)
            
            balancer_id = f"balancer_{uuid.uuid4().hex[:8]}"
            
            # Создаем конфигурацию узлов
            node_configs = []
            for i, node in enumerate(nodes):
                weight = weights[i] if i < len(weights) else 1.0
                node_configs.append({
                    "node": node,
                    "weight": weight,
                    "status": "healthy",
                    "connections": 0,
                    "response_time": 0.0
                })
            
            balancer_info = {
                "balancer_id": balancer_id,
                "algorithm": algorithm,
                "nodes": node_configs,
                "health_check": health_check,
                "created_at": time.time(),
                "status": "active",
                "total_requests": 0,
                "current_node_index": 0  # Для round_robin
            }
            
            # Сохраняем балансировщик в контексте
            if not hasattr(context, 'load_balancers'):
                context.load_balancers = {}
            context.load_balancers[balancer_id] = balancer_info
            
            return CommandResult(
                success=True,
                message=f"Балансировка настроена для {len(nodes)} узлов",
                data=balancer_info
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка настройки балансировки: {str(e)}",
                error=CommandError("BALANCE_ERROR", str(e))
            )


class RouteCommand(CloudNetworkCommand):
    """Команда настройки маршрутизации"""
    
    def __init__(self):
        super().__init__(
            name="route",
            description="Настраивает маршрутизацию сетевого трафика",
            parameters={
                "destination": "Целевая сеть или узел",
                "gateway": "Шлюз для маршрута",
                "metric": "Метрика маршрута",
                "interface": "Сетевой интерфейс"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            destination = kwargs.get("destination")
            gateway = kwargs.get("gateway")
            metric = int(kwargs.get("metric", 1))
            interface = kwargs.get("interface", "default")
            
            if not destination:
                return CommandResult(
                    success=False,
                    message="Не указано назначение маршрута",
                    error=CommandError("NO_DESTINATION", "Destination parameter required")
                )
            
            route_id = f"route_{uuid.uuid4().hex[:8]}"
            
            route_info = {
                "route_id": route_id,
                "destination": destination,
                "gateway": gateway,
                "metric": metric,
                "interface": interface,
                "created_at": time.time(),
                "status": "active",
                "packets_routed": 0,
                "bytes_routed": 0
            }
            
            # Сохраняем маршрут в контексте
            if not hasattr(context, 'network_routes'):
                context.network_routes = {}
            context.network_routes[route_id] = route_info
            
            return CommandResult(
                success=True,
                message=f"Маршрут к {destination} создан",
                data=route_info
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка создания маршрута: {str(e)}",
                error=CommandError("ROUTE_ERROR", str(e))
            )


# Регистрируем все сетевые команды
NETWORK_COMMANDS = [
    ConnectCommand(),
    DisconnectCommand(),
    SendCommand(),
    ReceiveCommand(),
    BroadcastCommand(),
    MulticastCommand(),
    TunnelCommand(),
    ProxyCommand(),
    BalanceCommand(),
    RouteCommand()
] 