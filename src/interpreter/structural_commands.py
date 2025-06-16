"""
Структурные команды AnamorphX

Команды для создания и управления нейронными структурами, топологиями и архитектурами.
"""

import time
import uuid
import math
import random
import copy
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum

from .commands import StructuralCommand, CommandResult, CommandError, NeuralEntity, SynapseConnection
from .runtime import ExecutionContext


# =============================================================================
# ENUMS И DATACLASSES
# =============================================================================

class NodeType(Enum):
    """Типы нейронных узлов"""
    BASIC = "basic"
    PERSISTENT = "persistent"
    GPU_ACCELERATED = "gpu_accelerated"
    MEMORY_OPTIMIZED = "memory_optimized"
    DISTRIBUTED = "distributed"


class BindingType(Enum):
    """Типы привязки данных"""
    PERSISTENT = "persistent"
    TEMPORARY = "temporary"
    STREAMING = "streaming"
    CACHED = "cached"


class ClusterState(Enum):
    """Состояния кластера"""
    FORMING = "forming"
    ACTIVE = "active"
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    DORMANT = "dormant"
    DISSOLVED = "dissolved"


class EvolutionStrategy(Enum):
    """Стратегии эволюции"""
    GENETIC = "genetic"
    GRADIENT = "gradient"
    REINFORCEMENT = "reinforcement"
    HYBRID = "hybrid"


@dataclass
class NeuralNode:
    """Нейронный узел"""
    id: str
    name: str
    node_type: NodeType
    created_at: float
    connections: Set[str] = field(default_factory=set)
    data: Dict[str, Any] = field(default_factory=dict)
    state: str = "inactive"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapseConnection:
    """Синаптическое соединение"""
    id: str
    source_id: str
    target_id: str
    weight: float
    created_at: float
    connection_type: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBinding:
    """Привязка данных к узлу"""
    id: str
    node_id: str
    data_key: str
    binding_type: BindingType
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeCluster:
    """Кластер узлов"""
    id: str
    name: str
    node_ids: Set[str]
    state: ClusterState
    created_at: float
    center_node_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# СТРУКТУРНЫЕ КОМАНДЫ (10 команд)
# =============================================================================

class NeuroCommand(StructuralCommand):
    """Создание нейронного узла"""
    
    def __init__(self):
        super().__init__("neuro", "Create a new neural node")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_name = kwargs.get('node_name')
            node_type = kwargs.get('type', 'basic')
            
            if not node_name:
                raise CommandError("Node name is required")
            
            # Проверка существования узла
            if node_name in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' already exists")
            
            # Проверка квоты узлов
            if len(context.neural_network.nodes) >= context.config.get('max_nodes', 1000):
                raise CommandError("Node quota exceeded")
            
            # Создание узла
            node_id = str(uuid.uuid4())
            node = NeuralNode(
                id=node_id,
                name=node_name,
                node_type=NodeType(node_type),
                created_at=time.time(),
                metadata={'created_by': 'neuro_command'}
            )
            
            # Добавление в сеть
            context.neural_network.nodes[node_name] = node
            context.neural_network.node_by_id[node_id] = node
            
            return CommandResult(
                success=True,
                data={
                    'node_id': node_id,
                    'node_name': node_name,
                    'node_type': node_type,
                    'created_at': node.created_at
                },
                message=f"Neural node '{node_name}' created successfully"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to create neural node: {e}"
            )


class SynapCommand(StructuralCommand):
    """Создание синаптического соединения"""
    
    def __init__(self):
        super().__init__("synap", "Create synaptic connection between nodes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get('from')
            target = kwargs.get('to')
            weight = kwargs.get('weight', 1.0)
            connection_type = kwargs.get('type', 'standard')
            
            if not source or not target:
                raise CommandError("Source and target nodes are required")
            
            # Проверка существования узлов
            if source not in context.neural_network.nodes:
                raise CommandError(f"Source node '{source}' not found")
            if target not in context.neural_network.nodes:
                raise CommandError(f"Target node '{target}' not found")
            
            # Создание соединения
            synapse_id = str(uuid.uuid4())
            source_node = context.neural_network.nodes[source]
            target_node = context.neural_network.nodes[target]
            
            synapse = SynapseConnection(
                id=synapse_id,
                source_id=source_node.id,
                target_id=target_node.id,
                weight=float(weight),
                created_at=time.time(),
                connection_type=connection_type,
                metadata={'created_by': 'synap_command'}
            )
            
            # Добавление соединения
            context.neural_network.synapses[synapse_id] = synapse
            source_node.connections.add(target_node.id)
            
            return CommandResult(
                success=True,
                data={
                    'synapse_id': synapse_id,
                    'source': source,
                    'target': target,
                    'weight': weight,
                    'type': connection_type
                },
                message=f"Synapse created: {source} -> {target}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to create synapse: {e}"
            )


class BindCommand(StructuralCommand):
    """Привязка данных к узлу"""
    
    def __init__(self):
        super().__init__("bind", "Bind data to neural node")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_name = kwargs.get('node')
            data_key = kwargs.get('key')
            value = kwargs.get('value')
            binding_type = kwargs.get('type', 'temporary')
            ttl = kwargs.get('ttl')  # time to live in seconds
            
            if not node_name or not data_key:
                raise CommandError("Node name and data key are required")
            
            # Проверка существования узла
            if node_name not in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' not found")
            
            node = context.neural_network.nodes[node_name]
            
            # Создание привязки
            binding_id = str(uuid.uuid4())
            expires_at = None
            if ttl:
                expires_at = time.time() + float(ttl)
            
            binding = DataBinding(
                id=binding_id,
                node_id=node.id,
                data_key=data_key,
                binding_type=BindingType(binding_type),
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                metadata={'created_by': 'bind_command'}
            )
            
            # Сохранение привязки
            if not hasattr(context.neural_network, 'bindings'):
                context.neural_network.bindings = {}
            context.neural_network.bindings[binding_id] = binding
            
            # Обновление данных узла
            node.data[data_key] = value
            
            return CommandResult(
                success=True,
                data={
                    'binding_id': binding_id,
                    'node': node_name,
                    'key': data_key,
                    'type': binding_type,
                    'expires_at': expires_at
                },
                message=f"Data bound to node '{node_name}'"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to bind data: {e}"
            )


class ClusterCommand(StructuralCommand):
    """Создание кластера узлов"""
    
    def __init__(self):
        super().__init__("cluster", "Create cluster of neural nodes")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            cluster_name = kwargs.get('name')
            node_names = kwargs.get('nodes', [])
            center_node = kwargs.get('center')
            
            if not cluster_name:
                raise CommandError("Cluster name is required")
            if not node_names:
                raise CommandError("At least one node is required")
            
            # Проверка существования узлов
            node_ids = set()
            for node_name in node_names:
                if node_name not in context.neural_network.nodes:
                    raise CommandError(f"Node '{node_name}' not found")
                node_ids.add(context.neural_network.nodes[node_name].id)
            
            # Проверка центрального узла
            center_node_id = None
            if center_node:
                if center_node not in context.neural_network.nodes:
                    raise CommandError(f"Center node '{center_node}' not found")
                center_node_id = context.neural_network.nodes[center_node].id
            
            # Создание кластера
            cluster_id = str(uuid.uuid4())
            cluster = NodeCluster(
                id=cluster_id,
                name=cluster_name,
                node_ids=node_ids,
                state=ClusterState.FORMING,
                created_at=time.time(),
                center_node_id=center_node_id,
                metadata={'created_by': 'cluster_command'}
            )
            
            # Сохранение кластера
            if not hasattr(context.neural_network, 'clusters'):
                context.neural_network.clusters = {}
            context.neural_network.clusters[cluster_id] = cluster
            
            # Активация кластера
            cluster.state = ClusterState.ACTIVE
            
            return CommandResult(
                success=True,
                data={
                    'cluster_id': cluster_id,
                    'name': cluster_name,
                    'nodes': list(node_names),
                    'center_node': center_node,
                    'state': cluster.state.value
                },
                message=f"Cluster '{cluster_name}' created with {len(node_names)} nodes"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to create cluster: {e}"
            )


class ExpandCommand(StructuralCommand):
    """Расширение кластера"""
    
    def __init__(self):
        super().__init__("expand", "Expand neural cluster")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            cluster_name = kwargs.get('cluster')
            new_nodes = kwargs.get('nodes', [])
            expansion_factor = kwargs.get('factor', 1.0)
            
            if not cluster_name:
                raise CommandError("Cluster name is required")
            
            # Поиск кластера
            cluster = None
            if hasattr(context.neural_network, 'clusters'):
                for c in context.neural_network.clusters.values():
                    if c.name == cluster_name:
                        cluster = c
                        break
            
            if not cluster:
                raise CommandError(f"Cluster '{cluster_name}' not found")
            
            # Проверка состояния кластера
            if cluster.state == ClusterState.DISSOLVED:
                raise CommandError("Cannot expand dissolved cluster")
            
            cluster.state = ClusterState.EXPANDING
            original_size = len(cluster.node_ids)
            
            # Добавление новых узлов
            if new_nodes:
                for node_name in new_nodes:
                    if node_name not in context.neural_network.nodes:
                        raise CommandError(f"Node '{node_name}' not found")
                    cluster.node_ids.add(context.neural_network.nodes[node_name].id)
            
            # Автоматическое расширение по фактору
            if expansion_factor > 1.0 and not new_nodes:
                available_nodes = []
                for node_name, node in context.neural_network.nodes.items():
                    if node.id not in cluster.node_ids:
                        available_nodes.append(node)
                
                target_size = int(original_size * expansion_factor)
                nodes_to_add = min(len(available_nodes), target_size - original_size)
                
                for i in range(nodes_to_add):
                    cluster.node_ids.add(available_nodes[i].id)
            
            cluster.state = ClusterState.ACTIVE
            new_size = len(cluster.node_ids)
            
            return CommandResult(
                success=True,
                data={
                    'cluster_name': cluster_name,
                    'original_size': original_size,
                    'new_size': new_size,
                    'expansion_factor': expansion_factor,
                    'added_nodes': new_size - original_size
                },
                message=f"Cluster '{cluster_name}' expanded from {original_size} to {new_size} nodes"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to expand cluster: {e}"
            )


class ContractCommand(StructuralCommand):
    """Сжатие кластера"""
    
    def __init__(self):
        super().__init__("contract", "Contract neural cluster")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            cluster_name = kwargs.get('cluster')
            remove_nodes = kwargs.get('remove', [])
            contraction_factor = kwargs.get('factor', 0.8)
            preserve_center = kwargs.get('preserve_center', True)
            
            if not cluster_name:
                raise CommandError("Cluster name is required")
            
            # Поиск кластера
            cluster = None
            if hasattr(context.neural_network, 'clusters'):
                for c in context.neural_network.clusters.values():
                    if c.name == cluster_name:
                        cluster = c
                        break
            
            if not cluster:
                raise CommandError(f"Cluster '{cluster_name}' not found")
            
            cluster.state = ClusterState.CONTRACTING
            original_size = len(cluster.node_ids)
            
            # Удаление конкретных узлов
            if remove_nodes:
                for node_name in remove_nodes:
                    if node_name in context.neural_network.nodes:
                        node_id = context.neural_network.nodes[node_name].id
                        if node_id in cluster.node_ids:
                            # Проверка центрального узла
                            if preserve_center and node_id == cluster.center_node_id:
                                continue
                            cluster.node_ids.discard(node_id)
            
            # Автоматическое сжатие по фактору
            elif contraction_factor < 1.0:
                target_size = max(1, int(original_size * contraction_factor))
                nodes_to_remove = original_size - target_size
                
                # Получение списка узлов для удаления (исключая центральный)
                removable_nodes = []
                for node_id in cluster.node_ids:
                    if not preserve_center or node_id != cluster.center_node_id:
                        removable_nodes.append(node_id)
                
                # Случайное удаление узлов
                random.shuffle(removable_nodes)
                for i in range(min(nodes_to_remove, len(removable_nodes))):
                    cluster.node_ids.discard(removable_nodes[i])
            
            cluster.state = ClusterState.ACTIVE
            new_size = len(cluster.node_ids)
            
            return CommandResult(
                success=True,
                data={
                    'cluster_name': cluster_name,
                    'original_size': original_size,
                    'new_size': new_size,
                    'contraction_factor': contraction_factor,
                    'removed_nodes': original_size - new_size
                },
                message=f"Cluster '{cluster_name}' contracted from {original_size} to {new_size} nodes"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to contract cluster: {e}"
            )


class MorphCommand(StructuralCommand):
    """Трансформация узла"""
    
    def __init__(self):
        super().__init__("morph", "Transform neural node structure")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_name = kwargs.get('node')
            new_type = kwargs.get('type')
            properties = kwargs.get('properties', {})
            preserve_connections = kwargs.get('preserve_connections', True)
            
            if not node_name:
                raise CommandError("Node name is required")
            if not new_type:
                raise CommandError("New node type is required")
            
            # Проверка существования узла
            if node_name not in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' not found")
            
            node = context.neural_network.nodes[node_name]
            old_type = node.node_type
            
            # Сохранение старых свойств
            old_connections = node.connections.copy() if preserve_connections else set()
            old_data = node.data.copy()
            
            # Трансформация узла
            node.node_type = NodeType(new_type)
            node.metadata['morphed_from'] = old_type.value
            node.metadata['morphed_at'] = time.time()
            
            # Применение новых свойств
            for key, value in properties.items():
                node.metadata[key] = value
            
            # Восстановление соединений если нужно
            if preserve_connections:
                node.connections = old_connections
            
            return CommandResult(
                success=True,
                data={
                    'node_name': node_name,
                    'old_type': old_type.value,
                    'new_type': new_type,
                    'properties': properties,
                    'connections_preserved': preserve_connections,
                    'connection_count': len(node.connections)
                },
                message=f"Node '{node_name}' morphed from {old_type.value} to {new_type}"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to morph node: {e}"
            )


class EvolveCommand(StructuralCommand):
    """Эволюционное развитие узла"""
    
    def __init__(self):
        super().__init__("evolve", "Evolve neural node using specified strategy")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            node_name = kwargs.get('node')
            strategy = kwargs.get('strategy', 'genetic')
            generations = kwargs.get('generations', 10)
            fitness_function = kwargs.get('fitness_function')
            mutation_rate = kwargs.get('mutation_rate', 0.1)
            
            if not node_name:
                raise CommandError("Node name is required")
            
            # Проверка существования узла
            if node_name not in context.neural_network.nodes:
                raise CommandError(f"Node '{node_name}' not found")
            
            node = context.neural_network.nodes[node_name]
            evolution_strategy = EvolutionStrategy(strategy)
            
            # Инициализация эволюции
            evolution_data = {
                'strategy': strategy,
                'generations': generations,
                'mutation_rate': mutation_rate,
                'start_time': time.time(),
                'fitness_history': []
            }
            
            # Симуляция эволюционного процесса
            current_fitness = random.uniform(0.5, 1.0)  # Начальная приспособленность
            
            for generation in range(generations):
                # Мутация (изменение весов соединений)
                if random.random() < mutation_rate:
                    # Случайное изменение метаданных узла
                    if 'evolution_score' not in node.metadata:
                        node.metadata['evolution_score'] = current_fitness
                    else:
                        node.metadata['evolution_score'] *= random.uniform(0.9, 1.1)
                
                # Оценка приспособленности
                if fitness_function:
                    # В реальной реализации здесь был бы вызов пользовательской функции
                    current_fitness = random.uniform(current_fitness * 0.9, current_fitness * 1.1)
                else:
                    current_fitness = min(1.0, current_fitness * random.uniform(1.0, 1.05))
                
                evolution_data['fitness_history'].append(current_fitness)
            
            # Обновление узла результатами эволюции
            node.metadata['evolved'] = True
            node.metadata['evolution_data'] = evolution_data
            node.metadata['final_fitness'] = current_fitness
            node.metadata['evolved_at'] = time.time()
            
            return CommandResult(
                success=True,
                data={
                    'node_name': node_name,
                    'strategy': strategy,
                    'generations': generations,
                    'final_fitness': current_fitness,
                    'improvement': current_fitness - evolution_data['fitness_history'][0] if evolution_data['fitness_history'] else 0,
                    'evolution_time': time.time() - evolution_data['start_time']
                },
                message=f"Node '{node_name}' evolved using {strategy} strategy over {generations} generations"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to evolve node: {e}"
            )


class PruneCommand(StructuralCommand):
    """Обрезка неактивных элементов"""
    
    def __init__(self):
        super().__init__("prune", "Remove inactive neural elements")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get('target', 'network')  # 'network', 'cluster', 'node'
            threshold = kwargs.get('threshold', 0.1)  # Порог активности
            preserve_critical = kwargs.get('preserve_critical', True)
            dry_run = kwargs.get('dry_run', False)
            
            pruned_elements = {
                'nodes': [],
                'synapses': [],
                'bindings': []
            }
            
            current_time = time.time()
            
            # Обрезка узлов
            nodes_to_remove = []
            for node_name, node in context.neural_network.nodes.items():
                # Проверка активности узла
                last_activity = node.metadata.get('last_activity', node.created_at)
                inactivity_time = current_time - last_activity
                
                # Критические узлы не удаляем
                if preserve_critical and node.metadata.get('critical', False):
                    continue
                
                # Проверка порога неактивности
                if inactivity_time > threshold * 3600:  # threshold в часах
                    nodes_to_remove.append(node_name)
                    pruned_elements['nodes'].append({
                        'name': node_name,
                        'id': node.id,
                        'inactive_hours': inactivity_time / 3600
                    })
            
            # Обрезка синапсов
            synapses_to_remove = []
            if hasattr(context.neural_network, 'synapses'):
                for synapse_id, synapse in context.neural_network.synapses.items():
                    # Проверка веса синапса
                    if abs(synapse.weight) < threshold:
                        synapses_to_remove.append(synapse_id)
                        pruned_elements['synapses'].append({
                            'id': synapse_id,
                            'weight': synapse.weight,
                            'source': synapse.source_id,
                            'target': synapse.target_id
                        })
            
            # Обрезка привязок данных
            bindings_to_remove = []
            if hasattr(context.neural_network, 'bindings'):
                for binding_id, binding in context.neural_network.bindings.items():
                    # Проверка истекших привязок
                    if binding.expires_at and current_time > binding.expires_at:
                        bindings_to_remove.append(binding_id)
                        pruned_elements['bindings'].append({
                            'id': binding_id,
                            'node_id': binding.node_id,
                            'key': binding.data_key,
                            'expired_hours': (current_time - binding.expires_at) / 3600
                        })
            
            # Выполнение обрезки (если не dry_run)
            if not dry_run:
                # Удаление узлов
                for node_name in nodes_to_remove:
                    node = context.neural_network.nodes[node_name]
                    del context.neural_network.nodes[node_name]
                    if node.id in context.neural_network.node_by_id:
                        del context.neural_network.node_by_id[node.id]
                
                # Удаление синапсов
                if hasattr(context.neural_network, 'synapses'):
                    for synapse_id in synapses_to_remove:
                        del context.neural_network.synapses[synapse_id]
                
                # Удаление привязок
                if hasattr(context.neural_network, 'bindings'):
                    for binding_id in bindings_to_remove:
                        del context.neural_network.bindings[binding_id]
            
            total_pruned = len(pruned_elements['nodes']) + len(pruned_elements['synapses']) + len(pruned_elements['bindings'])
            
            return CommandResult(
                success=True,
                data={
                    'target': target,
                    'threshold': threshold,
                    'dry_run': dry_run,
                    'pruned_elements': pruned_elements,
                    'total_pruned': total_pruned,
                    'nodes_removed': len(pruned_elements['nodes']),
                    'synapses_removed': len(pruned_elements['synapses']),
                    'bindings_removed': len(pruned_elements['bindings'])
                },
                message=f"Pruning completed: {total_pruned} elements {'would be' if dry_run else ''} removed"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to prune elements: {e}"
            )


class ForgeCommand(StructuralCommand):
    """Создание сложных структур"""
    
    def __init__(self):
        super().__init__("forge", "Create complex neural structures")
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            structure_type = kwargs.get('type', 'mesh')  # 'mesh', 'tree', 'ring', 'star'
            name = kwargs.get('name')
            size = kwargs.get('size', 5)
            properties = kwargs.get('properties', {})
            
            if not name:
                raise CommandError("Structure name is required")
            
            if size < 2:
                raise CommandError("Structure size must be at least 2")
            
            created_nodes = []
            created_synapses = []
            
            # Создание узлов структуры
            node_names = []
            for i in range(size):
                node_name = f"{name}_node_{i}"
                node_id = str(uuid.uuid4())
                
                node = NeuralNode(
                    id=node_id,
                    name=node_name,
                    node_type=NodeType.BASIC,
                    created_at=time.time(),
                    metadata={
                        'created_by': 'forge_command',
                        'structure_type': structure_type,
                        'structure_name': name,
                        **properties
                    }
                )
                
                context.neural_network.nodes[node_name] = node
                context.neural_network.node_by_id[node_id] = node
                created_nodes.append(node_name)
                node_names.append(node_name)
            
            # Создание соединений в зависимости от типа структуры
            if structure_type == 'mesh':
                # Полносвязная сеть
                for i in range(size):
                    for j in range(i + 1, size):
                        synapse_id = self._create_synapse(
                            context, node_names[i], node_names[j], 1.0
                        )
                        created_synapses.append(synapse_id)
            
            elif structure_type == 'tree':
                # Древовидная структура
                for i in range(1, size):
                    parent_idx = (i - 1) // 2
                    synapse_id = self._create_synapse(
                        context, node_names[parent_idx], node_names[i], 1.0
                    )
                    created_synapses.append(synapse_id)
            
            elif structure_type == 'ring':
                # Кольцевая структура
                for i in range(size):
                    next_idx = (i + 1) % size
                    synapse_id = self._create_synapse(
                        context, node_names[i], node_names[next_idx], 1.0
                    )
                    created_synapses.append(synapse_id)
            
            elif structure_type == 'star':
                # Звездообразная структура
                center_node = node_names[0]
                for i in range(1, size):
                    synapse_id = self._create_synapse(
                        context, center_node, node_names[i], 1.0
                    )
                    created_synapses.append(synapse_id)
            
            return CommandResult(
                success=True,
                data={
                    'structure_name': name,
                    'structure_type': structure_type,
                    'size': size,
                    'created_nodes': created_nodes,
                    'created_synapses': len(created_synapses),
                    'properties': properties
                },
                message=f"Complex structure '{name}' of type '{structure_type}' forged with {size} nodes"
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                message=f"Failed to forge structure: {e}"
            )
    
    def _create_synapse(self, context: ExecutionContext, source_name: str, target_name: str, weight: float) -> str:
        """Вспомогательный метод для создания синапса"""
        synapse_id = str(uuid.uuid4())
        source_node = context.neural_network.nodes[source_name]
        target_node = context.neural_network.nodes[target_name]
        
        synapse = SynapseConnection(
            id=synapse_id,
            source_id=source_node.id,
            target_id=target_node.id,
            weight=weight,
            created_at=time.time(),
            connection_type='forged',
            metadata={'created_by': 'forge_command'}
        )
        
        if not hasattr(context.neural_network, 'synapses'):
            context.neural_network.synapses = {}
        context.neural_network.synapses[synapse_id] = synapse
        source_node.connections.add(target_node.id)
        
        return synapse_id


# =============================================================================
# РЕГИСТРАЦИЯ КОМАНД
# =============================================================================

STRUCTURAL_COMMANDS = [
    NeuroCommand(),
    SynapCommand(),
    BindCommand(),
    ClusterCommand(),
    ExpandCommand(),
    ContractCommand(),
    MorphCommand(),
    EvolveCommand(),
    PruneCommand(),
    ForgeCommand()
] 