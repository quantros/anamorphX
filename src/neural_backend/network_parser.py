"""
AnamorphX Neural Network Parser
Парсинг network блоков AnamorphX в структурированные конфигурации
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class NeuronConfig:
    """Конфигурация нейрона (слоя)"""
    name: str
    layer_type: str  # 'dense', 'conv', 'pool', 'lstm', 'transformer', 'attention'
    activation: Optional[str] = None
    units: Optional[int] = None
    filters: Optional[int] = None
    kernel_size: Optional[int] = None
    padding: Optional[int] = None
    stride: Optional[int] = None
    pool_size: Optional[int] = None
    dropout: Optional[float] = None
    # Новые параметры для Transformer
    num_heads: Optional[int] = None
    embed_dim: Optional[int] = None
    ff_dim: Optional[int] = None
    # Параметры для ResNet
    skip_connection: Optional[bool] = None
    residual_type: Optional[str] = None


@dataclass
class NetworkConfig:
    """Конфигурация нейронной сети"""
    name: str
    neurons: List[NeuronConfig]
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None
    loss: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None


class NetworkParser:
    """Парсер network блоков AnamorphX"""
    
    def __init__(self):
        self.layer_type_patterns = {
            'dense': r'units:\s*(\d+)',
            'conv': r'filters:\s*(\d+)',
            'pool': r'pool_size:\s*(\d+)',
            'lstm': r'units:\s*(\d+).*?return_sequences',
            'transformer': r'num_heads:\s*(\d+)',
            'attention': r'embed_dim:\s*(\d+)',
        }
        
        self.supported_activations = {
            'relu', 'sigmoid', 'tanh', 'softmax', 'linear',
            'leaky_relu', 'gelu', 'swish'
        }
        
        self.supported_optimizers = {
            'adam', 'sgd', 'adamw', 'rmsprop', 'adagrad'
        }
        
        self.supported_losses = {
            'mse', 'categorical_crossentropy', 'binary_crossentropy',
            'sparse_categorical_crossentropy', 'huber', 'mae'
        }
    
    def parse_code(self, code: str) -> List[NetworkConfig]:
        """Парсинг AnamorphX кода для извлечения network блоков"""
        networks = []
        
        # Поиск network блоков
        network_pattern = r'network\s+(\w+)\s*\{([^}]+)\}'
        network_matches = re.finditer(network_pattern, code, re.MULTILINE | re.DOTALL)
        
        for match in network_matches:
            network_name = match.group(1)
            network_body = match.group(2)
            
            # Парсинг нейронов
            neurons = self._parse_neurons(network_body)
            
            # Парсинг параметров сети
            network_params = self._parse_network_params(network_body)
            
            network_config = NetworkConfig(
                name=network_name,
                neurons=neurons,
                **network_params
            )
            
            networks.append(network_config)
        
        return networks
    
    def _parse_neurons(self, network_body: str) -> List[NeuronConfig]:
        """Парсинг neuron блоков"""
        neurons = []
        
        # Поиск neuron блоков
        neuron_pattern = r'neuron\s+(\w+)\s*\{([^}]+)\}'
        neuron_matches = re.finditer(neuron_pattern, network_body, re.MULTILINE | re.DOTALL)
        
        for match in neuron_matches:
            neuron_name = match.group(1)
            neuron_body = match.group(2)
            
            # Определение типа слоя
            layer_type = self._detect_layer_type(neuron_body)
            
            # Парсинг параметров нейрона
            neuron_params = self._parse_neuron_params(neuron_body)
            
            neuron_config = NeuronConfig(
                name=neuron_name,
                layer_type=layer_type,
                **neuron_params
            )
            
            neurons.append(neuron_config)
        
        return neurons
    
    def _detect_layer_type(self, neuron_body: str) -> str:
        """Определение типа слоя по содержимому"""
        # Проверка на Transformer
        if 'num_heads:' in neuron_body or 'embed_dim:' in neuron_body:
            return 'transformer'
        
        # Проверка на Attention
        if 'attention_type:' in neuron_body:
            return 'attention'
        
        # Проверка на Convolutional
        if 'filters:' in neuron_body or 'kernel_size:' in neuron_body:
            return 'conv'
        
        # Проверка на Pooling
        if 'pool_size:' in neuron_body:
            return 'pool'
        
        # Проверка на LSTM
        if 'return_sequences:' in neuron_body or 'stateful:' in neuron_body:
            return 'lstm'
        
        # По умолчанию Dense
        return 'dense'
    
    def _parse_neuron_params(self, neuron_body: str) -> Dict[str, Any]:
        """Парсинг параметров нейрона"""
        params = {}
        
        # Основные параметры
        param_patterns = {
            'activation': r'activation:\s*(\w+)',
            'units': r'units:\s*(\d+)',
            'filters': r'filters:\s*(\d+)',
            'kernel_size': r'kernel_size:\s*(\d+)',
            'padding': r'padding:\s*(\d+)',
            'stride': r'stride:\s*(\d+)',
            'pool_size': r'pool_size:\s*(\d+)',
            'dropout': r'dropout:\s*([\d.]+)',
            # Transformer параметры
            'num_heads': r'num_heads:\s*(\d+)',
            'embed_dim': r'embed_dim:\s*(\d+)',
            'ff_dim': r'ff_dim:\s*(\d+)',
            # ResNet параметры
            'skip_connection': r'skip_connection:\s*(true|false)',
            'residual_type': r'residual_type:\s*(\w+)',
        }
        
        for param_name, pattern in param_patterns.items():
            match = re.search(pattern, neuron_body)
            if match:
                value = match.group(1)
                
                # Конвертация типов
                if param_name in ['units', 'filters', 'kernel_size', 'padding', 'stride', 'pool_size', 'num_heads', 'embed_dim', 'ff_dim']:
                    params[param_name] = int(value)
                elif param_name == 'dropout':
                    params[param_name] = float(value)
                elif param_name == 'skip_connection':
                    params[param_name] = value.lower() == 'true'
                else:
                    params[param_name] = value
        
        return params
    
    def _parse_network_params(self, network_body: str) -> Dict[str, Any]:
        """Парсинг параметров сети"""
        params = {}
        
        param_patterns = {
            'optimizer': r'optimizer:\s*(\w+)',
            'learning_rate': r'learning_rate:\s*([\d.]+)',
            'loss': r'loss:\s*(\w+)',
            'batch_size': r'batch_size:\s*(\d+)',
            'epochs': r'epochs:\s*(\d+)',
        }
        
        for param_name, pattern in param_patterns.items():
            match = re.search(pattern, network_body)
            if match:
                value = match.group(1)
                
                # Конвертация типов
                if param_name == 'learning_rate':
                    params[param_name] = float(value)
                elif param_name in ['batch_size', 'epochs']:
                    params[param_name] = int(value)
                else:
                    params[param_name] = value
        
        return params
    
    def validate_network(self, network: NetworkConfig) -> List[str]:
        """Валидация конфигурации сети"""
        errors = []
        
        # Проверка наличия нейронов
        if not network.neurons:
            errors.append(f"Network '{network.name}' has no neurons")
        
        # Проверка параметров сети
        if network.optimizer and network.optimizer not in self.supported_optimizers:
            errors.append(f"Unsupported optimizer: {network.optimizer}")
        
        if network.loss and network.loss not in self.supported_losses:
            errors.append(f"Unsupported loss function: {network.loss}")
        
        # Проверка нейронов
        for neuron in network.neurons:
            neuron_errors = self._validate_neuron(neuron)
            errors.extend(neuron_errors)
        
        return errors
    
    def _validate_neuron(self, neuron: NeuronConfig) -> List[str]:
        """Валидация конфигурации нейрона"""
        errors = []
        
        # Проверка активации
        if neuron.activation and neuron.activation not in self.supported_activations:
            errors.append(f"Unsupported activation in {neuron.name}: {neuron.activation}")
        
        # Проверка параметров по типу слоя
        if neuron.layer_type == 'dense':
            if not neuron.units:
                errors.append(f"Dense layer {neuron.name} missing 'units' parameter")
        
        elif neuron.layer_type == 'conv':
            if not neuron.filters:
                errors.append(f"Conv layer {neuron.name} missing 'filters' parameter")
        
        elif neuron.layer_type == 'pool':
            if not neuron.pool_size:
                errors.append(f"Pool layer {neuron.name} missing 'pool_size' parameter")
        
        elif neuron.layer_type == 'transformer':
            if not neuron.num_heads:
                errors.append(f"Transformer layer {neuron.name} missing 'num_heads' parameter")
            if not neuron.embed_dim:
                errors.append(f"Transformer layer {neuron.name} missing 'embed_dim' parameter")
        
        return errors
    
    def get_parsing_summary(self, networks: List[NetworkConfig]) -> Dict[str, Any]:
        """Получение сводки парсинга"""
        total_neurons = sum(len(net.neurons) for net in networks)
        layer_types = set()
        activations = set()
        
        for network in networks:
            for neuron in network.neurons:
                layer_types.add(neuron.layer_type)
                if neuron.activation:
                    activations.add(neuron.activation)
        
        return {
            'total_networks': len(networks),
            'total_neurons': total_neurons,
            'layer_types': list(layer_types),
            'activations': list(activations),
            'networks': [net.name for net in networks]
        }
