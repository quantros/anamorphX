"""
Улучшенная система типов AnamorphX

Расширенная статическая типизация, runtime валидация, кастомные конвертеры
и проверка совместимости с поддержкой пользовательских типов.
"""

import time
import inspect
import warnings
from typing import Any, Dict, List, Optional, Union, Type, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Импорт базовой системы типов
from .type_system import (
    AnamorphType, ValidationLevel, TypeCompatibility, TypeInfo, 
    TypeAnnotation, NeuronAnnotation, TensorAnnotation, LayerAnnotation
)


# =============================================================================
# РАСШИРЕННЫЕ АННОТАЦИИ ТИПОВ
# =============================================================================

class EnhancedTensorAnnotation(TensorAnnotation):
    """Улучшенная аннотация для тензоров с строгой проверкой размерностей"""
    
    def __init__(self, shape: List[int] = None, dtype: str = "float32", 
                 allow_dynamic_dims: bool = False, min_dims: int = None, max_dims: int = None):
        super().__init__(shape, dtype)
        self.allow_dynamic_dims = allow_dynamic_dims
        self.min_dims = min_dims
        self.max_dims = max_dims
    
    def validate(self, value: Any) -> bool:
        """Строгая валидация тензора"""
        if not hasattr(value, 'shape'):
            return False
        
        actual_shape = getattr(value, 'shape', [])
        
        # Проверка количества измерений
        if self.min_dims and len(actual_shape) < self.min_dims:
            return False
        if self.max_dims and len(actual_shape) > self.max_dims:
            return False
        
        # Проверка точных размерностей
        if self.type_info.dimensions:
            expected_shape = self.type_info.dimensions
            if len(actual_shape) != len(expected_shape):
                return False
            
            # Проверка размеров по каждой оси
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if expected == -1:  # -1 означает динамический размер
                    if not self.allow_dynamic_dims:
                        return False
                    continue
                if expected != actual:
                    return False
        
        # Проверка типа данных
        if hasattr(value, 'dtype'):
            expected_dtype = self.type_info.constraints.get('dtype', 'float32')
            actual_dtype = str(getattr(value, 'dtype'))
            if expected_dtype not in actual_dtype:
                return False
        
        return True


class SynapseAnnotation(TypeAnnotation):
    """Аннотация для синапсов"""
    
    def __init__(self, weight_range: Tuple[float, float] = None, 
                 connection_type: str = None, plasticity: bool = None):
        type_info = TypeInfo(
            name="Synapse",
            base_type=AnamorphType.SYNAPSE,
            python_type=object,
            constraints={
                'weight_range': weight_range,
                'connection_type': connection_type,
                'plasticity': plasticity
            }
        )
        super().__init__(type_info)
    
    def validate(self, value: Any) -> bool:
        """Валидация синапса"""
        # Проверка наличия основных атрибутов
        required_attrs = ['source', 'target', 'weight']
        for attr in required_attrs:
            if not hasattr(value, attr):
                return False
        
        # Проверка диапазона веса
        weight_range = self.type_info.constraints.get('weight_range')
        if weight_range:
            weight = getattr(value, 'weight', 0)
            if not (weight_range[0] <= weight <= weight_range[1]):
                return False
        
        # Проверка типа соединения
        connection_type = self.type_info.constraints.get('connection_type')
        if connection_type:
            actual_type = getattr(value, 'connection_type', None)
            if actual_type != connection_type:
                return False
        
        return True


class NetworkAnnotation(TypeAnnotation):
    """Аннотация для нейронных сетей"""
    
    def __init__(self, min_layers: int = None, max_layers: int = None,
                 required_layer_types: List[str] = None):
        type_info = TypeInfo(
            name="Network",
            base_type=AnamorphType.NETWORK,
            python_type=object,
            constraints={
                'min_layers': min_layers,
                'max_layers': max_layers,
                'required_layer_types': required_layer_types or []
            }
        )
        super().__init__(type_info)
    
    def validate(self, value: Any) -> bool:
        """Валидация нейронной сети"""
        if not hasattr(value, 'layers') and not hasattr(value, 'nodes'):
            return False
        
        # Получение слоев
        layers = getattr(value, 'layers', getattr(value, 'nodes', {}))
        if isinstance(layers, dict):
            layer_count = len(layers)
        elif isinstance(layers, list):
            layer_count = len(layers)
        else:
            return False
        
        # Проверка количества слоев
        min_layers = self.type_info.constraints.get('min_layers')
        max_layers = self.type_info.constraints.get('max_layers')
        
        if min_layers and layer_count < min_layers:
            return False
        if max_layers and layer_count > max_layers:
            return False
        
        return True


# =============================================================================
# СИСТЕМА КОНВЕРТАЦИИ ТИПОВ
# =============================================================================

class TypeConverter:
    """Базовый класс для конвертеров типов"""
    
    def __init__(self, from_type: AnamorphType, to_type: AnamorphType):
        self.from_type = from_type
        self.to_type = to_type
    
    @abstractmethod
    def can_convert(self, value: Any) -> bool:
        """Проверка возможности конвертации"""
        pass
    
    @abstractmethod
    def convert(self, value: Any) -> Any:
        """Выполнение конвертации"""
        pass


class ScalarToVectorConverter(TypeConverter):
    """Конвертер скаляра в вектор"""
    
    def __init__(self):
        super().__init__(AnamorphType.SCALAR, AnamorphType.VECTOR)
    
    def can_convert(self, value: Any) -> bool:
        return isinstance(value, (int, float))
    
    def convert(self, value: Any) -> List[float]:
        return [float(value)]


class VectorToTensorConverter(TypeConverter):
    """Конвертер вектора в тензор"""
    
    def __init__(self):
        super().__init__(AnamorphType.VECTOR, AnamorphType.TENSOR)
    
    def can_convert(self, value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)
    
    def convert(self, value: Any) -> 'MockTensor':
        """Конвертация в mock тензор"""
        return MockTensor(data=value, shape=[len(value)])


class NeuronToLayerConverter(TypeConverter):
    """Конвертер нейрона в слой"""
    
    def __init__(self):
        super().__init__(AnamorphType.NEURON, AnamorphType.LAYER)
    
    def can_convert(self, value: Any) -> bool:
        return hasattr(value, 'activate')
    
    def convert(self, value: Any) -> 'MockLayer':
        """Конвертация в mock слой с одним нейроном"""
        return MockLayer(neurons=[value])


# =============================================================================
# MOCK КЛАССЫ ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

class MockTensor:
    """Mock класс тензора"""
    
    def __init__(self, data: List, shape: List[int], dtype: str = "float32"):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.size = self._calculate_size()
        self.nbytes = self.size * 4  # 4 bytes per float32
    
    def _calculate_size(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size
    
    def __repr__(self):
        return f"MockTensor(shape={self.shape}, dtype={self.dtype})"


class MockLayer:
    """Mock класс слоя"""
    
    def __init__(self, neurons: List = None, input_shape: List[int] = None, 
                 output_shape: List[int] = None):
        self.neurons = neurons or []
        self.input_shape = input_shape or [1]
        self.output_shape = output_shape or [1]
    
    def forward(self, x):
        """Прямой проход"""
        return x
    
    def __repr__(self):
        return f"MockLayer(input_shape={self.input_shape}, output_shape={self.output_shape})"


class MockSynapse:
    """Mock класс синапса"""
    
    def __init__(self, source: str, target: str, weight: float = 1.0, 
                 connection_type: str = "excitatory"):
        self.source = source
        self.target = target
        self.weight = weight
        self.connection_type = connection_type
    
    def __repr__(self):
        return f"MockSynapse({self.source}->{self.target}, weight={self.weight})"


# =============================================================================
# РАСШИРЕННЫЙ СТАТИЧЕСКИЙ АНАЛИЗАТОР
# =============================================================================

class EnhancedStaticTypeChecker:
    """Расширенный статический анализатор типов"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.type_registry: Dict[str, TypeInfo] = {}
        self.function_signatures: Dict[str, Dict[str, TypeInfo]] = {}
        self.converters: List[TypeConverter] = []
        self.custom_validators: Dict[str, Callable] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        self._register_default_converters()
    
    def _register_default_converters(self):
        """Регистрация стандартных конвертеров"""
        self.converters.extend([
            ScalarToVectorConverter(),
            VectorToTensorConverter(),
            NeuronToLayerConverter()
        ])
    
    def register_converter(self, converter: TypeConverter):
        """Регистрация кастомного конвертера"""
        self.converters.append(converter)
    
    def register_custom_validator(self, type_name: str, validator: Callable[[Any], bool]):
        """Регистрация кастомного валидатора"""
        self.custom_validators[type_name] = validator
    
    def register_function_signature(self, func_name: str, param_types: Dict[str, str], 
                                  return_type: str = None):
        """Регистрация сигнатуры функции с типами"""
        signature = {}
        
        # Преобразование строковых типов в TypeInfo
        for param_name, type_name in param_types.items():
            if type_name in self.type_registry:
                signature[param_name] = self.type_registry[type_name]
            else:
                self.warnings.append(f"Unknown type '{type_name}' for parameter '{param_name}'")
        
        if return_type and return_type in self.type_registry:
            signature['return'] = self.type_registry[return_type]
        
        self.function_signatures[func_name] = signature
    
    def register_type(self, name: str, type_info: TypeInfo):
        """Регистрация типа в реестре"""
        self.type_registry[name] = type_info
    
    def check_function_call(self, func_name: str, args: Dict[str, Any]) -> List[str]:
        """Расширенная проверка вызова функции"""
        errors = []
        
        if func_name not in self.function_signatures:
            if self.validation_level != ValidationLevel.DISABLED:
                errors.append(f"Function '{func_name}' not registered for type checking")
            return errors
        
        signature = self.function_signatures[func_name]
        
        # Проверка обязательных параметров
        for param_name, expected_type in signature.items():
            if param_name == 'return':
                continue
            
            if param_name not in args:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Missing required parameter '{param_name}'")
                continue
            
            # Проверка типа аргумента
            actual_value = args[param_name]
            type_check_result = self._check_value_type(actual_value, expected_type)
            
            if not type_check_result['valid']:
                if type_check_result['convertible']:
                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append(f"Type mismatch for '{param_name}': {type_check_result['message']} (conversion available)")
                    else:
                        self.warnings.append(f"Auto-converting '{param_name}': {type_check_result['message']}")
                else:
                    errors.append(f"Type error for '{param_name}': {type_check_result['message']}")
        
        # Проверка лишних параметров
        expected_params = set(signature.keys()) - {'return'}
        actual_params = set(args.keys())
        extra_params = actual_params - expected_params
        
        if extra_params and self.validation_level == ValidationLevel.STRICT:
            errors.append(f"Unexpected parameters: {', '.join(extra_params)}")
        
        return errors
    
    def _check_value_type(self, value: Any, expected_type: TypeInfo) -> Dict[str, Any]:
        """Проверка соответствия значения ожидаемому типу"""
        result = {
            'valid': False,
            'convertible': False,
            'message': '',
            'converter': None
        }
        
        # Проверка точного соответствия типа
        if self._value_matches_type(value, expected_type):
            result['valid'] = True
            result['message'] = 'Type match'
            return result
        
        # Проверка возможности конвертации
        for converter in self.converters:
            if (converter.to_type == expected_type.base_type and 
                converter.can_convert(value)):
                result['convertible'] = True
                result['converter'] = converter
                result['message'] = f'Can convert from {converter.from_type.value} to {converter.to_type.value}'
                return result
        
        # Проверка кастомного валидатора
        if expected_type.name in self.custom_validators:
            if self.custom_validators[expected_type.name](value):
                result['valid'] = True
                result['message'] = 'Custom validator passed'
                return result
        
        result['message'] = f'Expected {expected_type.name}, got {type(value).__name__}'
        return result
    
    def _value_matches_type(self, value: Any, expected_type: TypeInfo) -> bool:
        """Проверка точного соответствия значения типу"""
        # Проверка Python типа
        if isinstance(expected_type.python_type, tuple):
            if not isinstance(value, expected_type.python_type):
                return False
        else:
            if not isinstance(value, expected_type.python_type):
                return False
        
        # Специальные проверки для конкретных типов
        if expected_type.base_type == AnamorphType.NEURON:
            return hasattr(value, 'activate')
        elif expected_type.base_type == AnamorphType.TENSOR:
            return hasattr(value, 'shape')
        elif expected_type.base_type == AnamorphType.LAYER:
            return hasattr(value, 'forward')
        elif expected_type.base_type == AnamorphType.SYNAPSE:
            return hasattr(value, 'source') and hasattr(value, 'target')
        
        return True


# Глобальная улучшенная система типов
advanced_type_system = None

def get_advanced_type_system():
    """Получение глобальной системы типов"""
    global advanced_type_system
    if advanced_type_system is None:
        from .type_system import global_type_system
        advanced_type_system = global_type_system
    return advanced_type_system


# Экспортируемые функции для улучшений
def create_enhanced_tensor_annotation(shape: List[int] = None, dtype: str = "float32", 
                                    allow_dynamic_dims: bool = False) -> EnhancedTensorAnnotation:
    """Создание улучшенной аннотации тензора"""
    return EnhancedTensorAnnotation(shape, dtype, allow_dynamic_dims)

def create_synapse_annotation(weight_range: Tuple[float, float] = None, 
                             connection_type: str = None) -> SynapseAnnotation:
    """Создание аннотации синапса"""
    return SynapseAnnotation(weight_range, connection_type)

def create_network_annotation(min_layers: int = None, max_layers: int = None) -> NetworkAnnotation:
    """Создание аннотации сети"""
    return NetworkAnnotation(min_layers, max_layers)

def register_custom_converter(converter: TypeConverter):
    """Регистрация кастомного конвертера"""
    type_system = get_advanced_type_system()
    if hasattr(type_system, 'static_checker'):
        type_system.static_checker.register_converter(converter)

def register_custom_validator(type_name: str, validator: Callable[[Any], bool]):
    """Регистрация кастомного валидатора"""
    type_system = get_advanced_type_system()
    if hasattr(type_system, 'runtime_validator'):
        type_system.runtime_validator.register_custom_validator(type_name, validator)

print("✅ Улучшенная система типов загружена с расширенными возможностями!") 