"""
Система типов AnamorphX - Приоритет 2

Статическая типизация, runtime валидация и проверка совместимости.
"""

import time
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from enum import Enum


class AnamorphType(Enum):
    """Базовые типы данных AnamorphX"""
    NEURON = "neuron"
    SYNAPSE = "synapse" 
    TENSOR = "tensor"
    SCALAR = "scalar"
    VECTOR = "vector"
    LAYER = "layer"
    NETWORK = "network"
    SIGNAL = "signal"


class ValidationLevel(Enum):
    """Уровни валидации типов"""
    STRICT = "strict"      # Строгая типизация
    MODERATE = "moderate"  # Умеренная типизация
    LOOSE = "loose"        # Слабая типизация
    DISABLED = "disabled"  # Отключена


class TypeCompatibility(Enum):
    """Совместимость типов"""
    COMPATIBLE = "compatible"
    CONVERTIBLE = "convertible"
    INCOMPATIBLE = "incompatible"


@dataclass
class TypeInfo:
    """Информация о типе"""
    name: str
    base_type: AnamorphType
    python_type: Type
    dimensions: Optional[List[int]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible_with(self, other: 'TypeInfo') -> TypeCompatibility:
        """Проверка совместимости типов"""
        if self.base_type == other.base_type:
            return TypeCompatibility.COMPATIBLE
        
        # Проверка конвертируемости
        convertible_pairs = [
            (AnamorphType.SCALAR, AnamorphType.VECTOR),
            (AnamorphType.VECTOR, AnamorphType.TENSOR),
            (AnamorphType.NEURON, AnamorphType.LAYER),
        ]
        
        for from_type, to_type in convertible_pairs:
            if (self.base_type == from_type and other.base_type == to_type) or \
               (self.base_type == to_type and other.base_type == from_type):
                return TypeCompatibility.CONVERTIBLE
        
        return TypeCompatibility.INCOMPATIBLE


class TypeAnnotation:
    """Базовая аннотация типа"""
    
    def __init__(self, type_info: TypeInfo):
        self.type_info = type_info
    
    def validate(self, value: Any) -> bool:
        """Валидация значения"""
        return True


class NeuronAnnotation(TypeAnnotation):
    """Аннотация для нейронов"""
    
    def __init__(self, activation: str = None, input_size: int = None):
        type_info = TypeInfo(
            name="Neuron",
            base_type=AnamorphType.NEURON,
            python_type=object,
            constraints={'activation': activation, 'input_size': input_size}
        )
        super().__init__(type_info)
    
    def validate(self, value: Any) -> bool:
        """Валидация нейрона"""
        if not hasattr(value, 'activate'):
            return False
        
        activation = self.type_info.constraints.get('activation')
        if activation and getattr(value, 'activation', None) != activation:
            return False
        
        return True


class TensorAnnotation(TypeAnnotation):
    """Аннотация для тензоров"""
    
    def __init__(self, shape: List[int] = None, dtype: str = "float32"):
        type_info = TypeInfo(
            name="Tensor",
            base_type=AnamorphType.TENSOR,
            python_type=object,
            dimensions=shape,
            constraints={'dtype': dtype}
        )
        super().__init__(type_info)
    
    def validate(self, value: Any) -> bool:
        """Валидация тензора"""
        if not hasattr(value, 'shape'):
            return False
        
        if self.type_info.dimensions:
            expected_shape = self.type_info.dimensions
            actual_shape = getattr(value, 'shape', [])
            if len(actual_shape) != len(expected_shape):
                return False
        
        return True


class LayerAnnotation(TypeAnnotation):
    """Аннотация для слоев"""
    
    def __init__(self, input_shape: List[int] = None, output_shape: List[int] = None):
        type_info = TypeInfo(
            name="Layer",
            base_type=AnamorphType.LAYER,
            python_type=object,
            constraints={'input_shape': input_shape, 'output_shape': output_shape}
        )
        super().__init__(type_info)
    
    def validate(self, value: Any) -> bool:
        """Валидация слоя"""
        return hasattr(value, 'forward')


class StaticTypeChecker:
    """Статический анализатор типов"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.type_registry: Dict[str, TypeInfo] = {}
        self.function_signatures: Dict[str, Dict[str, TypeInfo]] = {}
        self.errors: List[str] = []
    
    def register_type(self, name: str, type_info: TypeInfo):
        """Регистрация типа"""
        self.type_registry[name] = type_info
    
    def check_function_call(self, func_name: str, args: Dict[str, Any]) -> List[str]:
        """Проверка вызова функции"""
        errors = []
        
        if func_name not in self.function_signatures:
            if self.validation_level != ValidationLevel.DISABLED:
                errors.append(f"Function '{func_name}' not registered")
            return errors
        
        # Проверка типов параметров
        signature = self.function_signatures[func_name]
        for param_name, expected_type in signature.items():
            if param_name == 'return':
                continue
            
            if param_name not in args:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Missing parameter '{param_name}'")
                continue
        
        return errors
    
    def check_layer_compatibility(self, input_layer: Any, output_layer: Any) -> List[str]:
        """Проверка совместимости слоев"""
        errors = []
        
        if not hasattr(input_layer, 'output_shape'):
            errors.append("Input layer missing 'output_shape'")
            return errors
        
        if not hasattr(output_layer, 'input_shape'):
            errors.append("Output layer missing 'input_shape'")
            return errors
        
        input_shape = getattr(input_layer, 'output_shape')
        expected_shape = getattr(output_layer, 'input_shape')
        
        if input_shape != expected_shape:
            errors.append(f"Shape mismatch: {input_shape} -> {expected_shape}")
        
        return errors


class RuntimeValidator:
    """Runtime валидатор типов"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.type_annotations: Dict[str, TypeAnnotation] = {}
        self.validation_stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0
        }
    
    def register_annotation(self, name: str, annotation: TypeAnnotation):
        """Регистрация аннотации типа"""
        self.type_annotations[name] = annotation
    
    def validate_value(self, name: str, value: Any) -> bool:
        """Валидация значения"""
        self.validation_stats['total_checks'] += 1
        
        if self.validation_level == ValidationLevel.DISABLED:
            self.validation_stats['passed_checks'] += 1
            return True
        
        if name not in self.type_annotations:
            if self.validation_level == ValidationLevel.STRICT:
                self.validation_stats['failed_checks'] += 1
                return False
            else:
                self.validation_stats['passed_checks'] += 1
                return True
        
        annotation = self.type_annotations[name]
        is_valid = annotation.validate(value)
        
        if is_valid:
            self.validation_stats['passed_checks'] += 1
        else:
            self.validation_stats['failed_checks'] += 1
        
        return is_valid
    
    def validate_memory_usage(self, tensors: List[Any], max_memory_mb: float = 1000.0) -> bool:
        """Валидация использования памяти"""
        total_memory = 0.0
        
        for tensor in tensors:
            if hasattr(tensor, 'nbytes'):
                total_memory += tensor.nbytes / (1024 * 1024)
            elif hasattr(tensor, 'size'):
                total_memory += tensor.size * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return total_memory <= max_memory_mb
    
    def validate_network_dimensions(self, network_layers: List[Any]) -> List[str]:
        """Валидация размерностей сети"""
        errors = []
        
        for i in range(len(network_layers) - 1):
            current_layer = network_layers[i]
            next_layer = network_layers[i + 1]
            
            if hasattr(current_layer, 'output_shape') and hasattr(next_layer, 'input_shape'):
                output_shape = current_layer.output_shape
                input_shape = next_layer.input_shape
                
                if output_shape != input_shape:
                    errors.append(f"Dimension mismatch between layer {i} and {i+1}")
        
        return errors
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Отчет о валидации"""
        total = self.validation_stats['total_checks']
        success_rate = (self.validation_stats['passed_checks'] / total * 100) if total > 0 else 0
        
        return {
            'validation_level': self.validation_level.value,
            'total_checks': total,
            'success_rate': f"{success_rate:.2f}%",
            'statistics': self.validation_stats.copy()
        }


class AnamorphXTypeSystem:
    """Глобальная система типов AnamorphX"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.static_checker = StaticTypeChecker(validation_level)
        self.runtime_validator = RuntimeValidator(validation_level)
        self.validation_level = validation_level
        self.registered_types: Dict[str, TypeInfo] = {}
        
        self._initialize_builtin_types()
    
    def _initialize_builtin_types(self):
        """Инициализация встроенных типов"""
        builtin_types = [
            TypeInfo("Neuron", AnamorphType.NEURON, object),
            TypeInfo("Synapse", AnamorphType.SYNAPSE, object),
            TypeInfo("Tensor", AnamorphType.TENSOR, object),
            TypeInfo("Scalar", AnamorphType.SCALAR, (int, float)),
            TypeInfo("Vector", AnamorphType.VECTOR, list),
            TypeInfo("Layer", AnamorphType.LAYER, object),
            TypeInfo("Network", AnamorphType.NETWORK, dict),
        ]
        
        for type_info in builtin_types:
            self.register_type(type_info.name, type_info)
    
    def register_type(self, name: str, type_info: TypeInfo):
        """Регистрация типа"""
        self.registered_types[name] = type_info
        self.static_checker.register_type(name, type_info)
    
    def check_compatibility(self, type1: str, type2: str) -> TypeCompatibility:
        """Проверка совместимости типов"""
        if type1 not in self.registered_types or type2 not in self.registered_types:
            return TypeCompatibility.INCOMPATIBLE
        
        type_info1 = self.registered_types[type1]
        type_info2 = self.registered_types[type2]
        
        return type_info1.is_compatible_with(type_info2)
    
    def validate_function_call(self, func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация вызова функции"""
        errors = self.static_checker.check_function_call(func_name, args)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'validation_level': self.validation_level.value
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Статистика системы типов"""
        return {
            'registered_types': len(self.registered_types),
            'validation_level': self.validation_level.value,
            'builtin_types': list(self.registered_types.keys()),
            'runtime_validation': self.runtime_validator.get_validation_report()
        }


# Глобальная система типов
global_type_system = AnamorphXTypeSystem()

# Экспортируемые функции
def register_type(name: str, type_info: TypeInfo):
    """Регистрация типа"""
    global_type_system.register_type(name, type_info)

def check_compatibility(type1: str, type2: str) -> TypeCompatibility:
    """Проверка совместимости типов"""
    return global_type_system.check_compatibility(type1, type2)

def validate_function_call(func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Валидация вызова функции"""
    return global_type_system.validate_function_call(func_name, args)

def get_type_system_stats() -> Dict[str, Any]:
    """Статистика системы типов"""
    return global_type_system.get_system_stats()

print("🎯 Приоритет 2: Система типов - готова!")
