"""
Type System for AnamorphX Interpreter

Система типов для интерпретатора Anamorph с поддержкой:
- Базовых типов (int, float, string, bool, array, object)
- Нейронных типов (neuron, synapse, signal, pulse)
- Проверки типов во время выполнения
- Автоматического приведения типов
- Валидации нейронных операций
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


class TypeKind(Enum):
    """Виды типов в системе типов."""
    PRIMITIVE = auto()      # Примитивные типы
    COLLECTION = auto()     # Коллекции (array, object)
    FUNCTION = auto()       # Функциональные типы
    NEURAL = auto()         # Нейронные типы
    SIGNAL = auto()         # Сигнальные типы
    PULSE = auto()          # Импульсные типы
    UNION = auto()          # Объединения типов
    GENERIC = auto()        # Обобщенные типы
    VOID = auto()           # Пустой тип
    ANY = auto()            # Любой тип
    UNKNOWN = auto()        # Неизвестный тип


@dataclass
class Type:
    """Базовый класс для всех типов."""
    name: str
    kind: TypeKind
    nullable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.name + ("?" if self.nullable else "")
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        return (self.name == other.name and 
                self.kind == other.kind and 
                self.nullable == other.nullable)
    
    def __hash__(self) -> int:
        return hash((self.name, self.kind, self.nullable))
    
    def is_compatible_with(self, other: 'Type') -> bool:
        """Проверка совместимости типов."""
        # Any совместим с любым типом
        if self.kind == TypeKind.ANY or other.kind == TypeKind.ANY:
            return True
        
        # Точное совпадение
        if self == other:
            return True
        
        # Nullable совместимость
        if self.nullable and not other.nullable:
            return False
        
        # Числовые типы могут быть совместимы
        if (self.kind == TypeKind.PRIMITIVE and other.kind == TypeKind.PRIMITIVE and
            self.name in ['int', 'float', 'double'] and other.name in ['int', 'float', 'double']):
            return True
        
        return False
    
    def can_cast_to(self, target: 'Type') -> bool:
        """Проверка возможности приведения типа."""
        # Все типы можно привести к Any
        if target.kind == TypeKind.ANY:
            return True
        
        # Совместимые типы можно приводить
        if self.is_compatible_with(target):
            return True
        
        # Специальные правила приведения
        if self.kind == TypeKind.PRIMITIVE and target.kind == TypeKind.PRIMITIVE:
            return self._can_cast_primitive(target)
        
        return False
    
    def _can_cast_primitive(self, target: 'Type') -> bool:
        """Правила приведения примитивных типов."""
        cast_rules = {
            'int': ['float', 'double', 'string', 'bool'],
            'float': ['double', 'string', 'int'],
            'double': ['string', 'float', 'int'],
            'string': ['int', 'float', 'double', 'bool'],
            'bool': ['int', 'string']
        }
        
        return target.name in cast_rules.get(self.name, [])


@dataclass
class PrimitiveType(Type):
    """Примитивный тип данных."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    size: Optional[int] = None  # Размер в битах
    
    def __post_init__(self):
        self.kind = TypeKind.PRIMITIVE


@dataclass
class CollectionType(Type):
    """Тип коллекции (array, object)."""
    element_type: Optional[Type] = None
    key_type: Optional[Type] = None
    size: Optional[int] = None
    
    def __post_init__(self):
        self.kind = TypeKind.COLLECTION


@dataclass
class FunctionType(Type):
    """Функциональный тип."""
    parameter_types: List[Type] = field(default_factory=list)
    return_type: Optional[Type] = None
    is_async: bool = False
    is_neural: bool = False
    
    def __post_init__(self):
        self.kind = TypeKind.FUNCTION


@dataclass
class NeuralType(Type):
    """Нейронный тип."""
    activation_function: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    threshold: Optional[float] = None
    learning_rate: Optional[float] = None
    
    def __post_init__(self):
        self.kind = TypeKind.NEURAL


@dataclass
class SignalType(Type):
    """Тип сигнала."""
    data_type: Type = None
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    frequency: Optional[float] = None
    routing_rules: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.kind = TypeKind.SIGNAL


@dataclass
class PulseType(Type):
    """Тип импульса."""
    data_type: Type = None
    duration: Optional[float] = None
    pattern: Optional[str] = None
    decay_rate: Optional[float] = None
    
    def __post_init__(self):
        self.kind = TypeKind.PULSE


@dataclass
class UnionType(Type):
    """Объединение типов."""
    member_types: List[Type] = field(default_factory=list)
    
    def __post_init__(self):
        self.kind = TypeKind.UNION
    
    def is_compatible_with(self, other: 'Type') -> bool:
        """Union совместим если любой из членов совместим."""
        return any(member.is_compatible_with(other) for member in self.member_types)


class TypeSystem:
    """Центральная система типов."""
    
    def __init__(self):
        self.types: Dict[str, Type] = {}
        self.type_aliases: Dict[str, Type] = {}
        self.conversion_rules: Dict[Tuple[str, str], Callable] = {}
        self._setup_builtin_types()
        self._setup_conversion_rules()
    
    def _setup_builtin_types(self):
        """Инициализация встроенных типов."""
        # Примитивные типы
        self.register_type(PrimitiveType('int', TypeKind.PRIMITIVE, 
                                       min_value=-2**31, max_value=2**31-1, size=32))
        self.register_type(PrimitiveType('float', TypeKind.PRIMITIVE, size=32))
        self.register_type(PrimitiveType('double', TypeKind.PRIMITIVE, size=64))
        self.register_type(PrimitiveType('string', TypeKind.PRIMITIVE))
        self.register_type(PrimitiveType('bool', TypeKind.PRIMITIVE))
        self.register_type(PrimitiveType('byte', TypeKind.PRIMITIVE, 
                                       min_value=0, max_value=255, size=8))
        
        # Специальные типы
        self.register_type(Type('void', TypeKind.VOID))
        self.register_type(Type('any', TypeKind.ANY))
        self.register_type(Type('unknown', TypeKind.UNKNOWN))
        
        # Коллекции
        self.register_type(CollectionType('array', TypeKind.COLLECTION))
        self.register_type(CollectionType('object', TypeKind.COLLECTION))
        
        # Нейронные типы
        self.register_type(NeuralType('neuron', TypeKind.NEURAL, 
                                    activation_function='sigmoid', threshold=0.5))
        self.register_type(NeuralType('perceptron', TypeKind.NEURAL,
                                    activation_function='step', threshold=0.0))
        self.register_type(NeuralType('synapse', TypeKind.NEURAL))
        
        # Сигнальные типы
        int_type = self.get_type('int')
        float_type = self.get_type('float')
        
        self.register_type(SignalType('signal', TypeKind.SIGNAL, data_type=float_type))
        self.register_type(SignalType('int_signal', TypeKind.SIGNAL, data_type=int_type))
        self.register_type(SignalType('float_signal', TypeKind.SIGNAL, data_type=float_type))
        
        # Импульсные типы
        self.register_type(PulseType('pulse', TypeKind.PULSE, data_type=float_type))
        self.register_type(PulseType('int_pulse', TypeKind.PULSE, data_type=int_type))
        self.register_type(PulseType('float_pulse', TypeKind.PULSE, data_type=float_type))
        
        print("✅ Встроенные типы инициализированы")
    
    def _setup_conversion_rules(self):
        """Настройка правил преобразования типов."""
        # Числовые преобразования
        self.conversion_rules[('int', 'float')] = float
        self.conversion_rules[('int', 'double')] = float
        self.conversion_rules[('float', 'int')] = int
        self.conversion_rules[('double', 'int')] = int
        self.conversion_rules[('float', 'double')] = float
        self.conversion_rules[('double', 'float')] = float
        
        # Строковые преобразования
        self.conversion_rules[('int', 'string')] = str
        self.conversion_rules[('float', 'string')] = str
        self.conversion_rules[('bool', 'string')] = str
        self.conversion_rules[('string', 'int')] = lambda x: int(x) if x.isdigit() else 0
        self.conversion_rules[('string', 'float')] = lambda x: float(x) if x.replace('.', '').isdigit() else 0.0
        
        # Логические преобразования
        self.conversion_rules[('int', 'bool')] = bool
        self.conversion_rules[('float', 'bool')] = bool
        self.conversion_rules[('string', 'bool')] = lambda x: len(x) > 0
        self.conversion_rules[('bool', 'int')] = int
        
        print("✅ Правила преобразования типов настроены")
    
    def register_type(self, type_obj: Type):
        """Регистрация нового типа."""
        self.types[type_obj.name] = type_obj
    
    def get_type(self, name: str) -> Optional[Type]:
        """Получение типа по имени."""
        # Прямой поиск
        if name in self.types:
            return self.types[name]
        
        # Поиск в алиасах
        if name in self.type_aliases:
            return self.type_aliases[name]
        
        # Парсинг сложных типов
        return self._parse_complex_type(name)
    
    def _parse_complex_type(self, type_str: str) -> Optional[Type]:
        """Парсинг сложных типов."""
        # Array types: int[], string[], etc.
        if type_str.endswith('[]'):
            element_type_name = type_str[:-2]
            element_type = self.get_type(element_type_name)
            if element_type:
                return CollectionType(f'{element_type_name}_array', TypeKind.COLLECTION,
                                    element_type=element_type)
        
        # Generic types: Signal<int>, Pulse<float>, etc.
        if '<' in type_str and type_str.endswith('>'):
            base_name = type_str[:type_str.index('<')]
            inner_type_name = type_str[type_str.index('<')+1:-1]
            inner_type = self.get_type(inner_type_name)
            
            if inner_type:
                if base_name.lower() == 'signal':
                    return SignalType(f'signal_{inner_type_name}', TypeKind.SIGNAL, 
                                    data_type=inner_type)
                elif base_name.lower() == 'pulse':
                    return PulseType(f'pulse_{inner_type_name}', TypeKind.PULSE,
                                   data_type=inner_type)
                elif base_name.lower() == 'array':
                    return CollectionType(f'array_{inner_type_name}', TypeKind.COLLECTION,
                                        element_type=inner_type)
        
        return None
    
    def create_alias(self, alias_name: str, target_type: Type):
        """Создание алиаса типа."""
        self.type_aliases[alias_name] = target_type
    
    def are_compatible(self, type1: Type, type2: Type) -> bool:
        """Проверка совместимости типов."""
        return type1.is_compatible_with(type2)
    
    def can_convert(self, from_type: Type, to_type: Type) -> bool:
        """Проверка возможности преобразования."""
        return from_type.can_cast_to(to_type)
    
    def convert_value(self, value: Any, from_type: Type, to_type: Type) -> Any:
        """Преобразование значения из одного типа в другой."""
        # Если типы совместимы, возвращаем как есть
        if from_type.is_compatible_with(to_type):
            return value
        
        # Поиск правила преобразования
        conversion_key = (from_type.name, to_type.name)
        if conversion_key in self.conversion_rules:
            converter = self.conversion_rules[conversion_key]
            try:
                return converter(value)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot convert {value} from {from_type} to {to_type}: {e}")
        
        # Попытка универсального преобразования
        if to_type.name == 'string':
            return str(value)
        elif to_type.name == 'bool':
            return bool(value)
        
        raise TypeError(f"No conversion rule from {from_type} to {to_type}")
    
    def infer_type(self, value: Any) -> Type:
        """Автоматическое определение типа по значению."""
        if isinstance(value, int):
            return self.get_type('int')
        elif isinstance(value, float):
            return self.get_type('float')
        elif isinstance(value, str):
            return self.get_type('string')
        elif isinstance(value, bool):
            return self.get_type('bool')
        elif isinstance(value, list):
            if value:
                element_type = self.infer_type(value[0])
                return CollectionType(f'{element_type.name}_array', TypeKind.COLLECTION,
                                    element_type=element_type, size=len(value))
            else:
                return CollectionType('array', TypeKind.COLLECTION)
        elif isinstance(value, dict):
            return CollectionType('object', TypeKind.COLLECTION)
        elif value is None:
            return Type('void', TypeKind.VOID, nullable=True)
        else:
            return self.get_type('any')
    
    def validate_neural_operation(self, operation: str, operand_types: List[Type]) -> bool:
        """Валидация нейронных операций."""
        neural_operations = {
            'pulse': [TypeKind.SIGNAL, TypeKind.PULSE],
            'activate': [TypeKind.NEURAL],
            'train': [TypeKind.NEURAL, TypeKind.COLLECTION],
            'connect': [TypeKind.NEURAL, TypeKind.NEURAL],
            'disconnect': [TypeKind.NEURAL, TypeKind.NEURAL],
            'resonate': [TypeKind.SIGNAL, TypeKind.PRIMITIVE],
            'filter': [TypeKind.SIGNAL, TypeKind.FUNCTION],
            'encode': [TypeKind.COLLECTION, TypeKind.NEURAL],
            'decode': [TypeKind.NEURAL, TypeKind.COLLECTION]
        }
        
        if operation not in neural_operations:
            return False
        
        expected_kinds = neural_operations[operation]
        if len(operand_types) != len(expected_kinds):
            return False
        
        for operand_type, expected_kind in zip(operand_types, expected_kinds):
            if operand_type.kind != expected_kind:
                return False
        
        return True
    
    def get_builtin_types(self) -> List[str]:
        """Получение списка встроенных типов."""
        return list(self.types.keys())
    
    def get_type_info(self, type_name: str) -> Dict[str, Any]:
        """Получение информации о типе."""
        type_obj = self.get_type(type_name)
        if not type_obj:
            return {}
        
        info = {
            'name': type_obj.name,
            'kind': type_obj.kind.name,
            'nullable': type_obj.nullable,
            'metadata': type_obj.metadata
        }
        
        # Дополнительная информация в зависимости от типа
        if isinstance(type_obj, PrimitiveType):
            info.update({
                'min_value': type_obj.min_value,
                'max_value': type_obj.max_value,
                'size': type_obj.size
            })
        elif isinstance(type_obj, NeuralType):
            info.update({
                'activation_function': type_obj.activation_function,
                'input_size': type_obj.input_size,
                'output_size': type_obj.output_size,
                'threshold': type_obj.threshold
            })
        elif isinstance(type_obj, SignalType):
            info.update({
                'data_type': type_obj.data_type.name if type_obj.data_type else None,
                'intensity_range': type_obj.intensity_range,
                'frequency': type_obj.frequency
            })
        
        return info


class TypeChecker:
    """Проверка типов во время выполнения."""
    
    def __init__(self, type_system: TypeSystem):
        self.type_system = type_system
        self.type_cache: Dict[str, Type] = {}
        self.errors: List[str] = []
    
    def check_assignment(self, var_name: str, value: Any, declared_type: Optional[Type] = None) -> bool:
        """Проверка присваивания значения переменной."""
        inferred_type = self.type_system.infer_type(value)
        
        if declared_type:
            if not declared_type.is_compatible_with(inferred_type):
                self.errors.append(
                    f"Type mismatch in assignment to '{var_name}': "
                    f"expected {declared_type}, got {inferred_type}"
                )
                return False
        
        self.type_cache[var_name] = inferred_type
        return True
    
    def check_function_call(self, func_name: str, args: List[Any], 
                          expected_types: List[Type]) -> bool:
        """Проверка вызова функции."""
        if len(args) != len(expected_types):
            self.errors.append(
                f"Argument count mismatch in call to '{func_name}': "
                f"expected {len(expected_types)}, got {len(args)}"
            )
            return False
        
        for i, (arg, expected_type) in enumerate(zip(args, expected_types)):
            arg_type = self.type_system.infer_type(arg)
            if not arg_type.is_compatible_with(expected_type):
                self.errors.append(
                    f"Argument {i+1} type mismatch in call to '{func_name}': "
                    f"expected {expected_type}, got {arg_type}"
                )
                return False
        
        return True
    
    def check_neural_operation(self, operation: str, operands: List[Any]) -> bool:
        """Проверка нейронной операции."""
        operand_types = [self.type_system.infer_type(op) for op in operands]
        
        if not self.type_system.validate_neural_operation(operation, operand_types):
            self.errors.append(
                f"Invalid neural operation '{operation}' with types: "
                f"{[str(t) for t in operand_types]}"
            )
            return False
        
        return True
    
    def get_errors(self) -> List[str]:
        """Получение списка ошибок."""
        return self.errors.copy()
    
    def clear_errors(self):
        """Очистка ошибок."""
        self.errors.clear()
    
    def get_variable_type(self, var_name: str) -> Optional[Type]:
        """Получение типа переменной."""
        return self.type_cache.get(var_name)


# Экспорт основных классов
__all__ = [
    'TypeSystem',
    'TypeChecker',
    'Type',
    'PrimitiveType',
    'CollectionType',
    'FunctionType',
    'NeuralType',
    'SignalType',
    'PulseType',
    'UnionType',
    'TypeKind'
]


if __name__ == "__main__":
    # Демонстрация системы типов
    print("🎯 ДЕМОНСТРАЦИЯ СИСТЕМЫ ТИПОВ")
    print("=" * 50)
    
    # Создаем систему типов
    type_system = TypeSystem()
    type_checker = TypeChecker(type_system)
    
    print("📋 Встроенные типы:")
    for type_name in type_system.get_builtin_types():
        print(f"  {type_name}")
    
    print("\n🔍 Проверка совместимости типов:")
    int_type = type_system.get_type('int')
    float_type = type_system.get_type('float')
    string_type = type_system.get_type('string')
    neuron_type = type_system.get_type('neuron')
    
    tests = [
        (int_type, float_type, "int -> float"),
        (float_type, string_type, "float -> string"),
        (neuron_type, int_type, "neuron -> int"),
        (int_type, neuron_type, "int -> neuron")
    ]
    
    for type1, type2, desc in tests:
        compatible = type_system.are_compatible(type1, type2)
        convertible = type_system.can_convert(type1, type2)
        print(f"  {desc}: compatible={compatible}, convertible={convertible}")
    
    print("\n🔄 Автоматическое определение типов:")
    values = [42, 3.14, "hello", True, [1, 2, 3], {"key": "value"}]
    for value in values:
        inferred_type = type_system.infer_type(value)
        print(f"  {value} -> {inferred_type}")
    
    print("\n🧠 Проверка нейронных операций:")
    signal_type = type_system.get_type('signal')
    neural_operations = [
        ('pulse', [signal_type]),
        ('activate', [neuron_type]),
        ('train', [neuron_type, type_system.get_type('array')]),
        ('invalid_op', [int_type])
    ]
    
    for op, operand_types in neural_operations:
        valid = type_system.validate_neural_operation(op, operand_types)
        print(f"  {op}({[str(t) for t in operand_types]}): {valid}")
    
    print("\n✅ Демонстрация завершена") 