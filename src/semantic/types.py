"""
Type System for AnamorphX

This module provides comprehensive type checking and type inference
for the Anamorph neural programming language.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from abc import ABC, abstractmethod
from ..syntax.nodes import ASTNode, SourceLocation
from .errors import TypeError, SemanticErrorType


class TypeKind(Enum):
    """Kinds of types in the type system."""
    
    PRIMITIVE = auto()
    ARRAY = auto()
    OBJECT = auto()
    FUNCTION = auto()
    NEURAL = auto()
    SIGNAL = auto()
    PULSE = auto()
    UNION = auto()
    GENERIC = auto()
    VOID = auto()
    ANY = auto()
    UNKNOWN = auto()


@dataclass
class Type(ABC):
    """Base class for all types."""
    
    kind: TypeKind
    name: str
    nullable: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, kind: TypeKind, name: str, nullable: bool = False, **kwargs):
        """Initialize type with flexible kwargs support."""
        self.kind = kind
        self.name = name
        self.nullable = nullable
        self.attributes = kwargs
    
    @abstractmethod
    def is_compatible_with(self, other: 'Type') -> bool:
        """Check if this type is compatible with another type."""
        pass
    
    @abstractmethod
    def can_cast_to(self, other: 'Type') -> bool:
        """Check if this type can be cast to another type."""
        pass
    
    def __str__(self) -> str:
        suffix = "?" if self.nullable else ""
        return f"{self.name}{suffix}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        return (self.kind == other.kind and 
                self.name == other.name and 
                self.nullable == other.nullable)


@dataclass
class PrimitiveType(Type):
    """Primitive types like int, float, string, bool."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(kind=TypeKind.PRIMITIVE, name=name, **kwargs)
        self.size = kwargs.get('size', 0)
        self.min_value = kwargs.get('min_value')
        self.max_value = kwargs.get('max_value')
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, PrimitiveType):
            return self._is_primitive_compatible(other)
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        if isinstance(other, PrimitiveType):
            return self._can_cast_primitive(other)
        return False
    
    def _is_primitive_compatible(self, other: 'PrimitiveType') -> bool:
        """Check compatibility between primitive types."""
        # Exact match
        if self.name == other.name:
            return True
        
        # Numeric compatibility
        numeric_types = {'int', 'float', 'double', 'byte'}
        if self.name in numeric_types and other.name in numeric_types:
            return True
        
        return False
    
    def _can_cast_primitive(self, other: 'PrimitiveType') -> bool:
        """Check if primitive can be cast to another primitive."""
        cast_rules = {
            'int': {'float', 'double', 'string'},
            'float': {'int', 'double', 'string'},
            'double': {'int', 'float', 'string'},
            'string': {'int', 'float', 'double', 'bool'},
            'bool': {'string', 'int'},
            'byte': {'int', 'float', 'double'}
        }
        
        return other.name in cast_rules.get(self.name, set())


@dataclass
class ArrayType(Type):
    """Array type with element type."""
    
    def __init__(self, element_type: Type, size: Optional[int] = None, **kwargs):
        super().__init__(kind=TypeKind.ARRAY, name=f"{element_type.name}[]", **kwargs)
        self.element_type = element_type
        self.size = size
        self.is_dynamic = size is None
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, ArrayType):
            return self.element_type.is_compatible_with(other.element_type)
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        if isinstance(other, ArrayType):
            return self.element_type.can_cast_to(other.element_type)
        return False


@dataclass
class ObjectType(Type):
    """Object type with properties."""
    
    def __init__(self, name: str, properties: Dict[str, Type] = None, **kwargs):
        super().__init__(kind=TypeKind.OBJECT, name=name, **kwargs)
        self.properties = properties or {}
        self.methods: Dict[str, 'FunctionType'] = {}
    
    def add_property(self, name: str, prop_type: Type):
        """Add a property to the object type."""
        self.properties[name] = prop_type
    
    def add_method(self, name: str, method_type: 'FunctionType'):
        """Add a method to the object type."""
        self.methods[name] = method_type
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, ObjectType):
            # Structural compatibility
            for prop_name, prop_type in other.properties.items():
                if prop_name not in self.properties:
                    return False
                if not self.properties[prop_name].is_compatible_with(prop_type):
                    return False
            return True
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        return self.is_compatible_with(other)


@dataclass
class FunctionType(Type):
    """Function type with parameters and return type."""
    
    def __init__(self, parameter_types: List[Type], return_type: Type, **kwargs):
        param_str = ', '.join(str(p) for p in parameter_types)
        name = f"({param_str}) -> {return_type}"
        super().__init__(kind=TypeKind.FUNCTION, name=name, **kwargs)
        self.parameter_types = parameter_types
        self.return_type = return_type
        self.is_async = kwargs.get('is_async', False)
        self.is_generator = kwargs.get('is_generator', False)
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, FunctionType):
            # Check parameter compatibility (contravariant)
            if len(self.parameter_types) != len(other.parameter_types):
                return False
            
            for self_param, other_param in zip(self.parameter_types, other.parameter_types):
                if not other_param.is_compatible_with(self_param):
                    return False
            
            # Check return type compatibility (covariant)
            return self.return_type.is_compatible_with(other.return_type)
        
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        return self.is_compatible_with(other)


@dataclass
class NeuralType(Type):
    """Neural-specific types for neurons and neural networks."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(kind=TypeKind.NEURAL, name=name, **kwargs)
        self.activation_function = kwargs.get('activation_function', 'sigmoid')
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, NeuralType):
            return (self.input_size == other.input_size and 
                   self.output_size == other.output_size)
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        return self.is_compatible_with(other)


@dataclass
class SignalType(Type):
    """Signal type for neural communication."""
    
    def __init__(self, data_type: Type, frequency: float = 1.0, **kwargs):
        # Use custom name if provided, otherwise generate default
        name = kwargs.pop('name', f"Signal<{data_type.name}>")
        super().__init__(kind=TypeKind.SIGNAL, name=name, **kwargs)
        self.data_type = data_type
        self.frequency = frequency
        self.amplitude = kwargs.get('amplitude', 1.0)
        self.phase = kwargs.get('phase', 0.0)
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, SignalType):
            return self.data_type.is_compatible_with(other.data_type)
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        if isinstance(other, SignalType):
            return self.data_type.can_cast_to(other.data_type)
        return False


@dataclass
class PulseType(Type):
    """Pulse type for discrete neural events."""
    
    def __init__(self, data_type: Type, **kwargs):
        # Use custom name if provided, otherwise generate default
        name = kwargs.pop('name', f"Pulse<{data_type.name}>")
        super().__init__(kind=TypeKind.PULSE, name=name, **kwargs)
        self.data_type = data_type
        self.duration = kwargs.get('duration', 1.0)
        self.intensity = kwargs.get('intensity', 1.0)
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, PulseType):
            return self.data_type.is_compatible_with(other.data_type)
        return False
    
    def can_cast_to(self, other: Type) -> bool:
        if isinstance(other, PulseType):
            return self.data_type.can_cast_to(other.data_type)
        return False


class TypeSystem:
    """Central type system managing all types."""
    
    def __init__(self):
        self.types: Dict[str, Type] = {}
        self.type_aliases: Dict[str, Type] = {}
        self._setup_builtin_types()
    
    def _setup_builtin_types(self):
        """Setup built-in types."""
        # Primitive types
        self.register_type(PrimitiveType('int', size=32, min_value=-2**31, max_value=2**31-1))
        self.register_type(PrimitiveType('float', size=32))
        self.register_type(PrimitiveType('double', size=64))
        self.register_type(PrimitiveType('string'))
        self.register_type(PrimitiveType('bool'))
        self.register_type(PrimitiveType('byte', size=8, min_value=0, max_value=255))
        self.register_type(PrimitiveType('void'))
        self.register_type(PrimitiveType('any'))
        
        # Neural types
        self.register_type(NeuralType('neuron'))
        self.register_type(NeuralType('perceptron', input_size=1, output_size=1))
        self.register_type(NeuralType('lstm', input_size=1, output_size=1))
        
        # Signal types
        int_type = self.get_type('int')
        float_type = self.get_type('float')
        self.register_type(SignalType(int_type, name='IntSignal'))
        self.register_type(SignalType(float_type, name='FloatSignal'))
        
        # Pulse types
        self.register_type(PulseType(int_type, name='IntPulse'))
        self.register_type(PulseType(float_type, name='FloatPulse'))
    
    def register_type(self, type_obj: Type):
        """Register a new type."""
        self.types[type_obj.name] = type_obj
    
    def get_type(self, name: str) -> Optional[Type]:
        """Get a type by name."""
        # Check direct types
        if name in self.types:
            return self.types[name]
        
        # Check aliases
        if name in self.type_aliases:
            return self.type_aliases[name]
        
        # Try to parse complex types
        return self._parse_type_string(name)
    
    def _parse_type_string(self, type_str: str) -> Optional[Type]:
        """Parse complex type strings like 'int[]' or 'Signal<float>'."""
        # Array type
        if type_str.endswith('[]'):
            element_type_name = type_str[:-2]
            element_type = self.get_type(element_type_name)
            if element_type:
                return ArrayType(element_type)
        
        # Generic types like Signal<T> or Pulse<T>
        if '<' in type_str and type_str.endswith('>'):
            base_name = type_str[:type_str.index('<')]
            inner_type_name = type_str[type_str.index('<')+1:-1]
            inner_type = self.get_type(inner_type_name)
            
            if inner_type:
                if base_name == 'Signal':
                    return SignalType(inner_type)
                elif base_name == 'Pulse':
                    return PulseType(inner_type)
                elif base_name == 'Array':
                    return ArrayType(inner_type)
        
        return None
    
    def create_alias(self, alias_name: str, target_type: Type):
        """Create a type alias."""
        self.type_aliases[alias_name] = target_type
    
    def are_compatible(self, type1: Type, type2: Type) -> bool:
        """Check if two types are compatible."""
        return type1.is_compatible_with(type2)
    
    def can_cast(self, from_type: Type, to_type: Type) -> bool:
        """Check if one type can be cast to another."""
        return from_type.can_cast_to(to_type)


class TypeChecker:
    """Performs type checking on AST nodes."""
    
    def __init__(self, type_system: TypeSystem):
        self.type_system = type_system
        self.type_cache: Dict[ASTNode, Type] = {}
        self.errors: List[TypeError] = []
    
    def check_node(self, node: ASTNode) -> Optional[Type]:
        """Check the type of an AST node."""
        if node in self.type_cache:
            return self.type_cache[node]
        
        node_type = self._infer_node_type(node)
        if node_type:
            self.type_cache[node] = node_type
        
        return node_type
    
    def _infer_node_type(self, node: ASTNode) -> Optional[Type]:
        """Infer the type of a node based on its kind."""
        from ..syntax.nodes import NodeType
        
        if node.node_type == NodeType.INTEGER_LITERAL:
            return self.type_system.get_type('int')
        
        elif node.node_type == NodeType.FLOAT_LITERAL:
            return self.type_system.get_type('float')
        
        elif node.node_type == NodeType.STRING_LITERAL:
            return self.type_system.get_type('string')
        
        elif node.node_type == NodeType.BOOLEAN_LITERAL:
            return self.type_system.get_type('bool')
        
        elif node.node_type == NodeType.ARRAY_LITERAL:
            return self._infer_array_type(node)
        
        elif node.node_type == NodeType.BINARY_EXPRESSION:
            return self._infer_binary_expression_type(node)
        
        elif node.node_type == NodeType.UNARY_EXPRESSION:
            return self._infer_unary_expression_type(node)
        
        elif node.node_type == NodeType.CALL_EXPRESSION:
            return self._infer_call_expression_type(node)
        
        elif node.node_type == NodeType.SIGNAL_EXPRESSION:
            return self._infer_signal_type(node)
        
        return None
    
    def _infer_array_type(self, node: ASTNode) -> Optional[Type]:
        """Infer array type from array literal."""
        if not hasattr(node, 'elements') or not node.elements:
            return ArrayType(self.type_system.get_type('any'))
        
        # Infer from first element
        first_element_type = self.check_node(node.elements[0])
        if not first_element_type:
            return None
        
        # Check all elements are compatible
        for element in node.elements[1:]:
            element_type = self.check_node(element)
            if not element_type or not element_type.is_compatible_with(first_element_type):
                self.errors.append(TypeError(
                    f"Array elements have incompatible types",
                    expected_type=str(first_element_type),
                    actual_type=str(element_type) if element_type else "unknown",
                    location=element.location
                ))
                return None
        
        return ArrayType(first_element_type, size=len(node.elements))
    
    def _infer_binary_expression_type(self, node: ASTNode) -> Optional[Type]:
        """Infer type of binary expression."""
        left_type = self.check_node(node.left)
        right_type = self.check_node(node.right)
        
        if not left_type or not right_type:
            return None
        
        # Arithmetic operations
        if node.operator in ['+', '-', '*', '/', '%']:
            if (isinstance(left_type, PrimitiveType) and 
                isinstance(right_type, PrimitiveType)):
                return self._get_arithmetic_result_type(left_type, right_type)
        
        # Comparison operations
        elif node.operator in ['==', '!=', '<', '>', '<=', '>=']:
            return self.type_system.get_type('bool')
        
        # Logical operations
        elif node.operator in ['&&', '||']:
            return self.type_system.get_type('bool')
        
        return None
    
    def _get_arithmetic_result_type(self, left: PrimitiveType, right: PrimitiveType) -> Type:
        """Get result type of arithmetic operation."""
        # Type promotion rules
        if left.name == 'double' or right.name == 'double':
            return self.type_system.get_type('double')
        elif left.name == 'float' or right.name == 'float':
            return self.type_system.get_type('float')
        else:
            return self.type_system.get_type('int')
    
    def _infer_unary_expression_type(self, node: ASTNode) -> Optional[Type]:
        """Infer type of unary expression."""
        operand_type = self.check_node(node.operand)
        if not operand_type:
            return None
        
        if node.operator in ['+', '-']:
            return operand_type
        elif node.operator == '!':
            return self.type_system.get_type('bool')
        
        return None
    
    def _infer_call_expression_type(self, node: ASTNode) -> Optional[Type]:
        """Infer type of function call."""
        # This would need symbol table integration
        return self.type_system.get_type('any')
    
    def _infer_signal_type(self, node: ASTNode) -> Optional[Type]:
        """Infer type of signal expression."""
        if hasattr(node, 'data_type'):
            data_type = self.type_system.get_type(node.data_type)
            if data_type:
                return SignalType(data_type)
        
        return self.type_system.get_type('FloatSignal')


class TypeInference:
    """Performs type inference for variables and expressions."""
    
    def __init__(self, type_system: TypeSystem, type_checker: TypeChecker):
        self.type_system = type_system
        self.type_checker = type_checker
        self.inference_cache: Dict[str, Type] = {}
    
    def infer_variable_type(self, var_name: str, initializer: ASTNode) -> Optional[Type]:
        """Infer variable type from initializer."""
        if var_name in self.inference_cache:
            return self.inference_cache[var_name]
        
        inferred_type = self.type_checker.check_node(initializer)
        if inferred_type:
            self.inference_cache[var_name] = inferred_type
        
        return inferred_type
    
    def infer_function_return_type(self, function_body: ASTNode) -> Optional[Type]:
        """Infer function return type from body."""
        # This would analyze return statements in the function body
        return self.type_system.get_type('void')


# Type compatibility and coercion utilities
class TypeCompatibility:
    """Utilities for type compatibility checking."""
    
    @staticmethod
    def is_numeric(type_obj: Type) -> bool:
        """Check if type is numeric."""
        return (isinstance(type_obj, PrimitiveType) and 
                type_obj.name in {'int', 'float', 'double', 'byte'})
    
    @staticmethod
    def is_neural(type_obj: Type) -> bool:
        """Check if type is neural-related."""
        return isinstance(type_obj, (NeuralType, SignalType, PulseType))
    
    @staticmethod
    def can_auto_convert(from_type: Type, to_type: Type) -> bool:
        """Check if automatic conversion is possible."""
        if from_type == to_type:
            return True
        
        # Numeric conversions
        if TypeCompatibility.is_numeric(from_type) and TypeCompatibility.is_numeric(to_type):
            return True
        
        # Neural signal conversions
        if isinstance(from_type, SignalType) and isinstance(to_type, PulseType):
            return from_type.data_type.is_compatible_with(to_type.data_type)
        
        return False


class TypeCoercion:
    """Handles type coercion and conversion."""
    
    def __init__(self, type_system: TypeSystem):
        self.type_system = type_system
    
    def coerce_types(self, type1: Type, type2: Type) -> Optional[Type]:
        """Find common type for two types."""
        if type1 == type2:
            return type1
        
        # Numeric coercion
        if TypeCompatibility.is_numeric(type1) and TypeCompatibility.is_numeric(type2):
            return self._coerce_numeric_types(type1, type2)
        
        return None
    
    def _coerce_numeric_types(self, type1: Type, type2: Type) -> Type:
        """Coerce numeric types to common type."""
        type_hierarchy = ['byte', 'int', 'float', 'double']
        
        idx1 = type_hierarchy.index(type1.name) if type1.name in type_hierarchy else -1
        idx2 = type_hierarchy.index(type2.name) if type2.name in type_hierarchy else -1
        
        if idx1 >= 0 and idx2 >= 0:
            common_type_name = type_hierarchy[max(idx1, idx2)]
            return self.type_system.get_type(common_type_name)
        
        return type1


class TypeValidator:
    """Validates type usage and constraints."""
    
    def __init__(self, type_system: TypeSystem):
        self.type_system = type_system
        self.errors: List[TypeError] = []
    
    def validate_assignment(self, target_type: Type, value_type: Type, 
                          location: SourceLocation = None) -> bool:
        """Validate assignment compatibility."""
        if not target_type.is_compatible_with(value_type):
            self.errors.append(TypeError(
                f"Cannot assign {value_type} to {target_type}",
                expected_type=str(target_type),
                actual_type=str(value_type),
                location=location
            ))
            return False
        return True
    
    def validate_function_call(self, function_type: FunctionType, 
                             argument_types: List[Type],
                             location: SourceLocation = None) -> bool:
        """Validate function call arguments."""
        if len(argument_types) != len(function_type.parameter_types):
            self.errors.append(TypeError(
                f"Function expects {len(function_type.parameter_types)} arguments, got {len(argument_types)}",
                location=location
            ))
            return False
        
        for i, (param_type, arg_type) in enumerate(zip(function_type.parameter_types, argument_types)):
            if not param_type.is_compatible_with(arg_type):
                self.errors.append(TypeError(
                    f"Argument {i+1}: expected {param_type}, got {arg_type}",
                    expected_type=str(param_type),
                    actual_type=str(arg_type),
                    location=location
                ))
                return False
        
        return True
    
    def validate_neural_connection(self, source_type: Type, target_type: Type,
                                 location: SourceLocation = None) -> bool:
        """Validate neural connection compatibility."""
        if not isinstance(source_type, NeuralType) or not isinstance(target_type, NeuralType):
            self.errors.append(TypeError(
                "Neural connections require neural types",
                location=location
            ))
            return False
        
        if source_type.output_size != target_type.input_size:
            self.errors.append(TypeError(
                f"Neural connection size mismatch: output {source_type.output_size} != input {target_type.input_size}",
                location=location
            ))
            return False
        
        return True 