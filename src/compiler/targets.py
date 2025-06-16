"""
Target platform definitions and registry for the AnamorphX compiler.

This module provides abstractions for different compilation targets,
including Python, JavaScript, C++, and LLVM IR.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
import re
import functools

# Custom exceptions for better error handling
class TargetError(Exception):
    """Base exception for target-related errors."""
    pass

class TargetNotFoundError(TargetError):
    """Raised when a target platform is not found in registry."""
    pass

class InvalidIdentifierError(TargetError):
    """Raised when an identifier is invalid for the target platform."""
    pass

class TargetFeature(Enum):
    """Features that target platforms may support."""
    NEURAL_CONSTRUCTS = "neural_constructs"
    ASYNC_AWAIT = "async_await"
    STATIC_TYPING = "static_typing"
    GARBAGE_COLLECTION = "garbage_collection"
    MANUAL_MEMORY = "manual_memory"
    EXCEPTIONS = "exceptions"
    GENERICS = "generics"
    LAMBDAS = "lambdas"
    CLOSURES = "closures"
    REFLECTION = "reflection"
    METAPROGRAMMING = "metaprogramming"
    NEURAL_RUNTIME = "neural_runtime"
    PARALLEL_EXECUTION = "parallel_execution"
    CROSS_COMPILATION = "cross_compilation"

@dataclass
class TargetCapabilities:
    """Capabilities supported by a target platform."""
    features: Set[TargetFeature] = field(default_factory=set)
    max_identifier_length: int = 255
    supports_unicode_identifiers: bool = True
    neural_libraries: List[str] = field(default_factory=list)
    
    def supports_feature(self, feature: TargetFeature) -> bool:
        """Check if target supports a specific feature."""
        return feature in self.features
    
    def supports_neural_library(self, library: str) -> bool:
        """Check if target supports a specific neural library."""
        return library.lower() in [lib.lower() for lib in self.neural_libraries]

@dataclass
class TargetConfiguration:
    """Configuration options for code generation."""
    indent_size: int = 4
    use_tabs: bool = False
    line_ending: str = "\n"
    max_line_length: int = 100
    include_comments: bool = True
    include_type_hints: bool = True
    optimize_imports: bool = True
    format_code: bool = True
    neural_backend: Optional[str] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)

class TargetPlatform(ABC):
    """Abstract base class for compilation targets."""
    
    def __init__(self, name: str, capabilities: TargetCapabilities, 
                 config: Optional[TargetConfiguration] = None):
        self.name = name
        self.capabilities = capabilities
        self.config = config or TargetConfiguration()
        self._identifier_cache: Dict[str, bool] = {}  # Cache for identifier validation
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this target."""
        pass
    
    @abstractmethod
    def get_type_name(self, anamorph_type: str) -> str:
        """Convert AnamorphX type to target type."""
        pass
    
    @abstractmethod
    def format_identifier(self, name: str) -> str:
        """Format identifier according to target conventions."""
        pass
    
    @functools.lru_cache(maxsize=1000)  # Cache frequently validated identifiers
    def is_valid_identifier(self, name: str) -> bool:
        """Check if identifier is valid for this target with caching."""
        if name in self._identifier_cache:
            return self._identifier_cache[name]
        
        result = self._validate_identifier(name)
        self._identifier_cache[name] = result
        return result
    
    @abstractmethod
    def _validate_identifier(self, name: str) -> bool:
        """Internal method to validate identifier (to be cached)."""
        pass
    
    def validate_identifier(self, name: str) -> None:
        """Validate identifier and raise exception if invalid."""
        if not self.is_valid_identifier(name):
            raise InvalidIdentifierError(
                f"Invalid identifier '{name}' for target '{self.name}'"
            )
    
    def supports_feature(self, feature: TargetFeature) -> bool:
        """Check if target supports a feature."""
        return self.capabilities.supports_feature(feature)
    
    def format_code(self, code: str) -> str:
        """Format code according to target conventions."""
        if not self.config.format_code:
            return code
        return self._format_code_internal(code)
    
    def _format_code_internal(self, code: str) -> str:
        """Internal code formatting - can be overridden by subclasses."""
        # Basic formatting - subclasses should override for language-specific formatting
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                formatted_lines.append(stripped)
            else:
                formatted_lines.append('')
        
        return '\n'.join(formatted_lines)

class PythonTarget(TargetPlatform):
    """Python compilation target."""
    
    def __init__(self, config: Optional[TargetConfiguration] = None):
        capabilities = TargetCapabilities(
            features={
                TargetFeature.NEURAL_CONSTRUCTS,
                TargetFeature.ASYNC_AWAIT,
                TargetFeature.STATIC_TYPING,
                TargetFeature.GARBAGE_COLLECTION,
                TargetFeature.EXCEPTIONS,
                TargetFeature.GENERICS,
                TargetFeature.LAMBDAS,
                TargetFeature.CLOSURES,
                TargetFeature.REFLECTION,
                TargetFeature.METAPROGRAMMING,
                TargetFeature.NEURAL_RUNTIME,
                TargetFeature.PARALLEL_EXECUTION
            },
            neural_libraries=["tensorflow", "pytorch", "numpy", "scipy", "scikit-learn", "keras"]
        )
        super().__init__("python", capabilities, config)
        self.reserved_words = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'with', 'yield', 'async', 'await'
        }
    
    def get_file_extension(self) -> str:
        return ".py"
    
    def get_type_name(self, anamorph_type: str) -> str:
        type_mapping = {
            'int': 'int',
            'float': 'float',
            'string': 'str',
            'bool': 'bool',
            'list': 'List',
            'dict': 'Dict',
            'neuron': 'Neuron',
            'synapse': 'Synapse',
            'signal': 'Signal'
        }
        return type_mapping.get(anamorph_type, anamorph_type)
    
    def format_identifier(self, name: str) -> str:
        # Convert to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _validate_identifier(self, name: str) -> bool:
        if not name or name in self.reserved_words:
            return False
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        if len(name) > self.capabilities.max_identifier_length:
            return False
        return True
    
    def _format_code_internal(self, code: str) -> str:
        """Python-specific code formatting."""
        # This could integrate with black formatter if available
        try:
            import black
            return black.format_str(code, mode=black.FileMode())
        except ImportError:
            # Fallback to basic formatting
            return super()._format_code_internal(code)

class JavaScriptTarget(TargetPlatform):
    """JavaScript compilation target."""
    
    def __init__(self, config: Optional[TargetConfiguration] = None):
        capabilities = TargetCapabilities(
            features={
                TargetFeature.NEURAL_CONSTRUCTS,
                TargetFeature.ASYNC_AWAIT,
                TargetFeature.GARBAGE_COLLECTION,
                TargetFeature.EXCEPTIONS,
                TargetFeature.LAMBDAS,
                TargetFeature.CLOSURES,
                TargetFeature.NEURAL_RUNTIME,
                TargetFeature.PARALLEL_EXECUTION
            },
            neural_libraries=["tensorflow.js", "brain.js", "ml-matrix", "synaptic"]
        )
        super().__init__("javascript", capabilities, config)
        self.reserved_words = {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'export', 'extends', 'finally',
            'for', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new',
            'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var',
            'void', 'while', 'with', 'yield', 'async', 'await'
        }
    
    def get_file_extension(self) -> str:
        return ".js"
    
    def get_type_name(self, anamorph_type: str) -> str:
        type_mapping = {
            'int': 'number',
            'float': 'number',
            'string': 'string',
            'bool': 'boolean',
            'list': 'Array',
            'dict': 'Object',
            'neuron': 'Neuron',
            'synapse': 'Synapse',
            'signal': 'Signal'
        }
        return type_mapping.get(anamorph_type, anamorph_type)
    
    def format_identifier(self, name: str) -> str:
        # Convert to camelCase
        components = name.split('_')
        return components[0].lower() + ''.join(word.capitalize() for word in components[1:])
    
    def _validate_identifier(self, name: str) -> bool:
        if not name or name in self.reserved_words:
            return False
        if not re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', name):
            return False
        if len(name) > self.capabilities.max_identifier_length:
            return False
        return True

class CppTarget(TargetPlatform):
    """C++ compilation target."""
    
    def __init__(self, config: Optional[TargetConfiguration] = None):
        capabilities = TargetCapabilities(
            features={
                TargetFeature.NEURAL_CONSTRUCTS,
                TargetFeature.STATIC_TYPING,
                TargetFeature.MANUAL_MEMORY,
                TargetFeature.EXCEPTIONS,
                TargetFeature.GENERICS,
                TargetFeature.LAMBDAS,
                TargetFeature.NEURAL_RUNTIME,
                TargetFeature.PARALLEL_EXECUTION,
                TargetFeature.CROSS_COMPILATION
            },
            neural_libraries=["eigen", "opencv", "dlib", "caffe", "pytorch-cpp"]
        )
        super().__init__("cpp", capabilities, config)
        self.reserved_words = {
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
            'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class',
            'compl', 'const', 'constexpr', 'const_cast', 'continue', 'decltype',
            'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
            'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend',
            'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new',
            'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
            'private', 'protected', 'public', 'register', 'reinterpret_cast',
            'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
            'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local',
            'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union',
            'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t', 'while',
            'xor', 'xor_eq'
        }
    
    def get_file_extension(self) -> str:
        return ".cpp"
    
    def get_type_name(self, anamorph_type: str) -> str:
        type_mapping = {
            'int': 'int',
            'float': 'double',
            'string': 'std::string',
            'bool': 'bool',
            'list': 'std::vector',
            'dict': 'std::map',
            'neuron': 'Neuron',
            'synapse': 'Synapse',
            'signal': 'Signal'
        }
        return type_mapping.get(anamorph_type, anamorph_type)
    
    def format_identifier(self, name: str) -> str:
        # Convert to snake_case (C++ convention)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _validate_identifier(self, name: str) -> bool:
        if not name or name in self.reserved_words:
            return False
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        if len(name) > self.capabilities.max_identifier_length:
            return False
        return True

class LLVMTarget(TargetPlatform):
    """LLVM IR compilation target."""
    
    def __init__(self, config: Optional[TargetConfiguration] = None):
        capabilities = TargetCapabilities(
            features={
                TargetFeature.NEURAL_CONSTRUCTS,
                TargetFeature.STATIC_TYPING,
                TargetFeature.MANUAL_MEMORY,
                TargetFeature.NEURAL_RUNTIME,
                TargetFeature.PARALLEL_EXECUTION,
                TargetFeature.CROSS_COMPILATION
            },
            neural_libraries=["llvm-neural", "mlir"]
        )
        super().__init__("llvm", capabilities, config)
    
    def get_file_extension(self) -> str:
        return ".ll"
    
    def get_type_name(self, anamorph_type: str) -> str:
        type_mapping = {
            'int': 'i32',
            'float': 'double',
            'string': 'i8*',
            'bool': 'i1',
            'list': '%Array*',
            'dict': '%Map*',
            'neuron': '%Neuron*',
            'synapse': '%Synapse*',
            'signal': '%Signal*'
        }
        return type_mapping.get(anamorph_type, f'%{anamorph_type}*')
    
    def format_identifier(self, name: str) -> str:
        # LLVM uses % prefix for local variables and @ for globals
        return f"%{name}"
    
    def _validate_identifier(self, name: str) -> bool:
        if not name:
            return False
        # LLVM identifiers can contain most characters
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        return True

class TargetRegistry:
    """Registry for managing compilation targets."""
    
    def __init__(self):
        self._targets: Dict[str, TargetPlatform] = {}
        self._register_default_targets()
    
    def _register_default_targets(self):
        """Register default compilation targets."""
        self.register_target(PythonTarget())
        self.register_target(JavaScriptTarget())
        self.register_target(CppTarget())
        self.register_target(LLVMTarget())
    
    def register_target(self, target: TargetPlatform):
        """Register a compilation target."""
        self._targets[target.name] = target
    
    def get_target(self, name: str) -> TargetPlatform:
        """Get a compilation target by name."""
        if name not in self._targets:
            raise TargetNotFoundError(f"Target '{name}' not found in registry")
        return self._targets[name]
    
    def list_targets(self) -> List[str]:
        """List all registered target names."""
        return list(self._targets.keys())
    
    def get_targets_with_feature(self, feature: TargetFeature) -> List[TargetPlatform]:
        """Get all targets that support a specific feature."""
        return [target for target in self._targets.values() 
                if target.supports_feature(feature)]
    
    def get_targets_with_neural_library(self, library: str) -> List[TargetPlatform]:
        """Get all targets that support a specific neural library."""
        return [target for target in self._targets.values()
                if target.capabilities.supports_neural_library(library)]

# Global target registry instance
target_registry = TargetRegistry() 