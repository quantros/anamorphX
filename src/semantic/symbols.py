"""
Symbol Table and Symbol Management for AnamorphX

This module provides comprehensive symbol management for the Anamorph language,
including symbol tables, symbol types, and symbol resolution.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Iterator
from abc import ABC, abstractmethod
from ..syntax.nodes import ASTNode, SourceLocation, NodeType
from .errors import SymbolError, SemanticErrorType


class SymbolType(Enum):
    """Types of symbols in the symbol table."""
    
    VARIABLE = auto()
    FUNCTION = auto()
    PARAMETER = auto()
    NEURON = auto()
    SYNAPSE = auto()
    SIGNAL = auto()
    PULSE = auto()
    MODULE = auto()
    CLASS = auto()
    INTERFACE = auto()
    ENUM = auto()
    CONSTANT = auto()
    LABEL = auto()


class SymbolScope(Enum):
    """Scope levels for symbols."""
    
    GLOBAL = auto()
    MODULE = auto()
    FUNCTION = auto()
    BLOCK = auto()
    NEURON = auto()
    CLASS = auto()
    LOOP = auto()
    TRY = auto()


@dataclass
class Symbol:
    """Base symbol class representing any named entity."""
    
    name: str
    symbol_type: SymbolType
    location: Optional[SourceLocation] = None
    scope: Optional[SymbolScope] = None
    node: Optional[ASTNode] = None
    type_info: Optional[str] = None
    is_mutable: bool = True
    is_initialized: bool = False
    is_used: bool = False
    is_exported: bool = False
    is_imported: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    references: List[SourceLocation] = field(default_factory=list)
    
    def __init__(self, name: str, symbol_type: SymbolType, **kwargs):
        """Initialize symbol with flexible kwargs support."""
        self.name = name
        self.symbol_type = symbol_type
        self.location = kwargs.get('location')
        self.scope = kwargs.get('scope')
        self.node = kwargs.get('node')
        self.type_info = kwargs.get('type_info')
        self.is_mutable = kwargs.get('is_mutable', True)
        self.is_initialized = kwargs.get('is_initialized', False)
        self.is_used = kwargs.get('is_used', False)
        self.is_exported = kwargs.get('is_exported', False)
        self.is_imported = kwargs.get('is_imported', False)
        self.attributes = kwargs.get('attributes', {})
        self.references = kwargs.get('references', [])
        
        # Store any additional kwargs in attributes
        for key, value in kwargs.items():
            if key not in ['location', 'scope', 'node', 'type_info', 'is_mutable', 
                          'is_initialized', 'is_used', 'is_exported', 'is_imported', 
                          'attributes', 'references']:
                self.attributes[key] = value
    
    def __post_init__(self):
        """Initialize symbol after creation."""
        if self.location and not self.references:
            self.references.append(self.location)
    
    def add_reference(self, location: SourceLocation):
        """Add a reference to this symbol."""
        self.references.append(location)
        self.is_used = True
    
    def get_qualified_name(self, scope_prefix: str = "") -> str:
        """Get fully qualified name of the symbol."""
        if scope_prefix:
            return f"{scope_prefix}.{self.name}"
        return self.name
    
    def is_accessible_from(self, scope: SymbolScope) -> bool:
        """Check if symbol is accessible from given scope."""
        # Global symbols are accessible from anywhere
        if self.scope == SymbolScope.GLOBAL:
            return True
        
        # Module symbols are accessible within the module
        if self.scope == SymbolScope.MODULE:
            return scope in [SymbolScope.MODULE, SymbolScope.FUNCTION, 
                           SymbolScope.BLOCK, SymbolScope.NEURON]
        
        # Function parameters are only accessible within function
        if self.scope == SymbolScope.FUNCTION:
            return scope in [SymbolScope.FUNCTION, SymbolScope.BLOCK]
        
        # Block symbols are only accessible within block and nested blocks
        if self.scope == SymbolScope.BLOCK:
            return scope == SymbolScope.BLOCK
        
        return False


@dataclass
class VariableSymbol(Symbol):
    """Symbol representing a variable."""
    
    def __init__(self, name: str, type_info: str = None, **kwargs):
        super().__init__(
            name=name,
            symbol_type=SymbolType.VARIABLE,
            type_info=type_info,
            **kwargs
        )
        self.is_constant = kwargs.get('is_constant', False)
        self.default_value = kwargs.get('default_value')


@dataclass
class FunctionSymbol(Symbol):
    """Symbol representing a function."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(
            name=name,
            symbol_type=SymbolType.FUNCTION,
            **kwargs
        )
        self.parameters: List[Symbol] = kwargs.get('parameters', [])
        self.return_type: Optional[str] = kwargs.get('return_type')
        self.is_async: bool = kwargs.get('is_async', False)
        self.is_generator: bool = kwargs.get('is_generator', False)
        self.is_recursive: bool = False
        self.call_count: int = 0
    
    def add_parameter(self, param: Symbol):
        """Add a parameter to the function."""
        param.symbol_type = SymbolType.PARAMETER
        param.scope = SymbolScope.FUNCTION
        self.parameters.append(param)
    
    def get_signature(self) -> str:
        """Get function signature string."""
        param_types = [p.type_info or 'any' for p in self.parameters]
        params_str = ', '.join(param_types)
        return_str = self.return_type or 'void'
        return f"{self.name}({params_str}) -> {return_str}"


@dataclass
class NeuronSymbol(Symbol):
    """Symbol representing a neuron."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(
            name=name,
            symbol_type=SymbolType.NEURON,
            **kwargs
        )
        self.neuron_type: str = kwargs.get('neuron_type', 'basic')
        self.activation_function: str = kwargs.get('activation_function', 'sigmoid')
        self.threshold: float = kwargs.get('threshold', 0.5)
        self.synapses: List['SynapseSymbol'] = []
        self.input_signals: List[str] = []
        self.output_signals: List[str] = []
        self.state: Dict[str, Any] = {}
    
    def add_synapse(self, synapse: 'SynapseSymbol'):
        """Add a synapse connection to this neuron."""
        self.synapses.append(synapse)
    
    def get_connections(self) -> List[str]:
        """Get list of connected neuron names."""
        connections = []
        for synapse in self.synapses:
            if synapse.source_neuron == self.name:
                connections.append(synapse.target_neuron)
            elif synapse.target_neuron == self.name:
                connections.append(synapse.source_neuron)
        return connections


@dataclass
class SynapseSymbol(Symbol):
    """Symbol representing a synapse connection."""
    
    def __init__(self, name: str, source_neuron: str, target_neuron: str, **kwargs):
        super().__init__(
            name=name,
            symbol_type=SymbolType.SYNAPSE,
            **kwargs
        )
        self.source_neuron: str = source_neuron
        self.target_neuron: str = target_neuron
        self.weight: float = kwargs.get('weight', 1.0)
        self.delay: float = kwargs.get('delay', 0.0)
        self.plasticity: str = kwargs.get('plasticity', 'static')
        self.is_inhibitory: bool = kwargs.get('is_inhibitory', False)
    
    def get_connection_info(self) -> str:
        """Get connection information string."""
        direction = "inhibits" if self.is_inhibitory else "excites"
        return f"{self.source_neuron} {direction} {self.target_neuron} (weight: {self.weight})"


class SymbolTable:
    """Symbol table for managing symbols in a scope."""
    
    def __init__(self, name: str = "global", parent: Optional['SymbolTable'] = None):
        self.name = name
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self.children: List['SymbolTable'] = []
        self.scope_type: SymbolScope = SymbolScope.GLOBAL
        self.level = 0 if not parent else parent.level + 1
        
        if parent:
            parent.children.append(self)
    
    def define(self, symbol: Symbol) -> bool:
        """Define a new symbol in this table."""
        if symbol.name in self.symbols:
            # Symbol already exists
            existing = self.symbols[symbol.name]
            if existing.symbol_type == symbol.symbol_type:
                # Redefinition error
                return False
            elif self._can_overload(existing, symbol):
                # Function overloading allowed
                self._add_overload(existing, symbol)
                return True
            else:
                return False
        
        self.symbols[symbol.name] = symbol
        symbol.scope = self.scope_type
        return True
    
    def lookup(self, name: str, recursive: bool = True) -> Optional[Symbol]:
        """Look up a symbol by name."""
        if name in self.symbols:
            symbol = self.symbols[name]
            symbol.is_used = True
            return symbol
        
        if recursive and self.parent:
            return self.parent.lookup(name, recursive)
        
        return None
    
    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol only in this table."""
        return self.symbols.get(name)
    
    def _can_overload(self, existing: Symbol, new: Symbol) -> bool:
        """Check if symbols can be overloaded."""
        return (existing.symbol_type == SymbolType.FUNCTION and 
                new.symbol_type == SymbolType.FUNCTION)
    
    def _add_overload(self, existing: Symbol, new: Symbol):
        """Add function overload."""
        if not hasattr(existing, 'overloads'):
            existing.overloads = []
        existing.overloads.append(new)
    
    def get_all_symbols(self, include_children: bool = False) -> Dict[str, Symbol]:
        """Get all symbols in this table."""
        result = self.symbols.copy()
        
        if include_children:
            for child in self.children:
                child_symbols = child.get_all_symbols(include_children=True)
                for name, symbol in child_symbols.items():
                    qualified_name = f"{child.name}.{name}"
                    result[qualified_name] = symbol
        
        return result
    
    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[Symbol]:
        """Get all symbols of a specific type."""
        return [symbol for symbol in self.symbols.values() 
                if symbol.symbol_type == symbol_type]
    
    def get_unused_symbols(self) -> List[Symbol]:
        """Get all unused symbols."""
        return [symbol for symbol in self.symbols.values() 
                if not symbol.is_used and symbol.symbol_type != SymbolType.FUNCTION]
    
    def get_uninitialized_variables(self) -> List[Symbol]:
        """Get all uninitialized variables."""
        return [symbol for symbol in self.symbols.values() 
                if (symbol.symbol_type == SymbolType.VARIABLE and 
                    not symbol.is_initialized)]
    
    def create_child(self, name: str) -> 'SymbolTable':
        """Create a child symbol table."""
        return SymbolTable(name, self)
    
    def __contains__(self, name: str) -> bool:
        """Check if symbol exists in this table."""
        return name in self.symbols
    
    def __iter__(self) -> Iterator[Symbol]:
        """Iterate over symbols."""
        return iter(self.symbols.values())
    
    def __len__(self) -> int:
        """Get number of symbols."""
        return len(self.symbols)
    
    def __str__(self) -> str:
        """String representation of symbol table."""
        lines = [f"SymbolTable '{self.name}' (level {self.level}):"]
        for symbol in self.symbols.values():
            lines.append(f"  {symbol.symbol_type.name}: {symbol.name}")
        return '\n'.join(lines)


class SymbolResolver:
    """Resolves symbol references and builds symbol tables."""
    
    def __init__(self):
        self.global_table = SymbolTable("global")
        self.current_table = self.global_table
        self.table_stack: List[SymbolTable] = [self.global_table]
        self.unresolved_references: List[Tuple[str, SourceLocation]] = []
        self.neural_network: Dict[str, List[str]] = {}  # neuron connections
    
    def enter_scope(self, name: str, scope_type: SymbolScope = SymbolScope.BLOCK) -> SymbolTable:
        """Enter a new scope."""
        new_table = self.current_table.create_child(name)
        new_table.scope_type = scope_type
        self.current_table = new_table
        self.table_stack.append(new_table)
        return new_table
    
    def exit_scope(self) -> Optional[SymbolTable]:
        """Exit current scope."""
        if len(self.table_stack) <= 1:
            return None
        
        self.table_stack.pop()
        self.current_table = self.table_stack[-1]
        return self.current_table
    
    def define_symbol(self, symbol: Symbol) -> bool:
        """Define a symbol in current scope."""
        success = self.current_table.define(symbol)
        
        if not success:
            # Handle redefinition error
            existing = self.current_table.lookup_local(symbol.name)
            if existing:
                raise SymbolError(
                    f"Symbol '{symbol.name}' is already defined",
                    symbol_name=symbol.name,
                    location=symbol.location,
                    context={
                        'existing_location': existing.location,
                        'existing_type': existing.symbol_type.name,
                        'new_type': symbol.symbol_type.name
                    }
                )
        
        # Handle neural network connections
        if symbol.symbol_type == SymbolType.SYNAPSE:
            synapse = symbol
            if hasattr(synapse, 'source_neuron') and hasattr(synapse, 'target_neuron'):
                self._add_neural_connection(synapse.source_neuron, synapse.target_neuron)
        
        return success
    
    def resolve_symbol(self, name: str, location: SourceLocation = None) -> Optional[Symbol]:
        """Resolve a symbol reference."""
        symbol = self.current_table.lookup(name)
        
        if symbol:
            if location:
                symbol.add_reference(location)
            return symbol
        else:
            # Add to unresolved references
            if location:
                self.unresolved_references.append((name, location))
            return None
    
    def _add_neural_connection(self, source: str, target: str):
        """Add neural network connection."""
        if source not in self.neural_network:
            self.neural_network[source] = []
        if target not in self.neural_network[source]:
            self.neural_network[source].append(target)
    
    def validate_neural_network(self) -> List[str]:
        """Validate neural network for cycles and connectivity."""
        errors = []
        
        # Check for cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.neural_network.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for neuron in self.neural_network:
            if neuron not in visited:
                if has_cycle(neuron):
                    errors.append(f"Cycle detected in neural network involving '{neuron}'")
        
        return errors
    
    def get_symbol_statistics(self) -> Dict[str, Any]:
        """Get statistics about symbols."""
        all_symbols = self.global_table.get_all_symbols(include_children=True)
        
        stats = {
            'total_symbols': len(all_symbols),
            'by_type': {},
            'unused_symbols': 0,
            'uninitialized_variables': 0,
            'unresolved_references': len(self.unresolved_references),
            'neural_connections': len(self.neural_network)
        }
        
        for symbol in all_symbols.values():
            symbol_type = symbol.symbol_type.name
            stats['by_type'][symbol_type] = stats['by_type'].get(symbol_type, 0) + 1
            
            if not symbol.is_used:
                stats['unused_symbols'] += 1
            
            if (symbol.symbol_type == SymbolType.VARIABLE and 
                not symbol.is_initialized):
                stats['uninitialized_variables'] += 1
        
        return stats


class SymbolCollector:
    """Collects symbols from AST nodes."""
    
    def __init__(self, resolver: SymbolResolver):
        self.resolver = resolver
        self.errors: List[SymbolError] = []
    
    def collect_from_node(self, node: ASTNode) -> List[Symbol]:
        """Collect symbols from an AST node."""
        symbols = []
        
        if node.node_type == NodeType.VARIABLE_DECLARATION:
            symbol = self._create_variable_symbol(node)
            if symbol:
                symbols.append(symbol)
        
        elif node.node_type == NodeType.FUNCTION_DECLARATION:
            symbol = self._create_function_symbol(node)
            if symbol:
                symbols.append(symbol)
        
        elif node.node_type == NodeType.NEURON_DECLARATION:
            symbol = self._create_neuron_symbol(node)
            if symbol:
                symbols.append(symbol)
        
        elif node.node_type == NodeType.SYNAPSE_DECLARATION:
            symbol = self._create_synapse_symbol(node)
            if symbol:
                symbols.append(symbol)
        
        return symbols
    
    def _create_variable_symbol(self, node: ASTNode) -> Optional[VariableSymbol]:
        """Create variable symbol from declaration node."""
        try:
            return VariableSymbol(
                name=node.name,
                type_info=getattr(node, 'type_annotation', None),
                location=node.location,
                is_mutable=getattr(node, 'is_mutable', True),
                is_initialized=hasattr(node, 'initializer') and node.initializer is not None
            )
        except Exception as e:
            self.errors.append(SymbolError(f"Failed to create variable symbol: {e}"))
            return None
    
    def _create_function_symbol(self, node: ASTNode) -> Optional[FunctionSymbol]:
        """Create function symbol from declaration node."""
        try:
            symbol = FunctionSymbol(
                name=node.name,
                return_type=getattr(node, 'return_type', None),
                location=node.location,
                is_async=getattr(node, 'is_async', False)
            )
            
            # Add parameters
            if hasattr(node, 'parameters'):
                for param in node.parameters:
                    param_symbol = VariableSymbol(
                        name=param.name,
                        type_info=getattr(param, 'type_annotation', None),
                        location=param.location
                    )
                    symbol.add_parameter(param_symbol)
            
            return symbol
        except Exception as e:
            self.errors.append(SymbolError(f"Failed to create function symbol: {e}"))
            return None
    
    def _create_neuron_symbol(self, node: ASTNode) -> Optional[NeuronSymbol]:
        """Create neuron symbol from declaration node."""
        try:
            return NeuronSymbol(
                name=node.name,
                neuron_type=getattr(node, 'neuron_type', 'basic'),
                activation_function=getattr(node, 'activation_function', 'sigmoid'),
                threshold=getattr(node, 'threshold', 0.5),
                location=node.location
            )
        except Exception as e:
            self.errors.append(SymbolError(f"Failed to create neuron symbol: {e}"))
            return None
    
    def _create_synapse_symbol(self, node: ASTNode) -> Optional[SynapseSymbol]:
        """Create synapse symbol from declaration node."""
        try:
            return SynapseSymbol(
                name=getattr(node, 'name', f"{node.source}->{node.target}"),
                source_neuron=node.source,
                target_neuron=node.target,
                weight=getattr(node, 'weight', 1.0),
                delay=getattr(node, 'delay', 0.0),
                location=node.location
            )
        except Exception as e:
            self.errors.append(SymbolError(f"Failed to create synapse symbol: {e}"))
            return None 