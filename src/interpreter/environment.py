"""
Environment management for the AnamorphX interpreter.

This module provides environment and scope management for variable storage,
neural constructs, and signal processing during code execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Type
import threading
import time
from collections import defaultdict


class VariableType(Enum):
    """Types of variables in the environment."""
    PRIMITIVE = "primitive"
    OBJECT = "object"
    FUNCTION = "function"
    NEURON = "neuron"
    SYNAPSE = "synapse"
    SIGNAL = "signal"
    PULSE = "pulse"
    NEURAL_NETWORK = "neural_network"
    
    # Additional types for compatibility
    VARIABLE = "variable"
    BUILTIN = "builtin"
    PARAMETER = "parameter"
    IMPORT = "import"
    TYPE = "type"


class ScopeType(Enum):
    """Types of scopes in the environment."""
    GLOBAL = "global"
    FUNCTION = "function"
    CLASS = "class"
    BLOCK = "block"
    NEURAL = "neural"
    SIGNAL = "signal"


@dataclass
class Variable:
    """Represents a variable in the environment."""
    name: str
    value: Any
    var_type: VariableType
    is_constant: bool = False
    is_mutable: bool = True
    created_at: float = field(default_factory=time.time)
    accessed_count: int = 0
    modified_count: int = 0
    
    def access(self) -> Any:
        """Access the variable value."""
        self.accessed_count += 1
        return self.value
    
    def modify(self, new_value: Any) -> None:
        """Modify the variable value."""
        if self.is_constant:
            raise EnvironmentError(f"Cannot modify constant variable '{self.name}'")
        if not self.is_mutable:
            raise EnvironmentError(f"Cannot modify immutable variable '{self.name}'")
        
        self.value = new_value
        self.modified_count += 1
    
    def __str__(self) -> str:
        return f"Variable({self.name}: {self.var_type.value} = {self.value})"


class EnvironmentError(Exception):
    """Exception raised for environment-related errors."""
    pass


class Scope:
    """Represents a scope in the environment."""
    
    def __init__(self, name: str, scope_type: ScopeType, parent: Optional['Scope'] = None):
        self.name = name
        self.scope_type = scope_type
        self.parent = parent
        self.children: List['Scope'] = []
        self.variables: Dict[str, Variable] = {}
        self.created_at = time.time()
        
        # Add this scope as child of parent
        if parent:
            parent.children.append(self)
    
    def define_variable(self, name: str, value: Any, var_type: VariableType = VariableType.PRIMITIVE,
                       is_constant: bool = False, is_mutable: bool = True) -> Variable:
        """Define a new variable in this scope."""
        if name in self.variables:
            raise EnvironmentError(f"Variable '{name}' already defined in scope '{self.name}'")
        
        variable = Variable(
            name=name,
            value=value,
            var_type=var_type,
            is_constant=is_constant,
            is_mutable=is_mutable
        )
        self.variables[name] = variable
        return variable
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable from this scope or parent scopes."""
        if name in self.variables:
            return self.variables[name]
        
        # Search in parent scopes
        if self.parent:
            return self.parent.get_variable(name)
        
        return None
    
    def set_variable(self, name: str, value: Any) -> bool:
        """Set a variable value in this scope or parent scopes."""
        if name in self.variables:
            self.variables[name].modify(value)
            return True
        
        # Search in parent scopes
        if self.parent:
            return self.parent.set_variable(name, value)
        
        return False
    
    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in this scope or parent scopes."""
        return self.get_variable(name) is not None
    
    def list_variables(self, include_parent: bool = False) -> List[str]:
        """List all variable names in this scope."""
        names = list(self.variables.keys())
        
        if include_parent and self.parent:
            names.extend(self.parent.list_variables(include_parent=True))
        
        return names
    
    def clear(self) -> None:
        """Clear all variables in this scope."""
        self.variables.clear()
    
    def __str__(self) -> str:
        return f"Scope({self.name}: {self.scope_type.value}, {len(self.variables)} vars)"


class NeuralEnvironment:
    """Specialized environment for neural constructs."""
    
    def __init__(self):
        self.neurons: Dict[str, Any] = {}
        self.synapses: Dict[str, Any] = {}
        self.networks: Dict[str, Any] = {}
        self.connections: Dict[str, List[str]] = defaultdict(list)
        self.bindings: Dict[str, List[str]] = {}  # For grouped entities
        self._lock = threading.RLock()
    
    def register_neuron(self, name: str, neuron: Any) -> None:
        """Register a neuron in the neural environment."""
        with self._lock:
            if name in self.neurons:
                raise EnvironmentError(f"Neuron '{name}' already registered")
            self.neurons[name] = neuron
    
    def get_neuron(self, name: str) -> Any:
        """Get a neuron by name."""
        with self._lock:
            if name not in self.neurons:
                raise EnvironmentError(f"Neuron '{name}' not found")
            return self.neurons[name]
    
    def remove_neuron(self, name: str) -> bool:
        """Remove a neuron and its connections."""
        with self._lock:
            if name not in self.neurons:
                return False
            
            # Remove all connections involving this neuron
            connections_to_remove = []
            for conn_id, synapse in self.synapses.items():
                if synapse.get('source') == name or synapse.get('target') == name:
                    connections_to_remove.append(conn_id)
            
            for conn_id in connections_to_remove:
                del self.synapses[conn_id]
            
            # Remove from connections mapping
            if name in self.connections:
                del self.connections[name]
            
            # Remove neuron
            del self.neurons[name]
            return True
    
    def register_synapse(self, connection_id: str, synapse: Any) -> None:
        """Register a synapse connection."""
        with self._lock:
            if connection_id in self.synapses:
                raise EnvironmentError(f"Synapse '{connection_id}' already registered")
            self.synapses[connection_id] = synapse
            
            # Update connections mapping
            source = synapse.get('source')
            target = synapse.get('target')
            if source:
                self.connections[source].append(target)
    
    def get_synapse(self, connection_id: str) -> Any:
        """Get a synapse by connection ID."""
        with self._lock:
            if connection_id not in self.synapses:
                raise EnvironmentError(f"Synapse '{connection_id}' not found")
            return self.synapses[connection_id]
    
    def remove_synapse(self, connection_id: str) -> bool:
        """Remove a synapse connection."""
        with self._lock:
            if connection_id not in self.synapses:
                return False
            
            synapse = self.synapses[connection_id]
            source = synapse.get('source')
            target = synapse.get('target')
            
            # Update connections mapping
            if source and target in self.connections.get(source, []):
                self.connections[source].remove(target)
            
            del self.synapses[connection_id]
            return True
    
    def register_network(self, name: str, network: Any) -> None:
        """Register a neural network."""
        with self._lock:
            if name in self.networks:
                raise EnvironmentError(f"Network '{name}' already registered")
            self.networks[name] = network
    
    def get_network(self, name: str) -> Any:
        """Get a neural network by name."""
        with self._lock:
            if name not in self.networks:
                raise EnvironmentError(f"Network '{name}' not found")
            return self.networks[name]
    
    def create_binding(self, binding_name: str, entities: List[str]) -> None:
        """Create a binding between multiple neural entities."""
        with self._lock:
            # Validate that all entities exist
            for entity in entities:
                if entity not in self.neurons and entity not in self.networks:
                    raise EnvironmentError(f"Entity '{entity}' not found for binding")
            
            self.bindings[binding_name] = entities
    
    def remove_binding(self, binding_name: str) -> bool:
        """Remove a binding."""
        with self._lock:
            if binding_name in self.bindings:
                del self.bindings[binding_name]
                return True
            return False
    
    def get_connections(self, neuron_name: str) -> List[str]:
        """Get all connections for a neuron."""
        with self._lock:
            return self.connections.get(neuron_name, []).copy()
    
    def list_neurons(self) -> List[str]:
        """List all registered neurons."""
        with self._lock:
            return list(self.neurons.keys())
    
    def list_synapses(self) -> List[str]:
        """List all registered synapses."""
        with self._lock:
            return list(self.synapses.keys())
    
    def list_networks(self) -> List[str]:
        """List all registered networks."""
        with self._lock:
            return list(self.networks.keys())
    
    def clear(self) -> None:
        """Clear all neural constructs."""
        with self._lock:
            self.neurons.clear()
            self.synapses.clear()
            self.networks.clear()
            self.connections.clear()
            self.bindings.clear()


class SignalEnvironment:
    """Specialized environment for signal processing."""
    
    def __init__(self):
        self.signals: Dict[str, Any] = {}
        self.pulses: Dict[str, Any] = {}
        self.signal_queues: Dict[str, List[Any]] = defaultdict(list)
        self.signal_handlers: Dict[str, List[callable]] = defaultdict(list)
        self.active_pulses: Dict[str, Dict] = {}  # Track active pulse sessions
        self._lock = threading.RLock()
    
    def register_signal(self, name: str, signal: Any) -> None:
        """Register a signal."""
        with self._lock:
            self.signals[name] = signal
    
    def send_signal(self, signal_name: str, data: Any = None, target: str = "broadcast") -> None:
        """Send a signal."""
        with self._lock:
            signal_data = {
                'name': signal_name,
                'data': data,
                'target': target,
                'timestamp': time.time()
            }
            
            if target == "broadcast":
                # Send to all signal queues
                for queue_name in self.signal_queues:
                    self.signal_queues[queue_name].append(signal_data)
            else:
                # Send to specific target
                self.signal_queues[target].append(signal_data)
            
            # Call registered handlers
            for handler in self.signal_handlers.get(signal_name, []):
                try:
                    handler(signal_data)
                except Exception:
                    pass  # Don't let handler errors break signal sending
    
    def register_pulse(self, pulse_id: str, pulse_data: Dict) -> None:
        """Register an active pulse."""
        with self._lock:
            self.active_pulses[pulse_id] = pulse_data
    
    def remove_pulse(self, pulse_id: str) -> bool:
        """Remove an active pulse."""
        with self._lock:
            if pulse_id in self.active_pulses:
                del self.active_pulses[pulse_id]
                return True
            return False
    
    def get_pulse(self, pulse_id: str) -> Optional[Dict]:
        """Get pulse data."""
        with self._lock:
            return self.active_pulses.get(pulse_id)
    
    def register_handler(self, signal_name: str, handler: callable) -> None:
        """Register a signal handler."""
        with self._lock:
            self.signal_handlers[signal_name].append(handler)
    
    def unregister_handler(self, signal_name: str, handler: callable) -> bool:
        """Unregister a signal handler."""
        with self._lock:
            if signal_name in self.signal_handlers:
                try:
                    self.signal_handlers[signal_name].remove(handler)
                    return True
                except ValueError:
                    pass
            return False
    
    def get_signals(self, target: str) -> List[Any]:
        """Get all signals for a target."""
        with self._lock:
            signals = self.signal_queues.get(target, []).copy()
            self.signal_queues[target].clear()  # Clear after reading
            return signals
    
    def clear(self) -> None:
        """Clear all signal constructs."""
        with self._lock:
            self.signals.clear()
            self.pulses.clear()
            self.signal_queues.clear()
            self.signal_handlers.clear()
            self.active_pulses.clear()


class Environment:
    """Main environment for code execution."""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.global_scope = Scope("global", ScopeType.GLOBAL)
        self.current_scope = self.global_scope
        self.scope_stack: List[Scope] = [self.global_scope]
        
        # Specialized environments
        self.neural_env = NeuralEnvironment()
        self.signal_env = SignalEnvironment()
        
        # Environment metadata
        self.created_at = time.time()
        self.access_count = 0
        self._lock = threading.RLock()
        
        # Built-in variables
        self._setup_builtins()
    
    def _setup_builtins(self) -> None:
        """Setup built-in variables and constants."""
        builtins = {
            'True': True,
            'False': False,
            'None': None,
            '__version__': '1.0.0',
            '__interpreter__': 'AnamorphX'
        }
        
        for name, value in builtins.items():
            self.define_variable(name, value, VariableType.PRIMITIVE, is_constant=True)
    
    def push_scope(self, name: str, scope_type: ScopeType = ScopeType.BLOCK) -> Scope:
        """Push a new scope onto the scope stack."""
        with self._lock:
            new_scope = Scope(name, scope_type, self.current_scope)
            self.scope_stack.append(new_scope)
            self.current_scope = new_scope
            return new_scope
    
    def pop_scope(self) -> Optional[Scope]:
        """Pop the current scope from the scope stack."""
        with self._lock:
            if len(self.scope_stack) <= 1:
                raise EnvironmentError("Cannot pop global scope")
            
            popped_scope = self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
            return popped_scope
    
    def define_variable(self, name: str, value: Any, var_type: VariableType = VariableType.PRIMITIVE,
                       is_constant: bool = False, is_mutable: bool = True) -> Variable:
        """Define a variable in the current scope."""
        with self._lock:
            return self.current_scope.define_variable(name, value, var_type, is_constant, is_mutable)
    
    def get_variable(self, name: str) -> Any:
        """Get a variable value."""
        with self._lock:
            self.access_count += 1
            variable = self.current_scope.get_variable(name)
            
            if variable is None:
                # Check parent environment
                if self.parent:
                    return self.parent.get_variable(name)
                raise EnvironmentError(f"Variable '{name}' not found")
            
            return variable.access()
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value."""
        with self._lock:
            if not self.current_scope.set_variable(name, value):
                # Check parent environment
                if self.parent:
                    self.parent.set_variable(name, value)
                    return
                raise EnvironmentError(f"Variable '{name}' not found")
    
    def has_variable(self, name: str) -> bool:
        """Check if a variable exists."""
        with self._lock:
            if self.current_scope.has_variable(name):
                return True
            
            # Check parent environment
            if self.parent:
                return self.parent.has_variable(name)
            
            return False
    
    def reset(self) -> None:
        """Reset the environment to initial state."""
        with self._lock:
            # Clear all scopes except global
            self.scope_stack = [self.global_scope]
            self.current_scope = self.global_scope
            self.global_scope.clear()
            
            # Clear specialized environments
            self.neural_env.clear()
            self.signal_env.clear()
            
            # Reset metadata
            self.access_count = 0
            
            # Restore builtins
            self._setup_builtins()
    
    # =========================================================================
    # COMPATIBILITY METHODS FOR INTERPRETER
    # =========================================================================
    
    def define(self, name: str, value: Any, var_type: Union[VariableType, str] = VariableType.VARIABLE) -> Variable:
        """Define a variable (compatibility method)."""
        # Convert string var_type to VariableType if needed
        if isinstance(var_type, str):
            try:
                var_type = VariableType(var_type)
            except ValueError:
                var_type = VariableType.VARIABLE
        
        return self.define_variable(name, value, var_type)
    
    def get(self, name: str) -> Any:
        """Get a variable value (compatibility method)."""
        return self.get_variable(name)
    
    def set(self, name: str, value: Any) -> None:
        """Set a variable value (compatibility method)."""
        self.set_variable(name, value)
    
    def has(self, name: str) -> bool:
        """Check if a variable exists (compatibility method)."""
        return self.has_variable(name)
    
    def create_child(self) -> 'Environment':
        """Create a child environment."""
        return Environment(parent=self)
    
    def __str__(self) -> str:
        return f"Environment(scopes: {len(self.scope_stack)}, vars: {sum(len(s.variables) for s in self.scope_stack)})"
