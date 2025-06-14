"""
Built-in Functions for AST Interpreter.

This module provides all built-in functions available in the Anamorph
language interpreter.
"""

import math
import time
import random
import sys
import os
import asyncio
import threading
import json
import re
import hashlib
import base64
from typing import Any, Union, List, Dict, Optional, Iterable, Callable, Tuple, Set
from collections import defaultdict, deque
from functools import reduce, partial

# Add parent directories to path for comprehensive imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try comprehensive imports with fallbacks
try:
    from interpreter.ast_types import (
        ProgramState, SignalData, ExecutionState, ExecutionStats,
        DebugInfo, InterpreterConfig, ReturnException, BreakException,
        ContinueException, PulseException, RuntimeInterpreterError,
        NameInterpreterError, TypeInterpreterError, Literal
    )
    AST_TYPES_AVAILABLE = True
except ImportError:
    try:
        from ast_types import (
            ProgramState, SignalData, ExecutionState, ExecutionStats,
            DebugInfo, InterpreterConfig, ReturnException, BreakException,
            ContinueException, PulseException, RuntimeInterpreterError,
            NameInterpreterError, TypeInterpreterError, Literal
        )
        AST_TYPES_AVAILABLE = True
    except ImportError:
        # Create minimal fallback classes
        class ProgramState:
            def __init__(self):
                self.variables = {}
                self.functions = {}
                self.neurons = {}
                self.synapses = {}
                self.signal_queue = deque()
        
        class SignalData:
            def __init__(self, source, target, signal, intensity=1.0, signal_type="sync"):
                self.source = source
                self.target = target
                self.signal = signal
                self.intensity = intensity
                self.signal_type = signal_type
                self.timestamp = time.time()
        
        AST_TYPES_AVAILABLE = False

# Try to import environment module
try:
    from interpreter.environment import Environment, VariableType
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    try:
        from environment import Environment, VariableType
        ENVIRONMENT_AVAILABLE = True
    except ImportError:
        ENVIRONMENT_AVAILABLE = False

# Try to import syntax nodes
try:
    from syntax.nodes import (
        ASTNode, Expression, Statement, Literal as SyntaxLiteral,
        NeuralEntity, SynapseConnection, FunctionDeclaration
    )
    SYNTAX_AVAILABLE = True
except ImportError:
    # Mock classes for syntax nodes
    class ASTNode:
        def __init__(self):
            self.location = None
    
    class Expression(ASTNode):
        pass
    
    class Statement(ASTNode):
        pass
    
    class SyntaxLiteral(Expression):
        def __init__(self, value):
            super().__init__()
            self.value = value
    
    class NeuralEntity(Statement):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.state = {'activation': 0.0}
    
    class SynapseConnection(Statement):
        def __init__(self, source, target, weight=1.0):
            super().__init__()
            self.source = source
            self.target = target
            self.weight = weight
    
    class FunctionDeclaration(Statement):
        def __init__(self, name, params, body):
            super().__init__()
            self.name = name
            self.params = params
            self.body = body
    
    SYNTAX_AVAILABLE = False

class BuiltinFunctions:
    """Container for all built-in functions."""
    
    def __init__(self, program_state: ProgramState):
        """Initialize with program state reference."""
        self.program_state = program_state
    
    def get_all_builtins(self) -> Dict[str, callable]:
        """Get all built-in functions."""
        return {
            # Standard built-ins
            'print': self.builtin_print,
            'len': self.builtin_len,
            'type': self.builtin_type,
            'str': self.builtin_str,
            'int': self.builtin_int,
            'float': self.builtin_float,
            'bool': self.builtin_bool,
            'list': self.builtin_list,
            'dict': self.builtin_dict,
            'tuple': self.builtin_tuple,
            'set': self.builtin_set,
            'range': self.builtin_range,
            'enumerate': self.builtin_enumerate,
            'zip': self.builtin_zip,
            'map': self.builtin_map,
            'filter': self.builtin_filter,
            'sum': self.builtin_sum,
            'max': self.builtin_max,
            'min': self.builtin_min,
            'abs': self.builtin_abs,
            'round': self.builtin_round,
            'pow': self.builtin_pow,
            'divmod': self.builtin_divmod,
            'sorted': self.builtin_sorted,
            'reversed': self.builtin_reversed,
            'any': self.builtin_any,
            'all': self.builtin_all,
            
            # Math functions
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'exp': math.exp,
            'ceil': math.ceil,
            'floor': math.floor,
            
            # Neural-specific built-ins
            'activate': self.builtin_activate,
            'pulse': self.builtin_pulse,
            'resonate': self.builtin_resonate,
            'signal_strength': self.builtin_signal_strength,
            'connect_neurons': self.builtin_connect_neurons,
            'get_neuron_state': self.builtin_get_neuron_state,
            'reset_neuron': self.builtin_reset_neuron,
            
            # Utility functions
            'sleep': self.builtin_sleep,
            'time': self.builtin_time,
            'random': self.builtin_random,
            'randint': self.builtin_randint,
            'choice': self.builtin_choice,
        }
    
    # =========================================================================
    # STANDARD BUILT-IN FUNCTIONS
    # =========================================================================
    
    def builtin_print(self, *args, sep=' ', end='\n', file=None) -> None:
        """Built-in print function."""
        output = sep.join(str(arg) for arg in args)
        print(output, end=end, file=file)
    
    def builtin_len(self, obj) -> int:
        """Built-in len function."""
        return len(obj)
    
    def builtin_type(self, obj) -> str:
        """Built-in type function."""
        return type(obj).__name__
    
    def builtin_str(self, obj) -> str:
        """Built-in str function."""
        return str(obj)
    
    def builtin_int(self, obj, base=10) -> int:
        """Built-in int function."""
        if isinstance(obj, str) and base != 10:
            return int(obj, base)
        return int(obj)
    
    def builtin_float(self, obj) -> float:
        """Built-in float function."""
        return float(obj)
    
    def builtin_bool(self, obj) -> bool:
        """Built-in bool function."""
        return bool(obj)
    
    def builtin_list(self, iterable=None) -> list:
        """Built-in list function."""
        if iterable is None:
            return []
        return list(iterable)
    
    def builtin_dict(self, *args, **kwargs) -> dict:
        """Built-in dict function."""
        result = dict(*args)
        result.update(kwargs)
        return result
    
    def builtin_tuple(self, iterable=None) -> tuple:
        """Built-in tuple function."""
        if iterable is None:
            return ()
        return tuple(iterable)
    
    def builtin_set(self, iterable=None) -> set:
        """Built-in set function."""
        if iterable is None:
            return set()
        return set(iterable)
    
    def builtin_range(self, *args) -> range:
        """Built-in range function."""
        return range(*args)
    
    def builtin_enumerate(self, iterable, start=0) -> enumerate:
        """Built-in enumerate function."""
        return enumerate(iterable, start)
    
    def builtin_zip(self, *iterables) -> zip:
        """Built-in zip function."""
        return zip(*iterables)
    
    def builtin_map(self, function, *iterables) -> map:
        """Built-in map function."""
        return map(function, *iterables)
    
    def builtin_filter(self, function, iterable) -> filter:
        """Built-in filter function."""
        return filter(function, iterable)
    
    def builtin_sum(self, iterable, start=0) -> Union[int, float]:
        """Built-in sum function."""
        return sum(iterable, start)
    
    def builtin_max(self, *args, key=None, default=None) -> Any:
        """Built-in max function."""
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            if default is not None:
                try:
                    return max(args[0], key=key)
                except ValueError:
                    return default
            return max(args[0], key=key)
        return max(args, key=key)
    
    def builtin_min(self, *args, key=None, default=None) -> Any:
        """Built-in min function."""
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            if default is not None:
                try:
                    return min(args[0], key=key)
                except ValueError:
                    return default
            return min(args[0], key=key)
        return min(args, key=key)
    
    def builtin_abs(self, x) -> Union[int, float]:
        """Built-in abs function."""
        return abs(x)
    
    def builtin_round(self, x, ndigits=None) -> Union[int, float]:
        """Built-in round function."""
        if ndigits is None:
            return round(x)
        return round(x, ndigits)
    
    def builtin_pow(self, base, exp, mod=None) -> Union[int, float]:
        """Built-in pow function."""
        if mod is None:
            return pow(base, exp)
        return pow(base, exp, mod)
    
    def builtin_divmod(self, a, b) -> tuple:
        """Built-in divmod function."""
        return divmod(a, b)
    
    def builtin_sorted(self, iterable, key=None, reverse=False) -> list:
        """Built-in sorted function."""
        return sorted(iterable, key=key, reverse=reverse)
    
    def builtin_reversed(self, seq) -> reversed:
        """Built-in reversed function."""
        return reversed(seq)
    
    def builtin_any(self, iterable) -> bool:
        """Built-in any function."""
        return any(iterable)
    
    def builtin_all(self, iterable) -> bool:
        """Built-in all function."""
        return all(iterable)
    
    # =========================================================================
    # NEURAL-SPECIFIC BUILT-IN FUNCTIONS
    # =========================================================================
    
    def builtin_activate(self, neuron_name: str, signal: float = 0.0) -> float:
        """Activate a neuron."""
        if neuron_name in self.program_state.neurons:
            neuron = self.program_state.neurons[neuron_name]
            return neuron.activate(signal)
        else:
            raise NameError(f"Neuron not found: {neuron_name}")
    
    def builtin_pulse(self, signal: Any, target: str = "broadcast", 
                     intensity: float = 1.0) -> None:
        """Send a pulse signal."""
        signal_data = SignalData(
            source="builtin_pulse",
            target=target,
            signal=signal,
            intensity=intensity,
            signal_type="sync"
        )
        
        self.program_state.signal_queue.append(signal_data)
    
    def builtin_resonate(self, frequency: float, duration: float = 1.0) -> None:
        """Create resonance pattern."""
        # Simplified resonance implementation
        for neuron in self.program_state.neurons.values():
            neuron.activate(frequency * duration)
    
    def builtin_signal_strength(self, neuron_name: str) -> float:
        """Get signal strength of a neuron."""
        if neuron_name in self.program_state.neurons:
            neuron = self.program_state.neurons[neuron_name]
            return neuron.state.get('output', 0.0)
        return 0.0
    
    def builtin_connect_neurons(self, source: str, target: str, 
                               weight: float = 1.0, 
                               connection_type: str = "excitatory") -> bool:
        """Connect two neurons with a synapse."""
        if source not in self.program_state.neurons:
            raise NameError(f"Source neuron not found: {source}")
        
        if target not in self.program_state.neurons:
            raise NameError(f"Target neuron not found: {target}")
        
        # Create synapse connection (simplified)
        synapse_key = f"{source}->{target}"
        synapse_data = {
            'source': source,
            'target': target,
            'weight': weight,
            'connection_type': connection_type
        }
        
        self.program_state.synapses[synapse_key] = synapse_data
        return True
    
    def builtin_get_neuron_state(self, neuron_name: str) -> Dict[str, Any]:
        """Get complete state of a neuron."""
        if neuron_name in self.program_state.neurons:
            neuron = self.program_state.neurons[neuron_name]
            if hasattr(neuron, 'get_info'):
                return neuron.get_info()
            else:
                return {'name': neuron_name, 'state': 'active'}
        else:
            raise NameError(f"Neuron not found: {neuron_name}")
    
    def builtin_reset_neuron(self, neuron_name: str) -> bool:
        """Reset a neuron to initial state."""
        if neuron_name in self.program_state.neurons:
            neuron = self.program_state.neurons[neuron_name]
            if hasattr(neuron, 'state'):
                neuron.state.clear()
            if hasattr(neuron, 'activation_count'):
                neuron.activation_count = 0
            if hasattr(neuron, 'last_activation'):
                neuron.last_activation = 0.0
            return True
        else:
            raise NameError(f"Neuron not found: {neuron_name}")
    
    # =========================================================================
    # UTILITY BUILT-IN FUNCTIONS
    # =========================================================================
    
    def builtin_sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        time.sleep(seconds)
    
    def builtin_time(self) -> float:
        """Get current time."""
        return time.time()
    
    def builtin_random(self) -> float:
        """Get random float between 0 and 1."""
        return random.random()
    
    def builtin_randint(self, a: int, b: int) -> int:
        """Get random integer between a and b inclusive."""
        return random.randint(a, b)
    
    def builtin_choice(self, sequence) -> Any:
        """Choose random element from sequence."""
        return random.choice(sequence) 