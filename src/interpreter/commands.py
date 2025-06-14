"""
Enhanced Command System for AnamorphX Interpreter.

This module provides the complete command system with neural constructs,
async execution, and comprehensive environment integration.
"""

import asyncio
import math
import time
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .runtime import ExecutionContext
from .environment import VariableType


# =============================================================================
# BASE COMMAND CLASSES
# =============================================================================

class CommandCategory(Enum):
    """Command categories in AnamorphX."""
    STRUCTURAL = "structural"
    FLOW_CONTROL = "flow_control"
    SECURITY = "security"
    DATA_MANAGEMENT = "data_management"
    MACHINE_LEARNING = "machine_learning"
    CLOUD_NETWORK = "cloud_network"


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool = True
    value: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def failed(self) -> bool:
        return not self.success
    
    def add_side_effect(self, effect: str) -> None:
        self.side_effects.append(effect)
    
    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value


class CommandError(Exception):
    """Base exception for command errors."""
    pass


class Command:
    """Base class for all commands."""
    
    def __init__(self, name: str, category: CommandCategory, description: str = ""):
        self.name = name
        self.category = category
        self.description = description
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution = None
    
    def execute(self, context: ExecutionContext, *args, **kwargs) -> CommandResult:
        """Execute the command."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_args(self, *args, **kwargs) -> bool:
        """Validate command arguments."""
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get command execution statistics."""
        return {
            'name': self.name,
            'category': self.category.value,
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': (
                self.total_execution_time / max(self.execution_count, 1)
            ),
            'last_execution': self.last_execution
        }


class StructuralCommand(Command):
    """Base class for structural commands."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, CommandCategory.STRUCTURAL, description)


class FlowControlCommand(Command):
    """Base class for flow control commands."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, CommandCategory.FLOW_CONTROL, description)


class SecurityCommand(Command):
    """Base class for security commands."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, CommandCategory.SECURITY, description)


class DataManagementCommand(Command):
    """Base class for data management commands."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, CommandCategory.DATA_MANAGEMENT, description)


class MachineLearningCommand(Command):
    """Base class for machine learning commands."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, CommandCategory.MACHINE_LEARNING, description)


class CloudNetworkCommand(Command):
    """Base class for cloud/network commands."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, CommandCategory.CLOUD_NETWORK, description)


# =============================================================================
# NEURAL ENTITY MODELS
# =============================================================================

@dataclass
class NeuralEntity:
    """Enhanced neural entity with full functionality."""
    name: str
    entity_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activation: float = 0.0
    activation_count: int = 0
    
    def activate(self, input_signal: float = 0.0) -> float:
        """Activate the neuron with input signal."""
        self.last_activation = time.time()
        self.activation_count += 1
        
        # Apply activation function
        activation_func = self.parameters.get('activation', 'relu')
        bias = self.parameters.get('bias', 0.0)
        threshold = self.parameters.get('threshold', 0.5)
        
        # Calculate output based on activation function
        if activation_func == 'relu':
            output = max(0, input_signal + bias)
        elif activation_func == 'sigmoid':
            output = 1 / (1 + math.exp(-(input_signal + bias)))
        elif activation_func == 'tanh':
            output = math.tanh(input_signal + bias)
        elif activation_func == 'linear':
            output = input_signal + bias
        else:
            output = input_signal + bias
        
        # Apply threshold
        if output < threshold:
            output = 0.0
        
        self.state['output'] = output
        self.state['last_input'] = input_signal
        
        return output
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the entity."""
        return {
            'name': self.name,
            'type': self.entity_type,
            'parameters': self.parameters.copy(),
            'state': self.state.copy(),
            'created_at': self.created_at,
            'last_activation': self.last_activation,
            'activation_count': self.activation_count,
            'age': time.time() - self.created_at
        }


@dataclass
class SynapseConnection:
    """Enhanced synapse connection with learning capabilities."""
    source: str
    target: str
    weight: float
    connection_type: str = "excitatory"
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    signal_count: int = 0
    last_signal: float = 0.0
    
    def transmit_signal(self, signal: float) -> float:
        """Transmit signal through the synapse."""
        self.signal_count += 1
        self.last_signal = time.time()
        
        # Apply connection type
        if self.connection_type == "inhibitory":
            signal = -abs(signal)
        
        # Apply weight and parameters
        delay = self.parameters.get('delay', 0.0)
        strength = self.parameters.get('strength', 1.0)
        
        transmitted_signal = signal * self.weight * strength
        
        # Apply plasticity (Hebbian learning)
        if self.parameters.get('plasticity', True):
            learning_rate = self.parameters.get('learning_rate', 0.01)
            if signal > 0:  # Strengthen connection on positive signals
                self.weight += learning_rate * signal * 0.1
                self.weight = min(self.weight, 10.0)  # Cap maximum weight
        
        return transmitted_signal
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the connection."""
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'type': self.connection_type,
            'parameters': self.parameters.copy(),
            'created_at': self.created_at,
            'signal_count': self.signal_count,
            'last_signal': self.last_signal,
            'age': time.time() - self.created_at
        }


# =============================================================================
# ENHANCED STRUCTURAL COMMANDS
# =============================================================================

class EnhancedNeuroCommand(StructuralCommand):
    """Enhanced neuron creation command with full environment integration."""
    
    def __init__(self):
        super().__init__("neuro", "Create a new neuron with specified parameters")
    
    def validate_args(self, *args, **kwargs) -> bool:
        """Validate command arguments."""
        if not args or not isinstance(args[0], str):
            return False
        if len(args[0].strip()) == 0:
            return False
        return True
    
    def execute(self, context: ExecutionContext, name: str, 
                neuron_type: str = "basic", activation: str = "relu",
                bias: float = 0.0, threshold: float = 0.5,
                learning_rate: float = 0.01, **kwargs) -> CommandResult:
        """Execute enhanced neuro command."""
        try:
            # Check if neuron already exists
            if name in context.environment.neural_env.list_neurons():
                return CommandResult(
                    success=False,
                    error=f"Neuron '{name}' already exists"
                )
            
            # Create enhanced neuron entity
            entity = NeuralEntity(
                name=name,
                entity_type=neuron_type,
                parameters={
                    'activation': activation,
                    'bias': bias,
                    'threshold': threshold,
                    'learning_rate': learning_rate,
                    **kwargs
                },
                state={
                    'active': True, 
                    'output': 0.0,
                    'energy': 1.0,
                    'connections_in': 0,
                    'connections_out': 0
                }
            )
            
            # Register in neural environment
            context.environment.neural_env.register_neuron(name, entity)
            
            # Also register as variable in current scope
            context.environment.define_variable(
                name, entity, VariableType.NEURON, is_constant=False
            )
            
            result = CommandResult(
                success=True,
                value=entity,
                side_effects=[f"Created neuron '{name}' of type '{neuron_type}'"],
                metadata={
                    'neuron_id': name, 
                    'type': neuron_type,
                    'activation': activation,
                    'parameters': entity.parameters
                }
            )
            
            result.add_side_effect(f"Registered neuron '{name}' in environment")
            result.set_metadata('creation_time', entity.created_at)
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to create neuron '{name}': {str(e)}"
            )


class EnhancedSynapCommand(StructuralCommand):
    """Enhanced synapse creation command with learning capabilities."""
    
    def __init__(self):
        super().__init__("synap", "Create a synapse connection between neurons")
    
    def validate_args(self, *args, **kwargs) -> bool:
        """Validate command arguments."""
        if len(args) < 2:
            return False
        if not isinstance(args[0], str) or not isinstance(args[1], str):
            return False
        return True
    
    def execute(self, context: ExecutionContext, source: str, target: str, 
                weight: float = 1.0, synapse_type: str = "excitatory",
                delay: float = 0.0, plasticity: bool = True,
                learning_rate: float = 0.01, **kwargs) -> CommandResult:
        """Execute enhanced synap command."""
        try:
            # Verify neurons exist
            if source not in context.environment.neural_env.list_neurons():
                return CommandResult(
                    success=False, 
                    error=f"Source neuron '{source}' not found"
                )
            
            if target not in context.environment.neural_env.list_neurons():
                return CommandResult(
                    success=False, 
                    error=f"Target neuron '{target}' not found"
                )
            
            # Get neuron entities
            source_neuron = context.environment.neural_env.get_neuron(source)
            target_neuron = context.environment.neural_env.get_neuron(target)
            
            # Create enhanced synapse
            synapse = SynapseConnection(
                source=source,
                target=target,
                weight=weight,
                connection_type=synapse_type,
                parameters={
                    'delay': delay,
                    'plasticity': plasticity,
                    'learning_rate': learning_rate,
                    'strength': kwargs.get('strength', 1.0),
                    **kwargs
                }
            )
            
            connection_id = f"{source}->{target}"
            
            # Register synapse in environment
            context.environment.neural_env.register_synapse(connection_id, synapse)
            
            # Update neuron connection counts
            if hasattr(source_neuron, 'state'):
                source_neuron.state['connections_out'] = source_neuron.state.get('connections_out', 0) + 1
            if hasattr(target_neuron, 'state'):
                target_neuron.state['connections_in'] = target_neuron.state.get('connections_in', 0) + 1
            
            result = CommandResult(
                success=True,
                value=synapse,
                side_effects=[
                    f"Created synapse {source} -> {target} (weight: {weight})",
                    f"Updated connection counts for neurons"
                ],
                metadata={
                    'connection_id': connection_id, 
                    'weight': weight,
                    'type': synapse_type,
                    'source': source,
                    'target': target
                }
            )
            
            result.set_metadata('creation_time', synapse.created_at)
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to create synapse {source}->{target}: {str(e)}"
            )


class EnhancedPulseCommand(FlowControlCommand):
    """Enhanced pulse command with real signal propagation."""
    
    def __init__(self):
        super().__init__("pulse", "Send a neural pulse through the network")
    
    def execute(self, context: ExecutionContext, signal_data: Any = None, 
                target: str = "broadcast", intensity: float = 1.0,
                propagate: bool = True, **kwargs) -> CommandResult:
        """Execute enhanced pulse command."""
        try:
            pulse_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Create pulse data
            pulse_data = {
                'id': pulse_id,
                'data': signal_data,
                'intensity': intensity,
                'timestamp': timestamp,
                'source': kwargs.get('source', 'system'),
                'propagated_to': []
            }
            
            # Register pulse in signal environment
            context.environment.signal_env.register_pulse(pulse_id, pulse_data)
            
            propagated_neurons = []
            
            if target == "broadcast":
                # Send to all neurons
                neurons = context.environment.neural_env.list_neurons()
                for neuron_name in neurons:
                    self._activate_neuron(context, neuron_name, intensity, pulse_data)
                    propagated_neurons.append(neuron_name)
                    
                    if propagate:
                        # Propagate through connections
                        connections = context.environment.neural_env.get_connections(neuron_name)
                        for connected_neuron in connections:
                            if connected_neuron not in propagated_neurons:
                                # Get synapse and transmit signal
                                synapse_id = f"{neuron_name}->{connected_neuron}"
                                try:
                                    synapse = context.environment.neural_env.get_synapse(synapse_id)
                                    if hasattr(synapse, 'transmit_signal'):
                                        transmitted_intensity = synapse.transmit_signal(intensity)
                                    else:
                                        transmitted_intensity = intensity * synapse.get('weight', 1.0)
                                    
                                    self._activate_neuron(context, connected_neuron, transmitted_intensity, pulse_data)
                                    propagated_neurons.append(connected_neuron)
                                except:
                                    pass  # Synapse might not exist
            else:
                # Send to specific target
                if target in context.environment.neural_env.list_neurons():
                    self._activate_neuron(context, target, intensity, pulse_data)
                    propagated_neurons.append(target)
                else:
                    # Send as regular signal
                    context.environment.signal_env.send_signal(target, signal_data, target)
            
            pulse_data['propagated_to'] = propagated_neurons
            
            result = CommandResult(
                success=True,
                value=pulse_data,
                side_effects=[
                    f"Sent pulse with intensity {intensity}",
                    f"Propagated to {len(propagated_neurons)} neurons"
                ],
                metadata={
                    'pulse_id': pulse_id,
                    'intensity': intensity,
                    'target': target,
                    'propagated_count': len(propagated_neurons),
                    'propagated_to': propagated_neurons
                }
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to send pulse: {str(e)}"
            )
    
    def _activate_neuron(self, context: ExecutionContext, neuron_name: str, 
                        intensity: float, pulse_data: Dict) -> None:
        """Activate a specific neuron."""
        try:
            neuron = context.environment.neural_env.get_neuron(neuron_name)
            if hasattr(neuron, 'activate'):
                output = neuron.activate(intensity)
            else:
                # Handle dict-based neuron data
                if isinstance(neuron, dict):
                    neuron['state'] = neuron.get('state', {})
                    neuron['state']['last_input'] = intensity
                    neuron['state']['output'] = intensity  # Simplified activation
                    neuron['state']['last_activation'] = time.time()
                    output = intensity
                else:
                    output = intensity
            
            # Send activation signal
            context.environment.signal_env.send_signal(
                f"neuron_activated_{neuron_name}",
                {
                    'neuron': neuron_name,
                    'input': intensity,
                    'output': output,
                    'pulse_id': pulse_data['id']
                },
                neuron_name
            )
        except Exception:
            pass  # Don't let individual neuron failures break the pulse


# =============================================================================
# ENHANCED ASYNC COMMANDS
# =============================================================================

class EnhancedAsyncCommand(FlowControlCommand):
    """Enhanced async command with proper async execution."""
    
    def __init__(self):
        super().__init__("async", "Execute tasks asynchronously")
    
    def execute(self, context: ExecutionContext, task: Union[Callable, str], 
                task_name: str = None, timeout: float = None, **kwargs) -> CommandResult:
        """Execute async command synchronously (creates async task)."""
        try:
            if task_name is None:
                task_name = f"async_task_{int(time.time())}"
            
            # Create async task info
            task_info = {
                'name': task_name,
                'task': task,
                'created_at': time.time(),
                'status': 'created',
                'timeout': timeout,
                'kwargs': kwargs
            }
            
            # Store task info in context
            if not hasattr(context, 'async_tasks'):
                context.async_tasks = {}
            context.async_tasks[task_name] = task_info
            
            # For sync execution, we just prepare the task
            task_info['status'] = 'prepared'
            
            return CommandResult(
                success=True,
                value=task_info,
                side_effects=[f"Prepared async task '{task_name}'"],
                metadata={
                    'task_name': task_name,
                    'task_type': type(task).__name__,
                    'timeout': timeout
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to prepare async task: {str(e)}"
            )
    
    async def execute_async(self, context: ExecutionContext, task: Union[Callable, str], 
                           task_name: str = None, timeout: float = None, **kwargs) -> CommandResult:
        """Execute async command asynchronously."""
        try:
            if task_name is None:
                task_name = f"async_task_{int(time.time())}"
            
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(task):
                if timeout:
                    result = await asyncio.wait_for(task(**kwargs), timeout=timeout)
                else:
                    result = await task(**kwargs)
            elif callable(task):
                # Run in thread pool for CPU-bound tasks
                loop = asyncio.get_event_loop()
                if timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: task(**kwargs)), 
                        timeout=timeout
                    )
                else:
                    result = await loop.run_in_executor(None, lambda: task(**kwargs))
            else:
                # It's a command name - simulate execution
                await asyncio.sleep(0.1)  # Simulate async work
                result = f"Executed command: {task}"
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                success=True,
                value=result,
                execution_time=execution_time,
                side_effects=[f"Completed async task '{task_name}' in {execution_time:.3f}s"],
                metadata={
                    'task_name': task_name,
                    'execution_time': execution_time,
                    'timeout': timeout
                }
            )
            
        except asyncio.TimeoutError:
            return CommandResult(
                success=False,
                error=f"Async task '{task_name}' timed out after {timeout}s"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Async task '{task_name}' failed: {str(e)}"
            )


# =============================================================================
# ENHANCED COMMAND REGISTRY
# =============================================================================

class EnhancedCommandRegistry:
    """Enhanced command registry with better error handling and extensibility."""
    
    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._categories: Dict[CommandCategory, List[str]] = {
            category: [] for category in CommandCategory
        }
        self._aliases: Dict[str, str] = {}
        self._middleware: List[Callable] = []
        self._hooks: Dict[str, List[Callable]] = {
            'before_execute': [],
            'after_execute': [],
            'on_error': []
        }
        
        # Register enhanced commands
        self._register_enhanced_commands()
    
    def _register_enhanced_commands(self) -> None:
        """Register all enhanced commands."""
        commands = [
            EnhancedNeuroCommand(),
            EnhancedSynapCommand(),
            EnhancedPulseCommand(),
            EnhancedAsyncCommand(),
        ]
        
        for command in commands:
            self.register_command(command)
    
    def register_command(self, command: Command, aliases: Optional[List[str]] = None) -> None:
        """Register a command with enhanced error handling."""
        try:
            if command.name in self._commands:
                raise CommandError(f"Command '{command.name}' already registered")
            
            self._commands[command.name] = command
            self._categories[command.category].append(command.name)
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases:
                        raise CommandError(f"Alias '{alias}' already registered")
                    self._aliases[alias] = command.name
                    
        except Exception as e:
            raise CommandError(f"Failed to register command '{command.name}': {str(e)}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware for command execution."""
        self._middleware.append(middleware)
    
    def add_hook(self, hook_type: str, hook_func: Callable) -> None:
        """Add execution hook."""
        if hook_type in self._hooks:
            self._hooks[hook_type].append(hook_func)
    
    def execute_command(self, name: str, context: ExecutionContext, 
                       *args, **kwargs) -> CommandResult:
        """Execute command with enhanced error handling and hooks."""
        command = self.get_command(name)
        if not command:
            return CommandResult(
                success=False,
                error=f"Unknown command: {name}",
                metadata={'command_name': name}
            )
        
        try:
            # Execute before hooks
            for hook in self._hooks['before_execute']:
                try:
                    hook(command, context, args, kwargs)
                except Exception:
                    pass  # Don't let hook failures break execution
            
            # Apply middleware
            for middleware in self._middleware:
                try:
                    middleware(command, context, args, kwargs)
                except Exception:
                    pass
            
            # Validate arguments
            if not command.validate_args(*args, **kwargs):
                result = CommandResult(
                    success=False,
                    error=f"Invalid arguments for command: {name}",
                    metadata={'command_name': name, 'args': args, 'kwargs': kwargs}
                )
            else:
                # Execute command
                with command._execution_timer():
                    result = command.execute(context, *args, **kwargs)
                    result.set_metadata('command_name', name)
                    result.set_metadata('execution_stats', command.get_stats())
            
            # Execute after hooks
            for hook in self._hooks['after_execute']:
                try:
                    hook(command, context, result, args, kwargs)
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            error_result = CommandResult(
                success=False,
                error=f"Command execution failed: {str(e)}",
                metadata={'command_name': name, 'exception_type': type(e).__name__}
            )
            
            # Execute error hooks
            for hook in self._hooks['on_error']:
                try:
                    hook(command, context, e, args, kwargs)
                except Exception:
                    pass
            
            return error_result
    
    def get_command(self, name: str) -> Optional[Command]:
        """Get command with alias resolution."""
        if name in self._commands:
            return self._commands[name]
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        total_commands = len(self._commands)
        total_executions = sum(cmd.execution_count for cmd in self._commands.values())
        total_errors = sum(cmd.error_count for cmd in self._commands.values())
        
        category_stats = {}
        for category, command_names in self._categories.items():
            commands = [self._commands[name] for name in command_names]
            category_stats[category.value] = {
                'command_count': len(commands),
                'total_executions': sum(cmd.execution_count for cmd in commands),
                'total_errors': sum(cmd.error_count for cmd in commands),
                'average_execution_time': sum(cmd.total_execution_time for cmd in commands) / max(1, sum(cmd.execution_count for cmd in commands))
            }
        
        return {
            'total_commands': total_commands,
            'total_executions': total_executions,
            'total_errors': total_errors,
            'error_rate': total_errors / max(1, total_executions),
            'categories': category_stats,
            'middleware_count': len(self._middleware),
            'hooks': {hook_type: len(hooks) for hook_type, hooks in self._hooks.items()}
        } 