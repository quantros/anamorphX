"""
Neural Semantic Analysis for AnamorphX

This module provides specialized semantic analysis for neural constructs
in the Anamorph programming language.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from ..syntax.nodes import ASTNode, SourceLocation, NodeType
from .symbols import Symbol, SymbolType, NeuronSymbol, SynapseSymbol
from .types import Type, NeuralType, SignalType, PulseType
from .errors import NeuralError, SemanticErrorType


class NeuralAnalyzer:
    """Main analyzer for neural constructs."""
    
    def __init__(self, type_system, scope_manager, symbol_resolver):
        self.type_system = type_system
        self.scope_manager = scope_manager
        self.symbol_resolver = symbol_resolver
        self.neuron_analyzer = NeuronAnalyzer(self)
        self.synapse_analyzer = SynapseAnalyzer(self)
        self.signal_analyzer = SignalAnalyzer(self)
        self.pulse_analyzer = PulseAnalyzer(self)
        self.resonance_analyzer = ResonanceAnalyzer(self)
        self.network_validator = NeuralNetworkValidator(self)
        self.flow_analyzer = NeuralFlowAnalyzer(self)
        
        self.neural_network: Dict[str, List[str]] = {}
        self.signal_flows: Dict[str, List[str]] = {}
        self.errors: List[NeuralError] = []
    
    def analyze_neural_construct(self, node: ASTNode) -> bool:
        """Analyze a neural construct node."""
        try:
            if node.node_type == NodeType.NEURON_DECLARATION:
                return self.neuron_analyzer.analyze(node)
            elif node.node_type == NodeType.SYNAPSE_DECLARATION:
                return self.synapse_analyzer.analyze(node)
            elif node.node_type == NodeType.PULSE_STATEMENT:
                return self.pulse_analyzer.analyze(node)
            elif node.node_type == NodeType.SIGNAL_EXPRESSION:
                return self.signal_analyzer.analyze(node)
            elif node.node_type == NodeType.RESONATE_STATEMENT:
                return self.resonance_analyzer.analyze(node)
            
            return True
        except Exception as e:
            self.errors.append(NeuralError(
                f"Neural analysis failed: {e}",
                location=node.location
            ))
            return False
    
    def validate_neural_network(self) -> bool:
        """Validate the entire neural network."""
        return self.network_validator.validate()
    
    def analyze_signal_flow(self) -> Dict[str, Any]:
        """Analyze signal flow through the network."""
        return self.flow_analyzer.analyze()


class NeuronAnalyzer:
    """Analyzes neuron declarations and usage."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.neurons: Dict[str, NeuronSymbol] = {}
    
    def analyze(self, node: ASTNode) -> bool:
        """Analyze neuron declaration."""
        if not self._validate_neuron_declaration(node):
            return False
        
        # Create neuron symbol
        neuron_symbol = self._create_neuron_symbol(node)
        if not neuron_symbol:
            return False
        
        # Register neuron
        self.neurons[node.name] = neuron_symbol
        
        # Enter neuron scope
        neuron_scope = self.parent.scope_manager.create_neuron_scope(
            node.name, 
            getattr(node, 'neuron_type', 'basic')
        )
        
        # Analyze neuron body
        if hasattr(node, 'body') and node.body:
            success = self._analyze_neuron_body(node.body, neuron_scope)
            self.parent.scope_manager.exit_scope()
            return success
        
        self.parent.scope_manager.exit_scope()
        return True
    
    def _validate_neuron_declaration(self, node: ASTNode) -> bool:
        """Validate neuron declaration syntax and semantics."""
        # Check neuron name
        if not node.name or not node.name.isidentifier():
            self.parent.errors.append(NeuralError(
                f"Invalid neuron name '{node.name}'",
                neural_type='neuron',
                location=node.location
            ))
            return False
        
        # Check for duplicate neuron
        if node.name in self.neurons:
            self.parent.errors.append(NeuralError(
                f"Neuron '{node.name}' already declared",
                neural_type='neuron',
                location=node.location
            ))
            return False
        
        # Validate neuron type
        if hasattr(node, 'neuron_type'):
            valid_types = {'basic', 'perceptron', 'lstm', 'gru', 'cnn', 'rnn'}
            if node.neuron_type not in valid_types:
                self.parent.errors.append(NeuralError(
                    f"Unknown neuron type '{node.neuron_type}'",
                    neural_type='neuron',
                    location=node.location
                ))
                return False
        
        # Validate activation function
        if hasattr(node, 'activation_function'):
            valid_functions = {
                'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 
                'softmax', 'linear', 'step', 'gaussian'
            }
            if node.activation_function not in valid_functions:
                self.parent.errors.append(NeuralError(
                    f"Unknown activation function '{node.activation_function}'",
                    neural_type='neuron',
                    location=node.location
                ))
                return False
        
        # Validate threshold
        if hasattr(node, 'threshold'):
            if not isinstance(node.threshold, (int, float)):
                self.parent.errors.append(NeuralError(
                    f"Neuron threshold must be numeric, got {type(node.threshold).__name__}",
                    neural_type='neuron',
                    location=node.location
                ))
                return False
        
        return True
    
    def _create_neuron_symbol(self, node: ASTNode) -> Optional[NeuronSymbol]:
        """Create neuron symbol from declaration."""
        try:
            return NeuronSymbol(
                name=node.name,
                neuron_type=getattr(node, 'neuron_type', 'basic'),
                activation_function=getattr(node, 'activation_function', 'sigmoid'),
                threshold=getattr(node, 'threshold', 0.5),
                location=node.location
            )
        except Exception as e:
            self.parent.errors.append(NeuralError(
                f"Failed to create neuron symbol: {e}",
                neural_type='neuron',
                location=node.location
            ))
            return None
    
    def _analyze_neuron_body(self, body: ASTNode, scope) -> bool:
        """Analyze neuron body statements."""
        success = True
        
        if body.node_type == NodeType.BLOCK_STATEMENT:
            for stmt in getattr(body, 'statements', []):
                if not self.parent.analyze_neural_construct(stmt):
                    success = False
        else:
            if not self.parent.analyze_neural_construct(body):
                success = False
        
        return success


class SynapseAnalyzer:
    """Analyzes synapse declarations and connections."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.synapses: Dict[str, SynapseSymbol] = {}
    
    def analyze(self, node: ASTNode) -> bool:
        """Analyze synapse declaration."""
        if not self._validate_synapse_declaration(node):
            return False
        
        # Create synapse symbol
        synapse_symbol = self._create_synapse_symbol(node)
        if not synapse_symbol:
            return False
        
        # Register synapse
        synapse_name = f"{node.source}->{node.target}"
        self.synapses[synapse_name] = synapse_symbol
        
        # Update neural network topology
        self._update_network_topology(node.source, node.target)
        
        return True
    
    def _validate_synapse_declaration(self, node: ASTNode) -> bool:
        """Validate synapse declaration."""
        # Check source neuron exists
        source_symbol = self.parent.symbol_resolver.resolve_symbol(node.source)
        if not source_symbol:
            self.parent.errors.append(NeuralError(
                f"Source neuron '{node.source}' not found",
                neural_type='synapse',
                location=node.location
            ))
            return False
        
        if source_symbol.symbol_type != SymbolType.NEURON:
            self.parent.errors.append(NeuralError(
                f"'{node.source}' is not a neuron",
                neural_type='synapse',
                location=node.location
            ))
            return False
        
        # Check target neuron exists
        target_symbol = self.parent.symbol_resolver.resolve_symbol(node.target)
        if not target_symbol:
            self.parent.errors.append(NeuralError(
                f"Target neuron '{node.target}' not found",
                neural_type='synapse',
                location=node.location
            ))
            return False
        
        if target_symbol.symbol_type != SymbolType.NEURON:
            self.parent.errors.append(NeuralError(
                f"'{node.target}' is not a neuron",
                neural_type='synapse',
                location=node.location
            ))
            return False
        
        # Check for self-connection
        if node.source == node.target:
            self.parent.errors.append(NeuralError(
                f"Neuron '{node.source}' cannot connect to itself",
                neural_type='synapse',
                location=node.location
            ))
            return False
        
        # Validate weight
        if hasattr(node, 'weight') and node.weight is not None:
            if not isinstance(node.weight, (int, float)):
                self.parent.errors.append(NeuralError(
                    f"Synapse weight must be numeric",
                    neural_type='synapse',
                    location=node.location
                ))
                return False
            
            if abs(node.weight) > 100:  # Reasonable weight limit
                self.parent.errors.append(NeuralError(
                    f"Synapse weight {node.weight} may be too large",
                    neural_type='synapse',
                    location=node.location
                ))
        
        # Validate delay
        if hasattr(node, 'delay') and node.delay is not None:
            if not isinstance(node.delay, (int, float)) or node.delay < 0:
                self.parent.errors.append(NeuralError(
                    f"Synapse delay must be non-negative numeric value",
                    neural_type='synapse',
                    location=node.location
                ))
                return False
        
        return True
    
    def _create_synapse_symbol(self, node: ASTNode) -> Optional[SynapseSymbol]:
        """Create synapse symbol from declaration."""
        try:
            return SynapseSymbol(
                name=f"{node.source}->{node.target}",
                source_neuron=node.source,
                target_neuron=node.target,
                weight=getattr(node, 'weight', 1.0),
                delay=getattr(node, 'delay', 0.0),
                plasticity=getattr(node, 'plasticity', 'static'),
                is_inhibitory=getattr(node, 'is_inhibitory', False),
                location=node.location
            )
        except Exception as e:
            self.parent.errors.append(NeuralError(
                f"Failed to create synapse symbol: {e}",
                neural_type='synapse',
                location=node.location
            ))
            return None
    
    def _update_network_topology(self, source: str, target: str):
        """Update neural network topology."""
        if source not in self.parent.neural_network:
            self.parent.neural_network[source] = []
        if target not in self.parent.neural_network[source]:
            self.parent.neural_network[source].append(target)


class SignalAnalyzer:
    """Analyzes signal expressions and propagation."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.signals: Dict[str, Type] = {}
    
    def analyze(self, node: ASTNode) -> bool:
        """Analyze signal expression."""
        # Determine signal type
        signal_type = self._infer_signal_type(node)
        if not signal_type:
            return False
        
        # Validate signal parameters
        if not self._validate_signal_parameters(node):
            return False
        
        # Track signal flow
        self._track_signal_flow(node, signal_type)
        
        return True
    
    def _infer_signal_type(self, node: ASTNode) -> Optional[Type]:
        """Infer the type of signal."""
        if hasattr(node, 'data_type'):
            data_type = self.parent.type_system.get_type(node.data_type)
            if data_type:
                return SignalType(
                    data_type,
                    frequency=getattr(node, 'frequency', 1.0)
                )
        
        # Default to float signal
        float_type = self.parent.type_system.get_type('float')
        return SignalType(float_type) if float_type else None
    
    def _validate_signal_parameters(self, node: ASTNode) -> bool:
        """Validate signal parameters."""
        # Validate frequency
        if hasattr(node, 'frequency'):
            if not isinstance(node.frequency, (int, float)) or node.frequency <= 0:
                self.parent.errors.append(NeuralError(
                    f"Signal frequency must be positive numeric value",
                    neural_type='signal',
                    location=node.location
                ))
                return False
        
        # Validate amplitude
        if hasattr(node, 'amplitude'):
            if not isinstance(node.amplitude, (int, float)):
                self.parent.errors.append(NeuralError(
                    f"Signal amplitude must be numeric",
                    neural_type='signal',
                    location=node.location
                ))
                return False
        
        return True
    
    def _track_signal_flow(self, node: ASTNode, signal_type: Type):
        """Track signal flow through the network."""
        signal_id = f"signal_{id(node)}"
        self.signals[signal_id] = signal_type
        
        # Add to signal flow tracking
        if hasattr(node, 'source') and hasattr(node, 'target'):
            if node.source not in self.parent.signal_flows:
                self.parent.signal_flows[node.source] = []
            self.parent.signal_flows[node.source].append(node.target)


class PulseAnalyzer:
    """Analyzes pulse statements and timing."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.pulses: List[Dict[str, Any]] = []
    
    def analyze(self, node: ASTNode) -> bool:
        """Analyze pulse statement."""
        # Check if in neuron context
        neuron_scope = self.parent.scope_manager.find_enclosing_neuron()
        if not neuron_scope:
            self.parent.errors.append(NeuralError(
                "Pulse statement must be inside a neuron",
                neural_type='pulse',
                location=node.location
            ))
            return False
        
        # Validate pulse parameters
        if not self._validate_pulse_parameters(node):
            return False
        
        # Record pulse
        self._record_pulse(node, neuron_scope.name)
        
        return True
    
    def _validate_pulse_parameters(self, node: ASTNode) -> bool:
        """Validate pulse parameters."""
        # Validate frequency
        if hasattr(node, 'frequency'):
            if not isinstance(node.frequency, (int, float)) or node.frequency <= 0:
                self.parent.errors.append(NeuralError(
                    f"Pulse frequency must be positive",
                    neural_type='pulse',
                    location=node.location
                ))
                return False
        
        # Validate duration
        if hasattr(node, 'duration'):
            if not isinstance(node.duration, (int, float)) or node.duration <= 0:
                self.parent.errors.append(NeuralError(
                    f"Pulse duration must be positive",
                    neural_type='pulse',
                    location=node.location
                ))
                return False
        
        # Validate intensity
        if hasattr(node, 'intensity'):
            if not isinstance(node.intensity, (int, float)):
                self.parent.errors.append(NeuralError(
                    f"Pulse intensity must be numeric",
                    neural_type='pulse',
                    location=node.location
                ))
                return False
        
        return True
    
    def _record_pulse(self, node: ASTNode, neuron_name: str):
        """Record pulse for timing analysis."""
        pulse_info = {
            'neuron': neuron_name,
            'frequency': getattr(node, 'frequency', 1.0),
            'duration': getattr(node, 'duration', 1.0),
            'intensity': getattr(node, 'intensity', 1.0),
            'location': node.location
        }
        self.pulses.append(pulse_info)


class ResonanceAnalyzer:
    """Analyzes resonance patterns and synchronization."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.resonance_patterns: List[Dict[str, Any]] = []
    
    def analyze(self, node: ASTNode) -> bool:
        """Analyze resonance statement."""
        # Check if in neuron context
        neuron_scope = self.parent.scope_manager.find_enclosing_neuron()
        if not neuron_scope:
            self.parent.errors.append(NeuralError(
                "Resonate statement must be inside a neuron",
                neural_type='resonance',
                location=node.location
            ))
            return False
        
        # Validate resonance parameters
        if not self._validate_resonance_parameters(node):
            return False
        
        # Analyze resonance pattern
        self._analyze_resonance_pattern(node, neuron_scope.name)
        
        return True
    
    def _validate_resonance_parameters(self, node: ASTNode) -> bool:
        """Validate resonance parameters."""
        # Validate target neurons
        if hasattr(node, 'targets'):
            for target in node.targets:
                target_symbol = self.parent.symbol_resolver.resolve_symbol(target)
                if not target_symbol or target_symbol.symbol_type != SymbolType.NEURON:
                    self.parent.errors.append(NeuralError(
                        f"Resonance target '{target}' is not a neuron",
                        neural_type='resonance',
                        location=node.location
                    ))
                    return False
        
        # Validate frequency
        if hasattr(node, 'frequency'):
            if not isinstance(node.frequency, (int, float)) or node.frequency <= 0:
                self.parent.errors.append(NeuralError(
                    f"Resonance frequency must be positive",
                    neural_type='resonance',
                    location=node.location
                ))
                return False
        
        return True
    
    def _analyze_resonance_pattern(self, node: ASTNode, neuron_name: str):
        """Analyze resonance pattern for conflicts."""
        pattern = {
            'source': neuron_name,
            'targets': getattr(node, 'targets', []),
            'frequency': getattr(node, 'frequency', 1.0),
            'phase': getattr(node, 'phase', 0.0),
            'location': node.location
        }
        
        # Check for resonance conflicts
        for existing_pattern in self.resonance_patterns:
            if self._has_resonance_conflict(pattern, existing_pattern):
                self.parent.errors.append(NeuralError(
                    f"Resonance conflict between {neuron_name} and {existing_pattern['source']}",
                    neural_type='resonance',
                    location=node.location
                ))
        
        self.resonance_patterns.append(pattern)
    
    def _has_resonance_conflict(self, pattern1: Dict, pattern2: Dict) -> bool:
        """Check if two resonance patterns conflict."""
        # Check for overlapping targets
        targets1 = set(pattern1['targets'])
        targets2 = set(pattern2['targets'])
        
        if targets1.intersection(targets2):
            # Check frequency compatibility
            freq_diff = abs(pattern1['frequency'] - pattern2['frequency'])
            if freq_diff < 0.1:  # Too close frequencies
                return True
        
        return False


class NeuralNetworkValidator:
    """Validates neural network structure and properties."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.validation_errors: List[NeuralError] = []
    
    def validate(self) -> bool:
        """Validate the entire neural network."""
        success = True
        
        if not self._validate_connectivity():
            success = False
        
        if not self._validate_cycles():
            success = False
        
        if not self._validate_signal_compatibility():
            success = False
        
        if not self._validate_timing_constraints():
            success = False
        
        return success
    
    def _validate_connectivity(self) -> bool:
        """Validate network connectivity."""
        success = True
        
        # Check for isolated neurons
        all_neurons = set(self.parent.neuron_analyzer.neurons.keys())
        connected_neurons = set()
        
        for source, targets in self.parent.neural_network.items():
            connected_neurons.add(source)
            connected_neurons.update(targets)
        
        isolated = all_neurons - connected_neurons
        for neuron in isolated:
            self.validation_errors.append(NeuralError(
                f"Neuron '{neuron}' is not connected to the network",
                neural_type='network'
            ))
            success = False
        
        return success
    
    def _validate_cycles(self) -> bool:
        """Validate network for problematic cycles."""
        success = True
        
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.parent.neural_network.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for neuron in self.parent.neural_network:
            if neuron not in visited:
                if has_cycle(neuron):
                    self.validation_errors.append(NeuralError(
                        f"Cycle detected in neural network involving '{neuron}'",
                        neural_type='network'
                    ))
                    success = False
        
        return success
    
    def _validate_signal_compatibility(self) -> bool:
        """Validate signal type compatibility across connections."""
        success = True
        
        # Check signal type compatibility across synapses
        for synapse_name, synapse in self.parent.synapse_analyzer.synapses.items():
            source_neuron = self.parent.neuron_analyzer.neurons.get(synapse.source_neuron)
            target_neuron = self.parent.neuron_analyzer.neurons.get(synapse.target_neuron)
            
            if source_neuron and target_neuron:
                # Check output/input compatibility
                # This would require more detailed neuron type information
                pass
        
        return success
    
    def _validate_timing_constraints(self) -> bool:
        """Validate timing constraints and synchronization."""
        success = True
        
        # Check for timing conflicts in pulses
        pulses = self.parent.pulse_analyzer.pulses
        for i, pulse1 in enumerate(pulses):
            for pulse2 in pulses[i+1:]:
                if pulse1['neuron'] == pulse2['neuron']:
                    # Check for overlapping pulses
                    if abs(pulse1['frequency'] - pulse2['frequency']) < 0.01:
                        self.validation_errors.append(NeuralError(
                            f"Timing conflict in neuron '{pulse1['neuron']}'",
                            neural_type='timing'
                        ))
                        success = False
        
        return success


class NeuralFlowAnalyzer:
    """Analyzes signal flow and data propagation through neural networks."""
    
    def __init__(self, parent: NeuralAnalyzer):
        self.parent = parent
        self.flow_paths: List[List[str]] = []
        self.bottlenecks: List[str] = []
        self.dead_ends: List[str] = []
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze signal flow through the network."""
        self._find_flow_paths()
        self._identify_bottlenecks()
        self._find_dead_ends()
        
        return {
            'flow_paths': self.flow_paths,
            'bottlenecks': self.bottlenecks,
            'dead_ends': self.dead_ends,
            'network_depth': self._calculate_network_depth(),
            'connectivity_metrics': self._calculate_connectivity_metrics()
        }
    
    def _find_flow_paths(self):
        """Find all possible signal flow paths."""
        # Find input neurons (no incoming connections)
        all_neurons = set(self.parent.neuron_analyzer.neurons.keys())
        target_neurons = set()
        
        for targets in self.parent.neural_network.values():
            target_neurons.update(targets)
        
        input_neurons = all_neurons - target_neurons
        
        # Trace paths from each input neuron
        for input_neuron in input_neurons:
            paths = self._trace_paths_from(input_neuron)
            self.flow_paths.extend(paths)
    
    def _trace_paths_from(self, start_neuron: str, visited: Set[str] = None) -> List[List[str]]:
        """Trace all paths from a starting neuron."""
        if visited is None:
            visited = set()
        
        if start_neuron in visited:
            return []  # Cycle detected
        
        visited.add(start_neuron)
        paths = []
        
        targets = self.parent.neural_network.get(start_neuron, [])
        if not targets:
            # Dead end
            paths.append([start_neuron])
        else:
            for target in targets:
                sub_paths = self._trace_paths_from(target, visited.copy())
                for sub_path in sub_paths:
                    paths.append([start_neuron] + sub_path)
        
        return paths
    
    def _identify_bottlenecks(self):
        """Identify bottleneck neurons (high fan-in or fan-out)."""
        fan_in = {}
        fan_out = {}
        
        # Calculate fan-in and fan-out for each neuron
        for source, targets in self.parent.neural_network.items():
            fan_out[source] = len(targets)
            for target in targets:
                fan_in[target] = fan_in.get(target, 0) + 1
        
        # Identify bottlenecks
        for neuron in self.parent.neuron_analyzer.neurons:
            if fan_in.get(neuron, 0) > 5 or fan_out.get(neuron, 0) > 5:
                self.bottlenecks.append(neuron)
    
    def _find_dead_ends(self):
        """Find neurons with no outgoing connections."""
        all_neurons = set(self.parent.neuron_analyzer.neurons.keys())
        source_neurons = set(self.parent.neural_network.keys())
        
        self.dead_ends = list(all_neurons - source_neurons)
    
    def _calculate_network_depth(self) -> int:
        """Calculate maximum depth of the network."""
        if not self.flow_paths:
            return 0
        
        return max(len(path) for path in self.flow_paths)
    
    def _calculate_connectivity_metrics(self) -> Dict[str, float]:
        """Calculate network connectivity metrics."""
        num_neurons = len(self.parent.neuron_analyzer.neurons)
        num_connections = sum(len(targets) for targets in self.parent.neural_network.values())
        
        if num_neurons == 0:
            return {'density': 0.0, 'average_degree': 0.0}
        
        max_connections = num_neurons * (num_neurons - 1)
        density = num_connections / max_connections if max_connections > 0 else 0.0
        average_degree = num_connections / num_neurons
        
        return {
            'density': density,
            'average_degree': average_degree,
            'total_neurons': num_neurons,
            'total_connections': num_connections
        } 