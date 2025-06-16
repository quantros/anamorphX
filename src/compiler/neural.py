"""
Neural code generation for AnamorphX.

This module provides specialized code generation for neural network
constructs including neurons, synapses, signals, and pulses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto

from ..syntax.nodes import (
    ASTNode, NeuronDeclaration, SynapseDeclaration, SignalExpression,
    PulseStatement, Identifier, Expression, Statement
)
from .generator import GenerationContext
from .emitters import CodeEmitter


class NeuralArchitecture(Enum):
    """Neural network architectures."""
    
    FEEDFORWARD = auto()
    RECURRENT = auto()
    CONVOLUTIONAL = auto()
    TRANSFORMER = auto()
    CUSTOM = auto()


class ActivationFunction(Enum):
    """Neural activation functions."""
    
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"


class SignalType(Enum):
    """Types of neural signals."""
    
    SYNC = "sync"
    ASYNC = "async"
    PRIORITY = "priority"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class NeuronInfo:
    """Information about a neuron."""
    
    name: str
    neuron_type: str = "standard"
    activation_function: ActivationFunction = ActivationFunction.RELU
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'neuron_type': self.neuron_type,
            'activation_function': self.activation_function.value,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'parameters': self.parameters,
            'connections': self.connections
        }


@dataclass
class SynapseInfo:
    """Information about a synapse connection."""
    
    name: str
    source_neuron: str
    target_neuron: str
    weight: float = 1.0
    bias: float = 0.0
    connection_type: str = "dense"
    activation_function: Optional[ActivationFunction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source_neuron': self.source_neuron,
            'target_neuron': self.target_neuron,
            'weight': self.weight,
            'bias': self.bias,
            'connection_type': self.connection_type,
            'activation_function': self.activation_function.value if self.activation_function else None
        }


@dataclass
class SignalInfo:
    """Information about a neural signal."""
    
    name: str
    signal_type: SignalType
    data_type: str = "tensor"
    shape: Optional[Tuple[int, ...]] = None
    source: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'signal_type': self.signal_type.value,
            'data_type': self.data_type,
            'shape': list(self.shape) if self.shape else None,
            'source': self.source,
            'targets': self.targets
        }


@dataclass
class NeuralNetworkInfo:
    """Information about the entire neural network."""
    
    name: str
    architecture: NeuralArchitecture = NeuralArchitecture.CUSTOM
    neurons: Dict[str, NeuronInfo] = field(default_factory=dict)
    synapses: Dict[str, SynapseInfo] = field(default_factory=dict)
    signals: Dict[str, SignalInfo] = field(default_factory=dict)
    input_neurons: List[str] = field(default_factory=list)
    output_neurons: List[str] = field(default_factory=list)
    
    def add_neuron(self, neuron: NeuronInfo):
        """Add neuron to network."""
        self.neurons[neuron.name] = neuron
    
    def add_synapse(self, synapse: SynapseInfo):
        """Add synapse to network."""
        self.synapses[synapse.name] = synapse
        
        # Update neuron connections
        if synapse.source_neuron in self.neurons:
            self.neurons[synapse.source_neuron].connections.append(synapse.target_neuron)
    
    def add_signal(self, signal: SignalInfo):
        """Add signal to network."""
        self.signals[signal.name] = signal
    
    def get_network_topology(self) -> Dict[str, List[str]]:
        """Get network topology as adjacency list."""
        topology = {}
        
        for neuron_name in self.neurons:
            topology[neuron_name] = []
        
        for synapse in self.synapses.values():
            if synapse.source_neuron in topology:
                topology[synapse.source_neuron].append(synapse.target_neuron)
        
        return topology
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the neural network."""
        topology = self.get_network_topology()
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in topology.get(node, []):
                dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for neuron in self.neurons:
            if neuron not in visited:
                dfs(neuron, [neuron])
        
        return cycles
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'architecture': self.architecture.name,
            'neurons': {name: neuron.to_dict() for name, neuron in self.neurons.items()},
            'synapses': {name: synapse.to_dict() for name, synapse in self.synapses.items()},
            'signals': {name: signal.to_dict() for name, signal in self.signals.items()},
            'input_neurons': self.input_neurons,
            'output_neurons': self.output_neurons
        }


class NeuralCodeGenerator(ABC):
    """Abstract base class for neural code generators."""
    
    def __init__(self, context: GenerationContext):
        self.context = context
        self.network_info = NeuralNetworkInfo("main_network")
        self.emitter = self._create_emitter()
    
    @abstractmethod
    def _create_emitter(self) -> CodeEmitter:
        """Create appropriate code emitter."""
        pass
    
    @abstractmethod
    def generate_neural_runtime(self) -> str:
        """Generate neural runtime code."""
        pass
    
    @abstractmethod
    def generate_neuron_class(self, neuron_info: NeuronInfo) -> str:
        """Generate code for a neuron class."""
        pass
    
    @abstractmethod
    def generate_synapse_connection(self, synapse_info: SynapseInfo) -> str:
        """Generate code for a synapse connection."""
        pass
    
    @abstractmethod
    def generate_signal_processing(self, signal_info: SignalInfo) -> str:
        """Generate code for signal processing."""
        pass
    
    def analyze_neural_constructs(self, ast: ASTNode):
        """Analyze AST for neural constructs."""
        self._extract_neurons(ast)
        self._extract_synapses(ast)
        self._extract_signals(ast)
        self._analyze_network_topology()
    
    def _extract_neurons(self, ast: ASTNode):
        """Extract neuron declarations from AST."""
        if isinstance(ast, NeuronDeclaration):
            neuron_info = self._create_neuron_info(ast)
            self.network_info.add_neuron(neuron_info)
        
        # Recursively process child nodes
        for attr_name in dir(ast):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(ast, attr_name)
            
            if isinstance(attr_value, ASTNode):
                self._extract_neurons(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, ASTNode):
                        self._extract_neurons(item)
    
    def _extract_synapses(self, ast: ASTNode):
        """Extract synapse declarations from AST."""
        if isinstance(ast, SynapseDeclaration):
            for declarator in ast.declarations:
                synapse_info = self._create_synapse_info(declarator)
                self.network_info.add_synapse(synapse_info)
        
        # Recursively process child nodes
        for attr_name in dir(ast):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(ast, attr_name)
            
            if isinstance(attr_value, ASTNode):
                self._extract_synapses(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, ASTNode):
                        self._extract_synapses(item)
    
    def _extract_signals(self, ast: ASTNode):
        """Extract signal expressions from AST."""
        if isinstance(ast, SignalExpression):
            signal_info = self._create_signal_info(ast)
            self.network_info.add_signal(signal_info)
        
        # Recursively process child nodes
        for attr_name in dir(ast):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(ast, attr_name)
            
            if isinstance(attr_value, ASTNode):
                self._extract_signals(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, ASTNode):
                        self._extract_signals(item)
    
    def _create_neuron_info(self, neuron_decl: NeuronDeclaration) -> NeuronInfo:
        """Create neuron info from declaration."""
        neuron_info = NeuronInfo(name=neuron_decl.name.name)
        
        # Extract parameters
        for param in neuron_decl.parameters:
            param_name = param.name.name
            param_value = None
            
            if param.default_value:
                # Extract default value
                if hasattr(param.default_value, 'value'):
                    param_value = param.default_value.value
            
            neuron_info.parameters[param_name] = param_value
        
        # Determine neuron type and activation function from parameters
        if 'activation' in neuron_info.parameters:
            activation_name = neuron_info.parameters['activation']
            try:
                neuron_info.activation_function = ActivationFunction(activation_name)
            except ValueError:
                neuron_info.activation_function = ActivationFunction.RELU
        
        if 'input_size' in neuron_info.parameters:
            neuron_info.input_size = neuron_info.parameters['input_size']
        
        if 'output_size' in neuron_info.parameters:
            neuron_info.output_size = neuron_info.parameters['output_size']
        
        return neuron_info
    
    def _create_synapse_info(self, declarator) -> SynapseInfo:
        """Create synapse info from declarator."""
        synapse_name = declarator.id.name
        
        # Default synapse info
        synapse_info = SynapseInfo(
            name=synapse_name,
            source_neuron="unknown",
            target_neuron="unknown"
        )
        
        # Extract connection information from initializer
        if declarator.init:
            # This would parse the synapse initialization
            # For now, use placeholder values
            synapse_info.source_neuron = "input_neuron"
            synapse_info.target_neuron = "output_neuron"
            synapse_info.weight = 1.0
        
        return synapse_info
    
    def _create_signal_info(self, signal_expr: SignalExpression) -> SignalInfo:
        """Create signal info from expression."""
        signal_name = f"signal_{id(signal_expr)}"
        
        signal_type = SignalType.SYNC
        if signal_expr.signal_type == "async":
            signal_type = SignalType.ASYNC
        elif signal_expr.signal_type == "priority":
            signal_type = SignalType.PRIORITY
        elif signal_expr.signal_type == "streaming":
            signal_type = SignalType.STREAMING
        
        signal_info = SignalInfo(
            name=signal_name,
            signal_type=signal_type
        )
        
        # Extract target information
        if isinstance(signal_expr.target, Identifier):
            signal_info.targets.append(signal_expr.target.name)
        
        return signal_info
    
    def _analyze_network_topology(self):
        """Analyze network topology and detect issues."""
        # Detect cycles
        cycles = self.network_info.detect_cycles()
        if cycles:
            # Add warning about cycles
            for cycle in cycles:
                cycle_str = " -> ".join(cycle)
                self.context.error_handler.warnings.append(
                    f"Detected cycle in neural network: {cycle_str}"
                )
        
        # Identify input and output neurons
        topology = self.network_info.get_network_topology()
        
        for neuron_name in self.network_info.neurons:
            # Input neurons have no incoming connections
            has_incoming = any(neuron_name in connections for connections in topology.values())
            if not has_incoming:
                self.network_info.input_neurons.append(neuron_name)
            
            # Output neurons have no outgoing connections
            if not topology.get(neuron_name):
                self.network_info.output_neurons.append(neuron_name)
    
    def generate_network_code(self) -> str:
        """Generate complete neural network code."""
        code_sections = []
        
        # Neural runtime
        runtime_code = self.generate_neural_runtime()
        if runtime_code:
            code_sections.append(runtime_code)
        
        # Neuron classes
        for neuron_info in self.network_info.neurons.values():
            neuron_code = self.generate_neuron_class(neuron_info)
            if neuron_code:
                code_sections.append(neuron_code)
        
        # Synapse connections
        for synapse_info in self.network_info.synapses.values():
            synapse_code = self.generate_synapse_connection(synapse_info)
            if synapse_code:
                code_sections.append(synapse_code)
        
        # Signal processing
        for signal_info in self.network_info.signals.values():
            signal_code = self.generate_signal_processing(signal_info)
            if signal_code:
                code_sections.append(signal_code)
        
        # Network initialization
        init_code = self.generate_network_initialization()
        if init_code:
            code_sections.append(init_code)
        
        return "\n\n".join(code_sections)
    
    def generate_network_initialization(self) -> str:
        """Generate network initialization code."""
        lines = []
        
        lines.append("# Neural Network Initialization")
        lines.append("def initialize_network():")
        lines.append("    network = NeuralNetwork()")
        
        # Add neurons
        for neuron_name in self.network_info.neurons:
            lines.append(f"    {neuron_name}_instance = {neuron_name}()")
            lines.append(f"    network.add_neuron('{neuron_name}', {neuron_name}_instance)")
        
        # Add connections
        for synapse_info in self.network_info.synapses.values():
            lines.append(f"    network.connect('{synapse_info.source_neuron}', '{synapse_info.target_neuron}', weight={synapse_info.weight})")
        
        lines.append("    return network")
        
        return "\n".join(lines)


class NeuronGenerator:
    """Generator for individual neurons."""
    
    def __init__(self, context: GenerationContext):
        self.context = context
    
    def generate(self, neuron_info: NeuronInfo) -> str:
        """Generate code for a neuron."""
        if self.context.options.target_platform == "python":
            return self._generate_python_neuron(neuron_info)
        elif self.context.options.target_platform in ["javascript", "js"]:
            return self._generate_javascript_neuron(neuron_info)
        elif self.context.options.target_platform in ["cpp", "c++"]:
            return self._generate_cpp_neuron(neuron_info)
        else:
            return self._generate_python_neuron(neuron_info)
    
    def _generate_python_neuron(self, neuron_info: NeuronInfo) -> str:
        """Generate Python neuron code."""
        lines = []
        
        lines.append(f"class {neuron_info.name}(Neuron):")
        lines.append(f'    """Neural network neuron: {neuron_info.neuron_type}"""')
        lines.append("")
        lines.append("    def __init__(self):")
        lines.append("        super().__init__()")
        lines.append(f"        self.activation_function = '{neuron_info.activation_function.value}'")
        
        if neuron_info.input_size:
            lines.append(f"        self.input_size = {neuron_info.input_size}")
        
        if neuron_info.output_size:
            lines.append(f"        self.output_size = {neuron_info.output_size}")
        
        lines.append("")
        lines.append("    async def process_signal(self, signal: Signal) -> Signal:")
        lines.append("        # Apply activation function")
        lines.append(f"        activated_data = self.apply_activation(signal.data, '{neuron_info.activation_function.value}')")
        lines.append("        return Signal(data=activated_data, signal_type=signal.signal_type)")
        
        return "\n".join(lines)
    
    def _generate_javascript_neuron(self, neuron_info: NeuronInfo) -> str:
        """Generate JavaScript neuron code."""
        lines = []
        
        lines.append(f"class {neuron_info.name} extends Neuron {{")
        lines.append(f"    /**")
        lines.append(f"     * Neural network neuron: {neuron_info.neuron_type}")
        lines.append(f"     */")
        lines.append("")
        lines.append("    constructor() {")
        lines.append("        super();")
        lines.append(f"        this.activationFunction = '{neuron_info.activation_function.value}';")
        
        if neuron_info.input_size:
            lines.append(f"        this.inputSize = {neuron_info.input_size};")
        
        if neuron_info.output_size:
            lines.append(f"        this.outputSize = {neuron_info.output_size};")
        
        lines.append("    }")
        lines.append("")
        lines.append("    async processSignal(signal) {")
        lines.append("        // Apply activation function")
        lines.append(f"        const activatedData = this.applyActivation(signal.data, '{neuron_info.activation_function.value}');")
        lines.append("        return new Signal({ data: activatedData, signalType: signal.signalType });")
        lines.append("    }")
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_cpp_neuron(self, neuron_info: NeuronInfo) -> str:
        """Generate C++ neuron code."""
        lines = []
        
        lines.append(f"class {neuron_info.name} : public Neuron {{")
        lines.append("public:")
        lines.append(f"    {neuron_info.name}() {{")
        lines.append(f"        activation_function_ = ActivationFunction::{neuron_info.activation_function.name};")
        
        if neuron_info.input_size:
            lines.append(f"        input_size_ = {neuron_info.input_size};")
        
        if neuron_info.output_size:
            lines.append(f"        output_size_ = {neuron_info.output_size};")
        
        lines.append("    }")
        lines.append("")
        lines.append("    virtual std::future<Signal> processSignal(const Signal& signal) override {")
        lines.append("        // Apply activation function")
        lines.append("        auto activated_data = applyActivation(signal.data, activation_function_);")
        lines.append("        return std::async(std::launch::async, [=]() {")
        lines.append("            return Signal{activated_data, signal.signalType};")
        lines.append("        });")
        lines.append("")
        lines.append("private:")
        lines.append("    ActivationFunction activation_function_;")
        
        if neuron_info.input_size:
            lines.append(f"    size_t input_size_;")
        
        if neuron_info.output_size:
            lines.append(f"    size_t output_size_;")
        
        lines.append("};")
        
        return "\n".join(lines)


class SynapseGenerator:
    """Generator for synapses."""
    
    def __init__(self, context: GenerationContext):
        self.context = context
    
    def generate(self, synapse_info: SynapseInfo) -> str:
        """Generate code for a synapse."""
        if self.context.options.target_platform == "python":
            return self._generate_python_synapse(synapse_info)
        elif self.context.options.target_platform in ["javascript", "js"]:
            return self._generate_javascript_synapse(synapse_info)
        elif self.context.options.target_platform in ["cpp", "c++"]:
            return self._generate_cpp_synapse(synapse_info)
        else:
            return self._generate_python_synapse(synapse_info)
    
    def _generate_python_synapse(self, synapse_info: SynapseInfo) -> str:
        """Generate Python synapse code."""
        return f"""# Synapse: {synapse_info.source_neuron} -> {synapse_info.target_neuron}
{synapse_info.name} = Synapse(
    source='{synapse_info.source_neuron}',
    target='{synapse_info.target_neuron}',
    weight={synapse_info.weight},
    bias={synapse_info.bias},
    connection_type='{synapse_info.connection_type}'
)"""
    
    def _generate_javascript_synapse(self, synapse_info: SynapseInfo) -> str:
        """Generate JavaScript synapse code."""
        return f"""// Synapse: {synapse_info.source_neuron} -> {synapse_info.target_neuron}
const {synapse_info.name} = new Synapse({{
    source: '{synapse_info.source_neuron}',
    target: '{synapse_info.target_neuron}',
    weight: {synapse_info.weight},
    bias: {synapse_info.bias},
    connectionType: '{synapse_info.connection_type}'
}});"""
    
    def _generate_cpp_synapse(self, synapse_info: SynapseInfo) -> str:
        """Generate C++ synapse code."""
        return f"""// Synapse: {synapse_info.source_neuron} -> {synapse_info.target_neuron}
auto {synapse_info.name} = std::make_shared<Synapse>(
    "{synapse_info.source_neuron}",
    "{synapse_info.target_neuron}",
    {synapse_info.weight},
    {synapse_info.bias},
    ConnectionType::{synapse_info.connection_type.upper()}
);"""


class SignalGenerator:
    """Generator for signals."""
    
    def __init__(self, context: GenerationContext):
        self.context = context
    
    def generate(self, signal_info: SignalInfo) -> str:
        """Generate code for signal processing."""
        if self.context.options.target_platform == "python":
            return self._generate_python_signal(signal_info)
        elif self.context.options.target_platform in ["javascript", "js"]:
            return self._generate_javascript_signal(signal_info)
        elif self.context.options.target_platform in ["cpp", "c++"]:
            return self._generate_cpp_signal(signal_info)
        else:
            return self._generate_python_signal(signal_info)
    
    def _generate_python_signal(self, signal_info: SignalInfo) -> str:
        """Generate Python signal processing code."""
        lines = []
        
        lines.append(f"async def process_{signal_info.name}(data: Any) -> Signal:")
        lines.append(f'    """Process {signal_info.signal_type.value} signal."""')
        lines.append("    signal = Signal(")
        lines.append("        data=data,")
        lines.append(f"        signal_type='{signal_info.signal_type.value}',")
        lines.append("        timestamp=time.time()")
        lines.append("    )")
        lines.append("")
        
        if signal_info.signal_type == SignalType.ASYNC:
            lines.append("    # Asynchronous signal processing")
            lines.append("    await asyncio.sleep(0)  # Yield control")
        elif signal_info.signal_type == SignalType.PRIORITY:
            lines.append("    # Priority signal processing")
            lines.append("    signal.priority = 1")
        elif signal_info.signal_type == SignalType.STREAMING:
            lines.append("    # Streaming signal processing")
            lines.append("    signal.streaming = True")
        
        lines.append("    return signal")
        
        return "\n".join(lines)
    
    def _generate_javascript_signal(self, signal_info: SignalInfo) -> str:
        """Generate JavaScript signal processing code."""
        lines = []
        
        lines.append(f"async function process{signal_info.name.title()}(data) {{")
        lines.append(f"    /**")
        lines.append(f"     * Process {signal_info.signal_type.value} signal.")
        lines.append(f"     */")
        lines.append("    const signal = new Signal({")
        lines.append("        data: data,")
        lines.append(f"        signalType: '{signal_info.signal_type.value}',")
        lines.append("        timestamp: Date.now()")
        lines.append("    });")
        lines.append("")
        
        if signal_info.signal_type == SignalType.ASYNC:
            lines.append("    // Asynchronous signal processing")
            lines.append("    await new Promise(resolve => setTimeout(resolve, 0));")
        elif signal_info.signal_type == SignalType.PRIORITY:
            lines.append("    // Priority signal processing")
            lines.append("    signal.priority = 1;")
        elif signal_info.signal_type == SignalType.STREAMING:
            lines.append("    // Streaming signal processing")
            lines.append("    signal.streaming = true;")
        
        lines.append("    return signal;")
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_cpp_signal(self, signal_info: SignalInfo) -> str:
        """Generate C++ signal processing code."""
        lines = []
        
        lines.append(f"std::future<Signal> process_{signal_info.name}(const std::any& data) {{")
        lines.append(f"    /**")
        lines.append(f"     * Process {signal_info.signal_type.value} signal.")
        lines.append(f"     */")
        lines.append("    return std::async(std::launch::async, [=]() {")
        lines.append("        Signal signal{")
        lines.append("            data,")
        lines.append(f"            SignalType::{signal_info.signal_type.name},")
        lines.append("            std::chrono::system_clock::now()")
        lines.append("        };")
        lines.append("")
        
        if signal_info.signal_type == SignalType.ASYNC:
            lines.append("        // Asynchronous signal processing")
            lines.append("        std::this_thread::yield();")
        elif signal_info.signal_type == SignalType.PRIORITY:
            lines.append("        // Priority signal processing")
            lines.append("        signal.priority = 1;")
        elif signal_info.signal_type == SignalType.STREAMING:
            lines.append("        // Streaming signal processing")
            lines.append("        signal.streaming = true;")
        
        lines.append("        return signal;")
        lines.append("    });")
        lines.append("}")
        
        return "\n".join(lines)


class PulseGenerator:
    """Generator for pulses."""
    
    def __init__(self, context: GenerationContext):
        self.context = context
    
    def generate(self, pulse_stmt: PulseStatement) -> str:
        """Generate code for pulse generation."""
        if self.context.options.target_platform == "python":
            return self._generate_python_pulse(pulse_stmt)
        elif self.context.options.target_platform in ["javascript", "js"]:
            return self._generate_javascript_pulse(pulse_stmt)
        elif self.context.options.target_platform in ["cpp", "c++"]:
            return self._generate_cpp_pulse(pulse_stmt)
        else:
            return self._generate_python_pulse(pulse_stmt)
    
    def _generate_python_pulse(self, pulse_stmt: PulseStatement) -> str:
        """Generate Python pulse code."""
        lines = []
        
        lines.append("def generate_pulse():")
        lines.append('    """Generate neural pulse."""')
        lines.append("    pulse = Pulse(")
        lines.append("        amplitude=1.0,")
        lines.append("        frequency=1.0,")
        lines.append("        duration=0.1,")
        lines.append("        waveform='square'")
        lines.append("    )")
        
        if pulse_stmt.target:
            lines.append(f"    # Send pulse to target")
            lines.append(f"    pulse.send_to_target()")
        
        if pulse_stmt.condition:
            lines.append(f"    # Conditional pulse generation")
            lines.append(f"    if condition:")
            lines.append(f"        pulse.activate()")
        
        lines.append("    return pulse")
        
        return "\n".join(lines)
    
    def _generate_javascript_pulse(self, pulse_stmt: PulseStatement) -> str:
        """Generate JavaScript pulse code."""
        lines = []
        
        lines.append("function generatePulse() {")
        lines.append("    /**")
        lines.append("     * Generate neural pulse.")
        lines.append("     */")
        lines.append("    const pulse = new Pulse({")
        lines.append("        amplitude: 1.0,")
        lines.append("        frequency: 1.0,")
        lines.append("        duration: 0.1,")
        lines.append("        waveform: 'square'")
        lines.append("    });")
        
        if pulse_stmt.target:
            lines.append("    // Send pulse to target")
            lines.append("    pulse.sendToTarget();")
        
        if pulse_stmt.condition:
            lines.append("    // Conditional pulse generation")
            lines.append("    if (condition) {")
            lines.append("        pulse.activate();")
            lines.append("    }")
        
        lines.append("    return pulse;")
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_cpp_pulse(self, pulse_stmt: PulseStatement) -> str:
        """Generate C++ pulse code."""
        lines = []
        
        lines.append("Pulse generate_pulse() {")
        lines.append("    /**")
        lines.append("     * Generate neural pulse.")
        lines.append("     */")
        lines.append("    Pulse pulse{")
        lines.append("        1.0,  // amplitude")
        lines.append("        1.0,  // frequency")
        lines.append("        0.1,  // duration")
        lines.append("        WaveformType::SQUARE")
        lines.append("    };")
        
        if pulse_stmt.target:
            lines.append("    // Send pulse to target")
            lines.append("    pulse.send_to_target();")
        
        if pulse_stmt.condition:
            lines.append("    // Conditional pulse generation")
            lines.append("    if (condition) {")
            lines.append("        pulse.activate();")
            lines.append("    }")
        
        lines.append("    return pulse;")
        lines.append("}")
        
        return "\n".join(lines) 