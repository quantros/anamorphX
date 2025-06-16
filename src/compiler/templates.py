"""
Template system for AnamorphX code generation.

This module provides a flexible template engine for generating code
in different target languages with support for neural constructs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from string import Template
import re
import json


@dataclass
class TemplateContext:
    """Context for template rendering."""
    
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Callable] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    target_platform: str = "python"
    optimization_level: int = 1
    
    def set_variable(self, name: str, value: Any):
        """Set a template variable."""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a template variable."""
        return self.variables.get(name, default)
    
    def add_import(self, import_statement: str):
        """Add an import statement."""
        if import_statement not in self.imports:
            self.imports.append(import_statement)
    
    def register_function(self, name: str, func: Callable):
        """Register a template function."""
        self.functions[name] = func


class CodeTemplate(ABC):
    """Abstract base class for code templates."""
    
    def __init__(self, name: str, template_string: str):
        self.name = name
        self.template_string = template_string
        self._compiled_template = None
    
    @abstractmethod
    def render(self, context: TemplateContext) -> str:
        """Render the template with given context."""
        pass
    
    def validate(self) -> List[str]:
        """Validate template syntax and return any errors."""
        errors = []
        
        # Check for basic syntax issues
        if not self.template_string.strip():
            errors.append("Template is empty")
        
        # Check for unmatched braces/brackets
        open_braces = self.template_string.count('{')
        close_braces = self.template_string.count('}')
        if open_braces != close_braces:
            errors.append(f"Unmatched braces: {open_braces} open, {close_braces} close")
        
        return errors


class SimpleTemplate(CodeTemplate):
    """Simple string-based template using Python's Template class."""
    
    def __init__(self, name: str, template_string: str):
        super().__init__(name, template_string)
        self._compiled_template = Template(template_string)
    
    def render(self, context: TemplateContext) -> str:
        """Render using Python's Template substitution."""
        try:
            # Prepare template variables
            template_vars = context.variables.copy()
            
            # Add built-in functions
            template_vars.update({
                'imports': '\n'.join(context.imports),
                'target': context.target_platform,
                'optimization_level': context.optimization_level,
            })
            
            return self._compiled_template.safe_substitute(template_vars)
        except Exception as e:
            raise TemplateRenderError(f"Failed to render template '{self.name}': {e}")


class AdvancedTemplate(CodeTemplate):
    """Advanced template with control structures and functions."""
    
    def __init__(self, name: str, template_string: str):
        super().__init__(name, template_string)
        self._parse_template()
    
    def _parse_template(self):
        """Parse template for advanced features."""
        # This would implement a more sophisticated template parser
        # For now, we'll use a simple approach
        self._blocks = self._extract_blocks()
        self._conditionals = self._extract_conditionals()
        self._loops = self._extract_loops()
    
    def _extract_blocks(self) -> List[Dict[str, Any]]:
        """Extract template blocks."""
        blocks = []
        # Pattern for {% block name %}...{% endblock %}
        pattern = r'\{%\s*block\s+(\w+)\s*%\}(.*?)\{%\s*endblock\s*%\}'
        matches = re.finditer(pattern, self.template_string, re.DOTALL)
        
        for match in matches:
            blocks.append({
                'name': match.group(1),
                'content': match.group(2).strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        return blocks
    
    def _extract_conditionals(self) -> List[Dict[str, Any]]:
        """Extract conditional statements."""
        conditionals = []
        # Pattern for {% if condition %}...{% endif %}
        pattern = r'\{%\s*if\s+(.+?)\s*%\}(.*?)\{%\s*endif\s*%\}'
        matches = re.finditer(pattern, self.template_string, re.DOTALL)
        
        for match in matches:
            conditionals.append({
                'condition': match.group(1).strip(),
                'content': match.group(2).strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        return conditionals
    
    def _extract_loops(self) -> List[Dict[str, Any]]:
        """Extract loop statements."""
        loops = []
        # Pattern for {% for item in items %}...{% endfor %}
        pattern = r'\{%\s*for\s+(\w+)\s+in\s+(.+?)\s*%\}(.*?)\{%\s*endfor\s*%\}'
        matches = re.finditer(pattern, self.template_string, re.DOTALL)
        
        for match in matches:
            loops.append({
                'variable': match.group(1),
                'iterable': match.group(2).strip(),
                'content': match.group(3).strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        return loops
    
    def render(self, context: TemplateContext) -> str:
        """Render advanced template."""
        result = self.template_string
        
        # Process loops first (innermost to outermost)
        for loop in reversed(self._loops):
            result = self._render_loop(result, loop, context)
        
        # Process conditionals
        for conditional in reversed(self._conditionals):
            result = self._render_conditional(result, conditional, context)
        
        # Process variable substitutions
        result = self._render_variables(result, context)
        
        return result
    
    def _render_loop(self, template: str, loop: Dict[str, Any], context: TemplateContext) -> str:
        """Render a loop construct."""
        try:
            # Get the iterable from context
            iterable_name = loop['iterable']
            iterable = context.get_variable(iterable_name, [])
            
            if not isinstance(iterable, (list, tuple)):
                iterable = [iterable]
            
            # Render loop content for each item
            loop_content = []
            for item in iterable:
                # Create new context with loop variable
                loop_context = TemplateContext(
                    variables=context.variables.copy(),
                    functions=context.functions,
                    imports=context.imports,
                    target_platform=context.target_platform,
                    optimization_level=context.optimization_level
                )
                loop_context.set_variable(loop['variable'], item)
                
                # Render loop body
                rendered_content = self._render_variables(loop['content'], loop_context)
                loop_content.append(rendered_content)
            
            # Replace loop construct with rendered content
            loop_pattern = re.escape(template[loop['start']:loop['end']])
            return re.sub(loop_pattern, '\n'.join(loop_content), template, count=1)
            
        except Exception as e:
            raise TemplateRenderError(f"Failed to render loop: {e}")
    
    def _render_conditional(self, template: str, conditional: Dict[str, Any], context: TemplateContext) -> str:
        """Render a conditional construct."""
        try:
            # Evaluate condition
            condition = conditional['condition']
            condition_result = self._evaluate_condition(condition, context)
            
            # Replace conditional with content if true, empty string if false
            replacement = conditional['content'] if condition_result else ''
            
            conditional_pattern = re.escape(template[conditional['start']:conditional['end']])
            return re.sub(conditional_pattern, replacement, template, count=1)
            
        except Exception as e:
            raise TemplateRenderError(f"Failed to render conditional: {e}")
    
    def _evaluate_condition(self, condition: str, context: TemplateContext) -> bool:
        """Evaluate a template condition."""
        # Simple condition evaluation
        # In a real implementation, this would be more sophisticated
        
        # Handle simple variable checks
        if condition in context.variables:
            value = context.variables[condition]
            return bool(value)
        
        # Handle comparisons
        if ' == ' in condition:
            left, right = condition.split(' == ', 1)
            left_val = context.get_variable(left.strip(), left.strip())
            right_val = right.strip().strip('"\'')
            return str(left_val) == right_val
        
        if ' != ' in condition:
            left, right = condition.split(' != ', 1)
            left_val = context.get_variable(left.strip(), left.strip())
            right_val = right.strip().strip('"\'')
            return str(left_val) != right_val
        
        # Default to False for unknown conditions
        return False
    
    def _render_variables(self, template: str, context: TemplateContext) -> str:
        """Render variable substitutions."""
        # Pattern for {{ variable }}
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        
        def replace_var(match):
            var_name = match.group(1).strip()
            
            # Handle function calls
            if '(' in var_name and ')' in var_name:
                return self._call_template_function(var_name, context)
            
            # Handle simple variables
            return str(context.get_variable(var_name, f'{{{{ {var_name} }}}}'))
        
        return re.sub(pattern, replace_var, template)
    
    def _call_template_function(self, func_call: str, context: TemplateContext) -> str:
        """Call a template function."""
        # Parse function call
        func_match = re.match(r'(\w+)\((.*?)\)', func_call)
        if not func_match:
            return func_call
        
        func_name = func_match.group(1)
        args_str = func_match.group(2)
        
        # Get function from context
        if func_name not in context.functions:
            return func_call
        
        func = context.functions[func_name]
        
        # Parse arguments (simple implementation)
        args = []
        if args_str.strip():
            for arg in args_str.split(','):
                arg = arg.strip().strip('"\'')
                # Try to get from context, otherwise use as literal
                args.append(context.get_variable(arg, arg))
        
        try:
            result = func(*args)
            return str(result)
        except Exception:
            return func_call


class TemplateRenderError(Exception):
    """Error during template rendering."""
    pass


class TemplateEngine:
    """Template engine for managing and rendering templates."""
    
    def __init__(self):
        self.templates: Dict[str, CodeTemplate] = {}
        self.template_paths: List[str] = []
        self.default_context = TemplateContext()
    
    def register_template(self, template: CodeTemplate):
        """Register a template."""
        self.templates[template.name] = template
    
    def load_template(self, name: str, template_string: str, advanced: bool = False) -> CodeTemplate:
        """Load a template from string."""
        if advanced:
            template = AdvancedTemplate(name, template_string)
        else:
            template = SimpleTemplate(name, template_string)
        
        self.register_template(template)
        return template
    
    def get_template(self, name: str) -> Optional[CodeTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def render_template(self, name: str, context: Optional[TemplateContext] = None) -> str:
        """Render a template with context."""
        template = self.get_template(name)
        if not template:
            raise TemplateRenderError(f"Template '{name}' not found")
        
        if context is None:
            context = self.default_context
        
        return template.render(context)
    
    def list_templates(self) -> List[str]:
        """List all registered templates."""
        return list(self.templates.keys())
    
    def validate_all_templates(self) -> Dict[str, List[str]]:
        """Validate all templates and return errors."""
        errors = {}
        for name, template in self.templates.items():
            template_errors = template.validate()
            if template_errors:
                errors[name] = template_errors
        return errors


class PythonTemplates:
    """Python-specific code templates."""
    
    @staticmethod
    def get_templates() -> Dict[str, str]:
        """Get Python template definitions."""
        return {
            'module_header': '''"""
Generated AnamorphX module.
Target: Python $target_version
Generated at: $generation_time
"""

$imports

''',
            
            'class_definition': '''class $class_name:
    """$class_docstring"""
    
    def __init__(self$init_params):
        $init_body
    
$methods
''',
            
            'function_definition': '''def $function_name($parameters)$return_type:
    """$function_docstring"""
    $function_body
''',
            
            'neuron_class': '''class $neuron_name(Neuron):
    """Neural network neuron: $neuron_description"""
    
    def __init__(self, $neuron_params):
        super().__init__()
        $neuron_init_body
    
    async def process_signal(self, signal: Signal) -> Signal:
        """Process incoming signal."""
        $signal_processing_body
    
    def connect_synapse(self, target: 'Neuron', weight: float = 1.0):
        """Connect to another neuron via synapse."""
        synapse = Synapse(self, target, weight)
        self.add_synapse(synapse)
        return synapse
''',
            
            'synapse_definition': '''# Synapse connection: $source_neuron -> $target_neuron
$synapse_name = Synapse(
    source=$source_neuron,
    target=$target_neuron,
    weight=$synapse_weight,
    activation_function=$activation_function
)
''',
            
            'signal_processing': '''async def process_$signal_name(self, data: Any) -> Signal:
    """Process $signal_name signal."""
    signal = Signal(
        data=data,
        signal_type="$signal_type",
        timestamp=time.time()
    )
    
    $signal_processing_logic
    
    return signal
''',
            
            'pulse_generation': '''def generate_$pulse_name(self, $pulse_params) -> Pulse:
    """Generate $pulse_name pulse."""
    return Pulse(
        amplitude=$pulse_amplitude,
        frequency=$pulse_frequency,
        duration=$pulse_duration,
        waveform="$pulse_waveform"
    )
''',
        }


class JavaScriptTemplates:
    """JavaScript-specific code templates."""
    
    @staticmethod
    def get_templates() -> Dict[str, str]:
        """Get JavaScript template definitions."""
        return {
            'module_header': '''/**
 * Generated AnamorphX module.
 * Target: JavaScript $target_version
 * Generated at: $generation_time
 */

$imports

''',
            
            'class_definition': '''class $class_name {
    /**
     * $class_docstring
     */
    
    constructor($constructor_params) {
        $constructor_body
    }
    
$methods
}
''',
            
            'function_definition': '''function $function_name($parameters) {
    /**
     * $function_docstring
     */
    $function_body
}
''',
            
            'neuron_class': '''class $neuron_name extends Neuron {
    /**
     * Neural network neuron: $neuron_description
     */
    
    constructor($neuron_params) {
        super();
        $neuron_init_body
    }
    
    async processSignal(signal) {
        /**
         * Process incoming signal.
         */
        $signal_processing_body
    }
    
    connectSynapse(target, weight = 1.0) {
        /**
         * Connect to another neuron via synapse.
         */
        const synapse = new Synapse(this, target, weight);
        this.addSynapse(synapse);
        return synapse;
    }
}
''',
            
            'signal_processing': '''async function process$signal_name(data) {
    /**
     * Process $signal_name signal.
     */
    const signal = new Signal({
        data: data,
        signalType: "$signal_type",
        timestamp: Date.now()
    });
    
    $signal_processing_logic
    
    return signal;
}
''',
        }


class CppTemplates:
    """C++-specific code templates."""
    
    @staticmethod
    def get_templates() -> Dict[str, str]:
        """Get C++ template definitions."""
        return {
            'module_header': '''/**
 * Generated AnamorphX module.
 * Target: C++ $target_version
 * Generated at: $generation_time
 */

$includes

namespace anamorphx {

''',
            
            'class_definition': '''class $class_name {
public:
    /**
     * $class_docstring
     */
    
    $class_name($constructor_params);
    ~$class_name();
    
$public_methods

private:
$private_members
$private_methods
};
''',
            
            'function_definition': '''$return_type $function_name($parameters) {
    /**
     * $function_docstring
     */
    $function_body
}
''',
            
            'neuron_class': '''class $neuron_name : public Neuron {
public:
    /**
     * Neural network neuron: $neuron_description
     */
    
    $neuron_name($neuron_params);
    virtual ~$neuron_name();
    
    virtual std::future<Signal> processSignal(const Signal& signal) override;
    virtual void connectSynapse(std::shared_ptr<Neuron> target, double weight = 1.0);
    
private:
    $neuron_private_members
};
''',
        }


class TemplateRegistry:
    """Registry for managing template collections."""
    
    def __init__(self):
        self.engines: Dict[str, TemplateEngine] = {}
        self._register_builtin_templates()
    
    def _register_builtin_templates(self):
        """Register built-in template collections."""
        # Python templates
        python_engine = TemplateEngine()
        for name, template_str in PythonTemplates.get_templates().items():
            python_engine.load_template(name, template_str, advanced=True)
        self.engines['python'] = python_engine
        
        # JavaScript templates
        js_engine = TemplateEngine()
        for name, template_str in JavaScriptTemplates.get_templates().items():
            js_engine.load_template(name, template_str, advanced=True)
        self.engines['javascript'] = js_engine
        
        # C++ templates
        cpp_engine = TemplateEngine()
        for name, template_str in CppTemplates.get_templates().items():
            cpp_engine.load_template(name, template_str, advanced=True)
        self.engines['cpp'] = cpp_engine
    
    def get_engine(self, target: str) -> Optional[TemplateEngine]:
        """Get template engine for target platform."""
        return self.engines.get(target.lower())
    
    def register_engine(self, target: str, engine: TemplateEngine):
        """Register a template engine for a target."""
        self.engines[target.lower()] = engine
    
    def list_targets(self) -> List[str]:
        """List all registered template targets."""
        return list(self.engines.keys())


# Global template registry
template_registry = TemplateRegistry() 