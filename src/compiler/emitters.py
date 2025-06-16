"""
Code emitters for different target platforms.

This module provides platform-specific code emitters that convert
AST nodes into target language code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import re

from ..syntax.nodes import (
    ASTNode, Program, FunctionDeclaration, NeuronDeclaration, SynapseDeclaration,
    VariableDeclaration, VariableDeclarator, ClassDeclaration,
    IntegerLiteral, FloatLiteral, StringLiteral, BooleanLiteral, ArrayLiteral,
    Identifier, BinaryExpression, UnaryExpression, CallExpression,
    IfStatement, WhileStatement, ForStatement, ReturnStatement,
    BlockStatement, ExpressionStatement, SignalExpression, PulseStatement,
    ImportStatement, ExportStatement, TypeAnnotation
)
from ..syntax.enums import BinaryOperator, UnaryOperator
from .generator import GenerationContext


class CodeEmitter(ABC):
    """Abstract base class for code emitters."""
    
    def __init__(self, context: GenerationContext):
        self.context = context
        self.target = context.target
        self.indent_level = 0
        self.indent_string = context.target.configuration.indent_style
        
    @abstractmethod
    def emit(self, node: ASTNode) -> str:
        """Emit code for AST node."""
        pass
    
    def indent(self) -> str:
        """Get current indentation string."""
        return self.indent_string * self.indent_level
    
    def increase_indent(self):
        """Increase indentation level."""
        self.indent_level += 1
    
    def decrease_indent(self):
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)
    
    def emit_list(self, nodes: List[ASTNode], separator: str = ", ") -> str:
        """Emit a list of nodes with separator."""
        return separator.join(self.emit(node) for node in nodes if node)
    
    def emit_block(self, statements: List[ASTNode], include_braces: bool = False) -> str:
        """Emit a block of statements."""
        if not statements:
            return ""
        
        lines = []
        
        if include_braces:
            lines.append(self.indent() + "{")
            self.increase_indent()
        
        for stmt in statements:
            if stmt:
                stmt_code = self.emit(stmt)
                if stmt_code:
                    lines.append(self.indent() + stmt_code)
        
        if include_braces:
            self.decrease_indent()
            lines.append(self.indent() + "}")
        
        return "\n".join(lines)
    
    def mangle_identifier(self, name: str) -> str:
        """Mangle identifier for target platform."""
        return self.target.mangle_identifier(name)
    
    def get_type_name(self, type_name: str) -> str:
        """Get target-specific type name."""
        type_mapping = self.target.get_builtin_types()
        return type_mapping.get(type_name, type_name)


class PythonEmitter(CodeEmitter):
    """Python code emitter."""
    
    def emit(self, node: ASTNode) -> str:
        """Emit Python code for AST node."""
        if not node:
            return ""
        
        # Dispatch to specific emit methods
        method_name = f"emit_{node.__class__.__name__.lower()}"
        emit_method = getattr(self, method_name, self.emit_generic)
        return emit_method(node)
    
    def emit_program(self, node: Program) -> str:
        """Emit Python program."""
        lines = []
        
        # Add imports
        for import_stmt in self.context.import_statements:
            lines.append(import_stmt)
        
        if self.context.import_statements:
            lines.append("")  # Empty line after imports
        
        # Emit body
        for stmt in node.body:
            if stmt:
                stmt_code = self.emit(stmt)
                if stmt_code:
                    lines.append(stmt_code)
                    lines.append("")  # Empty line between top-level statements
        
        return "\n".join(lines).rstrip()
    
    def emit_functiondeclaration(self, node: FunctionDeclaration) -> str:
        """Emit Python function declaration."""
        lines = []
        
        # Function signature
        params = []
        for param in node.parameters:
            param_str = param.name.name
            if param.type_annotation:
                param_str += f": {self.emit(param.type_annotation)}"
            if param.default_value:
                param_str += f" = {self.emit(param.default_value)}"
            params.append(param_str)
        
        signature = f"def {self.mangle_identifier(node.name.name)}({', '.join(params)})"
        
        if node.return_type:
            signature += f" -> {self.emit(node.return_type)}"
        
        signature += ":"
        
        if node.is_async:
            signature = "async " + signature
        
        lines.append(signature)
        
        # Function body
        self.increase_indent()
        if node.body and hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code)
        else:
            lines.append(self.indent() + "pass")
        self.decrease_indent()
        
        self.context.generated_functions.append(node.name.name)
        return "\n".join(lines)
    
    def emit_neurondeclaration(self, node: NeuronDeclaration) -> str:
        """Emit Python neuron class."""
        lines = []
        
        # Class declaration
        class_name = self.mangle_identifier(node.name.name)
        lines.append(f"class {class_name}(Neuron):")
        
        self.increase_indent()
        
        # Constructor
        params = []
        for param in node.parameters:
            param_str = param.name.name
            if param.type_annotation:
                param_str += f": {self.emit(param.type_annotation)}"
            if param.default_value:
                param_str += f" = {self.emit(param.default_value)}"
            params.append(param_str)
        
        constructor_params = "self" + (", " + ", ".join(params) if params else "")
        lines.append(self.indent() + f"def __init__({constructor_params}):")
        
        self.increase_indent()
        lines.append(self.indent() + "super().__init__()")
        
        # Constructor body
        if node.body and hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code)
        else:
            lines.append(self.indent() + "pass")
        
        self.decrease_indent()
        
        # Process signal method
        lines.append("")
        lines.append(self.indent() + "async def process_signal(self, signal: Signal) -> Signal:")
        self.increase_indent()
        lines.append(self.indent() + "# Neural signal processing logic")
        lines.append(self.indent() + "return signal")
        self.decrease_indent()
        
        self.decrease_indent()
        
        self.context.generated_neurons.append(node.name.name)
        return "\n".join(lines)
    
    def emit_synapsedeclaration(self, node: SynapseDeclaration) -> str:
        """Emit Python synapse connections."""
        lines = []
        
        for declarator in node.declarations:
            synapse_name = self.mangle_identifier(declarator.id.name)
            
            if declarator.init:
                init_code = self.emit(declarator.init)
                lines.append(f"{synapse_name} = {init_code}")
            else:
                lines.append(f"{synapse_name} = Synapse()")
        
        return "\n".join(lines)
    
    def emit_variabledeclaration(self, node: VariableDeclaration) -> str:
        """Emit Python variable declaration."""
        lines = []
        
        for declarator in node.declarations:
            var_name = self.mangle_identifier(declarator.id.name)
            
            if declarator.init:
                init_code = self.emit(declarator.init)
                
                # Add type annotation if available
                if declarator.type_annotation and self.context.options.include_type_annotations:
                    type_code = self.emit(declarator.type_annotation)
                    lines.append(f"{var_name}: {type_code} = {init_code}")
                else:
                    lines.append(f"{var_name} = {init_code}")
            else:
                # Declaration without initialization
                if declarator.type_annotation and self.context.options.include_type_annotations:
                    type_code = self.emit(declarator.type_annotation)
                    lines.append(f"{var_name}: {type_code}")
                else:
                    lines.append(f"{var_name} = None")
        
        return "\n".join(lines)
    
    def emit_integerliteral(self, node: IntegerLiteral) -> str:
        """Emit Python integer literal."""
        return str(node.value)
    
    def emit_floatliteral(self, node: FloatLiteral) -> str:
        """Emit Python float literal."""
        return str(node.value)
    
    def emit_stringliteral(self, node: StringLiteral) -> str:
        """Emit Python string literal."""
        # Escape string and wrap in quotes
        escaped = node.value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    
    def emit_booleanliteral(self, node: BooleanLiteral) -> str:
        """Emit Python boolean literal."""
        return "True" if node.value else "False"
    
    def emit_arrayliteral(self, node: ArrayLiteral) -> str:
        """Emit Python list literal."""
        elements = self.emit_list(node.elements)
        return f"[{elements}]"
    
    def emit_identifier(self, node: Identifier) -> str:
        """Emit Python identifier."""
        return self.mangle_identifier(node.name)
    
    def emit_binaryexpression(self, node: BinaryExpression) -> str:
        """Emit Python binary expression."""
        left = self.emit(node.left)
        right = self.emit(node.right)
        
        # Map operators
        operator_map = {
            BinaryOperator.ADD: "+",
            BinaryOperator.SUBTRACT: "-",
            BinaryOperator.MULTIPLY: "*",
            BinaryOperator.DIVIDE: "/",
            BinaryOperator.MODULO: "%",
            BinaryOperator.POWER: "**",
            BinaryOperator.EQUAL: "==",
            BinaryOperator.NOT_EQUAL: "!=",
            BinaryOperator.LESS_THAN: "<",
            BinaryOperator.LESS_EQUAL: "<=",
            BinaryOperator.GREATER_THAN: ">",
            BinaryOperator.GREATER_EQUAL: ">=",
            BinaryOperator.LOGICAL_AND: "and",
            BinaryOperator.LOGICAL_OR: "or",
            BinaryOperator.BITWISE_AND: "&",
            BinaryOperator.BITWISE_OR: "|",
            BinaryOperator.BITWISE_XOR: "^",
            BinaryOperator.LEFT_SHIFT: "<<",
            BinaryOperator.RIGHT_SHIFT: ">>",
        }
        
        op = operator_map.get(node.operator, str(node.operator.value))
        return f"({left} {op} {right})"
    
    def emit_unaryexpression(self, node: UnaryExpression) -> str:
        """Emit Python unary expression."""
        operand = self.emit(node.operand)
        
        operator_map = {
            UnaryOperator.PLUS: "+",
            UnaryOperator.MINUS: "-",
            UnaryOperator.LOGICAL_NOT: "not ",
            UnaryOperator.BITWISE_NOT: "~",
        }
        
        op = operator_map.get(node.operator, str(node.operator.value))
        return f"({op}{operand})"
    
    def emit_callexpression(self, node: CallExpression) -> str:
        """Emit Python function call."""
        callee = self.emit(node.callee)
        args = self.emit_list(node.arguments)
        return f"{callee}({args})"
    
    def emit_ifstatement(self, node: IfStatement) -> str:
        """Emit Python if statement."""
        lines = []
        
        test = self.emit(node.test)
        lines.append(f"if {test}:")
        
        self.increase_indent()
        if hasattr(node.consequent, 'statements'):
            for stmt in node.consequent.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code)
        else:
            stmt_code = self.emit(node.consequent)
            if stmt_code:
                lines.append(self.indent() + stmt_code)
            else:
                lines.append(self.indent() + "pass")
        self.decrease_indent()
        
        if node.alternate:
            lines.append("else:")
            self.increase_indent()
            if hasattr(node.alternate, 'statements'):
                for stmt in node.alternate.statements:
                    if stmt:
                        stmt_code = self.emit(stmt)
                        if stmt_code:
                            lines.append(self.indent() + stmt_code)
            else:
                stmt_code = self.emit(node.alternate)
                if stmt_code:
                    lines.append(self.indent() + stmt_code)
                else:
                    lines.append(self.indent() + "pass")
            self.decrease_indent()
        
        return "\n".join(lines)
    
    def emit_whilestatement(self, node: WhileStatement) -> str:
        """Emit Python while loop."""
        lines = []
        
        test = self.emit(node.test)
        lines.append(f"while {test}:")
        
        self.increase_indent()
        if hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code)
        else:
            stmt_code = self.emit(node.body)
            if stmt_code:
                lines.append(self.indent() + stmt_code)
            else:
                lines.append(self.indent() + "pass")
        self.decrease_indent()
        
        return "\n".join(lines)
    
    def emit_returnstatement(self, node: ReturnStatement) -> str:
        """Emit Python return statement."""
        if node.argument:
            return f"return {self.emit(node.argument)}"
        return "return"
    
    def emit_expressionstatement(self, node: ExpressionStatement) -> str:
        """Emit Python expression statement."""
        return self.emit(node.expression)
    
    def emit_signalexpression(self, node: SignalExpression) -> str:
        """Emit Python signal expression."""
        target = self.emit(node.target)
        signal_type = node.signal_type
        
        if signal_type == "async":
            return f"await send_signal_async({target})"
        else:
            return f"send_signal({target})"
    
    def emit_pulsestatement(self, node: PulseStatement) -> str:
        """Emit Python pulse statement."""
        lines = []
        
        if node.target:
            target = self.emit(node.target)
            lines.append(f"pulse_target = {target}")
        
        if node.condition:
            condition = self.emit(node.condition)
            lines.append(f"if {condition}:")
            self.increase_indent()
            lines.append(self.indent() + "generate_pulse(pulse_target)")
            self.decrease_indent()
        else:
            lines.append("generate_pulse()")
        
        return "\n".join(lines)
    
    def emit_typeannotation(self, node: TypeAnnotation) -> str:
        """Emit Python type annotation."""
        type_name = self.get_type_name(node.type_name)
        
        if node.generic_args:
            args = self.emit_list(node.generic_args)
            return f"{type_name}[{args}]"
        
        return type_name
    
    def emit_generic(self, node: ASTNode) -> str:
        """Generic emit method for unsupported nodes."""
        return f"# Unsupported node: {node.__class__.__name__}"


class JavaScriptEmitter(CodeEmitter):
    """JavaScript code emitter."""
    
    def emit(self, node: ASTNode) -> str:
        """Emit JavaScript code for AST node."""
        if not node:
            return ""
        
        method_name = f"emit_{node.__class__.__name__.lower()}"
        emit_method = getattr(self, method_name, self.emit_generic)
        return emit_method(node)
    
    def emit_program(self, node: Program) -> str:
        """Emit JavaScript program."""
        lines = []
        
        # Add imports
        for import_stmt in self.context.import_statements:
            lines.append(import_stmt)
        
        if self.context.import_statements:
            lines.append("")
        
        # Emit body
        for stmt in node.body:
            if stmt:
                stmt_code = self.emit(stmt)
                if stmt_code:
                    lines.append(stmt_code)
                    lines.append("")
        
        return "\n".join(lines).rstrip()
    
    def emit_functiondeclaration(self, node: FunctionDeclaration) -> str:
        """Emit JavaScript function declaration."""
        lines = []
        
        # Function signature
        params = [param.name.name for param in node.parameters]
        signature = f"function {self.mangle_identifier(node.name.name)}({', '.join(params)})"
        
        if node.is_async:
            signature = "async " + signature
        
        lines.append(signature + " {")
        
        # Function body
        self.increase_indent()
        if node.body and hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code + ";")
        self.decrease_indent()
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def emit_neurondeclaration(self, node: NeuronDeclaration) -> str:
        """Emit JavaScript neuron class."""
        lines = []
        
        class_name = self.mangle_identifier(node.name.name)
        lines.append(f"class {class_name} extends Neuron {{")
        
        self.increase_indent()
        
        # Constructor
        params = [param.name.name for param in node.parameters]
        lines.append(self.indent() + f"constructor({', '.join(params)}) {{")
        
        self.increase_indent()
        lines.append(self.indent() + "super();")
        
        if node.body and hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code + ";")
        
        self.decrease_indent()
        lines.append(self.indent() + "}")
        
        # Process signal method
        lines.append("")
        lines.append(self.indent() + "async processSignal(signal) {")
        self.increase_indent()
        lines.append(self.indent() + "// Neural signal processing logic")
        lines.append(self.indent() + "return signal;")
        self.decrease_indent()
        lines.append(self.indent() + "}")
        
        self.decrease_indent()
        lines.append("}")
        
        return "\n".join(lines)
    
    def emit_variabledeclaration(self, node: VariableDeclaration) -> str:
        """Emit JavaScript variable declaration."""
        lines = []
        
        for declarator in node.declarations:
            var_name = self.mangle_identifier(declarator.id.name)
            
            if declarator.init:
                init_code = self.emit(declarator.init)
                
                # Use const for constants, let for variables
                keyword = "const" if node.kind == "const" else "let"
                lines.append(f"{keyword} {var_name} = {init_code};")
            else:
                lines.append(f"let {var_name};")
        
        return "\n".join(lines)
    
    def emit_booleanliteral(self, node: BooleanLiteral) -> str:
        """Emit JavaScript boolean literal."""
        return "true" if node.value else "false"
    
    def emit_binaryexpression(self, node: BinaryExpression) -> str:
        """Emit JavaScript binary expression."""
        left = self.emit(node.left)
        right = self.emit(node.right)
        
        operator_map = {
            BinaryOperator.LOGICAL_AND: "&&",
            BinaryOperator.LOGICAL_OR: "||",
        }
        
        op = operator_map.get(node.operator, str(node.operator.value))
        return f"({left} {op} {right})"
    
    def emit_unaryexpression(self, node: UnaryExpression) -> str:
        """Emit JavaScript unary expression."""
        operand = self.emit(node.operand)
        
        operator_map = {
            UnaryOperator.LOGICAL_NOT: "!",
        }
        
        op = operator_map.get(node.operator, str(node.operator.value))
        return f"({op}{operand})"
    
    def emit_generic(self, node: ASTNode) -> str:
        """Generic emit method - delegate to Python emitter for common nodes."""
        python_emitter = PythonEmitter(self.context)
        return python_emitter.emit(node)


class CppEmitter(CodeEmitter):
    """C++ code emitter."""
    
    def emit(self, node: ASTNode) -> str:
        """Emit C++ code for AST node."""
        if not node:
            return ""
        
        method_name = f"emit_{node.__class__.__name__.lower()}"
        emit_method = getattr(self, method_name, self.emit_generic)
        return emit_method(node)
    
    def emit_program(self, node: Program) -> str:
        """Emit C++ program."""
        lines = []
        
        # Add includes
        for include in self.context.import_statements:
            lines.append(include)
        
        if self.context.import_statements:
            lines.append("")
        
        # Namespace
        lines.append("namespace anamorphx {")
        lines.append("")
        
        # Emit body
        self.increase_indent()
        for stmt in node.body:
            if stmt:
                stmt_code = self.emit(stmt)
                if stmt_code:
                    lines.append(self.indent() + stmt_code)
                    lines.append("")
        self.decrease_indent()
        
        lines.append("} // namespace anamorphx")
        
        return "\n".join(lines).rstrip()
    
    def emit_functiondeclaration(self, node: FunctionDeclaration) -> str:
        """Emit C++ function declaration."""
        lines = []
        
        # Return type
        return_type = "void"
        if node.return_type:
            return_type = self.emit(node.return_type)
        
        # Parameters
        params = []
        for param in node.parameters:
            param_type = "auto"
            if param.type_annotation:
                param_type = self.emit(param.type_annotation)
            
            param_str = f"{param_type} {param.name.name}"
            if param.default_value:
                param_str += f" = {self.emit(param.default_value)}"
            params.append(param_str)
        
        signature = f"{return_type} {self.mangle_identifier(node.name.name)}({', '.join(params)})"
        lines.append(signature + " {")
        
        # Function body
        self.increase_indent()
        if node.body and hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if stmt:
                    stmt_code = self.emit(stmt)
                    if stmt_code:
                        lines.append(self.indent() + stmt_code + ";")
        self.decrease_indent()
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def emit_neurondeclaration(self, node: NeuronDeclaration) -> str:
        """Emit C++ neuron class."""
        lines = []
        
        class_name = self.mangle_identifier(node.name.name)
        lines.append(f"class {class_name} : public Neuron {{")
        
        lines.append("public:")
        self.increase_indent()
        
        # Constructor
        params = []
        for param in node.parameters:
            param_type = "auto"
            if param.type_annotation:
                param_type = self.emit(param.type_annotation)
            params.append(f"{param_type} {param.name.name}")
        
        lines.append(self.indent() + f"{class_name}({', '.join(params)});")
        lines.append(self.indent() + f"virtual ~{class_name}();")
        lines.append("")
        lines.append(self.indent() + "virtual std::future<Signal> processSignal(const Signal& signal) override;")
        
        self.decrease_indent()
        lines.append("};")
        
        return "\n".join(lines)
    
    def emit_variabledeclaration(self, node: VariableDeclaration) -> str:
        """Emit C++ variable declaration."""
        lines = []
        
        for declarator in node.declarations:
            var_name = self.mangle_identifier(declarator.id.name)
            
            # Type
            var_type = "auto"
            if declarator.type_annotation:
                var_type = self.emit(declarator.type_annotation)
            
            if declarator.init:
                init_code = self.emit(declarator.init)
                lines.append(f"{var_type} {var_name} = {init_code};")
            else:
                lines.append(f"{var_type} {var_name};")
        
        return "\n".join(lines)
    
    def emit_booleanliteral(self, node: BooleanLiteral) -> str:
        """Emit C++ boolean literal."""
        return "true" if node.value else "false"
    
    def emit_generic(self, node: ASTNode) -> str:
        """Generic emit method - delegate to Python emitter for common nodes."""
        python_emitter = PythonEmitter(self.context)
        return python_emitter.emit(node)


class LLVMEmitter(CodeEmitter):
    """LLVM IR code emitter."""
    
    def emit(self, node: ASTNode) -> str:
        """Emit LLVM IR code for AST node."""
        if not node:
            return ""
        
        method_name = f"emit_{node.__class__.__name__.lower()}"
        emit_method = getattr(self, method_name, self.emit_generic)
        return emit_method(node)
    
    def emit_program(self, node: Program) -> str:
        """Emit LLVM IR program."""
        lines = []
        
        # Module header
        lines.append("; Generated AnamorphX LLVM IR")
        lines.append("")
        
        # Target triple
        lines.append('target triple = "x86_64-unknown-linux-gnu"')
        lines.append("")
        
        # Emit body
        for stmt in node.body:
            if stmt:
                stmt_code = self.emit(stmt)
                if stmt_code:
                    lines.append(stmt_code)
                    lines.append("")
        
        return "\n".join(lines).rstrip()
    
    def emit_functiondeclaration(self, node: FunctionDeclaration) -> str:
        """Emit LLVM IR function."""
        lines = []
        
        # Function signature
        return_type = "void"
        if node.return_type:
            return_type = self.get_type_name(node.return_type.type_name)
        
        params = []
        for param in node.parameters:
            param_type = "i8*"
            if param.type_annotation:
                param_type = self.get_type_name(param.type_annotation.type_name)
            params.append(f"{param_type} %{param.name.name}")
        
        func_name = self.mangle_identifier(node.name.name)
        signature = f"define {return_type} @{func_name}({', '.join(params)}) {{"
        lines.append(signature)
        
        # Function body (simplified)
        lines.append("entry:")
        if return_type == "void":
            lines.append("  ret void")
        else:
            lines.append(f"  ret {return_type} 0")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def emit_generic(self, node: ASTNode) -> str:
        """Generic emit method for LLVM IR."""
        return f"; Unsupported node: {node.__class__.__name__}" 