"""
Scope Management for AnamorphX

This module provides comprehensive scope management for the Anamorph language,
including scope types, scope resolution, and scope validation.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from abc import ABC, abstractmethod
from ..syntax.nodes import ASTNode, SourceLocation
from .symbols import Symbol, SymbolTable, SymbolType, SymbolScope
from .errors import ScopeError, SemanticErrorType


class ScopeType(Enum):
    """Types of scopes in the language."""
    
    GLOBAL = auto()
    MODULE = auto()
    FUNCTION = auto()
    BLOCK = auto()
    NEURON = auto()
    CLASS = auto()
    LOOP = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()


@dataclass
class Scope(ABC):
    """Base class for all scopes."""
    
    name: str
    scope_type: ScopeType
    parent: Optional['Scope'] = None
    children: List['Scope'] = field(default_factory=list)
    symbol_table: Optional[SymbolTable] = None
    location: Optional[SourceLocation] = None
    level: int = 0
    
    def __post_init__(self):
        """Initialize scope after creation."""
        if self.parent:
            self.parent.children.append(self)
            self.level = self.parent.level + 1
        
        if not self.symbol_table:
            parent_table = self.parent.symbol_table if self.parent else None
            self.symbol_table = SymbolTable(self.name, parent_table)
    
    @abstractmethod
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Check if a symbol can be accessed from this scope."""
        pass
    
    @abstractmethod
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Check if a symbol can be declared in this scope."""
        pass
    
    def add_child(self, child: 'Scope'):
        """Add a child scope."""
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
    
    def find_symbol(self, name: str, recursive: bool = True) -> Optional[Symbol]:
        """Find a symbol in this scope or parent scopes."""
        return self.symbol_table.lookup(name, recursive)
    
    def declare_symbol(self, symbol: Symbol) -> bool:
        """Declare a symbol in this scope."""
        if not self.can_declare_symbol(symbol):
            return False
        return self.symbol_table.define(symbol)
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """Get all symbols accessible from this scope."""
        return self.symbol_table.get_all_symbols(include_children=True)
    
    def __str__(self) -> str:
        return f"{self.scope_type.name}({self.name})"


class GlobalScope(Scope):
    """Global scope containing module-level declarations."""
    
    def __init__(self, name: str = "global", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.GLOBAL, **kwargs)
        self.imports: Dict[str, str] = {}
        self.exports: Set[str] = set()
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Global scope can access all global symbols."""
        return symbol.scope in [SymbolScope.GLOBAL, SymbolScope.MODULE]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare most symbols in global scope."""
        return symbol.symbol_type not in [SymbolType.PARAMETER]
    
    def add_import(self, module_name: str, alias: str = None):
        """Add an import to this scope."""
        self.imports[alias or module_name] = module_name
    
    def add_export(self, symbol_name: str):
        """Add an export from this scope."""
        self.exports.add(symbol_name)


class FunctionScope(Scope):
    """Function scope containing parameters and local variables."""
    
    def __init__(self, name: str, return_type: str = None, **kwargs):
        super().__init__(name=name, scope_type=ScopeType.FUNCTION, **kwargs)
        self.return_type = return_type
        self.parameters: List[Symbol] = []
        self.local_variables: List[Symbol] = []
        self.has_return = False
        self.is_async = kwargs.get('is_async', False)
        self.is_generator = kwargs.get('is_generator', False)
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Function can access parameters, locals, and outer scope symbols."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE, 
            SymbolScope.FUNCTION, SymbolScope.BLOCK
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare variables and nested functions."""
        return symbol.symbol_type in [
            SymbolType.VARIABLE, SymbolType.FUNCTION, 
            SymbolType.PARAMETER, SymbolType.NEURON
        ]
    
    def add_parameter(self, param: Symbol):
        """Add a parameter to the function."""
        param.symbol_type = SymbolType.PARAMETER
        param.scope = SymbolScope.FUNCTION
        self.parameters.append(param)
        self.symbol_table.define(param)
    
    def add_local_variable(self, var: Symbol):
        """Add a local variable to the function."""
        self.local_variables.append(var)
        self.symbol_table.define(var)


class BlockScope(Scope):
    """Block scope for compound statements."""
    
    def __init__(self, name: str = "block", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.BLOCK, **kwargs)
        self.is_loop_body = kwargs.get('is_loop_body', False)
        self.is_try_block = kwargs.get('is_try_block', False)
        self.is_catch_block = kwargs.get('is_catch_block', False)
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Block can access symbols from outer scopes."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE,
            SymbolScope.FUNCTION, SymbolScope.BLOCK
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare variables in block scope."""
        return symbol.symbol_type in [
            SymbolType.VARIABLE, SymbolType.NEURON, SymbolType.SYNAPSE
        ]


class NeuronScope(Scope):
    """Scope for neuron declarations and neural networks."""
    
    def __init__(self, name: str, neuron_type: str = "basic", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.NEURON, **kwargs)
        self.neuron_type = neuron_type
        self.synapses: List[Symbol] = []
        self.signals: List[Symbol] = []
        self.state_variables: List[Symbol] = []
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Neuron can access neural-related symbols and outer scope."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE,
            SymbolScope.NEURON, SymbolScope.FUNCTION
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare neural-specific symbols."""
        return symbol.symbol_type in [
            SymbolType.SYNAPSE, SymbolType.SIGNAL, 
            SymbolType.PULSE, SymbolType.VARIABLE
        ]
    
    def add_synapse(self, synapse: Symbol):
        """Add a synapse to this neuron."""
        self.synapses.append(synapse)
        self.symbol_table.define(synapse)
    
    def add_signal(self, signal: Symbol):
        """Add a signal to this neuron."""
        self.signals.append(signal)
        self.symbol_table.define(signal)


class LoopScope(Scope):
    """Scope for loop constructs."""
    
    def __init__(self, name: str = "loop", loop_type: str = "for", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.LOOP, **kwargs)
        self.loop_type = loop_type  # for, while, do-while
        self.loop_variable: Optional[Symbol] = None
        self.break_allowed = True
        self.continue_allowed = True
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Loop can access outer scope symbols."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE,
            SymbolScope.FUNCTION, SymbolScope.BLOCK
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare loop variables."""
        return symbol.symbol_type in [SymbolType.VARIABLE]
    
    def set_loop_variable(self, var: Symbol):
        """Set the loop control variable."""
        self.loop_variable = var
        self.symbol_table.define(var)


class TryScope(Scope):
    """Scope for try-catch-finally blocks."""
    
    def __init__(self, name: str = "try", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.TRY, **kwargs)
        self.catch_blocks: List['CatchScope'] = []
        self.finally_block: Optional['FinallyScope'] = None
        self.exception_types: List[str] = []
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Try block can access outer scope symbols."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE,
            SymbolScope.FUNCTION, SymbolScope.BLOCK
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare variables in try block."""
        return symbol.symbol_type in [SymbolType.VARIABLE]
    
    def add_catch_block(self, catch_scope: 'CatchScope'):
        """Add a catch block to this try scope."""
        self.catch_blocks.append(catch_scope)
        catch_scope.parent = self


class CatchScope(Scope):
    """Scope for catch blocks."""
    
    def __init__(self, name: str = "catch", exception_type: str = "Exception", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.CATCH, **kwargs)
        self.exception_type = exception_type
        self.exception_variable: Optional[Symbol] = None
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Catch block can access outer scope and exception variable."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE,
            SymbolScope.FUNCTION, SymbolScope.BLOCK
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare variables in catch block."""
        return symbol.symbol_type in [SymbolType.VARIABLE]
    
    def set_exception_variable(self, var: Symbol):
        """Set the exception variable."""
        self.exception_variable = var
        self.symbol_table.define(var)


class FinallyScope(Scope):
    """Scope for finally blocks."""
    
    def __init__(self, name: str = "finally", **kwargs):
        super().__init__(name=name, scope_type=ScopeType.FINALLY, **kwargs)
    
    def can_access_symbol(self, symbol: Symbol) -> bool:
        """Finally block can access outer scope symbols."""
        return symbol.scope in [
            SymbolScope.GLOBAL, SymbolScope.MODULE,
            SymbolScope.FUNCTION, SymbolScope.BLOCK
        ]
    
    def can_declare_symbol(self, symbol: Symbol) -> bool:
        """Can declare variables in finally block."""
        return symbol.symbol_type in [SymbolType.VARIABLE]


class ScopeManager:
    """Manages scope hierarchy and resolution."""
    
    def __init__(self):
        self.global_scope = GlobalScope()
        self.current_scope = self.global_scope
        self.scope_stack: List[Scope] = [self.global_scope]
        self.scope_history: List[Scope] = []
    
    def enter_scope(self, scope: Scope) -> Scope:
        """Enter a new scope."""
        scope.parent = self.current_scope
        self.current_scope.add_child(scope)
        self.current_scope = scope
        self.scope_stack.append(scope)
        return scope
    
    def exit_scope(self) -> Optional[Scope]:
        """Exit current scope."""
        if len(self.scope_stack) <= 1:
            return None
        
        exited_scope = self.scope_stack.pop()
        self.scope_history.append(exited_scope)
        self.current_scope = self.scope_stack[-1]
        return exited_scope
    
    def create_function_scope(self, name: str, return_type: str = None) -> FunctionScope:
        """Create and enter a function scope."""
        scope = FunctionScope(name, return_type)
        return self.enter_scope(scope)
    
    def create_block_scope(self, name: str = "block") -> BlockScope:
        """Create and enter a block scope."""
        scope = BlockScope(name)
        return self.enter_scope(scope)
    
    def create_neuron_scope(self, name: str, neuron_type: str = "basic") -> NeuronScope:
        """Create and enter a neuron scope."""
        scope = NeuronScope(name, neuron_type)
        return self.enter_scope(scope)
    
    def create_loop_scope(self, name: str = "loop", loop_type: str = "for") -> LoopScope:
        """Create and enter a loop scope."""
        scope = LoopScope(name, loop_type)
        return self.enter_scope(scope)
    
    def create_try_scope(self, name: str = "try") -> TryScope:
        """Create and enter a try scope."""
        scope = TryScope(name)
        return self.enter_scope(scope)
    
    def find_symbol(self, name: str) -> Optional[Symbol]:
        """Find a symbol in current scope or parent scopes."""
        return self.current_scope.find_symbol(name)
    
    def declare_symbol(self, symbol: Symbol) -> bool:
        """Declare a symbol in current scope."""
        return self.current_scope.declare_symbol(symbol)
    
    def can_break(self) -> bool:
        """Check if break statement is allowed in current context."""
        for scope in reversed(self.scope_stack):
            if isinstance(scope, LoopScope):
                return scope.break_allowed
            elif isinstance(scope, FunctionScope):
                return False
        return False
    
    def can_continue(self) -> bool:
        """Check if continue statement is allowed in current context."""
        for scope in reversed(self.scope_stack):
            if isinstance(scope, LoopScope):
                return scope.continue_allowed
            elif isinstance(scope, FunctionScope):
                return False
        return False
    
    def find_enclosing_function(self) -> Optional[FunctionScope]:
        """Find the enclosing function scope."""
        for scope in reversed(self.scope_stack):
            if isinstance(scope, FunctionScope):
                return scope
        return None
    
    def find_enclosing_neuron(self) -> Optional[NeuronScope]:
        """Find the enclosing neuron scope."""
        for scope in reversed(self.scope_stack):
            if isinstance(scope, NeuronScope):
                return scope
        return None
    
    def find_enclosing_loop(self) -> Optional[LoopScope]:
        """Find the enclosing loop scope."""
        for scope in reversed(self.scope_stack):
            if isinstance(scope, LoopScope):
                return scope
        return None
    
    def get_scope_depth(self) -> int:
        """Get current scope depth."""
        return len(self.scope_stack) - 1
    
    def get_scope_path(self) -> List[str]:
        """Get path from global to current scope."""
        return [scope.name for scope in self.scope_stack]


class ScopeResolver:
    """Resolves scope-related issues and validates scope usage."""
    
    def __init__(self, scope_manager: ScopeManager):
        self.scope_manager = scope_manager
        self.errors: List[ScopeError] = []
    
    def resolve_identifier(self, name: str, location: SourceLocation = None) -> Optional[Symbol]:
        """Resolve an identifier to a symbol."""
        symbol = self.scope_manager.find_symbol(name)
        
        if not symbol:
            self.errors.append(ScopeError(
                f"Undefined identifier '{name}'",
                scope_name=self.scope_manager.current_scope.name,
                location=location
            ))
            return None
        
        # Check accessibility
        if not self.scope_manager.current_scope.can_access_symbol(symbol):
            self.errors.append(ScopeError(
                f"Symbol '{name}' is not accessible from current scope",
                scope_name=self.scope_manager.current_scope.name,
                location=location
            ))
            return None
        
        return symbol
    
    def validate_break_statement(self, location: SourceLocation = None) -> bool:
        """Validate break statement usage."""
        if not self.scope_manager.can_break():
            self.errors.append(ScopeError(
                "Break statement not allowed outside of loop",
                location=location
            ))
            return False
        return True
    
    def validate_continue_statement(self, location: SourceLocation = None) -> bool:
        """Validate continue statement usage."""
        if not self.scope_manager.can_continue():
            self.errors.append(ScopeError(
                "Continue statement not allowed outside of loop",
                location=location
            ))
            return False
        return True
    
    def validate_return_statement(self, location: SourceLocation = None) -> bool:
        """Validate return statement usage."""
        function_scope = self.scope_manager.find_enclosing_function()
        if not function_scope:
            self.errors.append(ScopeError(
                "Return statement not allowed outside of function",
                location=location
            ))
            return False
        
        function_scope.has_return = True
        return True
    
    def validate_neural_operation(self, operation: str, location: SourceLocation = None) -> bool:
        """Validate neural operation usage."""
        neuron_scope = self.scope_manager.find_enclosing_neuron()
        if not neuron_scope and operation in ['pulse', 'resonate']:
            self.errors.append(ScopeError(
                f"Neural operation '{operation}' not allowed outside of neuron",
                location=location
            ))
            return False
        return True


class ScopeValidator:
    """Validates scope-related semantic rules."""
    
    def __init__(self, scope_manager: ScopeManager):
        self.scope_manager = scope_manager
        self.errors: List[ScopeError] = []
    
    def validate_function_scope(self, function_scope: FunctionScope) -> bool:
        """Validate function scope rules."""
        valid = True
        
        # Check if non-void function has return statement
        if (function_scope.return_type and 
            function_scope.return_type != 'void' and 
            not function_scope.has_return):
            self.errors.append(ScopeError(
                f"Function '{function_scope.name}' must return a value",
                scope_name=function_scope.name
            ))
            valid = False
        
        return valid
    
    def validate_variable_usage(self, scope: Scope) -> bool:
        """Validate variable usage in scope."""
        valid = True
        
        # Check for unused variables
        for symbol in scope.symbol_table.get_unused_symbols():
            if symbol.symbol_type == SymbolType.VARIABLE:
                self.errors.append(ScopeError(
                    f"Variable '{symbol.name}' is declared but never used",
                    scope_name=scope.name,
                    location=symbol.location
                ))
        
        # Check for uninitialized variables
        for symbol in scope.symbol_table.get_uninitialized_variables():
            self.errors.append(ScopeError(
                f"Variable '{symbol.name}' is used before initialization",
                scope_name=scope.name,
                location=symbol.location
            ))
            valid = False
        
        return valid
    
    def validate_neural_scope(self, neuron_scope: NeuronScope) -> bool:
        """Validate neuron scope rules."""
        valid = True
        
        # Check for orphaned synapses
        for synapse in neuron_scope.synapses:
            if hasattr(synapse, 'source_neuron') and hasattr(synapse, 'target_neuron'):
                source_symbol = self.scope_manager.find_symbol(synapse.source_neuron)
                target_symbol = self.scope_manager.find_symbol(synapse.target_neuron)
                
                if not source_symbol or source_symbol.symbol_type != SymbolType.NEURON:
                    self.errors.append(ScopeError(
                        f"Synapse source neuron '{synapse.source_neuron}' not found",
                        scope_name=neuron_scope.name
                    ))
                    valid = False
                
                if not target_symbol or target_symbol.symbol_type != SymbolType.NEURON:
                    self.errors.append(ScopeError(
                        f"Synapse target neuron '{synapse.target_neuron}' not found",
                        scope_name=neuron_scope.name
                    ))
                    valid = False
        
        return valid
    
    def validate_all_scopes(self) -> bool:
        """Validate all scopes in the scope hierarchy."""
        valid = True
        
        def validate_scope_recursive(scope: Scope):
            nonlocal valid
            
            if isinstance(scope, FunctionScope):
                if not self.validate_function_scope(scope):
                    valid = False
            elif isinstance(scope, NeuronScope):
                if not self.validate_neural_scope(scope):
                    valid = False
            
            if not self.validate_variable_usage(scope):
                valid = False
            
            for child in scope.children:
                validate_scope_recursive(child)
        
        validate_scope_recursive(self.scope_manager.global_scope)
        return valid 