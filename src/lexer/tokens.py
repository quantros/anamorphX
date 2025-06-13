"""
Token definitions for Anamorph Neural Programming Language.

This module defines all token types, neural commands, and lexical elements
used by the Anamorph lexer.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional, Dict, Set
import re


class TokenType(Enum):
    """Enumeration of all token types in Anamorph language."""
    
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers
    IDENTIFIER = auto()
    
    # Neural Commands (80 commands)
    NEURO = auto()          # Define neuron (function)
    SYNAP = auto()          # Define synapse (variable)
    PULSE = auto()          # Send signal
    RESONATE = auto()       # Activate resonance (loops/events)
    DRIFT = auto()          # Shift state or context
    BIND = auto()           # Bind signal to action
    ECHO = auto()           # Send back signal
    FORGE = auto()          # Create new node
    PRUNE = auto()          # Remove node or connection
    FILTER = auto()         # Filter signals by condition
    GUARD = auto()          # Activate protection mode
    MASK = auto()           # Mask signal
    SCRAMBLE = auto()       # Scramble signal flow
    TRACE = auto()          # Log event
    QUANTA = auto()         # Manage data quantization
    PHASE = auto()          # Switch processing phase
    SYNC = auto()           # Synchronize nodes
    ASYNC = auto()          # Asynchronous activation
    FOLD = auto()           # Fold signal stream
    UNFOLD = auto()         # Unfold signal stream
    PULSEX = auto()         # Multi-pulse for simultaneous signals
    REFLECT = auto()        # Reflect signal on source node
    ABSORB = auto()         # Absorb signal
    DIFFUSE = auto()        # Diffuse signal across channels
    CLUSTER = auto()        # Group nodes into cluster
    EXPAND = auto()         # Expand cluster
    CONTRACT = auto()       # Contract cluster
    ENCODE = auto()         # Encode data
    DECODE = auto()         # Decode data
    MERGE = auto()          # Merge data from nodes
    SPLIT = auto()          # Split data into parts
    LOOP = auto()           # Start loop (cycle)
    HALT = auto()           # Stop node execution
    YIELD = auto()          # Yield control to another node
    SPAWN = auto()          # Create sub-node
    TAG = auto()            # Label (node identifier)
    QUERY = auto()          # Query node state
    RESPONSE = auto()       # Respond to query
    ENCRYPT = auto()        # Encrypt data
    DECRYPT = auto()        # Decrypt data
    CHECKPOINT = auto()     # Create state checkpoint
    ROLLBACK = auto()       # Rollback to checkpoint
    PULSEIF = auto()        # Conditional pulse
    WAIT = auto()           # Wait for signal or event
    TIME = auto()           # Track execution time
    JUMP = auto()           # Jump to label
    STACK = auto()          # Manage call stack
    POP = auto()            # Pop from stack
    PUSH = auto()           # Push to stack
    FLAG = auto()           # Set flag
    CLEARFLAG = auto()      # Clear flag
    TOGGLE = auto()         # Toggle state
    LISTEN = auto()         # Listen for external signal
    BROADCAST = auto()      # Broadcast signal to all channels
    FILTERIN = auto()       # Incoming filtration
    FILTEROUT = auto()      # Outgoing filtration
    AUTH = auto()           # Authorization check
    AUDIT = auto()          # Audit action recording
    THROTTLE = auto()       # Limit event frequency
    BAN = auto()            # Block address or node
    WHITELIST = auto()      # Allow access
    BLACKLIST = auto()      # Deny access
    MORPH = auto()          # Change node structure
    EVOLVE = auto()         # Node adaptation and learning
    SENSE = auto()          # Get sensor data
    ACT = auto()            # Perform action
    LOG = auto()            # Write to log
    ALERT = auto()          # Generate alert
    RESET = auto()          # Full state reset
    PATTERN = auto()        # Define behavior pattern
    TRAIN = auto()          # Train model or node
    INFER = auto()          # Perform inference (prediction)
    SCALEUP = auto()        # Scale up resources
    SCALEDOWN = auto()      # Scale down resources
    BACKUP = auto()         # Create backup
    RESTORE = auto()        # Restore from backup
    SNAPSHOT = auto()       # Create state snapshot
    MIGRATE = auto()        # Migrate nodes or data
    NOTIFY = auto()         # Send notification
    VALIDATE = auto()       # Validate input data
    
    # Operators
    ASSIGN = auto()         # =
    PLUS = auto()           # +
    MINUS = auto()          # -
    MULTIPLY = auto()       # *
    DIVIDE = auto()         # /
    MODULO = auto()         # %
    POWER = auto()          # **
    
    # Comparison
    EQUAL = auto()          # ==
    NOT_EQUAL = auto()      # !=
    LESS_THAN = auto()      # <
    LESS_EQUAL = auto()     # <=
    GREATER_THAN = auto()   # >
    GREATER_EQUAL = auto()  # >=
    
    # Logical
    AND = auto()            # &&
    OR = auto()             # ||
    NOT = auto()            # !
    
    # Bitwise
    BIT_AND = auto()        # &
    BIT_OR = auto()         # |
    BIT_XOR = auto()        # ^
    BIT_NOT = auto()        # ~
    LEFT_SHIFT = auto()     # <<
    RIGHT_SHIFT = auto()    # >>
    
    # Delimiters
    LEFT_PAREN = auto()     # (
    RIGHT_PAREN = auto()    # )
    LEFT_BRACE = auto()     # {
    RIGHT_BRACE = auto()    # }
    LEFT_BRACKET = auto()   # [
    RIGHT_BRACKET = auto()  # ]
    SEMICOLON = auto()      # ;
    COMMA = auto()          # ,
    DOT = auto()            # .
    COLON = auto()          # :
    ARROW = auto()          # ->
    DOUBLE_ARROW = auto()   # =>
    
    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    
    # Keywords
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()
    THROW = auto()
    
    # Types
    INT_TYPE = auto()
    FLOAT_TYPE = auto()
    STRING_TYPE = auto()
    BOOL_TYPE = auto()
    ARRAY_TYPE = auto()
    OBJECT_TYPE = auto()
    SIGNAL_TYPE = auto()
    NEURON_TYPE = auto()


@dataclass
class Token:
    """Represents a single token in the source code."""
    
    type: TokenType
    value: Any
    line: int
    column: int
    position: int
    length: int
    source_file: Optional[str] = None
    
    def __str__(self) -> str:
        return f"Token({self.type.name}, {repr(self.value)}, {self.line}:{self.column})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Neural commands mapping (case-insensitive)
NEURAL_COMMANDS: Dict[str, TokenType] = {
    # Core structural commands
    'neuro': TokenType.NEURO,
    'synap': TokenType.SYNAP,
    'pulse': TokenType.PULSE,
    'resonate': TokenType.RESONATE,
    'drift': TokenType.DRIFT,
    'bind': TokenType.BIND,
    'echo': TokenType.ECHO,
    'forge': TokenType.FORGE,
    'prune': TokenType.PRUNE,
    
    # Security and filtering
    'filter': TokenType.FILTER,
    'guard': TokenType.GUARD,
    'mask': TokenType.MASK,
    'scramble': TokenType.SCRAMBLE,
    'encrypt': TokenType.ENCRYPT,
    'decrypt': TokenType.DECRYPT,
    'auth': TokenType.AUTH,
    'audit': TokenType.AUDIT,
    'throttle': TokenType.THROTTLE,
    'ban': TokenType.BAN,
    'whitelist': TokenType.WHITELIST,
    'blacklist': TokenType.BLACKLIST,
    'validate': TokenType.VALIDATE,
    
    # Flow control
    'trace': TokenType.TRACE,
    'quanta': TokenType.QUANTA,
    'phase': TokenType.PHASE,
    'sync': TokenType.SYNC,
    'async': TokenType.ASYNC,
    'fold': TokenType.FOLD,
    'unfold': TokenType.UNFOLD,
    'pulsex': TokenType.PULSEX,
    'pulseif': TokenType.PULSEIF,
    'loop': TokenType.LOOP,
    'halt': TokenType.HALT,
    'yield': TokenType.YIELD,
    'wait': TokenType.WAIT,
    'jump': TokenType.JUMP,
    
    # Network and communication
    'reflect': TokenType.REFLECT,
    'absorb': TokenType.ABSORB,
    'diffuse': TokenType.DIFFUSE,
    'listen': TokenType.LISTEN,
    'broadcast': TokenType.BROADCAST,
    'filterin': TokenType.FILTERIN,
    'filterout': TokenType.FILTEROUT,
    'notify': TokenType.NOTIFY,
    
    # Data manipulation
    'encode': TokenType.ENCODE,
    'decode': TokenType.DECODE,
    'merge': TokenType.MERGE,
    'split': TokenType.SPLIT,
    'morph': TokenType.MORPH,
    
    # Clustering and scaling
    'cluster': TokenType.CLUSTER,
    'expand': TokenType.EXPAND,
    'contract': TokenType.CONTRACT,
    'scaleup': TokenType.SCALEUP,
    'scaledown': TokenType.SCALEDOWN,
    
    # Node management
    'spawn': TokenType.SPAWN,
    'tag': TokenType.TAG,
    'query': TokenType.QUERY,
    'response': TokenType.RESPONSE,
    
    # State management
    'checkpoint': TokenType.CHECKPOINT,
    'rollback': TokenType.ROLLBACK,
    'backup': TokenType.BACKUP,
    'restore': TokenType.RESTORE,
    'snapshot': TokenType.SNAPSHOT,
    'reset': TokenType.RESET,
    'migrate': TokenType.MIGRATE,
    
    # Stack operations
    'stack': TokenType.STACK,
    'pop': TokenType.POP,
    'push': TokenType.PUSH,
    
    # Flags and state
    'flag': TokenType.FLAG,
    'clearflag': TokenType.CLEARFLAG,
    'toggle': TokenType.TOGGLE,
    
    # Machine learning
    'evolve': TokenType.EVOLVE,
    'pattern': TokenType.PATTERN,
    'train': TokenType.TRAIN,
    'infer': TokenType.INFER,
    
    # I/O and monitoring
    'sense': TokenType.SENSE,
    'act': TokenType.ACT,
    'log': TokenType.LOG,
    'alert': TokenType.ALERT,
    'time': TokenType.TIME,
}

# Keywords mapping
KEYWORDS: Dict[str, TokenType] = {
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'elif': TokenType.ELIF,
    'while': TokenType.WHILE,
    'for': TokenType.FOR,
    'in': TokenType.IN,
    'return': TokenType.RETURN,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'try': TokenType.TRY,
    'catch': TokenType.CATCH,
    'finally': TokenType.FINALLY,
    'throw': TokenType.THROW,
    'true': TokenType.BOOLEAN,
    'false': TokenType.BOOLEAN,
    'null': TokenType.IDENTIFIER,  # Special null value
}

# Type keywords
TYPE_KEYWORDS: Dict[str, TokenType] = {
    'int': TokenType.INT_TYPE,
    'float': TokenType.FLOAT_TYPE,
    'string': TokenType.STRING_TYPE,
    'bool': TokenType.BOOL_TYPE,
    'array': TokenType.ARRAY_TYPE,
    'object': TokenType.OBJECT_TYPE,
    'signal': TokenType.SIGNAL_TYPE,
    'neuron': TokenType.NEURON_TYPE,
}

# Operators mapping
OPERATORS: Dict[str, TokenType] = {
    '=': TokenType.ASSIGN,
    '+': TokenType.PLUS,
    '-': TokenType.MINUS,
    '*': TokenType.MULTIPLY,
    '/': TokenType.DIVIDE,
    '%': TokenType.MODULO,
    '**': TokenType.POWER,
    '==': TokenType.EQUAL,
    '!=': TokenType.NOT_EQUAL,
    '<': TokenType.LESS_THAN,
    '<=': TokenType.LESS_EQUAL,
    '>': TokenType.GREATER_THAN,
    '>=': TokenType.GREATER_EQUAL,
    '&&': TokenType.AND,
    '||': TokenType.OR,
    '!': TokenType.NOT,
    '&': TokenType.BIT_AND,
    '|': TokenType.BIT_OR,
    '^': TokenType.BIT_XOR,
    '~': TokenType.BIT_NOT,
    '<<': TokenType.LEFT_SHIFT,
    '>>': TokenType.RIGHT_SHIFT,
}

# Delimiters mapping
DELIMITERS: Dict[str, TokenType] = {
    '(': TokenType.LEFT_PAREN,
    ')': TokenType.RIGHT_PAREN,
    '{': TokenType.LEFT_BRACE,
    '}': TokenType.RIGHT_BRACE,
    '[': TokenType.LEFT_BRACKET,
    ']': TokenType.RIGHT_BRACKET,
    ';': TokenType.SEMICOLON,
    ',': TokenType.COMMA,
    '.': TokenType.DOT,
    ':': TokenType.COLON,
    '->': TokenType.ARROW,
    '=>': TokenType.DOUBLE_ARROW,
}

# All reserved words (case-insensitive)
RESERVED_WORDS: Set[str] = set()
RESERVED_WORDS.update(NEURAL_COMMANDS.keys())
RESERVED_WORDS.update(KEYWORDS.keys())
RESERVED_WORDS.update(TYPE_KEYWORDS.keys())

# Regular expressions for token patterns
TOKEN_PATTERNS = {
    'FLOAT': re.compile(r'\d+\.\d+([eE][+-]?\d+)?'),
    'INTEGER': re.compile(r'\d+'),
    'STRING_DOUBLE': re.compile(r'"([^"\\]|\\.)*"'),
    'STRING_SINGLE': re.compile(r"'([^'\\]|\\.)*'"),
    'IDENTIFIER': re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*'),
    'COMMENT': re.compile(r'#.*'),
    'WHITESPACE': re.compile(r'[ \t]+'),
    'NEWLINE': re.compile(r'\n'),
    'MULTI_CHAR_OP': re.compile(r'(\*\*|==|!=|<=|>=|&&|\|\||<<|>>|->|=>)'),
    'SINGLE_CHAR': re.compile(r'[+\-*/%=<>!&|^~(){}[\];,.:?]'),
}

# Token categories for syntax highlighting and analysis
TOKEN_CATEGORIES = {
    'NEURAL_COMMANDS': set(NEURAL_COMMANDS.values()),
    'KEYWORDS': set(KEYWORDS.values()),
    'TYPES': set(TYPE_KEYWORDS.values()),
    'OPERATORS': set(OPERATORS.values()),
    'DELIMITERS': set(DELIMITERS.values()),
    'LITERALS': {TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING, TokenType.BOOLEAN},
    'IDENTIFIERS': {TokenType.IDENTIFIER},
}

def is_neural_command(token_type: TokenType) -> bool:
    """Check if token type is a neural command."""
    return token_type in TOKEN_CATEGORIES['NEURAL_COMMANDS']

def is_keyword(token_type: TokenType) -> bool:
    """Check if token type is a keyword."""
    return token_type in TOKEN_CATEGORIES['KEYWORDS']

def is_operator(token_type: TokenType) -> bool:
    """Check if token type is an operator."""
    return token_type in TOKEN_CATEGORIES['OPERATORS']

def is_literal(token_type: TokenType) -> bool:
    """Check if token type is a literal."""
    return token_type in TOKEN_CATEGORIES['LITERALS']

def get_token_category(token_type: TokenType) -> str:
    """Get the category name for a token type."""
    # Check specific categories first
    if token_type in TOKEN_CATEGORIES['NEURAL_COMMANDS']:
        return 'NEURAL_COMMANDS'
    elif token_type in TOKEN_CATEGORIES['KEYWORDS']:
        return 'KEYWORDS'
    elif token_type in TOKEN_CATEGORIES['TYPES']:
        return 'TYPES'
    elif token_type in TOKEN_CATEGORIES['OPERATORS']:
        return 'OPERATORS'
    elif token_type in TOKEN_CATEGORIES['DELIMITERS']:
        return 'DELIMITERS'
    elif token_type in TOKEN_CATEGORIES['LITERALS']:
        return 'LITERALS'
    elif token_type in TOKEN_CATEGORIES['IDENTIFIERS']:
        return 'IDENTIFIERS'
    else:
        return 'UNKNOWN' 