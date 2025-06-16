"""
Utility functions for AnamorphX code generation.

This module provides various utility functions for code formatting,
symbol management, dependency resolution, and import handling.
"""

import re
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ImportInfo:
    """Information about an import statement."""
    
    module_name: str
    imported_names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_relative: bool = False
    import_type: str = "import"  # "import", "from", "include", etc.
    
    def to_string(self, target_platform: str = "python") -> str:
        """Convert to import string for target platform."""
        if target_platform == "python":
            if self.import_type == "from":
                names = ", ".join(self.imported_names)
                alias_part = f" as {self.alias}" if self.alias else ""
                return f"from {self.module_name} import {names}{alias_part}"
            else:
                alias_part = f" as {self.alias}" if self.alias else ""
                return f"import {self.module_name}{alias_part}"
        
        elif target_platform in ["javascript", "js"]:
            if self.imported_names:
                names = ", ".join(self.imported_names)
                return f"import {{ {names} }} from '{self.module_name}';"
            else:
                alias_part = self.alias or self.module_name.split('/')[-1]
                return f"import {alias_part} from '{self.module_name}';"
        
        elif target_platform in ["cpp", "c++"]:
            return f"#include <{self.module_name}>"
        
        else:
            return f"// Import: {self.module_name}"


class CodeFormatter:
    """Code formatter for different target languages."""
    
    def __init__(self, target_platform: str = "python"):
        self.target_platform = target_platform
        self.indent_size = self._get_indent_size()
        self.line_ending = "\n"
        self.max_line_length = 100
    
    def _get_indent_size(self) -> int:
        """Get indent size for target platform."""
        indent_sizes = {
            "python": 4,
            "javascript": 2,
            "js": 2,
            "cpp": 4,
            "c++": 4,
            "llvm": 2,
        }
        return indent_sizes.get(self.target_platform, 4)
    
    def format_code(self, code: str) -> str:
        """Format code according to target platform conventions."""
        lines = code.split('\n')
        formatted_lines = []
        current_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                formatted_lines.append("")
                continue
            
            # Adjust indentation based on content
            if self._should_decrease_indent(stripped):
                current_indent = max(0, current_indent - 1)
            
            # Format the line
            formatted_line = self._format_line(stripped, current_indent)
            formatted_lines.append(formatted_line)
            
            # Adjust indentation for next line
            if self._should_increase_indent(stripped):
                current_indent += 1
        
        return self.line_ending.join(formatted_lines)
    
    def _should_increase_indent(self, line: str) -> bool:
        """Check if indentation should increase after this line."""
        if self.target_platform == "python":
            return line.endswith(':')
        elif self.target_platform in ["javascript", "js", "cpp", "c++"]:
            return line.endswith('{') or line.endswith('(')
        return False
    
    def _should_decrease_indent(self, line: str) -> bool:
        """Check if indentation should decrease for this line."""
        if self.target_platform == "python":
            return line.startswith(('except', 'elif', 'else', 'finally'))
        elif self.target_platform in ["javascript", "js", "cpp", "c++"]:
            return line.startswith(('}', ')')) or line.startswith('else')
        return False
    
    def _format_line(self, line: str, indent_level: int) -> str:
        """Format a single line with proper indentation."""
        indent = " " * (indent_level * self.indent_size)
        
        # Apply line-specific formatting
        formatted_line = self._apply_line_formatting(line)
        
        # Check line length
        full_line = indent + formatted_line
        if len(full_line) > self.max_line_length:
            full_line = self._wrap_long_line(full_line, indent_level)
        
        return full_line
    
    def _apply_line_formatting(self, line: str) -> str:
        """Apply target-specific line formatting."""
        if self.target_platform == "python":
            # Python-specific formatting
            line = self._format_python_line(line)
        elif self.target_platform in ["javascript", "js"]:
            # JavaScript-specific formatting
            line = self._format_javascript_line(line)
        elif self.target_platform in ["cpp", "c++"]:
            # C++-specific formatting
            line = self._format_cpp_line(line)
        
        return line
    
    def _format_python_line(self, line: str) -> str:
        """Format Python-specific line."""
        # Add spaces around operators
        line = re.sub(r'([^=!<>])=([^=])', r'\1 = \2', line)
        line = re.sub(r'([^=!<>])==([^=])', r'\1 == \2', line)
        line = re.sub(r'([^=!<>])!=([^=])', r'\1 != \2', line)
        
        # Format function calls
        line = re.sub(r'(\w+)\s*\(', r'\1(', line)
        
        return line
    
    def _format_javascript_line(self, line: str) -> str:
        """Format JavaScript-specific line."""
        # Ensure semicolons
        if not line.endswith((';', '{', '}', ':')):
            line += ';'
        
        # Format function declarations
        line = re.sub(r'function\s+(\w+)\s*\(', r'function \1(', line)
        
        return line
    
    def _format_cpp_line(self, line: str) -> str:
        """Format C++-specific line."""
        # Ensure semicolons for statements
        if not line.endswith((';', '{', '}', ':', '#')):
            if not line.startswith(('#include', 'namespace', 'class', 'struct')):
                line += ';'
        
        return line
    
    def _wrap_long_line(self, line: str, indent_level: int) -> str:
        """Wrap long lines according to target conventions."""
        if self.target_platform == "python":
            return self._wrap_python_line(line, indent_level)
        elif self.target_platform in ["javascript", "js"]:
            return self._wrap_javascript_line(line, indent_level)
        elif self.target_platform in ["cpp", "c++"]:
            return self._wrap_cpp_line(line, indent_level)
        
        return line
    
    def _wrap_python_line(self, line: str, indent_level: int) -> str:
        """Wrap Python long line."""
        # Simple wrapping for function calls
        if '(' in line and ')' in line:
            parts = line.split('(', 1)
            if len(parts) == 2:
                prefix = parts[0] + '('
                suffix = parts[1]
                
                # Split arguments
                args = suffix.rstrip(')').split(',')
                if len(args) > 1:
                    indent = " " * ((indent_level + 1) * self.indent_size)
                    wrapped_args = [args[0]]
                    for arg in args[1:]:
                        wrapped_args.append(f"\n{indent}{arg.strip()}")
                    
                    return prefix + ",".join(wrapped_args) + ")"
        
        return line
    
    def _wrap_javascript_line(self, line: str, indent_level: int) -> str:
        """Wrap JavaScript long line."""
        # Similar to Python but with different conventions
        return line
    
    def _wrap_cpp_line(self, line: str, indent_level: int) -> str:
        """Wrap C++ long line."""
        # C++ line wrapping
        return line


class SymbolMangler:
    """Symbol name mangler for different target platforms."""
    
    def __init__(self, target_platform: str = "python"):
        self.target_platform = target_platform
        self.reserved_words = self._get_reserved_words()
        self.naming_convention = self._get_naming_convention()
        self.symbol_cache: Dict[str, str] = {}
    
    def _get_reserved_words(self) -> Set[str]:
        """Get reserved words for target platform."""
        reserved_words = {
            "python": {
                'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
                'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
                'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
                'not', 'or', 'pass', 'print', 'raise', 'return', 'try',
                'while', 'with', 'yield', 'async', 'await', 'nonlocal'
            },
            "javascript": {
                'break', 'case', 'catch', 'class', 'const', 'continue',
                'debugger', 'default', 'delete', 'do', 'else', 'export',
                'extends', 'finally', 'for', 'function', 'if', 'import',
                'in', 'instanceof', 'new', 'return', 'super', 'switch',
                'this', 'throw', 'try', 'typeof', 'var', 'void', 'while',
                'with', 'yield', 'async', 'await', 'let'
            },
            "cpp": {
                'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto',
                'bitand', 'bitor', 'bool', 'break', 'case', 'catch',
                'char', 'char16_t', 'char32_t', 'class', 'compl',
                'const', 'constexpr', 'const_cast', 'continue',
                'decltype', 'default', 'delete', 'do', 'double',
                'dynamic_cast', 'else', 'enum', 'explicit', 'export',
                'extern', 'false', 'float', 'for', 'friend', 'goto',
                'if', 'inline', 'int', 'long', 'mutable', 'namespace',
                'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                'operator', 'or', 'or_eq', 'private', 'protected',
                'public', 'register', 'reinterpret_cast', 'return',
                'short', 'signed', 'sizeof', 'static', 'static_assert',
                'static_cast', 'struct', 'switch', 'template', 'this',
                'thread_local', 'throw', 'true', 'try', 'typedef',
                'typeid', 'typename', 'union', 'unsigned', 'using',
                'virtual', 'void', 'volatile', 'wchar_t', 'while',
                'xor', 'xor_eq'
            }
        }
        
        return reserved_words.get(self.target_platform, set())
    
    def _get_naming_convention(self) -> str:
        """Get naming convention for target platform."""
        conventions = {
            "python": "snake_case",
            "javascript": "camelCase",
            "js": "camelCase",
            "cpp": "camelCase",
            "c++": "camelCase",
        }
        return conventions.get(self.target_platform, "snake_case")
    
    def mangle_symbol(self, symbol_name: str, symbol_type: str = "variable") -> str:
        """Mangle a symbol name for the target platform."""
        if symbol_name in self.symbol_cache:
            return self.symbol_cache[symbol_name]
        
        # Clean the symbol name
        cleaned = self._clean_symbol_name(symbol_name)
        
        # Apply naming convention
        converted = self._apply_naming_convention(cleaned, symbol_type)
        
        # Handle reserved words
        if converted.lower() in self.reserved_words:
            converted = self._handle_reserved_word(converted)
        
        # Ensure uniqueness
        final_name = self._ensure_uniqueness(converted)
        
        # Cache the result
        self.symbol_cache[symbol_name] = final_name
        
        return final_name
    
    def _clean_symbol_name(self, name: str) -> str:
        """Clean symbol name of invalid characters."""
        # Remove invalid characters
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it starts with letter or underscore
        if cleaned and cleaned[0].isdigit():
            cleaned = '_' + cleaned
        
        # Remove consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Remove trailing underscores
        cleaned = cleaned.rstrip('_')
        
        return cleaned or '_unnamed'
    
    def _apply_naming_convention(self, name: str, symbol_type: str) -> str:
        """Apply naming convention based on symbol type."""
        if self.naming_convention == "snake_case":
            return self._to_snake_case(name)
        elif self.naming_convention == "camelCase":
            if symbol_type == "class":
                return self._to_pascal_case(name)
            else:
                return self._to_camel_case(name)
        elif self.naming_convention == "PascalCase":
            return self._to_pascal_case(name)
        
        return name
    
    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        # Insert underscores before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _to_camel_case(self, name: str) -> str:
        """Convert to camelCase."""
        components = name.split('_')
        return components[0].lower() + ''.join(word.capitalize() for word in components[1:])
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase."""
        components = name.split('_')
        return ''.join(word.capitalize() for word in components)
    
    def _handle_reserved_word(self, name: str) -> str:
        """Handle reserved words by adding suffix."""
        return name + '_'
    
    def _ensure_uniqueness(self, name: str) -> str:
        """Ensure symbol name uniqueness."""
        if name not in self.symbol_cache.values():
            return name
        
        # Add numeric suffix
        counter = 1
        while f"{name}_{counter}" in self.symbol_cache.values():
            counter += 1
        
        return f"{name}_{counter}"


class DependencyResolver:
    """Resolves dependencies between modules and symbols."""
    
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = {}
        self.resolved_order: List[str] = []
        self.circular_dependencies: List[Tuple[str, str]] = []
    
    def add_dependency(self, dependent: str, dependency: str):
        """Add a dependency relationship."""
        if dependent not in self.dependencies:
            self.dependencies[dependent] = set()
        self.dependencies[dependent].add(dependency)
    
    def resolve_dependencies(self) -> List[str]:
        """Resolve dependencies and return topological order."""
        visited = set()
        temp_visited = set()
        self.resolved_order = []
        self.circular_dependencies = []
        
        def visit(node: str):
            if node in temp_visited:
                # Circular dependency detected
                return False
            
            if node in visited:
                return True
            
            temp_visited.add(node)
            
            for dependency in self.dependencies.get(node, set()):
                if not visit(dependency):
                    self.circular_dependencies.append((node, dependency))
                    return False
            
            temp_visited.remove(node)
            visited.add(node)
            self.resolved_order.append(node)
            
            return True
        
        # Visit all nodes
        for node in self.dependencies:
            if node not in visited:
                visit(node)
        
        return self.resolved_order
    
    def has_circular_dependencies(self) -> bool:
        """Check if there are circular dependencies."""
        return len(self.circular_dependencies) > 0
    
    def get_circular_dependencies(self) -> List[Tuple[str, str]]:
        """Get list of circular dependencies."""
        return self.circular_dependencies


class ImportManager:
    """Manages import statements for generated code."""
    
    def __init__(self, target_platform: str = "python"):
        self.target_platform = target_platform
        self.imports: Dict[str, ImportInfo] = {}
        self.import_order: List[str] = []
    
    def add_import(self, module_name: str, imported_names: Optional[List[str]] = None, 
                   alias: Optional[str] = None, is_relative: bool = False):
        """Add an import statement."""
        import_key = self._generate_import_key(module_name, imported_names, alias)
        
        if import_key not in self.imports:
            import_info = ImportInfo(
                module_name=module_name,
                imported_names=imported_names or [],
                alias=alias,
                is_relative=is_relative,
                import_type="from" if imported_names else "import"
            )
            
            self.imports[import_key] = import_info
            self.import_order.append(import_key)
    
    def add_from_import(self, module_name: str, imported_names: List[str], 
                        alias: Optional[str] = None):
        """Add a 'from module import names' statement."""
        self.add_import(module_name, imported_names, alias)
    
    def add_standard_import(self, module_name: str, alias: Optional[str] = None):
        """Add a standard 'import module' statement."""
        self.add_import(module_name, None, alias)
    
    def _generate_import_key(self, module_name: str, imported_names: Optional[List[str]], 
                            alias: Optional[str]) -> str:
        """Generate unique key for import."""
        key_parts = [module_name]
        
        if imported_names:
            key_parts.append(",".join(sorted(imported_names)))
        
        if alias:
            key_parts.append(f"as_{alias}")
        
        return "|".join(key_parts)
    
    def generate_import_statements(self) -> List[str]:
        """Generate all import statements in proper order."""
        statements = []
        
        # Group imports by type
        standard_imports = []
        third_party_imports = []
        local_imports = []
        
        for import_key in self.import_order:
            import_info = self.imports[import_key]
            statement = import_info.to_string(self.target_platform)
            
            if self._is_standard_library(import_info.module_name):
                standard_imports.append(statement)
            elif self._is_third_party(import_info.module_name):
                third_party_imports.append(statement)
            else:
                local_imports.append(statement)
        
        # Combine in proper order
        if standard_imports:
            statements.extend(sorted(standard_imports))
            statements.append("")
        
        if third_party_imports:
            statements.extend(sorted(third_party_imports))
            statements.append("")
        
        if local_imports:
            statements.extend(sorted(local_imports))
        
        return [stmt for stmt in statements if stmt]  # Remove empty strings
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if module is from standard library."""
        if self.target_platform == "python":
            standard_modules = {
                'os', 'sys', 'time', 'datetime', 'json', 'math', 'random',
                'collections', 'itertools', 'functools', 'operator',
                'typing', 'dataclasses', 'enum', 'abc', 'asyncio'
            }
            return module_name.split('.')[0] in standard_modules
        
        elif self.target_platform in ["javascript", "js"]:
            # JavaScript doesn't have a standard library in the same sense
            return False
        
        elif self.target_platform in ["cpp", "c++"]:
            standard_headers = {
                'iostream', 'vector', 'string', 'memory', 'algorithm',
                'functional', 'thread', 'future', 'chrono', 'any'
            }
            return module_name in standard_headers
        
        return False
    
    def _is_third_party(self, module_name: str) -> bool:
        """Check if module is third-party."""
        # This would be more sophisticated in a real implementation
        third_party_modules = {
            'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn',
            'requests', 'flask', 'django', 'fastapi'
        }
        return module_name.split('.')[0] in third_party_modules
    
    def clear_imports(self):
        """Clear all imports."""
        self.imports.clear()
        self.import_order.clear()
    
    def get_import_count(self) -> int:
        """Get number of import statements."""
        return len(self.imports)


def generate_unique_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique identifier."""
    import time
    import random
    
    # Use timestamp and random number for uniqueness
    timestamp = str(int(time.time() * 1000))
    random_part = str(random.randint(1000, 9999))
    
    # Create hash
    hash_input = f"{prefix}_{timestamp}_{random_part}"
    hash_object = hashlib.md5(hash_input.encode())
    hash_hex = hash_object.hexdigest()
    
    # Return prefix + truncated hash
    unique_id = hash_hex[:length]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def escape_string_literal(value: str, target_platform: str = "python") -> str:
    """Escape string literal for target platform."""
    if target_platform == "python":
        # Python string escaping
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace("'", "\\'")
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\t', '\\t')
        escaped = escaped.replace('\r', '\\r')
        return f'"{escaped}"'
    
    elif target_platform in ["javascript", "js"]:
        # JavaScript string escaping
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace("'", "\\'")
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\t', '\\t')
        escaped = escaped.replace('\r', '\\r')
        return f'"{escaped}"'
    
    elif target_platform in ["cpp", "c++"]:
        # C++ string escaping
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\t', '\\t')
        escaped = escaped.replace('\r', '\\r')
        return f'"{escaped}"'
    
    return f'"{value}"'


def calculate_code_metrics(code: str) -> Dict[str, Any]:
    """Calculate basic code metrics."""
    lines = code.split('\n')
    
    total_lines = len(lines)
    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*')))
    code_lines = total_lines - blank_lines - comment_lines
    
    # Character count
    total_chars = len(code)
    
    # Word count (approximate)
    words = re.findall(r'\b\w+\b', code)
    word_count = len(words)
    
    return {
        'total_lines': total_lines,
        'code_lines': code_lines,
        'blank_lines': blank_lines,
        'comment_lines': comment_lines,
        'total_characters': total_chars,
        'word_count': word_count,
        'average_line_length': total_chars / total_lines if total_lines > 0 else 0
    } 