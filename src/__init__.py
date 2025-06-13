"""
Anamorph Neural Programming Language Implementation.

This package contains the complete implementation of the Anamorph language
including lexer, parser, interpreter, signal processing system, and more.
"""

__version__ = "0.1.0"
__author__ = "anamorphX Team"
__email__ = "team@anamorph.dev"
__description__ = "Neural Programming Language with unique signal processing"

# Version information
VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'stage': 'alpha',
    'build': 1
}

def get_version() -> str:
    """Get the current version string."""
    version = f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"
    if VERSION_INFO['stage'] != 'release':
        version += f"-{VERSION_INFO['stage']}.{VERSION_INFO['build']}"
    return version

# Package metadata
__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'VERSION_INFO',
    'get_version'
] 