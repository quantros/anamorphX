from __future__ import annotations

"""Simple grammar support for AnamorphX parser."""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ProductionRule:
    """Represents a single production of symbols."""
    symbols: Tuple[str, ...]

@dataclass
class GrammarRule:
    """Grammar rule consisting of one or more productions."""
    name: str
    productions: List[ProductionRule]

class GrammarValidator:
    """Validates a collection of grammar rules."""

    def __init__(self, rules: List[GrammarRule]):
        self.rules = {r.name: r for r in rules}

    def validate(self) -> bool:
        for rule in self.rules.values():
            for production in rule.productions:
                for symbol in production.symbols:
                    if symbol.isupper():
                        # Token names are assumed to be upper-case
                        continue
                    if symbol not in self.rules:
                        raise ValueError(f"Undefined symbol: {symbol}")
        return True

def validate_grammar(rules: List[GrammarRule]) -> bool:
    """Convenience wrapper for grammar validation."""
    return GrammarValidator(rules).validate()
