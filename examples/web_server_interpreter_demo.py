"""Demo interpreter for AnamorphX web server example."""

import sys
from pathlib import Path

# Ensure src package is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.lexer import tokenize


def main():
    """Tokenize and simulate running the web server example."""
    example_path = Path(__file__).parent / "web_server.amph"
    source = example_path.read_text()
    tokens = tokenize(source)
    print(f"Tokenized {len(tokens)} tokens from {example_path.name}.")
    for tok in tokens[:15]:
        print(tok)

    print("\nSimulating server loop (placeholder)...")
    for path in ["/", "/api/data", "/static/index.html"]:
        print(f"Handling request for {path}")


if __name__ == "__main__":
    main()

