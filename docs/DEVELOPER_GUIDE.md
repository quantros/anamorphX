# Developer Guide

This guide explains how to set up the project, run the IDE and execute the test suite.

## Setup
1. Clone the repository
   ```bash
   git clone https://github.com/quantros/anamorphX.git
   cd anamorphX
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Running the IDE
Start the full IDE with:
```bash
python run_full_ml_interpreter_ide.py
```

## Generating PyTorch Code
Define a `network` block in an `.anamorph` file and use the IDE command **Generate PyTorch** (Ctrl+Shift+G) to create a model.

## Running Tests
Execute all tests with:
```bash
pytest -q
```

