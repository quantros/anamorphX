import sys
import types
from pathlib import Path

import pytest


def test_neural_translator_basic(tmp_path, monkeypatch):
    # Stub PyTorchGenerator to avoid importing template code
    pg_module = types.ModuleType("src.neural_backend.pytorch_generator")
    class DummyPG:
        def generate_model_class(self, network):
            return "# model"
        def generate_training_script(self, network):
            return "# train"
    pg_module.PyTorchGenerator = DummyPG
    monkeypatch.setitem(sys.modules, "src.neural_backend.pytorch_generator", pg_module)

    # Stub interpreter.ast_interpreter required by anamorph_core
    ast_mod = types.ModuleType("src.interpreter.ast_interpreter")
    ast_mod.ASTInterpreter = object
    monkeypatch.setitem(sys.modules, "src.interpreter.ast_interpreter", ast_mod)
    interpreter_pkg = types.ModuleType("src.interpreter")
    interpreter_pkg.ast_interpreter = ast_mod
    monkeypatch.setitem(sys.modules, "src.interpreter", interpreter_pkg)

    from src.anamorph_core import NeuralTranslator

    translator = NeuralTranslator(output_dir=str(tmp_path))
    network_code = (
        "network TestNet {\n"
        "    optimizer: adam\n"
        "    learning_rate: 0.001\n"
        "    loss: mse\n"
        "    batch_size: 2\n"
        "    epochs: 10\n"
        "}"
    )
    result = translator.translate_code(network_code)
    assert result["success"] is True
    for file_path in result["files_generated"]:
        assert Path(file_path).exists()
        Path(file_path).unlink()
