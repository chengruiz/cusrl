import inspect
import sys

import numpy as np
import pytest
import torch

from cusrl.utils.misc import import_module, import_obj, set_global_seed, to_numpy, wrap_method_with_signature


def test_import_module_from_path_sets_argv_and_restores_it(tmp_path):
    script = tmp_path / "sample_module.py"
    script.write_text("import sys\nCAPTURED_ARGV = sys.argv.copy()\nVALUE = 3\n")
    original_argv = sys.argv.copy()

    module = import_module(path=str(script), args=["--flag", "value"])

    assert module.VALUE == 3
    assert module.CAPTURED_ARGV == [str(script), "--flag", "value"]
    assert sys.argv == original_argv
    assert import_module(path=str(script)) is module


def test_import_module_rejects_ambiguous_or_missing_inputs(tmp_path):
    script = tmp_path / "missing.py"

    with pytest.raises(ValueError, match="cannot be specified together"):
        import_module(module_name="sys", path=str(script))
    with pytest.raises(FileNotFoundError):
        import_module(path=str(script))


def test_import_obj_resolves_nested_qualified_names():
    assert import_obj("collections", "Counter")("aba")["a"] == 2


def test_to_numpy_preserves_arrays_and_detaches_tensors():
    array = np.array([1.0, 2.0])
    tensor = torch.tensor([3.0, 4.0], requires_grad=True)

    assert to_numpy(array) is array
    assert np.allclose(to_numpy(tensor), np.array([3.0, 4.0]))


def test_wrap_method_with_signature_forwards_arguments():
    class Target:
        def call(self, value, *, scale=1):
            return value * scale

    target = Target()
    wrapped = wrap_method_with_signature(target, "call", arg_names=("value",), kwarg_names=("scale",))

    assert str(inspect.signature(wrapped)) == "(value, scale)"
    assert wrapped(3, scale=4) == 12


def test_set_global_seed_bounds_pythonhashseed_and_toggles_determinism():
    import os
    import subprocess

    set_global_seed(2**32 + 7, deterministic=True)
    assert os.environ["PYTHONHASHSEED"] == "7"
    assert torch.are_deterministic_algorithms_enabled()

    set_global_seed(-1, deterministic=False)
    assert os.environ["PYTHONHASHSEED"] == str(2**32 - 1)
    assert not torch.are_deterministic_algorithms_enabled()

    result = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True)
    assert result.returncode == 0
