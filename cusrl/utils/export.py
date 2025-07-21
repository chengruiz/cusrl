import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import yaml
from torch import nn

from cusrl.module import Module

__all__ = ["ExportGraph"]

GetNumTensorsInputType = torch.Tensor | Iterable[torch.Tensor] | Iterable["GetNumTensorsInputType"]


class ExportGraph(nn.Module):
    def __init__(self, output_names: Iterable[str] = ()):
        super().__init__()
        self.output_names = list(output_names)
        self.info = {}
        self._named_submodules = {}

    def forward(self, *args, **kwargs):
        if args:
            kwargs.update(dict(args))
        if self.output_names is None:
            self.output_names = sorted(kwargs.keys())
        return tuple(kwargs[name] for name in self.output_names)

    def add_module_to_graph(
        self,
        module: nn.Module,
        module_name: str,
        input_names: str | Iterable[str] | Mapping[str, str],
        output_names: str | Iterable[str],
        method_name: str = "__call__",
        extra_kwargs: dict[str, Any] | None = None,
        info: dict[str, Any] | None = None,
        expose_outputs: bool = True,
        prepend: bool = False,
    ):
        if isinstance(input_names, str):
            input_names = {input_names: input_names}
        elif not isinstance(input_names, Mapping):
            input_names = {name: name for name in input_names}
        if isinstance(output_names, str):
            output_names = (output_names,)
        if module_name in self._named_submodules:
            raise ValueError(f"Module with name '{module_name}' already exists in the graph.")
        self._named_submodules[module_name] = module
        self.add_module(module_name, module)

        def hook(_: nn.Module, args: tuple, kwargs: dict[str, Any]):
            if args:
                kwargs.update(dict(args))
            if input_names is not None:
                inputs = {input_name: kwargs[arg_name] for input_name, arg_name in input_names.items()}
            else:
                inputs = kwargs
            outputs = getattr(module, method_name)(**inputs, **(extra_kwargs or {}))
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            outputs = [output for output in outputs if output is not None]
            named_outputs = {name: output for name, output in zip(output_names, outputs, strict=True)}
            if isinstance(module, Module):
                prefix = f"{module_name}." if module_name else ""
                named_outputs.update({
                    f"{prefix}{name}": value for name, value in module.intermediate_repr.items() if name not in outputs
                })
            kwargs.update(named_outputs)
            return (), kwargs

        if info is not None:
            self.info.update(info)
        if expose_outputs:
            for output_name in output_names:
                if output_name not in self.output_names:
                    self.output_names.append(output_name)

        self.register_forward_pre_hook(hook, prepend=prepend, with_kwargs=True)

    def export(
        self,
        inputs: dict[str, Any],
        output_dir: str,
        graph_name: str,
        dynamo: bool = False,
        verbose: bool = True,
    ):
        outputs = self(**inputs)
        input_names, output_names = [], []
        for name, input in inputs.items():
            if (num_tensors := get_num_tensors(input)) == 1:
                input_names.append(name)
            else:
                input_names.extend(f"{name}_{i}" for i in range(num_tensors))
        for output, name in zip(outputs, self.output_names):
            if (num_tensors := get_num_tensors(output)) == 1:
                output_names.append(name)
            else:
                output_names.extend(f"{name}_{i}" for i in range(num_tensors))

        unoptimized_model_path = f"{output_dir}/{graph_name}_unoptimized.onnx"
        optimized_model_path = f"{output_dir}/{graph_name}.onnx"
        torch.onnx.export(
            self,
            args=tuple(inputs.items()),
            f=unoptimized_model_path,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            external_data=False,
            dynamic_axes=None,
            dynamo=dynamo,
            report=verbose,
            optimize=False,
            verify=True,
            artifacts_dir=output_dir,
        )

        import onnx

        onnx.checker.check_model(unoptimized_model_path, full_check=True)
        onnx_model = onnx.load(unoptimized_model_path)
        self.info["input_name"] = [input.name for input in onnx_model.graph.input]
        self.info["output_name"] = [output.name for output in onnx_model.graph.output]
        with open(f"{output_dir}/{graph_name}.yml", "w") as f:
            yaml.safe_dump(self.info, f)

        optimizers = ["onnxslim", "onnxoptimizer"]
        for optimizer in optimizers:
            try:
                getattr(self, f"_optimize_onnx_model_with_{optimizer}")(onnx_model, optimized_model_path, verbose)
            except Exception as error:
                if verbose:
                    print(f"\033[1;33mFailed to optimize ONNX model with {optimizer}: {error}.\033[0m")
                continue
            if verbose:
                print(f"\033[1;32mOptimized ONNX model with {optimizer}.\033[0m")
            break
        else:
            if verbose:
                print("\033[1;33mFailed to optimize ONNX model.\033[0m")
            os.rename(unoptimized_model_path, optimized_model_path)
        onnx.checker.check_model(optimized_model_path, full_check=True)

    def _optimize_onnx_model_with_onnxslim(self, onnx_model, output_path: str, verbose: bool = True):
        import onnxslim

        onnxslim.slim(onnx_model, output_path, verbose=verbose)

    def _optimize_onnx_model_with_onnxoptimizer(self, onnx_model: Any, output_path: str, verbose: bool = True):
        import onnx
        import onnxoptimizer

        optimized_model = onnxoptimizer.optimize(onnx_model)
        onnx.save(optimized_model, output_path)


def get_num_tensors(tensor_list: GetNumTensorsInputType) -> int:
    if isinstance(tensor_list, torch.Tensor):
        return 1
    return sum(get_num_tensors(sublist) for sublist in tensor_list)
