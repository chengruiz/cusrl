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
        self.module_list = nn.ModuleList()
        self.output_names = list(output_names)
        self.info = {}

    def forward(self, **kwargs):
        if self.output_names is None:
            self.output_names = sorted(kwargs.keys())
        return tuple(kwargs[name] for name in self.output_names)

    def add_module_to_graph(
        self,
        module: nn.Module,
        input_names: str | Iterable[str] | Mapping[str, str],
        output_names: str | Iterable[str],
        module_name: str = "",
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
        self.module_list.append(module)

        def hook(_: nn.Module, args: tuple, kwargs: dict[str, Any]):
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

        if info is not None:
            self.info.update(info)
        if expose_outputs:
            for output_name in output_names:
                if output_name not in self.output_names:
                    self.output_names.append(output_name)

        self.register_forward_pre_hook(hook, prepend=prepend, with_kwargs=True)

    def export(
        self,
        kwargs: dict[str, Any],
        output_dir: str,
        graph_name: str,
        verbose: bool = True,
    ):
        outputs = self(**kwargs)
        output_names = []
        for output, name in zip(outputs, self.output_names):
            if (num_tensors := get_num_tensors(output)) == 1:
                output_names.append(name)
            else:
                output_names.extend(f"{name}_{i}" for i in range(num_tensors))

        onnx_program = torch.onnx.export(
            self,
            kwargs=kwargs,
            f=f"{output_dir}/{graph_name}.onnx",
            verbose=verbose,
            output_names=output_names,
            external_data=False,
            dynamic_axes=None,
            dynamo=True,
            report=verbose,
            optimize=False,
            verify=True,
            artifacts_dir=output_dir,
        )
        if onnx_program is None:
            raise RuntimeError("ONNX export failed.")

        self.info["input_name"] = [input.name for input in onnx_program.model.graph.inputs]
        self.info["output_name"] = [output.name for output in onnx_program.model.graph.outputs]
        with open(f"{output_dir}/{graph_name}.yml", "w") as f:
            yaml.safe_dump(self.info, f)


def get_num_tensors(tensor_list: GetNumTensorsInputType) -> int:
    if isinstance(tensor_list, torch.Tensor):
        return 1
    return sum(get_num_tensors(sublist) for sublist in tensor_list)
