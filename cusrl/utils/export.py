from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import yaml
from torch import nn

import cusrl

__all__ = [
    "ExportSpec",
    "get_num_tensors",
    "remove_none_output_forward_hook",
]


GetNumTensorsInputType = torch.Tensor | Iterable[torch.Tensor] | Iterable["GetNumTensorsInputType"]


class MethodForwarder(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        method_name: str | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.module = module
        self.method_name = method_name
        self.extra_kwargs = extra_kwargs or {}

    def forward(self, *args, **kwargs):
        if self.method_name:
            return getattr(self.module, self.method_name)(*args, **kwargs, **self.extra_kwargs)
        return self.module(*args, **kwargs, **self.extra_kwargs)


@dataclass(slots=True)
class ExportSpec:
    module: "cusrl.Module"
    module_name: str
    inputs: dict[str, Any]
    input_names: list[str]
    output_names: list[str]
    method_name: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    init_memory: Any = None
    dynamic_shapes: dict[str, Any] | None = None
    configuration: dict[str, Any] = field(default_factory=dict)

    def export(self, output_dir: str, verbose: bool = True):
        self._legacy_export(output_dir, verbose=verbose)
        self.configuration["input_name"] = self.input_names
        self.configuration["output_name"] = self.output_names
        with open(f"{output_dir}/{self.module_name}.yml", "w") as f:
            yaml.safe_dump(self.configuration, f)

    def _dynamo_export(self, output_dir: str, verbose: bool = True):
        # To be tested further
        torch.onnx.export(
            MethodForwarder(self.module, self.method_name, self.extra_kwargs),
            f=f"{output_dir}/{self.module_name}.onnx",
            kwargs=self.inputs,
            verbose=verbose,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamo=True,
            dynamic_shapes=self.dynamic_shapes,
            report=True,
            optimize=True,
            verify=True,
            fallback=False,
            artifacts_dir=f"{output_dir}/{self.module_name}_artifacts",
        )

    def _legacy_export(self, output_dir: str, verbose: bool = True):
        torch.onnx.export(
            MethodForwarder(self.module, self.method_name, self.extra_kwargs),
            args=tuple(self.inputs.values()),
            f=f"{output_dir}/{self.module_name}.onnx",
            input_names=self.input_names,
            output_names=self.output_names,
            dynamo=False,
            verbose=verbose,
        )


def get_num_tensors(tensor_list: GetNumTensorsInputType) -> int:
    if isinstance(tensor_list, torch.Tensor):
        return 1
    return sum(get_num_tensors(sublist) for sublist in tensor_list)


def remove_none_output_forward_hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
    if isinstance(output, tuple):
        return tuple(out for out in output if out is not None)
    return output
