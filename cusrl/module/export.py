import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import torch
import yaml
from torch import nn

from cusrl.module.module import Module
from cusrl.utils.dict_utils import prefix_dict_keys
from cusrl.utils.nest import flatten_nested, get_schema, iterate_nested, reconstruct_nested
from cusrl.utils.typing import NestedTensor

__all__ = ["FlowGraph"]

GetNumTensorsInputType = torch.Tensor | Iterable[torch.Tensor] | Iterable["GetNumTensorsInputType"]


class FlowGraph(nn.Module):
    """A dynamic computation graph module that chains sub-modules using forward
    pre-hooks.

    This class allows for the construction of complex, modular architectures by
    treating execution as a flow of data through a sequence of nodes. Instead of
    a rigid `forward` method, it uses a shared context (dictionary) that passes
    through registered hooks.
    """

    def __init__(self, graph_name: str, output_names: Iterable[str] = ()):
        super().__init__()
        self.graph_name = graph_name
        self.output_names = list(output_names)
        self.info = {}
        self.named_nodes = {}

    def forward(self, *args, **kwargs):
        if args:
            kwargs.update(dict(args))
        if self.output_names is None:
            self.output_names = sorted(kwargs.keys())
        return tuple(kwargs[name] for name in self.output_names)

    def add_node(
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
        """Registers a module as a node in the computation graph.

        The module is added as a child submodule and a pre-hook is registered to
        handle its execution. The hook extracts specified inputs from the shared
        context, runs the module's method, and updates the context with the
        outputs.

        Args:
            module (nn.Module):
                The PyTorch module to add to the graph.
            module_name (str):
                A unique name for this node in the graph.
            input_names (str | Iterable[str] | Mapping[str, str]):
                Keys to extract from the context to pass to the module. This
                should be a string or an iterable of strings, representing both
                the context key and the module argument name; or a mapping from
                module argument names to context keys.
            output_names (str | Iterable[str]):
                Keys to assign to the module's outputs in the context.
            method_name (str):
                The method of the module to call. Defaults to ``"__call__"``.
            extra_kwargs (dict[str, Any] | None):
                Static keyword arguments to always pass to the module.
            info (dict[str, Any] | None):
                Optional metadata dictionary to store with the graph.
            expose_outputs (bool):
                If ``True``, adds the outputs of this node to the graph's final
                outputs.
            prepend (bool):
                If ``True``, inserts this node at the beginning of the execution
                chain.

        Raises:
            ValueError: If a node with `module_name` already exists.
        """

        if isinstance(input_names, str):
            input_names = {input_names: input_names}
        elif not isinstance(input_names, Mapping):
            input_names = {name: name for name in input_names}
        if isinstance(output_names, str):
            output_names = (output_names,)
        if module_name in self.named_nodes:
            raise ValueError(f"Module with name '{module_name}' already exists in the graph.")
        self.named_nodes[module_name] = module
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
                named_outputs.update(prefix_dict_keys(module.intermediate_repr, prefix))
            kwargs.update(named_outputs)
            return (), kwargs

        if info is not None:
            self.info.update(info)
        if expose_outputs:
            for output_name in output_names:
                if output_name not in self.output_names:
                    self.output_names.append(output_name)

        self.register_forward_pre_hook(hook, prepend=prepend, with_kwargs=True)

    def export_jit(
        self,
        example_inputs: dict[str, Any],
        output_dir: str,
        optimize: bool = True,
    ):
        self.eval()

        # Flatten inputs
        flattened_inputs: dict[str, torch.Tensor] = flatten_nested(example_inputs)

        # Build the wrappers, trace and save them
        stateless_wrapper = StatelessWrapper(self, example_inputs, self.output_names)
        traced_stateless_module = torch.jit.trace_module(
            stateless_wrapper, inputs={"forward": (flattened_inputs,)}, strict=False
        )
        if optimize:
            traced_stateless_module = torch.jit.optimize_for_inference(traced_stateless_module)
        torch.jit.save(traced_stateless_module, f"{output_dir}/{self.graph_name}_stateless.pt")

        stateful_wrapper = StatefulWrapper(stateless_wrapper, example_inputs)
        non_memory_inputs = tuple(flattened_inputs[name] for name in stateful_wrapper.input_names)
        traced_stateful_module = torch.jit.trace_module(
            stateful_wrapper,
            inputs={
                "forward": non_memory_inputs,
                "reset": torch.rand(example_inputs["observation"].size(-2)) < 0.5,
            },
            strict=False,
        )
        if optimize:
            traced_stateful_module = torch.jit.optimize_for_inference(traced_stateful_module, other_methods=["reset"])
        torch.jit.save(traced_stateful_module, f"{output_dir}/{self.graph_name}.pt")

        # Save additional information
        info = self.info.copy()
        flattened_outputs = stateless_wrapper(flattened_inputs)
        info["inputs"] = [{name: tuple(value.shape)} for name, value in flattened_inputs.items()]
        info["outputs"] = [{name: tuple(value.shape)} for name, value in flattened_outputs.items()]
        with open(f"{output_dir}/{self.graph_name}.yml", "w") as f:
            yaml.safe_dump(info, f)

    def export_onnx(
        self,
        example_inputs: dict[str, Any],
        output_dir: str,
        dynamo: bool = False,
        optimize: bool = True,
        verbose: bool = True,
        opset_version: int | None = None,
        **kwargs: Any,
    ):
        self.eval()

        example_outputs = self(**example_inputs)

        # Get the actual name for each input / output tensor
        input_names = [name for name, _ in iterate_nested(example_inputs)]
        output_names = [name for name, _ in iterate_nested(dict(zip(self.output_names, example_outputs)))]

        # Export the model to ONNX format
        model_path = f"{output_dir}/{self.graph_name}.onnx"
        torch.onnx.export(
            self,
            args=() if dynamo else tuple(example_inputs.items()),
            kwargs=example_inputs if dynamo else None,
            f=model_path,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=None,
            dynamo=dynamo,
            external_data=False,
            report=verbose,
            optimize=False,
            verify=True,
            artifacts_dir=output_dir,
            **kwargs,
        )

        # Save additional information
        import onnx

        onnx.checker.check_model(model_path, full_check=True)
        onnx_model = onnx.load(model_path)
        info = self.info.copy()
        info["inputs"] = [
            {input.name: _get_onnx_tensor_shape(input.type.tensor_type)} for input in onnx_model.graph.input
        ]
        info["outputs"] = [
            {output.name: _get_onnx_tensor_shape(output.type.tensor_type)} for output in onnx_model.graph.output
        ]
        with open(f"{output_dir}/{self.graph_name}.yml", "w") as f:
            yaml.safe_dump(info, f)

        # Optimize the ONNX model
        if optimize and _optimize_onnx_model(onnx_model, model_path, verbose):
            onnx.checker.check_model(model_path, full_check=True)


def _get_onnx_tensor_shape(tensor_type: Any) -> tuple[int | str, ...]:
    """Extracts the shape of an ONNX tensor type."""
    if not tensor_type.HasField("shape"):
        raise ValueError("Tensor type does not have a shape defined.")
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return tuple(shape)


def _optimize_onnx_model(model, output_path: str, verbose: bool = True) -> bool:
    """Attempts to optimize the ONNX model using available optimizers."""

    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    try:
        import onnx
        import onnxoptimizer

        optimized_model = onnxoptimizer.optimize(model)
        onnx.save(optimized_model, output_path)
        print_if_verbose("\033[1;32mOptimized ONNX model with onnxoptimizer.\033[0m")
        return True
    except ModuleNotFoundError:
        pass

    try:
        import onnx
        import onnxslim
        from onnxslim.utils import print_model_info_as_table, summarize_model

        original_info = summarize_model(model, "Before Optimization")
        start_time = time.time()
        optimized_model = cast(onnx.ModelProto, onnxslim.slim(model, verbose=verbose))  # type: ignore
        end_time = time.time()
        onnx.save(optimized_model, output_path)

        if verbose:
            slimmed_info = summarize_model(optimized_model, "After Optimization")
            elapsed_time = end_time - start_time
            print_model_info_as_table([original_info, slimmed_info], elapsed_time)
            print("\033[1;32mOptimized ONNX model with onnxslim.\033[0m")
        return True
    except ModuleNotFoundError:
        pass

    print_if_verbose(
        "\033[1;33mFailed to optimize ONNX model. Run `pip install onnxoptimizer`"
        "or `pip install onnxslim` to install an optimizer.\033[0m"
    )
    return False


def get_num_tensors(tensor_list: GetNumTensorsInputType) -> int:
    if isinstance(tensor_list, torch.Tensor):
        return 1
    return sum(get_num_tensors(sublist) for sublist in tensor_list)


class StatelessWrapper(nn.Module):
    def __init__(
        self,
        module: FlowGraph,
        example_inputs: dict[str, NestedTensor],
        output_names: Sequence[str],
    ):
        super().__init__()
        self.module = module
        self.input_schema = get_schema(example_inputs)
        self.output_names = output_names

    @torch.no_grad()
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reconstructed_inputs = cast(dict[str, Any], reconstruct_nested(inputs, self.input_schema))
        # Convert outputs to a flatten dictionary keyed by corresponding output names
        return flatten_nested(dict(zip(self.output_names, self.module(**reconstructed_inputs))))


class StatefulWrapper(nn.Module):
    """Wraps a `StatelessWrapper` to create a stateful `nn.Module`.

    This class manages internal memory buffers, making a stateless module
    stateful. It identifies memory tensors in the `example_inputs` by the prefix
    'memory_in'. For each 'memory_in' tensor, it expects the corresponding
    'memory_out' tensor from the wrapped module's output, which it uses to
    update its internal state.

    The `forward` method's signature is dynamically generated based on the
    non-memory input names found in `example_inputs`. This allows for a clean,
    explicit forward pass when using the wrapped module.

    Args:
        stateless_wrapper (StatelessWrapper):
            The stateless module to be wrapped.
        example_inputs (dict[str, NestedTensor]):
            A dictionary of example inputs. Keys starting with 'memory_in' are
            recognized as states.
    """

    def __init__(
        self,
        stateless_wrapper: StatelessWrapper,
        example_inputs: dict[str, NestedTensor],
    ):
        super().__init__()
        self.module = stateless_wrapper

        input_names = []
        output_names = list(stateless_wrapper(flatten_nested(example_inputs)).keys())
        memory_names = []
        for name, value in iterate_nested(example_inputs):
            if not name.startswith("memory_in"):
                input_names.append(name)
                continue
            memory_names.append(name)
            output_names.remove(f"memory_out{name.removeprefix('memory_in')}")
            self.register_buffer(name.replace(".", "__"), value)
        self.input_names: tuple[str, ...] = tuple(input_names)
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.memory_names: tuple[str, ...] = tuple(memory_names)
        forward_input_signature = f"{', '.join(self.input_names)}"
        self.forward = eval(
            f"lambda {forward_input_signature}: self._forward({forward_input_signature})",
            {"self": self},
        )

    @torch.no_grad()
    def _forward(self, *args, **kwargs):
        # Add memories to inputs
        memories = {name: getattr(self, name.replace(".", "__")) for name in self.memory_names}
        inputs = memories | dict(zip(self.input_names, args)) | kwargs
        # Inference the module
        outputs = self.module(inputs)
        # Update memories with outputs
        for memory_name in self.memory_names:
            memory_out_name = f"memory_out{memory_name.removeprefix('memory_in')}"
            self._get_memory(memory_name).copy_(outputs.pop(memory_out_name))
        # Return the remaining outputs
        outputs = tuple(outputs[name] for name in self.output_names)
        return outputs[0] if len(outputs) == 1 else outputs

    def reset(self, indices: torch.Tensor):
        for memory_name in self.memory_names:
            memory = self._get_memory(memory_name)
            memory[..., indices, :] = 0
        return indices

    def _get_memory(self, memory_name: str) -> torch.Tensor:
        return getattr(self, memory_name.replace(".", "__"))
