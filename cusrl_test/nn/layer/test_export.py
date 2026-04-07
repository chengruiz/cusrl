import pytest
import torch
from torch import nn

from cusrl.nn.layer.export import FlowGraph, StatefulWrapper, StatelessWrapper, get_num_tensors
from cusrl.nn.module.module import Module


class AddBias(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class WithIntermediate(Module):
    def __init__(self):
        super().__init__(input_dim=2, output_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x + 1
        self.intermediate_repr["hidden"] = output * 3
        return output


class StatefulAccumulator(nn.Module):
    def forward(self, observation: torch.Tensor, memory_in: dict[str, torch.Tensor]):
        next_hidden = memory_in["hidden"] + observation
        return next_hidden * 2, {"hidden": next_hidden}


def test_flow_graph_executes_nodes_in_order_and_tracks_metadata():
    graph = FlowGraph("toy_graph")
    graph.add_node(
        AddBias(),
        module_name="add_bias",
        input_names={"x": "input", "bias": "bias"},
        output_names="hidden",
        expose_outputs=False,
        info={"stage": "preprocess"},
    )
    graph.add_node(
        Double(),
        module_name="doubler",
        input_names={"x": "hidden"},
        output_names="output",
    )
    input = torch.tensor([[1.0, 2.0]])
    bias = torch.tensor([[0.5, -0.5]])

    (output,) = graph(input=input, bias=bias)

    assert torch.allclose(output, (input + bias) * 2)
    assert graph.info == {"stage": "preprocess"}
    assert list(graph.named_nodes) == ["add_bias", "doubler"]
    assert graph.output_names == ["output"]


def test_flow_graph_exposes_intermediate_repr_from_cusrl_module():
    graph = FlowGraph("repr_graph", output_names=["output", "feature.hidden"])
    graph.add_node(
        WithIntermediate(),
        module_name="feature",
        input_names={"x": "input"},
        output_names="output",
    )
    input = torch.tensor([[1.0, 2.0]])

    output, hidden = graph(input=input)

    assert torch.allclose(output, input + 1)
    assert torch.allclose(hidden, (input + 1) * 3)


def test_flow_graph_rejects_duplicate_module_names():
    graph = FlowGraph("dup_graph")
    graph.add_node(AddBias(), module_name="node", input_names={"x": "input", "bias": "bias"}, output_names="output")

    with pytest.raises(ValueError, match="already used"):
        graph.add_node(AddBias(), module_name="node", input_names={"x": "input", "bias": "bias"}, output_names="copy")


def test_stateless_and_stateful_wrappers_flatten_nested_memory_inputs():
    graph = FlowGraph("stateful_graph", output_names=["action", "memory_out"])
    graph.add_node(
        StatefulAccumulator(),
        module_name="core",
        input_names={"observation": "observation", "memory_in": "memory_in"},
        output_names=("action", "memory_out"),
    )
    example_inputs = {
        "observation": torch.zeros(2, 3),
        "memory_in": {"hidden": torch.zeros(2, 3)},
    }
    stateless = StatelessWrapper(graph, example_inputs, graph.output_names)
    observation = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])

    stateless_output = stateless(observation=observation, memory_in__hidden=torch.zeros(2, 3))
    stateful = StatefulWrapper(stateless, example_inputs)
    action = stateful(observation=observation)

    assert set(stateless_output) == {"action", "memory_out__hidden"}
    assert torch.allclose(stateless_output["memory_out__hidden"], observation)
    assert torch.allclose(action, observation * 2)
    assert torch.allclose(stateful.memory_in__hidden, observation)

    next_observation = torch.full((2, 3), 2.0)
    next_action = stateful(observation=next_observation)
    assert torch.allclose(next_action, (observation + next_observation) * 2)
    assert torch.allclose(stateful.memory_in__hidden, observation + next_observation)

    stateful.reset(torch.tensor([True, False]))
    assert torch.allclose(stateful.memory_in__hidden[0], torch.zeros(3))
    assert torch.allclose(stateful.memory_in__hidden[1], observation[1] + next_observation[1])


def test_get_num_tensors_counts_nested_tensor_structures():
    nested = [torch.zeros(1), (torch.ones(1), [torch.randn(1), torch.randn(1)])]

    assert get_num_tensors(nested) == 4
