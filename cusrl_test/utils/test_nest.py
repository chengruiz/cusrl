import pytest

from cusrl.utils.misc import MISSING
from cusrl.utils.nest import (
    flatten_nested,
    get_schema,
    iterate_nested,
    map_nested,
    reconstruct_nested,
    zip_nested,
)


def sample_nested() -> dict[str, object]:
    return {
        "a": 1,
        "b": {
            "c": [10, 20],
            "d": (30, {"e": 40}),
        },
    }


def test_get_schema_returns_paths_for_every_leaf():
    assert get_schema(sample_nested()) == {
        "a": "a",
        "b": {
            "c": ["b.c.0", "b.c.1"],
            "d": ("b.d.0", {"e": "b.d.1.e"}),
        },
    }


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        (0, "root"),
        (1, {"a": "root/a", "b": "root/b"}),
        (
            2,
            {
                "a": "root/a",
                "b": {
                    "c": "root/b/c",
                    "d": "root/b/d",
                },
            },
        ),
    ],
)
def test_get_schema_honors_prefix_separator_and_max_depth(max_depth: int, expected: object):
    assert get_schema(sample_nested(), prefix="root", max_depth=max_depth, separator="/") == expected


def test_iterate_nested_returns_leaf_items_in_stable_order():
    assert list(iterate_nested(sample_nested())) == [
        ("a", 1),
        ("b.c.0", 10),
        ("b.c.1", 20),
        ("b.d.0", 30),
        ("b.d.1.e", 40),
    ]


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        (0, [("root", sample_nested())]),
        (1, [("root.a", 1), ("root.b", sample_nested()["b"])]),
    ],
)
def test_iterate_nested_stops_recursing_at_max_depth(max_depth: int, expected: list[tuple[str, object]]):
    assert list(iterate_nested(sample_nested(), prefix="root", max_depth=max_depth)) == expected


def test_flatten_nested_matches_iterate_nested_output():
    nested = sample_nested()

    assert flatten_nested(nested) == dict(iterate_nested(nested))
    assert flatten_nested(nested, prefix="root", max_depth=1) == {
        "root.a": 1,
        "root.b": nested["b"],
    }


def test_map_nested_transforms_leaves_and_preserves_container_types():
    mapped = map_nested(lambda value: value + 1, sample_nested())

    assert mapped == {
        "a": 2,
        "b": {
            "c": [11, 21],
            "d": (31, {"e": 41}),
        },
    }
    assert isinstance(mapped["b"]["c"], list)
    assert isinstance(mapped["b"]["d"], tuple)
    assert isinstance(mapped["b"]["d"][1], dict)


def test_reconstruct_nested_round_trips_flattened_data_and_schema():
    nested = sample_nested()
    flattened = flatten_nested(nested)
    schema = get_schema(nested)

    assert reconstruct_nested(flattened, schema) == nested
    assert reconstruct_nested({"leaf": 123}, "leaf") == 123


def test_zip_nested_with_single_argument_matches_iterate_nested():
    expected = list(iterate_nested(sample_nested(), prefix="root", max_depth=1))
    actual = [(key, values[0]) for key, values in zip_nested(sample_nested(), prefix="root", max_depth=1)]

    assert actual == expected


def test_zip_nested_merges_matching_structures():
    result = list(
        zip_nested(
            {"x": [1, 2], "y": {"z": 3}},
            {"x": [10, 20], "y": {"z": 30}},
        )
    )

    assert result == [
        ("x.0", (1, 10)),
        ("x.1", (2, 20)),
        ("y.z", (3, 30)),
    ]


def test_zip_nested_fills_missing_values_for_misaligned_keys_and_lengths():
    sequence_result = list(zip_nested({"x": [1, 2, 3]}, {"x": [10, 20]}))
    assert sequence_result[:2] == [("x.0", (1, 10)), ("x.1", (2, 20))]
    assert sequence_result[2][0] == "x.2"
    assert sequence_result[2][1][0] == 3
    assert sequence_result[2][1][1] is MISSING

    mapping_result = list(zip_nested({"a": 1, "b": 2}, {"a": 10, "c": 30}))
    assert mapping_result[0] == ("a", (1, 10))
    assert mapping_result[1][0] == "b"
    assert mapping_result[1][1][0] == 2
    assert mapping_result[1][1][1] is MISSING
    assert mapping_result[2][0] == "c"
    assert mapping_result[2][1][0] is MISSING
    assert mapping_result[2][1][1] == 30


def test_zip_nested_honors_max_depth_and_empty_input():
    assert list(
        zip_nested(
            {"x": [1, 2], "y": {"z": 3}},
            {"x": [10, 20], "y": {"z": 30}},
            max_depth=1,
        )
    ) == [
        ("x", ([1, 2], [10, 20])),
        ("y", ({"z": 3}, {"z": 30})),
    ]
    assert list(zip_nested()) == []
