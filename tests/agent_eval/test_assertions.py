from __future__ import annotations

import pytest

from ragqa.agent_eval import AssertionSpec, evaluate_assertion, resolve_json_path


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        ("equals", 2026),
        ("not_equals", 2025),
        ("contains", "search"),
        ("not_contains", "delete"),
        ("regex", r"^semantic"),
        ("in", ["semantic search", "keyword search"]),
    ],
)
def test_string_assertion_operators(operator: str, value: object) -> None:
    document = {
        "arguments": {"query": "semantic search", "filters": {"year": 2026}}
    }
    path = "arguments.filters.year" if operator in {"equals", "not_equals"} else "arguments.query"
    assertion = AssertionSpec(path=path, operator=operator, value=value)
    assert evaluate_assertion(assertion, document) is True


@pytest.mark.parametrize("operator", ["gte", "greater_than_or_equal"])
def test_greater_than_or_equal_aliases(operator: str) -> None:
    assertion = AssertionSpec(path="arguments.limit", operator=operator, value=5)
    assert evaluate_assertion(assertion, {"arguments": {"limit": 5}}) is True


@pytest.mark.parametrize("operator", ["lte", "less_than_or_equal"])
def test_less_than_or_equal_aliases(operator: str) -> None:
    assertion = AssertionSpec(path="arguments.limit", operator=operator, value=5)
    assert evaluate_assertion(assertion, {"arguments": {"limit": 4}}) is True


def test_exists_distinguishes_missing_from_null() -> None:
    document = {"arguments": {"optional": None}}
    assert evaluate_assertion(
        AssertionSpec(path="arguments.optional", operator="exists"), document
    )
    assert evaluate_assertion(
        AssertionSpec(path="arguments.missing", operator="exists", value=False),
        document,
    )


def test_json_path_supports_list_indexes_without_attribute_access() -> None:
    found, value = resolve_json_path({"items": [{"id": "a"}]}, "items.0.id")
    assert found is True
    assert value == "a"
    assert resolve_json_path({}, "__class__.__mro__") == (False, None)


def test_invalid_path_and_regex_fail_safely() -> None:
    with pytest.raises(ValueError, match="Invalid JSON path"):
        resolve_json_path({}, "arguments..query")
    assertion = AssertionSpec(path="value", operator="regex", value="[")
    assert evaluate_assertion(assertion, {"value": "text"}) is False
    not_contains = AssertionSpec(
        path="value", operator="not_contains", value="text"
    )
    assert evaluate_assertion(not_contains, {"value": 42}) is False
