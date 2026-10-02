from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ragqa.agent_eval.models import AssertionSpec


def resolve_json_path(document: Any, path: str) -> tuple[bool, Any]:
    """Resolve dict keys and list indexes with dot notation, without code execution."""

    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"Invalid JSON path: {path!r}")

    current = document
    for part in parts:
        if isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
            continue

        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if not part.isdecimal():
                return False, None
            index = int(part)
            if index >= len(current):
                return False, None
            current = current[index]
            continue

        return False, None

    return True, current


def evaluate_assertion(assertion: AssertionSpec, document: Any) -> bool:
    """Evaluate one deterministic assertion against a JSON-compatible value."""

    try:
        found, actual = resolve_json_path(document, assertion.path)
    except ValueError:
        return False

    operator = assertion.operator
    expected = assertion.value

    if operator == "exists":
        should_exist = expected if isinstance(expected, bool) else True
        return found is should_exist

    if not found:
        return False

    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "contains":
        return _contains(actual, expected)
    if operator == "not_contains":
        return _is_container(actual) and not _contains(actual, expected)
    if operator == "regex":
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        try:
            return re.search(expected, actual) is not None
        except re.error:
            return False
    if operator == "in":
        return _contains(expected, actual)
    if operator in {"gte", "greater_than_or_equal"}:
        return _compare(actual, expected, comparison="gte")
    if operator in {"lte", "less_than_or_equal"}:
        return _compare(actual, expected, comparison="lte")

    return False


def _contains(container: Any, item: Any) -> bool:
    if not _is_container(container):
        return False
    try:
        return item in container
    except TypeError:
        return False


def _is_container(value: Any) -> bool:
    return isinstance(value, (str, bytes, Mapping, Sequence, set, frozenset))


def _compare(actual: Any, expected: Any, comparison: str) -> bool:
    try:
        if comparison == "gte":
            return bool(actual >= expected)
        return bool(actual <= expected)
    except TypeError:
        return False
