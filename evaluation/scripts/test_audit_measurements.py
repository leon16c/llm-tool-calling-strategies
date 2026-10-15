from audit_measurements import (
    canonical,
    dependency_findings,
    numeric_values,
    offline_schema,
    projection,
)
from jsonschema import Draft202012Validator


def test_numeric_equivalence_preserves_boolean_string_and_array_distinctions():
    assert canonical(numeric_values({"x": 3.0})) == canonical(numeric_values({"x": 3}))
    assert canonical(numeric_values(True)) != canonical(numeric_values(1))
    assert canonical(numeric_values("3")) != canonical(numeric_values(3))
    assert canonical(numeric_values([1, 2])) != canonical(numeric_values([2, 1]))


def test_legacy_projection_can_hide_bad_dependency_but_separate_audit_detects_it():
    log = {
        "tool_calls": [
            {
                "index": 0,
                "tool_name": "get_sales_data",
                "status": "executed",
                "arguments": {},
                "result_ref": "source:123",
            },
            {
                "index": 1,
                "tool_name": "aggregate_data",
                "status": "executed",
                "arguments": {"data_id": "invented"},
                "result_ref": "aggregate:456",
            },
        ]
    }
    assert projection(log)[1]["arguments"] == {}
    assert dependency_findings(log) == [
        {"index": 1, "expected": "source:123", "actual": "invented"}
    ]
    log["tool_calls"][1]["arguments"]["data_id"] = "source:123"
    assert dependency_findings(log) == []


def test_offline_schema_checks_proposed_calls_and_retains_each_violation():
    validators = {
        "tool": Draft202012Validator(
            {
                "type": "object",
                "required": ["x"],
                "additionalProperties": False,
                "properties": {"x": {"type": "number"}},
            }
        )
    }
    log = {
        "tool_calls": [
            {"index": 0, "status": "proposed", "tool_name": "tool", "arguments": {"unexpected": 1}}
        ]
    }
    findings = offline_schema(log, validators)
    assert len(findings) == 1
    assert findings[0]["status"] == "proposed"
    assert {e["rule"] for e in findings[0]["errors"]} == {"required", "additionalProperties"}


def test_projection_retains_call_order_and_optionally_status():
    log = {
        "tool_calls": [
            {"tool_name": "a", "arguments": {"value": 3.0}, "status": "executed"},
            {"tool_name": "b", "arguments": {}, "status": "proposed"},
        ]
    }
    assert [c["tool_name"] for c in projection(log)] == ["a", "b"]
    assert projection(log, statuses=True)[1]["status"] == "proposed"
    assert "status" not in projection(log)[1]
    assert "3.0" in canonical(projection(log))
