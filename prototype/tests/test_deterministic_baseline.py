from typing import Any

from toolcall_eval.deterministic_baseline import DeterministicBaseline
from toolcall_eval.models import EvaluationRequest
from toolcall_eval.oracle import load_requests


def _stable_projection(run_log: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (tool_call["tool_name"], tool_call["arguments"])
        for tool_call in run_log["tool_calls"]
    ]


def test_baseline_uses_expected_tool_sequence_for_all_requests() -> None:
    runner = DeterministicBaseline()

    for request in load_requests():
        run_log = runner.run(request)
        actual_sequence = [tool_call["tool_name"] for tool_call in run_log["tool_calls"]]
        assert actual_sequence == request.expected_tool_sequence


def test_successful_controlled_requests_have_no_failures() -> None:
    runner = DeterministicBaseline()

    for request in load_requests():
        run_log = runner.run(request)
        assert run_log["status"] == "success"
        assert run_log["validation_events"] == []
        assert run_log["errors"] == []
        assert run_log["metrics"]["task_success"] is True
        assert run_log["metrics"]["schema_validation_failures"] == 0
        assert run_log["metrics"]["tool_execution_failures"] == 0


def test_baseline_tool_calls_are_reproducible_for_same_request() -> None:
    runner = DeterministicBaseline()
    request = load_requests()[0]

    first = runner.run(request)
    second = runner.run(request)

    assert _stable_projection(first) == _stable_projection(second)


def test_invalid_tool_parameters_are_rejected_before_follow_up_execution() -> None:
    runner = DeterministicBaseline()
    request = EvaluationRequest(
        id="RX01",
        category="invalid",
        user_request="Invalid controlled request.",
        expected_tool_sequence=["get_sales_data", "aggregate_data", "format_report"],
        expected_parameters={
            "timeframe": "unsupported_period",
            "dimensions": ["region"],
            "group_by": ["region"],
            "metrics": ["revenue"],
            "output_format": "markdown",
        },
        primary_evaluation_focus="rejection behavior",
    )

    run_log = runner.run(request)

    assert run_log["status"] == "rejected"
    assert len(run_log["tool_calls"]) == 1
    assert run_log["tool_calls"][0]["status"] == "rejected"
    assert run_log["validation_events"][0]["event_type"] == "schema_validation_failed"
    assert run_log["metrics"]["task_success"] is False
