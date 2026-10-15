from typing import Any

from toolcall_eval.fixed_sequence_llm_parameters import ModelInfo
from toolcall_eval.llm_tool_selection import (
    PREVIOUS_RESULT_PLACEHOLDER,
    LLMToolSelection,
    PlannedToolCall,
)
from toolcall_eval.models import EvaluationRequest
from toolcall_eval.ollama_mapper import OllamaToolCallPlanner
from toolcall_eval.oracle import load_requests
from toolcall_eval.validators import RunLogValidator


class UnsupportedToolPlanner:
    model_info = ModelInfo(provider="test", name="unsupported_tool_planner")

    def plan_tool_calls(self, request: EvaluationRequest) -> list[PlannedToolCall]:
        return [PlannedToolCall(tool_name="send_email", arguments={})]


class InvalidDataDependencyPlanner:
    model_info = ModelInfo(provider="test", name="invalid_data_dependency_planner")

    def plan_tool_calls(self, request: EvaluationRequest) -> list[PlannedToolCall]:
        return [
            PlannedToolCall(
                tool_name="get_sales_data",
                arguments={
                    "timeframe": "last_30_days",
                    "dimensions": ["region"],
                },
            ),
            PlannedToolCall(
                tool_name="aggregate_data",
                arguments={
                    "data_id": "invented_dataset",
                    "group_by": ["region"],
                    "metrics": ["revenue"],
                },
            ),
            PlannedToolCall(
                tool_name="detect_anomalies",
                arguments={
                    "data_id": PREVIOUS_RESULT_PLACEHOLDER,
                    "method": "z_score",
                    "threshold": 3.0,
                },
            ),
            PlannedToolCall(
                tool_name="format_report",
                arguments={
                    "data_id": PREVIOUS_RESULT_PLACEHOLDER,
                    "output_format": "markdown",
                },
            )
        ]


class MissingReportPlanner:
    model_info = ModelInfo(provider="test", name="missing_report_planner")

    def plan_tool_calls(self, request: EvaluationRequest) -> list[PlannedToolCall]:
        return [
            PlannedToolCall(
                tool_name="get_sales_data",
                arguments={
                    "timeframe": "previous_quarter",
                    "dimensions": ["category"],
                },
            ),
            PlannedToolCall(
                tool_name="aggregate_data",
                arguments={
                    "data_id": PREVIOUS_RESULT_PLACEHOLDER,
                    "group_by": ["category"],
                    "metrics": ["revenue", "units_sold"],
                },
            ),
        ]


def _stable_projection(run_log: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (tool_call["tool_name"], tool_call["arguments"])
        for tool_call in run_log["tool_calls"]
    ]


def test_tool_selection_variant_uses_planned_tool_sequence_for_all_requests() -> None:
    runner = LLMToolSelection()

    for request in load_requests():
        run_log = runner.run(request)
        actual_sequence = [tool_call["tool_name"] for tool_call in run_log["tool_calls"]]
        assert actual_sequence == request.expected_tool_sequence


def test_tool_selection_variant_marks_tool_calls_as_llm_sourced() -> None:
    runner = LLMToolSelection()
    run_log = runner.run(load_requests()[0])

    assert run_log["variant"] == "llm_tool_selection"
    assert run_log["model"]["provider"] == "local"
    assert {tool_call["source"] for tool_call in run_log["tool_calls"]} == {"llm"}


def test_tool_selection_variant_run_logs_match_schema() -> None:
    runner = LLMToolSelection()
    validator = RunLogValidator()

    for request in load_requests():
        run_log = runner.run(request)
        assert validator.validate(run_log) == []


def test_controlled_tool_selection_planner_is_reproducible_for_same_request() -> None:
    runner = LLMToolSelection()
    request = load_requests()[0]

    first = runner.run(request)
    second = runner.run(request)

    assert _stable_projection(first) == _stable_projection(second)


def test_unsupported_llm_selected_tool_is_rejected() -> None:
    runner = LLMToolSelection(planner=UnsupportedToolPlanner())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert run_log["tool_calls"] == []
    assert run_log["errors"][0]["code"] == "E1"
    assert run_log["validation_events"][0]["event_type"] == "tool_call_rejected"


def test_missing_llm_selected_tool_call_is_rejected() -> None:
    runner = LLMToolSelection(planner=MissingReportPlanner())

    run_log = runner.run(load_requests()[1])

    assert run_log["status"] == "rejected"
    assert [tool_call["status"] for tool_call in run_log["tool_calls"]] == [
        "proposed",
        "proposed",
    ]
    assert run_log["errors"][0]["code"] == "E2"
    assert run_log["validation_events"][0]["event_type"] == "semantic_validation_failed"


def test_invalid_llm_planned_data_dependency_is_rejected() -> None:
    runner = LLMToolSelection(planner=InvalidDataDependencyPlanner())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert len(run_log["tool_calls"]) == 2
    assert run_log["tool_calls"][0]["status"] == "executed"
    assert run_log["tool_calls"][1]["status"] == "rejected"
    assert run_log["errors"][0]["code"] == "E3"
    assert run_log["validation_events"][0]["event_type"] == "semantic_validation_failed"


def test_ollama_tool_call_planner_uses_generated_plan_without_network_call() -> None:
    generated = """
    {
      "tool_calls": [
        {
          "tool_name": "get_sales_data",
          "arguments": {
            "timeframe": "previous_quarter",
            "dimensions": ["category"]
          }
        }
      ]
    }
    """
    planner = OllamaToolCallPlanner(model="test-model", generate=lambda prompt: generated)

    planned_tool_calls = planner.plan_tool_calls(load_requests()[1])

    assert planner.model_info.provider == "ollama"
    assert planner.model_info.name == "test-model"
    assert planned_tool_calls == [
        PlannedToolCall(
            tool_name="get_sales_data",
            arguments={
                "timeframe": "previous_quarter",
                "dimensions": ["category"],
            },
        )
    ]
