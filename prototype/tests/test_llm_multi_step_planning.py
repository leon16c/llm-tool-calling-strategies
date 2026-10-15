from typing import Any

from toolcall_eval.fixed_sequence_llm_parameters import ModelInfo
from toolcall_eval.llm_multi_step_planning import (
    LLMMultiStepPlanning,
    MultiStepDecision,
)
from toolcall_eval.llm_tool_selection import PREVIOUS_RESULT_PLACEHOLDER, PlannedToolCall
from toolcall_eval.mock_tools import execute_tool
from toolcall_eval.models import EvaluationRequest, ToolCall
from toolcall_eval.ollama_mapper import OllamaMultiStepPlanner
from toolcall_eval.oracle import load_requests
from toolcall_eval.validators import RunLogValidator


class EarlyFinishPlanner:
    model_info = ModelInfo(provider="test", name="early_finish_planner")

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision:
        return MultiStepDecision(finish=True)


class WrongOrderPlanner:
    model_info = ModelInfo(provider="test", name="wrong_order_planner")

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision:
        return MultiStepDecision(
            finish=False,
            tool_call=PlannedToolCall(
                tool_name="aggregate_data",
                arguments={
                    "data_id": PREVIOUS_RESULT_PLACEHOLDER,
                    "group_by": ["region"],
                    "metrics": ["revenue"],
                },
            ),
        )


class InvalidDataDependencyStepPlanner:
    model_info = ModelInfo(provider="test", name="invalid_data_dependency_step_planner")

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision:
        if not executed_tool_calls:
            return MultiStepDecision(
                finish=False,
                tool_call=PlannedToolCall(
                    tool_name="get_sales_data",
                    arguments={
                        "timeframe": "last_30_days",
                        "dimensions": ["region"],
                    },
                ),
            )
        return MultiStepDecision(
            finish=False,
            tool_call=PlannedToolCall(
                tool_name="aggregate_data",
                arguments={
                    "data_id": "invented_dataset",
                    "group_by": ["region"],
                    "metrics": ["revenue"],
                },
            ),
        )


class ConcretePreviousResultStepPlanner:
    model_info = ModelInfo(provider="test", name="concrete_previous_result_step_planner")

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision:
        if not executed_tool_calls:
            return MultiStepDecision(
                finish=False,
                tool_call=PlannedToolCall(
                    tool_name="get_sales_data",
                    arguments={
                        "timeframe": "previous_quarter",
                        "dimensions": ["category"],
                    },
                ),
            )
        if len(executed_tool_calls) == 1:
            return MultiStepDecision(
                finish=False,
                tool_call=PlannedToolCall(
                    tool_name="aggregate_data",
                    arguments={
                        "data_id": executed_tool_calls[-1].result_ref,
                        "group_by": ["category"],
                        "metrics": ["revenue", "units_sold"],
                    },
                ),
            )
        if len(executed_tool_calls) == 2:
            return MultiStepDecision(
                finish=False,
                tool_call=PlannedToolCall(
                    tool_name="format_report",
                    arguments={
                        "data_id": executed_tool_calls[-1].result_ref,
                        "output_format": "markdown",
                    },
                ),
            )
        return MultiStepDecision(finish=True)


class NonTerminatingPlanner:
    model_info = ModelInfo(provider="test", name="non_terminating_planner")

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision:
        return MultiStepDecision(
            finish=False,
            tool_call=PlannedToolCall(
                tool_name="get_sales_data",
                arguments={
                    "timeframe": "last_30_days",
                    "dimensions": ["region"],
                },
            ),
        )


def _stable_projection(run_log: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (tool_call["tool_name"], tool_call["arguments"])
        for tool_call in run_log["tool_calls"]
    ]


def test_multi_step_variant_uses_iterative_oracle_sequence_for_all_requests() -> None:
    runner = LLMMultiStepPlanning()

    for request in load_requests():
        run_log = runner.run(request)
        actual_sequence = [tool_call["tool_name"] for tool_call in run_log["tool_calls"]]
        assert actual_sequence == request.expected_tool_sequence


def test_multi_step_variant_marks_tool_calls_as_llm_sourced() -> None:
    runner = LLMMultiStepPlanning()
    run_log = runner.run(load_requests()[0])

    assert run_log["variant"] == "llm_multi_step_planning"
    assert run_log["model"]["provider"] == "local"
    assert {tool_call["source"] for tool_call in run_log["tool_calls"]} == {"llm"}


def test_multi_step_variant_run_logs_match_schema() -> None:
    runner = LLMMultiStepPlanning()
    validator = RunLogValidator()

    for request in load_requests():
        run_log = runner.run(request)
        assert validator.validate(run_log) == []


def test_controlled_multi_step_planner_is_reproducible_for_same_request() -> None:
    runner = LLMMultiStepPlanning()
    request = load_requests()[0]

    first = runner.run(request)
    second = runner.run(request)

    assert _stable_projection(first) == _stable_projection(second)


def test_multi_step_planner_finishing_too_early_is_rejected() -> None:
    runner = LLMMultiStepPlanning(planner=EarlyFinishPlanner())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert run_log["tool_calls"] == []
    assert run_log["errors"][0]["code"] == "E2"


def test_multi_step_planner_wrong_next_tool_is_rejected() -> None:
    runner = LLMMultiStepPlanning(planner=WrongOrderPlanner())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert run_log["tool_calls"][0]["status"] == "proposed"
    assert run_log["errors"][0]["code"] == "E7"


def test_multi_step_planner_invalid_data_dependency_is_rejected() -> None:
    runner = LLMMultiStepPlanning(planner=InvalidDataDependencyStepPlanner())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert len(run_log["tool_calls"]) == 2
    assert run_log["tool_calls"][0]["status"] == "executed"
    assert run_log["tool_calls"][1]["status"] == "rejected"
    assert run_log["errors"][0]["code"] == "E3"


def test_multi_step_planner_accepts_concrete_immediate_result_reference() -> None:
    runner = LLMMultiStepPlanning(planner=ConcretePreviousResultStepPlanner())

    run_log = runner.run(load_requests()[1])

    expected_get_sales_arguments = {
        "timeframe": "previous_quarter",
        "dimensions": ["category"],
    }
    expected_get_sales_ref = execute_tool("get_sales_data", expected_get_sales_arguments)
    expected_aggregate_arguments = {
        "data_id": expected_get_sales_ref,
        "group_by": ["category"],
        "metrics": ["revenue", "units_sold"],
    }
    expected_aggregate_ref = execute_tool("aggregate_data", expected_aggregate_arguments)

    assert run_log["status"] == "success"
    assert run_log["tool_calls"][1]["arguments"]["data_id"] == expected_get_sales_ref
    assert run_log["tool_calls"][2]["arguments"]["data_id"] == expected_aggregate_ref


def test_multi_step_planner_non_termination_is_rejected() -> None:
    runner = LLMMultiStepPlanning(planner=NonTerminatingPlanner(), max_steps=1)

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert run_log["errors"][0]["code"] == "E10"


def test_ollama_multi_step_planner_uses_generated_decision_without_network_call() -> None:
    generated = """
    {
      "finish": false,
      "tool_call": {
        "tool_name": "get_sales_data",
        "arguments": {
          "timeframe": "previous_quarter",
          "dimensions": ["category"]
        }
      }
    }
    """
    planner = OllamaMultiStepPlanner(model="test-model", generate=lambda prompt: generated)

    decision = planner.plan_next_step(load_requests()[1], [])

    assert planner.model_info.provider == "ollama"
    assert planner.model_info.name == "test-model"
    assert decision == MultiStepDecision(
        finish=False,
        tool_call=PlannedToolCall(
            tool_name="get_sales_data",
            arguments={
                "timeframe": "previous_quarter",
                "dimensions": ["category"],
            },
        ),
    )
