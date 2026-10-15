from typing import Any

from toolcall_eval.fixed_sequence_llm_parameters import (
    FixedSequenceLLMParameters,
    ModelInfo,
)
from toolcall_eval.models import EvaluationRequest, ToolName
from toolcall_eval.ollama_mapper import OllamaParameterMapper, parse_json_object
from toolcall_eval.oracle import load_requests
from toolcall_eval.validators import RunLogValidator


class InvalidFirstCallMapper:
    model_info = ModelInfo(provider="test", name="invalid_first_call_mapper")

    def map_parameters(
        self,
        request: EvaluationRequest,
        tool_name: ToolName,
        previous_result_ref: str | None,
    ) -> dict[str, Any]:
        return {"timeframe": "unsupported_period", "dimensions": ["region"]}


class FailingMapper:
    model_info = ModelInfo(provider="test", name="failing_mapper")

    def map_parameters(
        self,
        request: EvaluationRequest,
        tool_name: ToolName,
        previous_result_ref: str | None,
    ) -> dict[str, Any]:
        raise ValueError("no parameter object produced")


def _stable_projection(run_log: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (tool_call["tool_name"], tool_call["arguments"])
        for tool_call in run_log["tool_calls"]
    ]


def test_fixed_sequence_variant_uses_expected_tool_sequence_for_all_requests() -> None:
    runner = FixedSequenceLLMParameters()

    for request in load_requests():
        run_log = runner.run(request)
        actual_sequence = [tool_call["tool_name"] for tool_call in run_log["tool_calls"]]
        assert actual_sequence == request.expected_tool_sequence


def test_fixed_sequence_variant_marks_tool_calls_as_llm_sourced() -> None:
    runner = FixedSequenceLLMParameters()
    run_log = runner.run(load_requests()[0])

    assert run_log["variant"] == "fixed_sequence_llm_parameters"
    assert run_log["model"]["provider"] == "local"
    assert {tool_call["source"] for tool_call in run_log["tool_calls"]} == {"llm"}


def test_fixed_sequence_variant_run_logs_match_schema() -> None:
    runner = FixedSequenceLLMParameters()
    validator = RunLogValidator()

    for request in load_requests():
        run_log = runner.run(request)
        assert validator.validate(run_log) == []


def test_controlled_mapper_is_reproducible_for_same_request() -> None:
    runner = FixedSequenceLLMParameters()
    request = load_requests()[0]

    first = runner.run(request)
    second = runner.run(request)

    assert _stable_projection(first) == _stable_projection(second)


def test_invalid_llm_mapped_parameters_are_rejected_before_follow_up_execution() -> None:
    runner = FixedSequenceLLMParameters(mapper=InvalidFirstCallMapper())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert len(run_log["tool_calls"]) == 1
    assert run_log["tool_calls"][0]["status"] == "rejected"
    assert run_log["tool_calls"][0]["source"] == "llm"
    assert run_log["validation_events"][0]["event_type"] == "schema_validation_failed"
    assert run_log["metrics"]["task_success"] is False


def test_mapper_failure_is_classified_as_incomplete_parameter_mapping() -> None:
    runner = FixedSequenceLLMParameters(mapper=FailingMapper())

    run_log = runner.run(load_requests()[0])

    assert run_log["status"] == "rejected"
    assert run_log["tool_calls"][0]["status"] == "rejected"
    assert run_log["validation_events"][0]["event_type"] == "semantic_validation_failed"
    assert run_log["errors"][0]["code"] == "E5"


def test_ollama_json_parser_accepts_markdown_fenced_json() -> None:
    generated = """```json
{
  "timeframe": "previous_quarter",
  "dimensions": ["category"]
}
```"""

    assert parse_json_object(generated) == {
        "timeframe": "previous_quarter",
        "dimensions": ["category"],
    }


def test_ollama_mapper_uses_generated_json_without_network_call() -> None:
    mapper = OllamaParameterMapper(
        model="test-model",
        generate=lambda prompt: '{"timeframe": "previous_quarter", "dimensions": ["category"]}',
    )

    arguments = mapper.map_parameters(load_requests()[1], "get_sales_data", None)

    assert mapper.model_info.provider == "ollama"
    assert mapper.model_info.name == "test-model"
    assert arguments == {
        "timeframe": "previous_quarter",
        "dimensions": ["category"],
    }
