from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import uuid4

from toolcall_eval.mock_tools import execute_tool
from toolcall_eval.models import EvaluationRequest, ToolCall, ToolName
from toolcall_eval.validators import ToolSchemaValidator

VARIANT = "fixed_sequence_llm_parameters"


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    name: str
    temperature: float = 0.0
    seed: int | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "name": self.name,
            "temperature": self.temperature,
            "seed": self.seed,
        }


class ParameterMapper(Protocol):
    model_info: ModelInfo

    def map_parameters(
        self,
        request: EvaluationRequest,
        tool_name: ToolName,
        previous_result_ref: str | None,
    ) -> dict[str, Any]: ...


class OracleBackedParameterMapper:
    """Controlled stand-in for an LLM mapper until an external LLM client is added."""

    model_info = ModelInfo(
        provider="local",
        name="oracle_backed_parameter_mapper",
        temperature=0.0,
        seed=0,
    )

    def map_parameters(
        self,
        request: EvaluationRequest,
        tool_name: ToolName,
        previous_result_ref: str | None,
    ) -> dict[str, Any]:
        parameters = request.expected_parameters
        if tool_name == "get_sales_data":
            return {
                "timeframe": parameters["timeframe"],
                "dimensions": parameters["dimensions"],
            }
        if tool_name == "aggregate_data":
            return {
                "data_id": self._require_previous_result(previous_result_ref, tool_name),
                "group_by": parameters["group_by"],
                "metrics": parameters["metrics"],
            }
        if tool_name == "detect_anomalies":
            return {
                "data_id": self._require_previous_result(previous_result_ref, tool_name),
                "method": parameters["method"],
                "threshold": parameters["threshold"],
            }
        if tool_name == "compare_previous_period":
            return {
                "data_id": self._require_previous_result(previous_result_ref, tool_name),
                "previous_period": parameters["previous_period"],
            }
        if tool_name == "format_report":
            return {
                "data_id": self._require_previous_result(previous_result_ref, tool_name),
                "output_format": parameters["output_format"],
            }
        raise ValueError(f"Unsupported tool: {tool_name}")

    def _require_previous_result(self, previous_result_ref: str | None, tool_name: ToolName) -> str:
        if previous_result_ref is None:
            raise ValueError(f"{tool_name} requires a previous data_id.")
        return previous_result_ref


class FixedSequenceLLMParameters:
    def __init__(
        self,
        mapper: ParameterMapper | None = None,
        validator: ToolSchemaValidator | None = None,
    ) -> None:
        self._mapper = mapper or OracleBackedParameterMapper()
        self._validator = validator or ToolSchemaValidator()

    def run(self, request: EvaluationRequest, repetition_index: int = 0) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        tool_calls: list[ToolCall] = []
        validation_events: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        schema_failures = 0
        previous_result_ref: str | None = None

        for index, tool_name in enumerate(request.expected_tool_sequence):
            try:
                arguments = self._mapper.map_parameters(request, tool_name, previous_result_ref)
            except (KeyError, TypeError, ValueError) as exc:
                validation_events.append(
                    {
                        "index": len(validation_events),
                        "tool_call_index": index,
                        "event_type": "semantic_validation_failed",
                        "message": f"Parameter mapper failed for {tool_name}: {exc}",
                    }
                )
                errors.append(
                    {
                        "code": "E5",
                        "tool_call_index": index,
                        "description": f"Could not map parameters for {tool_name}.",
                    }
                )
                tool_calls.append(
                    ToolCall(
                        index=index,
                        tool_name=tool_name,
                        arguments={},
                        source="llm",
                        status="rejected",
                    )
                )
                break

            validation_errors = self._validator.validate_tool_call(tool_name, arguments)
            if validation_errors:
                schema_failures += len(validation_errors)
                validation_events.extend(
                    {
                        "index": len(validation_events),
                        "tool_call_index": index,
                        "event_type": "schema_validation_failed",
                        "message": message,
                    }
                    for message in validation_errors
                )
                errors.append(
                    {
                        "code": "E4",
                        "tool_call_index": index,
                        "description": f"Invalid mapped parameters for {tool_name}.",
                    }
                )
                status: Literal["executed", "rejected"] = "rejected"
                result_ref = None
            else:
                status = "executed"
                result_ref = execute_tool(tool_name, arguments)
                previous_result_ref = result_ref

            tool_calls.append(
                ToolCall(
                    index=index,
                    tool_name=tool_name,
                    arguments=arguments,
                    source="llm",
                    status=status,
                    result_ref=result_ref,
                )
            )

            if status == "rejected":
                break

        completed_at = datetime.now(UTC)
        run_status = "success" if not errors else "rejected"
        final_response = self._format_final_response(request, previous_result_ref)

        return {
            "schema_version": "0.1.0",
            "run_id": str(uuid4()),
            "request_id": request.id,
            "repetition_index": repetition_index,
            "variant": VARIANT,
            "model": self._mapper.model_info.to_log_dict(),
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
            "latency_ms": int((completed_at - started_at).total_seconds() * 1000),
            "status": run_status,
            "input": {
                "user_request": request.user_request,
                "expected_tool_sequence": request.expected_tool_sequence,
            },
            "tool_calls": [tool_call.to_log_dict() for tool_call in tool_calls],
            "validation_events": validation_events,
            "errors": errors,
            "metrics": {
                "task_success": run_status == "success",
                "schema_validation_failures": schema_failures,
                "tool_execution_failures": 0,
                "retry_count": 0,
                "fallback_count": 0,
            },
            "output": {
                "final_response": final_response,
                "output_hash": execute_tool("format_report", {"final_response": final_response}),
            },
        }

    def _format_final_response(self, request: EvaluationRequest, data_id: str | None) -> str:
        return (
            f"{request.id} completed by fixed-sequence LLM parameter mapping "
            f"with result {data_id}."
        )
