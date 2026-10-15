from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from toolcall_eval.fixed_sequence_llm_parameters import ModelInfo, OracleBackedParameterMapper
from toolcall_eval.mock_tools import execute_tool
from toolcall_eval.models import EvaluationRequest, ToolCall, ToolName
from toolcall_eval.validators import ToolSchemaValidator

VARIANT = "llm_tool_selection"
PREVIOUS_RESULT_PLACEHOLDER = "$previous_result"
TOOL_NAMES = {
    "get_sales_data",
    "aggregate_data",
    "detect_anomalies",
    "compare_previous_period",
    "format_report",
}


@dataclass(frozen=True)
class PlannedToolCall:
    tool_name: str
    arguments: dict[str, Any]


class ToolCallPlanner(Protocol):
    model_info: ModelInfo

    def plan_tool_calls(self, request: EvaluationRequest) -> list[PlannedToolCall]: ...


class OracleBackedToolCallPlanner:
    """Controlled stand-in for an LLM planner until external LLM runs are executed."""

    model_info = ModelInfo(
        provider="local",
        name="oracle_backed_tool_call_planner",
        temperature=0.0,
        seed=0,
    )

    def __init__(self) -> None:
        self._parameter_mapper = OracleBackedParameterMapper()

    def plan_tool_calls(self, request: EvaluationRequest) -> list[PlannedToolCall]:
        return [
            PlannedToolCall(
                tool_name=tool_name,
                arguments=self._parameter_mapper.map_parameters(
                    request,
                    tool_name,
                    PREVIOUS_RESULT_PLACEHOLDER,
                ),
            )
            for tool_name in request.expected_tool_sequence
        ]


class LLMToolSelection:
    def __init__(
        self,
        planner: ToolCallPlanner | None = None,
        validator: ToolSchemaValidator | None = None,
    ) -> None:
        self._planner = planner or OracleBackedToolCallPlanner()
        self._validator = validator or ToolSchemaValidator()

    def run(self, request: EvaluationRequest, repetition_index: int = 0) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        tool_calls: list[ToolCall] = []
        validation_events: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        schema_failures = 0
        last_data_id: str | None = None

        try:
            planned_tool_calls = self._planner.plan_tool_calls(request)
        except (KeyError, TypeError, ValueError) as exc:
            validation_events.append(
                {
                    "index": 0,
                    "event_type": "semantic_validation_failed",
                    "message": f"Tool-call planner failed: {exc}",
                }
            )
            errors.append(
                {
                    "code": "E3",
                    "description": "Could not produce a valid tool-call plan.",
                }
            )
            planned_tool_calls = []

        if not errors:
            sequence_error = self._validate_planned_sequence(planned_tool_calls, request)
            if sequence_error is not None:
                validation_events.append(
                    {
                        "index": len(validation_events),
                        "event_type": "semantic_validation_failed",
                        "message": sequence_error["message"],
                    }
                )
                errors.append(
                    {
                        "code": sequence_error["code"],
                        "description": sequence_error["description"],
                    }
                )
                tool_calls.extend(
                    ToolCall(
                        index=index,
                        tool_name=cast(ToolName, planned_call.tool_name),
                        arguments=planned_call.arguments,
                        source="llm",
                        status="proposed",
                    )
                    for index, planned_call in enumerate(planned_tool_calls)
                    if planned_call.tool_name in TOOL_NAMES
                )

        if not errors:
            for index, planned_call in enumerate(planned_tool_calls):
                if planned_call.tool_name not in TOOL_NAMES:
                    validation_events.append(
                        {
                            "index": len(validation_events),
                            "tool_call_index": index,
                            "event_type": "tool_call_rejected",
                            "message": f"Unsupported tool selected: {planned_call.tool_name}",
                        }
                    )
                    errors.append(
                        {
                            "code": "E1",
                            "tool_call_index": index,
                            "description": "The planner selected an unsupported tool.",
                        }
                    )
                    break

                tool_name = cast(ToolName, planned_call.tool_name)
                arguments = dict(planned_call.arguments)

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
                            "description": f"Invalid planned parameters for {tool_name}.",
                        }
                    )
                    tool_calls.append(
                        ToolCall(
                            index=index,
                            tool_name=tool_name,
                            arguments=arguments,
                            source="llm",
                            status="rejected",
                        )
                    )
                    break

                semantic_error = self._resolve_data_reference(arguments, last_data_id)
                if semantic_error is not None:
                    validation_events.append(
                        {
                            "index": len(validation_events),
                            "tool_call_index": index,
                            "event_type": "semantic_validation_failed",
                            "message": semantic_error,
                        }
                    )
                    errors.append(
                        {
                            "code": "E3",
                            "tool_call_index": index,
                            "description": f"Invalid data dependency for {tool_name}.",
                        }
                    )
                    tool_calls.append(
                        ToolCall(
                            index=index,
                            tool_name=tool_name,
                            arguments=arguments,
                            source="llm",
                            status="rejected",
                        )
                    )
                    break

                result_ref = execute_tool(tool_name, arguments)
                last_data_id = result_ref
                tool_calls.append(
                    ToolCall(
                        index=index,
                        tool_name=tool_name,
                        arguments=arguments,
                        source="llm",
                        status="executed",
                        result_ref=result_ref,
                    )
                )

        completed_at = datetime.now(UTC)
        run_status = "success" if not errors else "rejected"
        final_response = self._format_final_response(request, last_data_id)

        return {
            "schema_version": "0.1.0",
            "run_id": str(uuid4()),
            "request_id": request.id,
            "repetition_index": repetition_index,
            "variant": VARIANT,
            "model": self._planner.model_info.to_log_dict(),
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

    def _validate_planned_sequence(
        self,
        planned_tool_calls: list[PlannedToolCall],
        request: EvaluationRequest,
    ) -> dict[str, str] | None:
        planned_sequence = [planned_call.tool_name for planned_call in planned_tool_calls]
        expected_sequence = request.expected_tool_sequence
        if planned_sequence == expected_sequence:
            return None
        if any(tool_name not in TOOL_NAMES for tool_name in planned_sequence):
            return None
        if self._is_subsequence(planned_sequence, expected_sequence):
            return {
                "code": "E2",
                "description": "The planner omitted at least one required tool call.",
                "message": (
                    "Planned tool sequence omits required calls: "
                    f"planned={planned_sequence}, expected={expected_sequence}"
                ),
            }
        if self._is_subsequence(expected_sequence, planned_sequence):
            return {
                "code": "E3",
                "description": "The planner added at least one unnecessary tool call.",
                "message": (
                    "Planned tool sequence contains unnecessary calls: "
                    f"planned={planned_sequence}, expected={expected_sequence}"
                ),
            }
        if sorted(planned_sequence) == sorted(expected_sequence):
            return {
                "code": "E7",
                "description": "The planner selected the right tools in the wrong order.",
                "message": (
                    "Planned tool sequence has invalid order: "
                    f"planned={planned_sequence}, expected={expected_sequence}"
                ),
            }
        return {
            "code": "E1",
            "description": "The planner selected an incorrect tool sequence.",
            "message": (
                "Planned tool sequence differs from the expected sequence: "
                f"planned={planned_sequence}, expected={expected_sequence}"
            ),
        }

    def _is_subsequence(self, shorter: Sequence[str], longer: Sequence[str]) -> bool:
        cursor = 0
        for item in longer:
            if cursor < len(shorter) and shorter[cursor] == item:
                cursor += 1
        return cursor == len(shorter)

    def _resolve_data_reference(
        self,
        arguments: dict[str, Any],
        last_data_id: str | None,
    ) -> str | None:
        if "data_id" not in arguments:
            return None
        if last_data_id is None:
            return "data_id was provided before any tool produced a dataset."
        if arguments["data_id"] != PREVIOUS_RESULT_PLACEHOLDER:
            return "data_id must reference the immediately previous tool result."
        arguments["data_id"] = last_data_id
        return None

    def _format_final_response(self, request: EvaluationRequest, data_id: str | None) -> str:
        return f"{request.id} completed by LLM tool selection with result {data_id}."
