from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from toolcall_eval.fixed_sequence_llm_parameters import ModelInfo, OracleBackedParameterMapper
from toolcall_eval.llm_tool_selection import (
    PREVIOUS_RESULT_PLACEHOLDER,
    TOOL_NAMES,
    PlannedToolCall,
)
from toolcall_eval.mock_tools import execute_tool
from toolcall_eval.models import EvaluationRequest, ToolCall, ToolName
from toolcall_eval.validators import ToolSchemaValidator

VARIANT = "llm_multi_step_planning"


@dataclass(frozen=True)
class MultiStepDecision:
    finish: bool
    tool_call: PlannedToolCall | None = None


class MultiStepPlanner(Protocol):
    model_info: ModelInfo

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision: ...


class OracleBackedMultiStepPlanner:
    """Controlled stand-in for iterative LLM planning."""

    model_info = ModelInfo(
        provider="local",
        name="oracle_backed_multi_step_planner",
        temperature=0.0,
        seed=0,
    )

    def __init__(self) -> None:
        self._parameter_mapper = OracleBackedParameterMapper()

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[ToolCall],
    ) -> MultiStepDecision:
        next_index = len(executed_tool_calls)
        if next_index >= len(request.expected_tool_sequence):
            return MultiStepDecision(finish=True)

        tool_name = request.expected_tool_sequence[next_index]
        return MultiStepDecision(
            finish=False,
            tool_call=PlannedToolCall(
                tool_name=tool_name,
                arguments=self._parameter_mapper.map_parameters(
                    request,
                    tool_name,
                    PREVIOUS_RESULT_PLACEHOLDER,
                ),
            ),
        )


class LLMMultiStepPlanning:
    def __init__(
        self,
        planner: MultiStepPlanner | None = None,
        validator: ToolSchemaValidator | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._planner = planner or OracleBackedMultiStepPlanner()
        self._validator = validator or ToolSchemaValidator()
        self._max_steps = max_steps

    def run(self, request: EvaluationRequest, repetition_index: int = 0) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        tool_calls: list[ToolCall] = []
        validation_events: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        schema_failures = 0
        last_data_id: str | None = None
        max_steps = self._max_steps or len(request.expected_tool_sequence) + 2

        for step_index in range(max_steps):
            executed_tool_calls = [
                tool_call for tool_call in tool_calls if tool_call.status == "executed"
            ]
            try:
                decision = self._planner.plan_next_step(request, executed_tool_calls)
            except (KeyError, TypeError, ValueError) as exc:
                validation_events.append(
                    {
                        "index": len(validation_events),
                        "event_type": "semantic_validation_failed",
                        "message": f"Multi-step planner failed: {exc}",
                    }
                )
                errors.append(
                    {
                        "code": "E3",
                        "description": "Could not produce a valid next-step decision.",
                    }
                )
                break

            if decision.finish:
                if len(executed_tool_calls) == len(request.expected_tool_sequence):
                    break
                validation_events.append(
                    {
                        "index": len(validation_events),
                        "event_type": "semantic_validation_failed",
                        "message": (
                            "Planner stopped before all required calls were executed: "
                            f"executed={len(executed_tool_calls)}, "
                            f"expected={len(request.expected_tool_sequence)}"
                        ),
                    }
                )
                errors.append(
                    {
                        "code": "E2",
                        "description": "The planner stopped before all required calls.",
                    }
                )
                break

            if decision.tool_call is None:
                validation_events.append(
                    {
                        "index": len(validation_events),
                        "event_type": "semantic_validation_failed",
                        "message": "Planner did not finish and did not provide a tool call.",
                    }
                )
                errors.append(
                    {
                        "code": "E10",
                        "description": "The planner returned an inconsistent next-step decision.",
                    }
                )
                break

            planned_call = decision.tool_call
            sequence_error = self._validate_next_tool(planned_call.tool_name, step_index, request)
            if sequence_error is not None:
                validation_events.append(
                    {
                        "index": len(validation_events),
                        "tool_call_index": step_index,
                        "event_type": sequence_error["event_type"],
                        "message": sequence_error["message"],
                    }
                )
                errors.append(
                    {
                        "code": sequence_error["code"],
                        "tool_call_index": step_index,
                        "description": sequence_error["description"],
                    }
                )
                if planned_call.tool_name in TOOL_NAMES:
                    tool_calls.append(
                        ToolCall(
                            index=step_index,
                            tool_name=cast(ToolName, planned_call.tool_name),
                            arguments=planned_call.arguments,
                            source="llm",
                            status="proposed",
                        )
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
                        "tool_call_index": step_index,
                        "event_type": "schema_validation_failed",
                        "message": message,
                    }
                    for message in validation_errors
                )
                errors.append(
                    {
                        "code": "E4",
                        "tool_call_index": step_index,
                        "description": f"Invalid planned parameters for {tool_name}.",
                    }
                )
                tool_calls.append(
                    ToolCall(
                        index=step_index,
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
                        "tool_call_index": step_index,
                        "event_type": "semantic_validation_failed",
                        "message": semantic_error,
                    }
                )
                errors.append(
                    {
                        "code": "E3",
                        "tool_call_index": step_index,
                        "description": f"Invalid data dependency for {tool_name}.",
                    }
                )
                tool_calls.append(
                    ToolCall(
                        index=step_index,
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
                    index=step_index,
                    tool_name=tool_name,
                    arguments=arguments,
                    source="llm",
                    status="executed",
                    result_ref=result_ref,
                )
            )
        else:
            validation_events.append(
                {
                    "index": len(validation_events),
                    "event_type": "semantic_validation_failed",
                    "message": f"Planner exceeded the maximum step count ({max_steps}).",
                }
            )
            errors.append(
                {
                    "code": "E10",
                    "description": (
                        "The planner did not terminate within the configured step limit."
                    ),
                }
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

    def _validate_next_tool(
        self,
        planned_tool_name: str,
        step_index: int,
        request: EvaluationRequest,
    ) -> dict[str, str] | None:
        if planned_tool_name not in TOOL_NAMES:
            return {
                "code": "E1",
                "description": "The planner selected an unsupported tool.",
                "event_type": "tool_call_rejected",
                "message": f"Unsupported tool selected: {planned_tool_name}",
            }
        if step_index >= len(request.expected_tool_sequence):
            return {
                "code": "E3",
                "description": "The planner added at least one unnecessary tool call.",
                "event_type": "semantic_validation_failed",
                "message": (
                    "Planner selected unnecessary tool after completion: "
                    f"{planned_tool_name}"
                ),
            }
        expected_tool_name = request.expected_tool_sequence[step_index]
        if planned_tool_name == expected_tool_name:
            return None
        if planned_tool_name in request.expected_tool_sequence:
            return {
                "code": "E7",
                "description": "The planner selected the right tool in the wrong order.",
                "event_type": "semantic_validation_failed",
                "message": (
                    "Planner selected an invalid next tool: "
                    f"planned={planned_tool_name}, expected={expected_tool_name}"
                ),
            }
        return {
            "code": "E1",
            "description": "The planner selected an incorrect tool.",
            "event_type": "tool_call_rejected",
            "message": (
                "Planner selected a tool that is not required for the request: "
                f"planned={planned_tool_name}, expected={expected_tool_name}"
            ),
        }

    def _resolve_data_reference(
        self,
        arguments: dict[str, Any],
        last_data_id: str | None,
    ) -> str | None:
        if "data_id" not in arguments:
            return None
        if last_data_id is None:
            return "data_id was provided before any tool produced a dataset."
        if arguments["data_id"] == last_data_id:
            return None
        if arguments["data_id"] != PREVIOUS_RESULT_PLACEHOLDER:
            return "data_id must reference the immediately previous tool result."
        arguments["data_id"] = last_data_id
        return None

    def _format_final_response(self, request: EvaluationRequest, data_id: str | None) -> str:
        return f"{request.id} completed by LLM multi-step planning with result {data_id}."
