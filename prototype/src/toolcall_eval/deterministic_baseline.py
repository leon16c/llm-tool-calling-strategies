from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from toolcall_eval.mock_tools import execute_tool
from toolcall_eval.models import EvaluationRequest, ToolCall, ToolName
from toolcall_eval.validators import ToolSchemaValidator

VARIANT = "deterministic_workflow"


class DeterministicBaseline:
    def __init__(self, validator: ToolSchemaValidator | None = None) -> None:
        self._validator = validator or ToolSchemaValidator()

    def run(self, request: EvaluationRequest, repetition_index: int = 0) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        tool_calls: list[ToolCall] = []
        validation_events: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        schema_failures = 0
        last_data_id: str | None = None

        for index, tool_name in enumerate(request.expected_tool_sequence):
            arguments = self._build_arguments(tool_name, request.expected_parameters, last_data_id)
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
                        "description": f"Invalid parameters for {tool_name}.",
                    }
                )
                status: Literal["executed", "rejected"] = "rejected"
                result_ref = None
            else:
                status = "executed"
                result_ref = execute_tool(tool_name, arguments)
                last_data_id = result_ref

            tool_calls.append(
                ToolCall(
                    index=index,
                    tool_name=tool_name,
                    arguments=arguments,
                    status=status,
                    result_ref=result_ref,
                )
            )

            if status == "rejected":
                break

        completed_at = datetime.now(UTC)
        run_status = "success" if schema_failures == 0 else "rejected"
        final_response = self._format_final_response(request, last_data_id)

        return {
            "schema_version": "0.1.0",
            "run_id": str(uuid4()),
            "request_id": request.id,
            "repetition_index": repetition_index,
            "variant": VARIANT,
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

    def _build_arguments(
        self,
        tool_name: ToolName,
        parameters: dict[str, Any],
        last_data_id: str | None,
    ) -> dict[str, Any]:
        if tool_name == "get_sales_data":
            return {
                "timeframe": parameters["timeframe"],
                "dimensions": parameters["dimensions"],
            }
        if tool_name == "aggregate_data":
            return {
                "data_id": self._require_data_id(last_data_id, tool_name),
                "group_by": parameters["group_by"],
                "metrics": parameters["metrics"],
            }
        if tool_name == "detect_anomalies":
            return {
                "data_id": self._require_data_id(last_data_id, tool_name),
                "method": parameters["method"],
                "threshold": parameters["threshold"],
            }
        if tool_name == "compare_previous_period":
            return {
                "data_id": self._require_data_id(last_data_id, tool_name),
                "previous_period": parameters["previous_period"],
            }
        if tool_name == "format_report":
            return {
                "data_id": self._require_data_id(last_data_id, tool_name),
                "output_format": parameters["output_format"],
            }
        raise ValueError(f"Unsupported tool: {tool_name}")

    def _require_data_id(self, last_data_id: str | None, tool_name: ToolName) -> str:
        if last_data_id is None:
            raise ValueError(f"{tool_name} requires a previous data_id.")
        return last_data_id

    def _format_final_response(self, request: EvaluationRequest, data_id: str | None) -> str:
        return f"{request.id} completed by deterministic workflow with result {data_id}."
