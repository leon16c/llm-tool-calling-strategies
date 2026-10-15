from dataclasses import dataclass
from typing import Any, Literal

ToolName = Literal[
    "get_sales_data",
    "aggregate_data",
    "detect_anomalies",
    "compare_previous_period",
    "format_report",
]

Variant = Literal[
    "deterministic_workflow",
    "fixed_sequence_llm_parameters",
    "llm_tool_selection",
    "llm_multi_step_planning",
]


@dataclass(frozen=True)
class EvaluationRequest:
    id: str
    category: str
    user_request: str
    expected_tool_sequence: list[ToolName]
    expected_parameters: dict[str, Any]
    primary_evaluation_focus: str


@dataclass(frozen=True)
class ToolCall:
    index: int
    tool_name: ToolName
    arguments: dict[str, Any]
    source: Literal["deterministic", "llm", "validator", "fallback"] = "deterministic"
    status: Literal["proposed", "executed", "rejected", "failed"] = "executed"
    result_ref: str | None = None
    latency_ms: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "source": self.source,
            "status": self.status,
            "latency_ms": self.latency_ms,
        }
        if self.result_ref is not None:
            data["result_ref"] = self.result_ref
        return data

