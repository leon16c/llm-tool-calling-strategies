import json
from pathlib import Path
from typing import Any, cast

from toolcall_eval.models import EvaluationRequest, ToolName
from toolcall_eval.paths import REQUEST_ORACLE_PATH


def load_requests(path: Path = REQUEST_ORACLE_PATH) -> list[EvaluationRequest]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationRequest(
            id=item["id"],
            category=item["category"],
            user_request=item["user_request"],
            expected_tool_sequence=cast(list[ToolName], item["expected_tool_sequence"]),
            expected_parameters=cast(dict[str, Any], item["expected_parameters"]),
            primary_evaluation_focus=item["primary_evaluation_focus"],
        )
        for item in data["requests"]
    ]


def get_request(request_id: str, path: Path = REQUEST_ORACLE_PATH) -> EvaluationRequest:
    for request in load_requests(path):
        if request.id == request_id:
            return request
    raise KeyError(f"Unknown request id: {request_id}")

