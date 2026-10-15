import hashlib
import json
from typing import Any

from toolcall_eval.models import ToolName


def stable_result_ref(tool_name: ToolName, arguments: dict[str, Any]) -> str:
    serialized = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"{tool_name}:{digest}"


def execute_tool(tool_name: ToolName, arguments: dict[str, Any]) -> str:
    return stable_result_ref(tool_name, arguments)

