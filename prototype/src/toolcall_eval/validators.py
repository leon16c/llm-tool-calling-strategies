import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from toolcall_eval.paths import RUN_LOG_SCHEMA_PATH, TOOLS_SCHEMA_PATH


class ToolSchemaValidator:
    def __init__(self, schema_path: Path = TOOLS_SCHEMA_PATH) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._tool_validators = {
            tool["name"]: Draft202012Validator(tool["parameters_schema"])
            for tool in schema["tools"]
        }

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> list[str]:
        validator = self._tool_validators[tool_name]
        return [error.message for error in sorted(validator.iter_errors(arguments), key=str)]


class RunLogValidator:
    def __init__(self, schema_path: Path = RUN_LOG_SCHEMA_PATH) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._validator = Draft202012Validator(schema)

    def validate(self, run_log: dict[str, Any]) -> list[str]:
        return [error.message for error in sorted(self._validator.iter_errors(run_log), key=str)]

