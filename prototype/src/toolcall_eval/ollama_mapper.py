import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from toolcall_eval.fixed_sequence_llm_parameters import ModelInfo
from toolcall_eval.llm_tool_selection import PREVIOUS_RESULT_PLACEHOLDER, PlannedToolCall
from toolcall_eval.models import EvaluationRequest, ToolName
from toolcall_eval.paths import TOOLS_SCHEMA_PATH

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaParameterMapper:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        generate: Callable[[str], str] | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("TOOLCALL_EVAL_LLM_BASE_URL") or DEFAULT_BASE_URL)
        self._model = model or os.getenv("TOOLCALL_EVAL_LLM_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._seed = seed
        self._generate = generate or self._generate_with_ollama
        self._schemas = _load_tool_parameter_schemas()
        self.model_info = ModelInfo(
            provider="ollama",
            name=self._model,
            temperature=temperature,
            seed=seed,
        )

    def map_parameters(
        self,
        request: EvaluationRequest,
        tool_name: ToolName,
        previous_result_ref: str | None,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(request, tool_name, previous_result_ref)
        return parse_json_object(self._generate(prompt))

    def _build_prompt(
        self,
        request: EvaluationRequest,
        tool_name: ToolName,
        previous_result_ref: str | None,
    ) -> str:
        schema = json.dumps(self._schemas[tool_name], indent=2, sort_keys=True)
        previous_ref = previous_result_ref or "null"
        return f"""You map one controlled reporting request to the arguments for one tool call.

Return exactly one JSON object.
Do not wrap the object in another property.
Do not use Markdown fences.
Do not add explanations.
Use only fields allowed by the schema.
Do not include data_id unless the schema explicitly lists data_id as a property.
If the schema requires data_id, use exactly this previous_result_ref value: {previous_ref}

User request:
{request.user_request}

Current tool:
{tool_name}

Parameter JSON Schema:
{schema}
"""

    def _generate_with_ollama(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self._temperature,
            },
        }
        if self._seed is not None:
            payload["options"]["seed"] = self._seed

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ValueError(f"Ollama request failed: {exc}") from exc

        generated = response_data.get("response")
        if not isinstance(generated, str):
            raise ValueError("Ollama response did not contain a string response field.")
        return generated


class OllamaToolCallPlanner:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        generate: Callable[[str], str] | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("TOOLCALL_EVAL_LLM_BASE_URL") or DEFAULT_BASE_URL)
        self._model = model or os.getenv("TOOLCALL_EVAL_LLM_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._seed = seed
        self._generate = generate or self._generate_with_ollama
        self._tools_schema = json.loads(TOOLS_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.model_info = ModelInfo(
            provider="ollama",
            name=self._model,
            temperature=temperature,
            seed=seed,
        )

    def plan_tool_calls(self, request: EvaluationRequest) -> list[PlannedToolCall]:
        prompt = self._build_prompt(request)
        decoded = parse_json_object(self._generate(prompt))
        raw_tool_calls = decoded.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            raise ValueError("Generated JSON must contain a tool_calls array.")

        planned_tool_calls: list[PlannedToolCall] = []
        for index, item in enumerate(raw_tool_calls):
            if not isinstance(item, dict):
                raise ValueError(f"tool_calls[{index}] must be an object.")
            tool_name = item.get("tool_name")
            arguments = item.get("arguments")
            if not isinstance(tool_name, str):
                raise ValueError(f"tool_calls[{index}].tool_name must be a string.")
            if not isinstance(arguments, dict):
                raise ValueError(f"tool_calls[{index}].arguments must be an object.")
            planned_tool_calls.append(
                PlannedToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )
        return planned_tool_calls

    def _build_prompt(self, request: EvaluationRequest) -> str:
        schema = json.dumps(self._tools_schema, indent=2, sort_keys=True)
        return f"""You select and parameterize tool calls for a controlled reporting request.

Return exactly one JSON object with this shape:
{{
  "tool_calls": [
    {{
      "tool_name": "get_sales_data",
      "arguments": {{}}
    }}
  ]
}}

Do not use Markdown fences.
Do not add explanations.
Use only tool names from the tool set.
Each arguments object must satisfy the selected tool's parameters_schema.
For any tool argument named data_id, use exactly "{PREVIOUS_RESULT_PLACEHOLDER}".
Do not invent dataset identifiers.
Use the minimum necessary chronological tool calls to satisfy the request.

User request:
{request.user_request}

Tool set:
{schema}
"""

    def _generate_with_ollama(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self._temperature,
            },
        }
        if self._seed is not None:
            payload["options"]["seed"] = self._seed

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ValueError(f"Ollama request failed: {exc}") from exc

        generated = response_data.get("response")
        if not isinstance(generated, str):
            raise ValueError("Ollama response did not contain a string response field.")
        return generated


class OllamaMultiStepPlanner:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        generate: Callable[[str], str] | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("TOOLCALL_EVAL_LLM_BASE_URL") or DEFAULT_BASE_URL)
        self._model = model or os.getenv("TOOLCALL_EVAL_LLM_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._seed = seed
        self._generate = generate or self._generate_with_ollama
        self._tools_schema = json.loads(TOOLS_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.model_info = ModelInfo(
            provider="ollama",
            name=self._model,
            temperature=temperature,
            seed=seed,
        )

    def plan_next_step(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[Any],
    ) -> Any:
        from toolcall_eval.llm_multi_step_planning import MultiStepDecision

        prompt = self._build_prompt(request, executed_tool_calls)
        decoded = parse_json_object(self._generate(prompt))
        finish = decoded.get("finish")
        if not isinstance(finish, bool):
            raise ValueError("Generated JSON must contain a boolean finish field.")
        raw_tool_call = decoded.get("tool_call")
        if finish:
            if raw_tool_call is not None:
                raise ValueError("tool_call must be null when finish is true.")
            return MultiStepDecision(finish=True)
        if not isinstance(raw_tool_call, dict):
            raise ValueError("tool_call must be an object when finish is false.")

        tool_name = raw_tool_call.get("tool_name")
        arguments = raw_tool_call.get("arguments")
        if not isinstance(tool_name, str):
            raise ValueError("tool_call.tool_name must be a string.")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call.arguments must be an object.")
        return MultiStepDecision(
            finish=False,
            tool_call=PlannedToolCall(
                tool_name=tool_name,
                arguments=arguments,
            ),
        )

    def _build_prompt(
        self,
        request: EvaluationRequest,
        executed_tool_calls: list[Any],
    ) -> str:
        schema = json.dumps(self._tools_schema, indent=2, sort_keys=True)
        executed = json.dumps(
            [tool_call.to_log_dict() for tool_call in executed_tool_calls],
            indent=2,
            sort_keys=True,
        )
        return f"""You are controlling a backend reporting workflow one step at a time.

Return exactly one JSON object with this shape:
{{
  "finish": false,
  "tool_call": {{
    "tool_name": "get_sales_data",
    "arguments": {{}}
  }}
}}

If the request is fully satisfied by the executed tool calls, return:
{{
  "finish": true,
  "tool_call": null
}}

Do not use Markdown fences.
Do not add explanations.
Use only tool names from the tool set.
The next tool_call.arguments object must satisfy the selected tool's parameters_schema.
For any tool argument named data_id, use exactly "{PREVIOUS_RESULT_PLACEHOLDER}".
Do not invent dataset identifiers.
Choose only the single next chronological tool call.
Do not repeat a tool that has already produced the needed intermediate result.
Every completed reporting request must end with format_report.
After get_sales_data and aggregate_data have produced a simple summary, call format_report.
If the latest executed tool call is format_report, the request is complete.
In that case, you must return finish true with tool_call null.
Use markdown as the default output_format unless the user explicitly asks for json or html.

User request:
{request.user_request}

Executed tool calls:
{executed}

Tool set:
{schema}
"""

    def _generate_with_ollama(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self._temperature,
            },
        }
        if self._seed is not None:
            payload["options"]["seed"] = self._seed

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ValueError(f"Ollama request failed: {exc}") from exc

        generated = response_data.get("response")
        if not isinstance(generated, str):
            raise ValueError("Ollama response did not contain a string response field.")
        return generated


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_markdown_code_fence(text.strip())
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Generated text is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Generated JSON must be an object.")
    return decoded


def _strip_markdown_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


def _load_tool_parameter_schemas() -> dict[str, dict[str, Any]]:
    schema = json.loads(TOOLS_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {tool["name"]: tool["parameters_schema"] for tool in schema["tools"]}
