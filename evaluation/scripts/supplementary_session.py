"""Record prompt-guidance and sampling experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from toolcall_eval.fixed_sequence_llm_parameters import FixedSequenceLLMParameters
from toolcall_eval.llm_multi_step_planning import LLMMultiStepPlanning
from toolcall_eval.llm_tool_selection import LLMToolSelection
from toolcall_eval.ollama_mapper import (
    OllamaMultiStepPlanner,
    OllamaParameterMapper,
    OllamaToolCallPlanner,
)
from toolcall_eval.oracle import load_requests
from toolcall_eval.validators import RunLogValidator

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("V2", "V3", "V4")
BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_event(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def api_text(
    base_url: str, endpoint: str, payload: dict[str, Any] | None = None
) -> str:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=120 if endpoint == "generate" else 10
        ) as response:
            return str(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        # Preserve the exception contract of the original Ollama adapters.
        raise ValueError(f"Ollama request failed: {exc}") from exc


def runtime_snapshot(base_url: str, model: str) -> dict[str, Any]:
    runtime = json.loads(api_text(base_url, "version"))
    tags = json.loads(api_text(base_url, "tags"))
    matches = [item for item in tags["models"] if item["name"] == model]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise ValueError(f"Expected exactly one installed model with a digest: {model}")
    return {
        "observed_at": utc_now(),
        "version": runtime,
        "model": matches[0],
        "show": json.loads(api_text(base_url, "show", {"model": model})),
        "residency": json.loads(api_text(base_url, "ps")),
    }


def configuration_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Residency and expiry are observations, not fixed configuration identity.
    return {
        "version": snapshot["version"],
        "digest": snapshot["model"]["digest"],
        "show": snapshot["show"],
    }


def artifact_paths() -> list[Path]:
    paths: set[Path] = set()
    for folder, pattern in [("prototype/src", "*.py"), ("prototype/schemas", "*.json")]:
        paths.update((ROOT / folder).rglob(pattern))
    for name in [
        "evaluation/requests/initial_requests.json",
        "evaluation/requests/initial_requests.schema.json",
        "prototype/pyproject.toml",
        "prototype/uv.lock",
        "evaluation/scripts/measurement_session.py",
        "evaluation/scripts/test_measurement_session.py",
        "evaluation/repeated_measurement_plan.md",
    ]:
        path = ROOT / name
        if path.exists():
            paths.add(path)
    for name in [
        "evaluation/scripts/supplementary_session.py",
        "evaluation/scripts/test_supplementary_session.py",
        "evaluation/supplementary_protocol.md",
    ]:
        paths.add(ROOT / name)
    return sorted(paths)


def source_hashes() -> dict[str, str]:
    return {
        p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in artifact_paths()
    }


def schedule(
    request_ids: list[str], repetitions: int, variants: list[str]
) -> list[dict[str, Any]]:
    if repetitions < 1 or not request_ids or not variants:
        raise ValueError("Requests, variants and positive repetitions are required.")
    if len(set(request_ids)) != len(request_ids) or len(set(variants)) != len(variants):
        raise ValueError("Duplicate requests or variants are not allowed.")
    if any(v not in VARIANTS for v in variants):
        raise ValueError("Unknown variant.")
    result: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for index, rid in enumerate(sorted(request_ids)):
            offset = (repetition + index) % len(variants)
            for variant in variants[offset:] + variants[:offset]:
                result.append(
                    {
                        "position": len(result),
                        "request_id": rid,
                        "variant": variant,
                        "repetition_index": repetition,
                        "task_id": f"{variant}_{rid}_rep{repetition:02d}",
                    }
                )
    return result


class RecordedGeneration:
    """Record request/response before the existing adapter parses model output."""

    def __init__(
        self,
        directory: Path,
        *,
        model: str = MODEL,
        base_url: str = BASE_URL,
        transport: Callable[[dict[str, Any]], str] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        treatment: str = "original",
    ) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.temperature = temperature
        self.seed = seed
        self.treatment = treatment
        self.model = model
        self.base_url = base_url
        self.transport = transport or (
            lambda payload: api_text(base_url, "generate", payload)
        )
        self.calls = 0
        self.infrastructure_errors = 0

    def __call__(self, prompt: str) -> str:
        index = self.calls
        self.calls += 1
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        prefix = f"{index:04d}"
        write_new(self.directory / f"{prefix}.request.json", payload)
        event: dict[str, Any] = {
            "index": index,
            "endpoint": f"{self.base_url.rstrip('/')}/api/generate",
            "prompt_sha256": digest(prompt.encode("utf-8")),
            "started_at": utc_now(),
        }
        started = time.perf_counter_ns()
        try:
            raw = self.transport(payload)
            event["http_response_elapsed_ms"] = (
                time.perf_counter_ns() - started
            ) / 1_000_000
            # Preserve the entire HTTP JSON body, including context, token counts and timings.
            with (self.directory / f"{prefix}.response.txt").open("xb") as handle:
                handle.write(raw.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            data = json.loads(raw)
            if not isinstance(data, dict) or not isinstance(data.get("response"), str):
                raise ValueError(
                    "Ollama response did not contain a string response field."
                )
            event["telemetry"] = {
                k: v for k, v in data.items() if k not in {"response", "context"}
            }
            event["response_sha256"] = digest(raw.encode("utf-8"))
            event["state"] = "received"
            return str(data["response"])
        except BaseException as exc:
            self.infrastructure_errors += 1
            event.update(
                {"state": "error", "error_type": type(exc).__name__, "error": str(exc)}
            )
            raise
        finally:
            event["completed_at"] = utc_now()
            event["elapsed_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            write_new(self.directory / f"{prefix}.event.json", event)


def build_runner(variant: str, recorder: RecordedGeneration) -> Any:
    if variant == "V2":
        return FixedSequenceLLMParameters(
            mapper=OllamaParameterMapper(
                model=recorder.model,
                base_url=recorder.base_url,
                temperature=recorder.temperature,
                seed=recorder.seed,
                generate=recorder,
            )
        )
    if variant == "V3":
        return LLMToolSelection(
            planner=(
                CompletionToolCallPlanner
                if recorder.treatment == "completion"
                else OllamaToolCallPlanner
            )(
                model=recorder.model,
                base_url=recorder.base_url,
                temperature=recorder.temperature,
                seed=recorder.seed,
                generate=recorder,
            )
        )
    if variant == "V4":
        return LLMMultiStepPlanning(
            planner=OllamaMultiStepPlanner(
                model=recorder.model,
                base_url=recorder.base_url,
                temperature=recorder.temperature,
                seed=recorder.seed,
                generate=recorder,
            )
        )
    raise ValueError(f"Unknown variant: {variant}")


def prepare_session(
    directory: Path,
    *,
    request_ids: list[str],
    repetitions: int,
    variants: list[str],
    hardware_path: Path,
    purpose: str,
    experiment: str,
    base_url: str = BASE_URL,
    model: str = MODEL,
) -> None:
    known = {r.id for r in load_requests()}
    if not set(request_ids) <= known:
        raise ValueError("Unknown request IDs.")
    tasks = experiment_schedule(request_ids, repetitions, experiment)
    hardware = read_json(hardware_path)
    snapshot = runtime_snapshot(base_url, model)
    paths = artifact_paths()
    recorded_hashes = source_hashes()
    directory.mkdir(parents=True, exist_ok=False)
    for path in paths:
        target = directory / "artifacts" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(path.read_bytes())
    if recorded_hashes != source_hashes():
        raise RuntimeError("Source changed during preparation.")
    write_new(directory / "hardware.json", hardware)
    write_new(directory / "runtime_prepared.json", snapshot)
    manifest = {
        "session_schema_version": "supplementary-experiments-1",
        "experiment": experiment,
        "prepared_at": utc_now(),
        "purpose": purpose,
        "base_url": base_url,
        "model": model,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ["jsonschema", "toolcall-eval"]},
        "source_sha256": recorded_hashes,
        "hardware_sha256": digest((directory / "hardware.json").read_bytes()),
        "configuration_identity": configuration_identity(snapshot),
        "generation": {
            "stream": False,
            "format": "json",
            "options": "temperature and seed are declared per scheduled task",
        },
        "inherited_settings": (
            "Other generation/server options are inherited; /api/show and /api/ps are captured. "
            "Server process environment is not captured."
        ),
        "order": (
            "Repetition, sorted request ID, rotating variant order by "
            "(repetition + request index). Sequential execution."
        ),
        "retries": 0,
        "fallbacks": 0,
        "warmup": "No implicit warmup; any smoke session is separate.",
        "schedule": tasks,
    }
    write_new(directory / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "prepared": str(directory),
                "scheduled_runs": len(tasks),
                "purpose": purpose,
            }
        )
    )


def run_session(directory: Path) -> None:
    manifest = read_json(directory / "manifest.json")
    if platform.python_version() != manifest["python"]:
        raise ValueError("Python version changed since preparation.")
    if {name: version(name) for name in manifest["packages"]} != manifest["packages"]:
        raise ValueError("Package versions changed since preparation.")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError(
            "Source hashes changed since preparation; prepare a new session."
        )
    for name, expected in manifest["source_sha256"].items():
        if digest((directory / "artifacts" / name).read_bytes()) != expected:
            raise ValueError(f"Snapshot mismatch: {name}")
    if (
        digest((directory / "hardware.json").read_bytes())
        != manifest["hardware_sha256"]
    ):
        raise ValueError("Hardware evidence changed.")
    requests = {request.id: request for request in load_requests()}
    validator = RunLogValidator()
    # Exclusive start prevents accidental overwrite or implicit retries/resumption.
    write_new(
        directory / "started.json",
        {
            "started_at": utc_now(),
            "manifest_sha256": digest((directory / "manifest.json").read_bytes()),
        },
    )
    ledger = directory / "ledger.jsonl"
    completed = 0
    final_state = "aborted"
    final_error: dict[str, str] | None = None
    try:
        for task in manifest["schedule"]:
            if source_hashes() != manifest["source_sha256"]:
                raise RuntimeError("Source changed during session.")
            snapshot = runtime_snapshot(manifest["base_url"], manifest["model"])
            task_dir = directory / "runs" / task["task_id"]
            write_new(task_dir / "runtime_before.json", snapshot)
            if configuration_identity(snapshot) != manifest["configuration_identity"]:
                raise RuntimeError(
                    "Runtime/model configuration drift; stopping session."
                )
            append_event(ledger, {**task, "state": "started", "at": utc_now()})
            recorder = RecordedGeneration(
                task_dir / "model_calls",
                model=manifest["model"],
                base_url=manifest["base_url"],
                temperature=task["temperature"],
                seed=task["seed"],
                treatment=task["treatment"],
            )
            started = time.perf_counter_ns()
            try:
                runner = build_runner(task["variant"], recorder)
                log = runner.run(requests[task["request_id"]], task["repetition_index"])
                expected_variant = {
                    "V2": "fixed_sequence_llm_parameters",
                    "V3": "llm_tool_selection",
                    "V4": "llm_multi_step_planning",
                }
                if (
                    log["variant"] != expected_variant[task["variant"]]
                    or log["request_id"] != task["request_id"]
                    or log["repetition_index"] != task["repetition_index"]
                ):
                    raise ValueError("Run identity differs from scheduled task.")
                errors = validator.validate(log)
                if errors:
                    write_new(task_dir / "invalid_run_log.json", log)
                    raise ValueError(f"Run-log schema errors: {errors}")
                write_new(task_dir / "run_log.json", log)
                after = runtime_snapshot(manifest["base_url"], manifest["model"])
                write_new(task_dir / "runtime_after.json", after)
                if configuration_identity(after) != manifest["configuration_identity"]:
                    raise RuntimeError("Runtime/model changed during run.")
                if recorder.infrastructure_errors:
                    raise RuntimeError(
                        "Generation transport/envelope failure; not a model-quality result."
                    )
            except BaseException as exc:
                append_event(
                    ledger,
                    {
                        **task,
                        "state": "error",
                        "at": utc_now(),
                        "model_calls": recorder.calls,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            append_event(
                ledger,
                {
                    **task,
                    "state": "completed",
                    "at": utc_now(),
                    "run_status": log["status"],
                    "run_id": log["run_id"],
                    "model_calls": recorder.calls,
                    "recorded_run_elapsed_ms": (time.perf_counter_ns() - started)
                    / 1_000_000,
                },
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(manifest["schedule"]),
                        "task": task["task_id"],
                        "status": log["status"],
                        "model_calls": recorder.calls,
                    }
                ),
                flush=True,
            )
        if source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("Source changed during final run.")
        final_state = "completed"
    except BaseException as exc:
        final_error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_new(
            directory / "finished.json",
            {
                "state": final_state,
                "at": utc_now(),
                "completed_runs": completed,
                "scheduled_runs": len(manifest["schedule"]),
                "error": final_error,
            },
        )


# Only these lines are added to the complete-plan prompt; the output shape is unchanged.
COMPLETION_GUIDANCE = """Do not repeat a tool that has already produced the needed intermediate result.
Every completed reporting request must end with format_report.
After get_sales_data and aggregate_data have produced a simple summary, call format_report.
The final planned tool call must be format_report. Do not add another tool call after it.
Use markdown as the default output_format unless the user explicitly asks for json or html.
"""
SEEDS = (11, 22, 33, 44, 55)


class CompletionToolCallPlanner(OllamaToolCallPlanner):
    def _build_prompt(self, request):
        original = super()._build_prompt(request)
        marker = "\nUser request:\n"
        if original.count(marker) != 1:
            raise ValueError("Prompt insertion marker changed")
        return original.replace(marker, "\n" + COMPLETION_GUIDANCE + marker, 1)


def experiment_schedule(request_ids, repetitions, experiment):
    if experiment == "sampling":
        if not 1 <= repetitions <= len(SEEDS):
            raise ValueError("Sampling supports one to five predefined seeds")
        tasks = schedule(request_ids, repetitions, ["V2", "V4"])
        for task in tasks:
            task.update(
                temperature=0.7,
                seed=SEEDS[task["repetition_index"]],
                treatment="original",
            )
        return tasks
    if experiment != "prompt" or repetitions != 1:
        raise ValueError("Prompt comparison requires one repetition")
    # Validate IDs and uniqueness through the original schedule helper.
    schedule(request_ids, 1, ["V3"])
    tasks = []
    for index, rid in enumerate(sorted(request_ids)):
        arms = (
            ["original", "completion"] if index % 2 == 0 else ["completion", "original"]
        )
        for arm in arms:
            tasks.append(
                dict(
                    position=len(tasks),
                    request_id=rid,
                    variant="V3",
                    repetition_index=0,
                    task_id=f"V3_{arm}_{rid}_rep00",
                    temperature=0.0,
                    seed=0,
                    treatment=arm,
                )
            )
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--experiment", choices=["prompt", "sampling"])
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--request", action="append", dest="requests")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--purpose", choices=["smoke", "measurement"], default="measurement"
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.hardware is None or args.experiment is None:
            parser.error("Preparation requires --hardware and --experiment")
        repetitions = (
            args.repetitions
            if args.repetitions is not None
            else (5 if args.experiment == "sampling" else 1)
        )
        prepare_session(
            args.session,
            request_ids=args.requests or [r.id for r in load_requests()],
            repetitions=repetitions,
            variants=[],
            hardware_path=args.hardware,
            purpose=args.purpose,
            experiment=args.experiment,
        )
    else:
        run_session(args.session)


if __name__ == "__main__":
    main()
