"""Offline audit of historical logs; never executes tools or calls a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation" / "audits" / "2026-10-09"
SETS = {
    "V1": ("deterministic_baseline", "deterministic_baseline"),
    "V2": ("fixed_sequence_llm_parameters_ollama", "fixed_sequence_llm_parameters"),
    "V3": ("llm_tool_selection_ollama", "llm_tool_selection_oracle"),
    "V4": ("llm_multi_step_planning_ollama", "llm_multi_step_planning_oracle"),
    "V2_repeated": (
        "fixed_sequence_llm_parameters_ollama_repeated",
        "fixed_sequence_llm_parameters",
    ),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    # Deliberately retains 3 versus 3.0, array order and call order.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def projection(log: dict[str, Any], *, statuses: bool = False) -> list[dict[str, Any]]:
    """Legacy-compatible projection, NOT a semantic or dependency verdict."""
    result = []
    for call in log["tool_calls"]:
        item = {
            "tool_name": call["tool_name"],
            "arguments": {k: v for k, v in call["arguments"].items() if k != "data_id"},
        }
        if statuses:
            item["status"] = call["status"]
        result.append(item)
    return result


def numeric_values(value: Any) -> Any:
    """Collapse integral floats only; preserve booleans, strings and array order."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [numeric_values(item) for item in value]
    if isinstance(value, dict):
        return {key: numeric_values(item) for key, item in value.items()}
    return value


def dependency_findings(log: dict[str, Any]) -> list[dict[str, Any]]:
    """Check executed calls against actual preceding executed result references."""
    previous = None
    findings = []
    for call in log["tool_calls"]:
        if call["status"] != "executed":
            continue
        if call["tool_name"] != "get_sales_data":
            supplied = call["arguments"].get("data_id")
            if previous is None or supplied != previous:
                findings.append({"index": call["index"], "expected": previous, "actual": supplied})
        previous = call.get("result_ref")
    return findings


def offline_schema(log: dict[str, Any], validators: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for call in log["tool_calls"]:
        validator = validators.get(call["tool_name"])
        if validator is None:
            findings.append({"index": call["index"], "unsupported_tool": call["tool_name"]})
            continue
        errors = sorted(validator.iter_errors(call["arguments"]), key=lambda e: str(e.path))
        if errors:
            findings.append(
                {
                    "index": call["index"],
                    "status": call["status"],
                    "errors": [
                        {"path": list(e.path), "rule": e.validator, "message": e.message}
                        for e in errors
                    ],
                }
            )
    return findings


def input_paths() -> list[Path]:
    paths = set()
    for directory, pattern in [
        ("prototype/src", "*.py"),
        ("prototype/tests", "*.py"),
        ("prototype/schemas", "*.*"),
        ("prototype/logs", "*.json"),
        ("evaluation/requests", "*.*"),
        ("thesis/chapters/generated", "*.md"),
    ]:
        paths.update(p for p in (ROOT / directory).rglob(pattern) if p.is_file())
    for name in [
        "expose_final.md",
        "prototype/pyproject.toml",
        "prototype/uv.lock",
        "thesis/abstract.md",
        "thesis/build.ps1",
        "thesis/metadata.yaml",
    ]:
        path = ROOT / name
        if path.exists():
            paths.add(path)
    return sorted(paths)


def hashes() -> dict[str, str]:
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in input_paths()
    }


def audit() -> dict[str, Any]:
    schemas = read(ROOT / "prototype/schemas/tools.json")["tools"]
    validators = {s["name"]: Draft202012Validator(s["parameters_schema"]) for s in schemas}
    log_validator = Draft202012Validator(read(ROOT / "prototype/schemas/run_log.schema.json"))
    requests = {
        r["id"]: r for r in read(ROOT / "evaluation/requests/initial_requests.json")["requests"]
    }
    result = {}
    for label, (directory, reference_directory) in SETS.items():
        logs = [read(p) for p in sorted((ROOT / "prototype/logs" / directory).glob("R*.json"))]
        expected_count = 150 if label == "V2_repeated" else 30
        if len(logs) != expected_count:
            raise ValueError(f"{label}: expected {expected_count} logs, found {len(logs)}")
        rows = []
        signatures: dict[str, set[str]] = defaultdict(set)
        statuses: dict[str, set[str]] = defaultdict(set)
        repetitions: dict[str, set[int]] = defaultdict(set)
        for log in logs:
            log_validator.validate(log)
            rid = log["request_id"]
            request = requests[rid]
            if log["repetition_index"] in repetitions[rid]:
                raise ValueError(f"Duplicate repetition: {label}/{rid}")
            repetitions[rid].add(log["repetition_index"])
            reference = read(ROOT / "prototype/logs" / reference_directory / f"{rid}.json")
            actual, expected = projection(log), projection(reference)
            offline = offline_schema(log, validators)
            signatures[rid].add(canonical(projection(log, statuses=True)))
            statuses[rid].add(log["status"])
            row = {
                "request_id": rid,
                "repetition_index": log["repetition_index"],
                "status": log["status"],
                "category": request["category"],
                "expected_length": len(request["expected_tool_sequence"]),
                "logged_calls": len(log["tool_calls"]),
                "executed_calls": sum(c["status"] == "executed" for c in log["tool_calls"]),
                "proposed_calls": sum(c["status"] == "proposed" for c in log["tool_calls"]),
                "exact_projection_match": canonical(actual) == canonical(expected),
                "numeric_value_projection_match": (
                    canonical(numeric_values(actual)) == canonical(numeric_values(expected))
                ),
                "logged_schema_violations": log["metrics"]["schema_validation_failures"],
                "offline_schema_findings": offline,
                "executed_dependency_findings": dependency_findings(log),
                "errors": log["errors"],
            }
            if label == "V2" and log["status"] == "success" and not row["exact_projection_match"]:
                row.update(
                    {
                        "user_request": request["user_request"],
                        "expected": expected,
                        "actual": actual,
                        "semantic_review": "pending",
                    }
                )
            rows.append(row)
        if set(repetitions) != set(requests):
            raise ValueError(f"{label}: request coverage differs from oracle")
        expected_reps = set(range(5)) if label == "V2_repeated" else {0}
        if any(indices != expected_reps for indices in repetitions.values()):
            raise ValueError(f"{label}: incomplete repetition coverage")
        result[label] = {
            "directory": directory,
            "summary": {
                "runs": len(rows),
                "validated_success": sum(r["status"] == "success" for r in rows),
                "exact_projection_matches": sum(r["exact_projection_match"] for r in rows),
                "numeric_value_projection_matches": sum(
                    r["numeric_value_projection_match"] for r in rows
                ),
                "logged_schema_violations": sum(r["logged_schema_violations"] for r in rows),
                "runs_with_offline_schema_findings": sum(
                    bool(r["offline_schema_findings"]) for r in rows
                ),
                "calls_with_offline_schema_findings": sum(
                    len(r["offline_schema_findings"]) for r in rows
                ),
                "runs_with_executed_dependency_findings": sum(
                    bool(r["executed_dependency_findings"]) for r in rows
                ),
                "call_status_counts": dict(
                    Counter(c["status"] for log in logs for c in log["tool_calls"])
                ),
            },
            "unique_legacy_signatures_per_request": {
                k: len(v) for k, v in sorted(signatures.items())
            },
            "unique_statuses_per_request": {k: len(v) for k, v in sorted(statuses.items())},
            "rows": rows,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-baseline", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "baseline_manifest.json"
    if args.capture_baseline:
        manifest = {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "note": (
                "Hashes capture current artifacts; HEAD does not prove historical run provenance."
            ),
            "sha256": hashes(),
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    baseline = read(manifest_path)
    if baseline["sha256"] != hashes():
        raise ValueError(
            "Frozen inputs changed; use a separately versioned audit for changed inputs."
        )
    results = audit()
    payload = {
        "audit_version": "1",
        "audit_environment": {
            "python": platform.python_version(),
            "jsonschema": version("jsonschema"),
        },
        "baseline_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sets": results,
    }
    (OUT / "measurement_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
