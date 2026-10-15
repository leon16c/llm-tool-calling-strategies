"""Verify supplementary sessions and compare prompt arms or sampling traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from audit_measurements import (
    canonical,
    dependency_findings,
    numeric_values,
    projection,
)
from jsonschema import Draft202012Validator
from analyze_matched_deviations import decision_content
from supplementary_session import CompletionToolCallPlanner, experiment_schedule
from toolcall_eval.ollama_mapper import (
    OllamaParameterMapper,
    OllamaToolCallPlanner,
    OllamaMultiStepPlanner,
)
from toolcall_eval.models import ToolCall
from toolcall_eval.oracle import load_requests

VARIANT_NAMES = {
    "V2": "fixed_sequence_llm_parameters",
    "V3": "llm_tool_selection",
    "V4": "llm_multi_step_planning",
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_nearest_rank_ms": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": mean(values),
        "median_ms": median(values),
        "p95_nearest_rank_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def verify_ledger(tasks: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    require(
        len({t["task_id"] for t in tasks}) == len(tasks), "Duplicate scheduled task"
    )
    require(
        len(events) == len(tasks) * 2, "Incomplete or unexpected ledger event count"
    )
    for position, task in enumerate(tasks):
        require(task["position"] == position, "Non-contiguous schedule")
        for offset, state in enumerate(["started", "completed"]):
            event = events[position * 2 + offset]
            require(
                event["state"] == state, f"Unexpected ledger state: {task['task_id']}"
            )
            for key in [
                "task_id",
                "position",
                "variant",
                "request_id",
                "repetition_index",
            ]:
                require(event[key] == task[key], f"Ledger identity mismatch: {key}")


def expected_projection(
    request: dict[str, Any], tools: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": name,
            "arguments": {
                key: request["expected_parameters"][key]
                for key in tools[name]["properties"]
                if key != "data_id"
            },
        }
        for name in request["expected_tool_sequence"]
    ]


def group_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    result: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_request[row["request_id"]].append(row)
        request_results = []
        for rid, observations in sorted(by_request.items()):
            repetitions = [r["repetition_index"] for r in observations]
            require(
                len(set(repetitions)) == len(repetitions),
                f"Duplicate repetition: {variant}/{rid}",
            )
            signatures = Counter(r["legacy_signature"] for r in observations)
            request_results.append(
                {
                    "request_id": rid,
                    "repetitions": len(observations),
                    "unique_traces": len(signatures),
                    "trace_frequencies": dict(sorted(signatures.items())),
                    "unique_run_statuses": len({r["status"] for r in observations}),
                    "unique_error_code_sequences": len(
                        {canonical(r["error_codes"]) for r in observations}
                    ),
                    "successful_runs": sum(
                        r["status"] == "success" for r in observations
                    ),
                    "category": observations[0]["category"],
                    "expected_length": observations[0]["expected_length"],
                }
            )
        result[variant] = {
            "runs": len(items),
            "request_count": len(request_results),
            "validated_success": sum(r["status"] == "success" for r in items),
            "exact_projection_matches": sum(r["exact_projection_match"] for r in items),
            "numeric_value_matches": sum(r["numeric_value_match"] for r in items),
            "requests_with_trace_variation": sum(
                r["unique_traces"] > 1 for r in request_results
            ),
            "requests_with_status_variation": sum(
                r["unique_run_statuses"] > 1 for r in request_results
            ),
            "requests_with_error_code_variation": sum(
                r["unique_error_code_sequences"] > 1 for r in request_results
            ),
            "primary_error_counts": dict(
                Counter(r["error_codes"][0] for r in items if r["error_codes"])
            ),
            "model_calls": sum(r["model_calls"] for r in items),
            "runs_with_dependency_findings": sum(
                bool(r["dependency_findings"]) for r in items
            ),
            "logged_schema_violations": sum(r["schema_violations"] for r in items),
            "latency_by_status": {
                status: latency_stats(
                    [r["latency_ms"] for r in items if r["status"] == status]
                )
                for status in ["success", "rejected"]
            },
            "requests": request_results,
        }
    return result


def analyze(directory: Path) -> dict[str, Any]:
    manifest = read(directory / "manifest.json")
    finished = read(directory / "finished.json")
    started = read(directory / "started.json")
    tasks = manifest["schedule"]
    require(
        manifest["session_schema_version"] == "supplementary-experiments-1",
        "Wrong session schema",
    )
    ids = sorted({t["request_id"] for t in tasks})
    repetitions = len({t["repetition_index"] for t in tasks})
    require(
        tasks == experiment_schedule(ids, repetitions, manifest["experiment"]),
        "Schedule differs from declared protocol",
    )
    require(
        finished["state"] == "completed",
        "Session not completed; no complete-series analysis",
    )
    require(finished["error"] is None, "Session has an abort reason")
    require(
        finished["completed_runs"] == finished["scheduled_runs"] == len(tasks),
        "Coverage mismatch",
    )
    require(
        started["manifest_sha256"] == sha(directory / "manifest.json"),
        "Manifest changed after start",
    )
    artifacts = directory / "artifacts"
    for name, expected in manifest["source_sha256"].items():
        require(sha(artifacts / name) == expected, f"Source snapshot mismatch: {name}")
    require(
        sha(directory / "hardware.json") == manifest["hardware_sha256"],
        "Hardware evidence changed",
    )
    ledger = [
        json.loads(line)
        for line in (directory / "ledger.jsonl").read_text().splitlines()
    ]
    verify_ledger(tasks, ledger)
    require(
        {p.name for p in (directory / "runs").iterdir() if p.is_dir()}
        == {t["task_id"] for t in tasks},
        "Unexpected or missing run directory",
    )
    requests = {
        r["id"]: r
        for r in read(artifacts / "evaluation/requests/initial_requests.json")[
            "requests"
        ]
    }
    tools = {
        t["name"]: t["parameters_schema"]
        for t in read(artifacts / "prototype/schemas/tools.json")["tools"]
    }
    request_objects = {
        r.id: r
        for r in load_requests(artifacts / "evaluation/requests/initial_requests.json")
    }
    for name, expected in manifest["source_sha256"].items():
        if name.startswith(("prototype/src/", "prototype/schemas/")) or name.endswith(
            "supplementary_session.py"
        ):
            require(
                sha(Path(__file__).resolve().parents[2] / name) == expected,
                "Current prompt/replay source differs from snapshot",
            )
    log_validator = Draft202012Validator(
        read(artifacts / "prototype/schemas/run_log.schema.json")
    )
    rows = []
    run_ids: set[str] = set()
    file_hashes = {
        name: sha(directory / name)
        for name in [
            "manifest.json",
            "started.json",
            "finished.json",
            "ledger.jsonl",
            "hardware.json",
            "runtime_prepared.json",
        ]
    }
    for position, task in enumerate(tasks):
        task_dir = directory / "runs" / task["task_id"]
        log = read(task_dir / "run_log.json")
        log_validator.validate(log)
        require(log["run_id"] not in run_ids, "Duplicate run ID")
        run_ids.add(log["run_id"])
        require(
            log["request_id"] == task["request_id"]
            and log["repetition_index"] == task["repetition_index"]
            and log["variant"] == VARIANT_NAMES[task["variant"]],
            "Run identity mismatch",
        )
        completion = ledger[position * 2 + 1]
        require(
            completion["run_id"] == log["run_id"]
            and completion["run_status"] == log["status"],
            "Ledger/log outcome mismatch",
        )
        require(
            log["model"]
            == {
                "provider": "ollama",
                "name": manifest["model"],
                "temperature": task["temperature"],
                "seed": task["seed"],
            },
            "Model/log settings mismatch",
        )
        contexts = set()
        for filename in ["runtime_before.json", "runtime_after.json"]:
            snapshot = read(task_dir / filename)
            identity = {
                "version": snapshot["version"],
                "digest": snapshot["model"]["digest"],
                "show": snapshot["show"],
            }
            require(
                identity == manifest["configuration_identity"], "Runtime/model drift"
            )
            for loaded in snapshot["residency"].get("models", []):
                if loaded.get("digest") == snapshot["model"]["digest"]:
                    contexts.add(loaded.get("context_length"))
        calls_dir = task_dir / "model_calls"
        expected_files = {
            f"{i:04d}.{suffix}"
            for i in range(completion["model_calls"])
            for suffix in ["request.json", "response.txt", "event.json"]
        }
        require(
            {p.name for p in calls_dir.iterdir()} == expected_files,
            "Incomplete model-call recording",
        )
        http_times, finish_decisions, telemetry, model_texts, decisions = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(completion["model_calls"]):
            prefix = f"{i:04d}"
            payload = read(calls_dir / f"{prefix}.request.json")
            response_path = calls_dir / f"{prefix}.response.txt"
            response = read(response_path)
            event = read(calls_dir / f"{prefix}.event.json")
            require(
                event["state"] == "received" and event["index"] == i,
                "Model transport failure/index mismatch",
            )
            require(
                event["prompt_sha256"]
                == hashlib.sha256(payload["prompt"].encode()).hexdigest(),
                "Prompt hash mismatch",
            )
            require(
                event["response_sha256"] == sha(response_path), "Response hash mismatch"
            )
            expected_payload = {
                "model": manifest["model"],
                "prompt": payload["prompt"],
                "stream": False,
                "format": "json",
                "options": {"temperature": task["temperature"], "seed": task["seed"]},
            }
            require(payload == expected_payload, "Generation payload changed")
            obj = request_objects[task["request_id"]]
            if task["variant"] == "V3":
                adapter = (
                    CompletionToolCallPlanner()
                    if task["treatment"] == "completion"
                    else OllamaToolCallPlanner()
                )
                expected_prompt = adapter._build_prompt(obj)
            elif task["variant"] == "V2":
                previous = [
                    c
                    for c in log["tool_calls"]
                    if c["status"] == "executed" and c["index"] < i
                ]
                expected_prompt = OllamaParameterMapper()._build_prompt(
                    obj,
                    obj.expected_tool_sequence[i],
                    previous[-1]["result_ref"] if previous else None,
                )
            else:
                previous = [
                    ToolCall(**c)
                    for c in log["tool_calls"]
                    if c["status"] == "executed" and c["index"] < i
                ]
                expected_prompt = OllamaMultiStepPlanner()._build_prompt(obj, previous)
            require(
                payload["prompt"] == expected_prompt,
                "Prompt differs from declared treatment",
            )
            model_texts.append(response["response"])
            decisions.append(decision_content(response["response"]))
            require(isinstance(response.get("response"), str), "Missing model response")
            require(
                event["telemetry"]
                == {
                    k: v
                    for k, v in response.items()
                    if k not in {"response", "context"}
                },
                "Telemetry differs from saved response",
            )
            http_times.append(event["http_response_elapsed_ms"])
            telemetry.append(event["telemetry"])
            if task["variant"] == "V4":
                raw = response["response"].strip()
                lines = raw.splitlines()
                if (
                    len(lines) >= 2
                    and lines[0].startswith("```")
                    and lines[-1].startswith("```")
                ):
                    raw = "\n".join(lines[1:-1]).strip()
                try:
                    decision = json.loads(raw)
                    finish_decisions.append(
                        decision.get("finish")
                        if isinstance(decision, dict)
                        else "invalid_shape"
                    )
                except json.JSONDecodeError:
                    finish_decisions.append("invalid_json")
        request = requests[task["request_id"]]
        actual, expected = projection(log), expected_projection(request, tools)
        rows.append(
            {
                **task,
                "run_id": log["run_id"],
                "status": log["status"],
                "legacy_signature": canonical(projection(log, statuses=True)),
                "raw_text_signature": canonical(model_texts),
                "decision_execution_signature": canonical(
                    {
                        "decisions": decisions,
                        "calls": [
                            {
                                k: c[k]
                                for k in [
                                    "tool_name",
                                    "arguments",
                                    "status",
                                    "result_ref",
                                ]
                                if k in c
                            }
                            for c in log["tool_calls"]
                        ],
                        "status": log["status"],
                        "errors": log["errors"],
                    }
                ),
                "report_in_proposed_plan": any(
                    c.get("tool_name") == "format_report"
                    for c in (
                        decisions[0].get("value", {}).get("tool_calls", [])
                        if task["variant"] == "V3"
                        and decisions[0].get("kind") == "json"
                        and isinstance(decisions[0].get("value"), dict)
                        else []
                    )
                ),
                "proposed_sequence_agrees": (
                    [
                        c.get("tool_name")
                        for c in decisions[0]["value"].get("tool_calls", [])
                    ]
                    == request["expected_tool_sequence"]
                )
                if task["variant"] == "V3"
                and decisions[0].get("kind") == "json"
                and isinstance(decisions[0].get("value"), dict)
                else None,
                "prescribed_sequence_completed": [
                    c["tool_name"]
                    for c in log["tool_calls"]
                    if c["status"] == "executed"
                ][: len(request["expected_tool_sequence"])]
                == request["expected_tool_sequence"],
                "exact_projection_match": canonical(actual) == canonical(expected),
                "numeric_value_match": canonical(numeric_values(actual))
                == canonical(numeric_values(expected)),
                "dependency_findings": dependency_findings(log),
                "error_codes": [e["code"] for e in log["errors"]],
                "schema_violations": log["metrics"]["schema_validation_failures"],
                "latency_ms": log["latency_ms"],
                "model_calls": completion["model_calls"],
                "http_call_elapsed_ms": http_times,
                "telemetry": telemetry,
                "finish_decisions": finish_decisions,
                "observed_context_lengths": sorted(contexts, key=str),
                "category": request["category"],
                "expected_length": len(request["expected_tool_sequence"]),
            }
        )
        for path in task_dir.rglob("*"):
            if path.is_file():
                file_hashes[path.relative_to(directory).as_posix()] = sha(path)
    summary_rows = [
        {
            **r,
            "variant": f"V3_{r['treatment']}" if r["variant"] == "V3" else r["variant"],
        }
        for r in rows
    ]
    summaries = group_results(summary_rows)
    for key, summary in summaries.items():
        selected = [r for r in summary_rows if r["variant"] == key]
        summary["report_in_proposed_plan"] = sum(
            r["report_in_proposed_plan"] for r in selected
        )
        summary["proposed_sequence_agrees"] = sum(
            r["proposed_sequence_agrees"] is True for r in selected
        )
        summary["prescribed_sequence_completed"] = sum(
            r["prescribed_sequence_completed"] for r in selected
        )
        for name in ("raw_text_signature", "decision_execution_signature"):
            summary[f"requests_with_{name}_variation"] = sum(
                len({r[name] for r in selected if r["request_id"] == rid}) > 1
                for rid in ids
            )
    require(set(ids) <= set(requests), "Unknown requests")
    if manifest["purpose"] == "measurement":
        require(set(ids) == set(requests), "Not the full request set")
        require(
            repetitions == (5 if manifest["experiment"] == "sampling" else 1),
            "Incomplete measurement repetitions",
        )
    h2, prompt_pairs = None, None
    if manifest["experiment"] == "sampling":
        pairs = []
        for rid in ids:
            pair = {"request_id": rid}
            for label, signature in (
                ("primary", "legacy_signature"),
                ("raw", "raw_text_signature"),
                ("decision", "decision_execution_signature"),
            ):
                for variant in ("V2", "V4"):
                    pair[f"{variant.lower()}_{label}_unique_traces"] = len(
                        {
                            r[signature]
                            for r in rows
                            if r["variant"] == variant and r["request_id"] == rid
                        }
                    )
            pairs.append(pair)
        h2 = {"paired_requests": len(pairs), "pairs": pairs}
        for label in ("primary", "raw", "decision"):
            h2[label] = {
                "v4_more": sum(
                    p[f"v4_{label}_unique_traces"] > p[f"v2_{label}_unique_traces"]
                    for p in pairs
                ),
                "equal": sum(
                    p[f"v4_{label}_unique_traces"] == p[f"v2_{label}_unique_traces"]
                    for p in pairs
                ),
                "v4_fewer": sum(
                    p[f"v4_{label}_unique_traces"] < p[f"v2_{label}_unique_traces"]
                    for p in pairs
                ),
            }
    else:
        prompt_pairs = []
        for rid in ids:
            arms = {r["treatment"]: r for r in rows if r["request_id"] == rid}
            prompt_pairs.append(
                {
                    "request_id": rid,
                    **{
                        arm: {
                            k: r[k]
                            for k in [
                                "status",
                                "report_in_proposed_plan",
                                "proposed_sequence_agrees",
                                "exact_projection_match",
                            ]
                        }
                        for arm, r in arms.items()
                    },
                }
            )
    return {
        "analysis_version": "supplementary-experiments-1",
        "experiment": manifest["experiment"],
        "purpose": manifest["purpose"],
        "configuration_identity": manifest["configuration_identity"],
        "session": directory.as_posix(),
        "verified_complete": True,
        "scheduled_runs": len(tasks),
        "verified_runs": len(rows),
        "model_calls": sum(r["model_calls"] for r in rows),
        "analysis_source_sha256": {
            Path(__file__).name: sha(Path(__file__)),
            "audit_measurements.py": sha(
                Path(__file__).with_name("audit_measurements.py")
            ),
        },
        "input_sha256": file_hashes,
        "variants": summaries,
        "h2": h2,
        "prompt_pairs": prompt_pairs,
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verified_runs": result["verified_runs"],
                "model_calls": result["model_calls"],
                "variants": {
                    k: {
                        a: b
                        for a, b in v.items()
                        if a not in ("requests", "latency_by_status")
                    }
                    for k, v in result["variants"].items()
                },
                "h2": {k: v for k, v in (result["h2"] or {}).items() if k != "pairs"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
