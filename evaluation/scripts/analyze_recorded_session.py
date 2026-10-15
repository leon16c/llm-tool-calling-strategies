"""Verify a completed recorded session before computing repeatability and run outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from audit_measurements import canonical, dependency_findings, numeric_values, projection
from jsonschema import Draft202012Validator

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
    require(len({t["task_id"] for t in tasks}) == len(tasks), "Duplicate scheduled task")
    require(len(events) == len(tasks) * 2, "Incomplete or unexpected ledger event count")
    for position, task in enumerate(tasks):
        require(task["position"] == position, "Non-contiguous schedule")
        for offset, state in enumerate(["started", "completed"]):
            event = events[position * 2 + offset]
            require(event["state"] == state, f"Unexpected ledger state: {task['task_id']}")
            for key in ["task_id", "position", "variant", "request_id", "repetition_index"]:
                require(event[key] == task[key], f"Ledger identity mismatch: {key}")


def expected_projection(request: dict[str, Any], tools: dict[str, Any]) -> list[dict[str, Any]]:
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
                len(set(repetitions)) == len(repetitions), f"Duplicate repetition: {variant}/{rid}"
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
                    "successful_runs": sum(r["status"] == "success" for r in observations),
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
            "requests_with_trace_variation": sum(r["unique_traces"] > 1 for r in request_results),
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
            "runs_with_dependency_findings": sum(bool(r["dependency_findings"]) for r in items),
            "logged_schema_violations": sum(r["schema_violations"] for r in items),
            "latency_by_status": {
                status: latency_stats([r["latency_ms"] for r in items if r["status"] == status])
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
    require(finished["state"] == "completed", "Session not completed; no complete-series analysis")
    require(finished["error"] is None, "Session has an abort reason")
    require(
        finished["completed_runs"] == finished["scheduled_runs"] == len(tasks), "Coverage mismatch"
    )
    require(
        started["manifest_sha256"] == sha(directory / "manifest.json"),
        "Manifest changed after start",
    )
    artifacts = directory / "artifacts"
    for name, expected in manifest["source_sha256"].items():
        require(sha(artifacts / name) == expected, f"Source snapshot mismatch: {name}")
    require(
        sha(directory / "hardware.json") == manifest["hardware_sha256"], "Hardware evidence changed"
    )
    ledger = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()]
    verify_ledger(tasks, ledger)
    require(
        {p.name for p in (directory / "runs").iterdir() if p.is_dir()}
        == {t["task_id"] for t in tasks},
        "Unexpected or missing run directory",
    )
    requests = {
        r["id"]: r
        for r in read(artifacts / "evaluation/requests/initial_requests.json")["requests"]
    }
    tools = {
        t["name"]: t["parameters_schema"]
        for t in read(artifacts / "prototype/schemas/tools.json")["tools"]
    }
    log_validator = Draft202012Validator(read(artifacts / "prototype/schemas/run_log.schema.json"))
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
            completion["run_id"] == log["run_id"] and completion["run_status"] == log["status"],
            "Ledger/log outcome mismatch",
        )
        contexts = set()
        for filename in ["runtime_before.json", "runtime_after.json"]:
            snapshot = read(task_dir / filename)
            identity = {
                "version": snapshot["version"],
                "digest": snapshot["model"]["digest"],
                "show": snapshot["show"],
            }
            require(identity == manifest["configuration_identity"], "Runtime/model drift")
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
        http_times, finish_decisions, telemetry = [], [], []
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
                event["prompt_sha256"] == hashlib.sha256(payload["prompt"].encode()).hexdigest(),
                "Prompt hash mismatch",
            )
            require(event["response_sha256"] == sha(response_path), "Response hash mismatch")
            expected_payload = {
                "model": manifest["model"],
                "prompt": payload["prompt"],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 0},
            }
            require(payload == expected_payload, "Generation payload changed")
            require(isinstance(response.get("response"), str), "Missing model response")
            require(
                event["telemetry"]
                == {k: v for k, v in response.items() if k not in {"response", "context"}},
                "Telemetry differs from saved response",
            )
            http_times.append(event["http_response_elapsed_ms"])
            telemetry.append(event["telemetry"])
            if task["variant"] == "V4":
                raw = response["response"].strip()
                lines = raw.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                    raw = "\n".join(lines[1:-1]).strip()
                try:
                    decision = json.loads(raw)
                    finish_decisions.append(
                        decision.get("finish") if isinstance(decision, dict) else "invalid_shape"
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
    summaries = group_results(rows)
    coverage = {
        (variant, row["request_id"]): row["repetitions"]
        for variant, summary in summaries.items()
        for row in summary["requests"]
    }
    require(set(v for v, _ in coverage) == {"V2", "V3", "V4"}, "Missing comparison variant")
    require({rid for _, rid in coverage} == set(requests), "Not the full request set")
    require(
        len(coverage) == len(requests) * 3 and set(coverage.values()) == {5},
        "Not five repetitions for every variant/request",
    )
    require(
        all(
            {
                row["repetition_index"]
                for row in rows
                if row["variant"] == variant and row["request_id"] == rid
            }
            == set(range(5))
            for variant, rid in coverage
        ),
        "Unexpected repetition indices",
    )
    v2 = {r["request_id"]: r for r in summaries["V2"]["requests"]}
    pairs = [
        {
            "request_id": r["request_id"],
            "v2_unique_traces": v2[r["request_id"]]["unique_traces"],
            "v4_unique_traces": r["unique_traces"],
        }
        for r in summaries["V4"]["requests"]
    ]
    h2 = {
        "paired_requests": len(pairs),
        "v4_more_unique_traces": sum(r["v4_unique_traces"] > r["v2_unique_traces"] for r in pairs),
        "equal_unique_traces": sum(r["v4_unique_traces"] == r["v2_unique_traces"] for r in pairs),
        "v4_fewer_unique_traces": sum(r["v4_unique_traces"] < r["v2_unique_traces"] for r in pairs),
        "pairs": pairs,
    }
    return {
        "analysis_version": "1",
        "session": directory.as_posix(),
        "verified_complete": True,
        "scheduled_runs": len(tasks),
        "verified_runs": len(rows),
        "model_calls": sum(r["model_calls"] for r in rows),
        "analysis_source_sha256": {
            Path(__file__).name: sha(Path(__file__)),
            "audit_measurements.py": sha(Path(__file__).with_name("audit_measurements.py")),
        },
        "input_sha256": file_hashes,
        "variants": summaries,
        "h2": h2,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = analyze(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result, args.output), encoding="utf-8")
    print(
        json.dumps(
            {
                "verified_runs": result["verified_runs"],
                "model_calls": result["model_calls"],
                "variants": {
                    k: {a: b for a, b in v.items() if a != "requests"}
                    for k, v in result["variants"].items()
                },
                "h2": {k: v for k, v in result["h2"].items() if k != "pairs"},
            },
            indent=2,
        )
    )


def render_report(result: dict[str, Any], output: Path) -> str:
    variants = result["variants"]
    paired = result["h2"]
    lines = [
        "# Matched repeated-run results",
        "",
        f"Session: `{result['session']}`. Machine-readable results: `{output.as_posix()}`.",
        "",
        f"Verified {result['verified_runs']} of {result['scheduled_runs']} scheduled runs and "
        f"{result['model_calls']} recorded model calls. All source snapshots, run identities, "
        "ledger events, model-call payloads, response hashes and runtime/model identities passed "
        "the completed-session checks. Smoke observations are excluded.",
        "",
        "## Repeatability and H2",
        "",
        "Each request was executed five times per variant. The primary signature preserves ordered "
        "tool names, call status and argument values/representations while dropping data_id. "
        "Run-status and error-code variation are separate measures. This is the legacy metric "
        "specified in the measurement plan, not a newly selected outcome measure.",
        "",
        "| Variant | Requests | Requests with trace variation | Requests with status variation |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, summary in variants.items():
        lines.append(
            f"| {name} | {summary['request_count']} | "
            f"{summary['requests_with_trace_variation']} | "
            f"{summary['requests_with_status_variation']} |"
        )
    lines += [
        "",
        f"In the {paired['paired_requests']} paired requests, V4 had more distinct traces "
        f"than V2 in {paired['v4_more_unique_traces']} cases, the same count in "
        f"{paired['equal_unique_traces']} and fewer in {paired['v4_fewer_unique_traces']}.",
        "",
    ]
    if paired["v4_more_unique_traces"] == 0 and paired["v4_fewer_unique_traces"] == 0:
        lines += [
            "The measured trace counts provide no support for H2's predicted increase in "
            "V4 variability relative to V2 under this configuration. This is an observed "
            "result over five repetitions, not proof of universal deterministic behavior.",
            "",
        ]
    else:
        lines += [
            "Interpret these paired counts together with the per-request results below; "
            "they describe this configuration and do not isolate control authority from "
            "prompting, context representation or orchestration.",
            "",
        ]
    lines += [
        "## Execution and oracle agreement",
        "",
        "| Variant | Runs | Validated successes | Exact projection matches | "
        "Numeric-value matches | Model calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in variants.items():
        lines.append(
            f"| {name} | {summary['runs']} | {summary['validated_success']} | "
            f"{summary['exact_projection_matches']} | {summary['numeric_value_matches']} | "
            f"{summary['model_calls']} |"
        )
    lines += [
        "",
        "These are separate operational measures, not semantic task-success rates. "
        "The previous semantic review concerns historical V2 cases and must not be "
        "automatically transferred to changed new traces.",
        "",
        "### Primary errors",
        "",
    ]
    for name, summary in variants.items():
        lines.append(
            f"- {name}: `{canonical(summary['primary_error_counts'])}`; logged schema "
            f"violations: {summary['logged_schema_violations']}; runs with executed "
            f"dependency findings: {summary['runs_with_dependency_findings']}."
        )
    lines += [
        "",
        "Primary errors follow runner check precedence. Zero logged schema violations "
        "does not mean every proposed call reached schema validation.",
        "",
        "## Per-request repeatability",
        "",
        "Cells show distinct legacy traces / successful repetitions (out of five).",
        "",
        "| Request | V2 | V3 | V4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    per_variant = {v: {r["request_id"]: r for r in s["requests"]} for v, s in variants.items()}
    for rid in sorted(per_variant["V2"]):
        cells = [
            f"{per_variant[v][rid]['unique_traces']} / {per_variant[v][rid]['successful_runs']}"
            for v in ["V2", "V3", "V4"]
        ]
        lines.append(f"| {rid} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Instrumented runner latency",
        "",
        "Times are milliseconds. P95 uses the nearest-rank definition. These durations "
        "include recording inside model callbacks and are not pooled with historical "
        "uninstrumented latencies. Different failure depths still prevent a simple "
        "efficiency ranking.",
        "",
        "| Variant | Outcome | n | Mean | Median | P95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, summary in variants.items():
        for status, stats in summary["latency_by_status"].items():
            values = [
                "—" if stats[k] is None else f"{stats[k]:.1f}"
                for k in ["mean_ms", "median_ms", "p95_nearest_rank_ms"]
            ]
            lines.append(f"| {variant} | {status} | {stats['n']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Scope and limitations",
        "",
        "- This is the separately recorded Ollama 0.40.1 series. Do not pool it with "
        "the earlier runtime/configuration observations or attribute differences solely "
        "to runtime changes.",
        "- Thirty controlled tasks with five repetitions remain thirty task cases per variant, "
        "not 150 independent tasks sampled from general backend usage.",
        "- Rejected traces are truncated by the validators. Stable observed prefixes do not "
        "prove stable hypothetical unconstrained continuations.",
        "- Low failure rates and repeatability are distinct. Repeated invalid decisions can "
        "be fully stable under the measured signature.",
        "- The legacy signature omits data IDs and explicit finish decisions. Dependency "
        "findings and raw V4 decisions are retained separately in the JSON evidence.",
        "- Server inherited settings and transient cache/device behavior are not completely "
        "controlled. Recorded metadata and checks bound, rather than eliminate, these limitations.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
