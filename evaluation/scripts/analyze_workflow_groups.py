"""Descriptive workflow strata from saved logs; no tool execution or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_measurements import SETS, canonical, numeric_values, projection

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "V1": "deterministic_workflow",
    "V2": "fixed_sequence_llm_parameters",
    "V3": "llm_tool_selection",
    "V4": "llm_multi_step_planning",
}


def validate_coverage(logs, requests, repetitions, variant):
    expected = {(rid, rep) for rid in requests for rep in range(repetitions)}
    identities = [(log["request_id"], log["repetition_index"]) for log in logs]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Missing, duplicate or unexpected request/repetition")
    for log in logs:
        if log["variant"] != variant or log["status"] not in {"success", "rejected"}:
            raise ValueError("Unexpected variant or outcome")
        request = requests[log["request_id"]]
        if log["input"]["expected_tool_sequence"] != request["expected_tool_sequence"]:
            raise ValueError("Log/oracle sequence mismatch")
        if log["input"]["user_request"] != request["user_request"]:
            raise ValueError("Log/oracle user text mismatch")


def summarize(rows):
    return {
        "request_ids": sorted({row["request_id"] for row in rows}),
        "requests": len({row["request_id"] for row in rows}),
        "runs": len(rows),
        "validated_success": sum(row["status"] == "success" for row in rows),
        "exact_matches": sum(row["exact_match"] for row in rows),
        "numeric_matches": sum(row["numeric_match"] for row in rows),
        "primary_errors": dict(
            sorted(
                Counter(
                    row["primary_error"] for row in rows if row["primary_error"]
                ).items()
            )
        ),
    }


def group_rows(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {label: summarize(items) for label, items in sorted(groups.items())}


def analyze(root=ROOT):
    hashes = {}

    def read(path):
        data = path.read_bytes()
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    oracle = read(root / "evaluation/requests/initial_requests.json")["requests"]
    requests = {r["id"]: r for r in oracle}
    if len(requests) != 30 or len(oracle) != 30:
        raise ValueError("Expected 30 distinct requests")
    datasets = {}
    for label, (directory, reference) in SETS.items():
        variant = label.split("_")[0]
        paths = sorted((root / "prototype/logs" / directory).glob("R*.json"))
        datasets["historical_" + label] = (
            variant,
            5 if "repeated" in label else 1,
            reference,
            paths,
        )
    for variant in ("V2", "V3", "V4"):
        paths = sorted(
            (root / "evaluation/experiments/2026-10-09_matched/runs").glob(
                f"{variant}_*/run_log.json"
            )
        )
        datasets["matched_" + variant] = (variant, 5, SETS[variant][1], paths)

    result = {}
    for label, (variant, repetitions, reference, paths) in datasets.items():
        logs = [read(path) for path in paths]
        validate_coverage(logs, requests, repetitions, VARIANTS[variant])
        rows = []
        for log in logs:
            rid = log["request_id"]
            request = requests[rid]
            ref = read(root / "prototype/logs" / reference / f"{rid}.json")
            if ref["request_id"] != rid or ref["status"] != "success":
                raise ValueError("Invalid reference identity/status")
            actual, expected = projection(log), projection(ref)
            rows.append(
                {
                    "request_id": rid,
                    "repetition_index": log["repetition_index"],
                    "category": request["category"],
                    "expected_length": len(request["expected_tool_sequence"]),
                    "expected_workflow": " → ".join(request["expected_tool_sequence"]),
                    "status": log["status"],
                    "exact_match": canonical(actual) == canonical(expected),
                    "numeric_match": canonical(numeric_values(actual))
                    == canonical(numeric_values(expected)),
                    "primary_error": log["errors"][0]["code"]
                    if log["errors"]
                    else None,
                }
            )
        result[label] = {
            "repetitions": repetitions,
            "overall": summarize(rows),
            "rows": rows,
        }
        for key in ("category", "expected_length", "expected_workflow"):
            result[label][key] = group_rows(rows, key)
    return {"analysis_version": "1", "input_sha256": hashes, "datasets": result}


def render_report(result):
    lines = [
        "# Workflow category and expected-length analysis",
        "",
        "Descriptive, retrospective stratification of existing logs; no new model calls.",
        "",
        "Categories are the original oracle labels, not newly adjudicated semantic classes. "
        "Length is the expected number of tool calls, excluding finish decisions; it is not "
        "the observed prefix length or model-call count. Exact workflow groups use oracle sequences.",
        "",
        "Historical and matched datasets remain separate. Counts of repeated runs are not "
        "independent task samples. Exact/numeric projections are not semantic success rates. "
        "Primary errors follow runtime check precedence. Small groups and confounded workflow "
        "content/length preclude a causal length or complexity claim.",
        "",
        "Coverage, variant identity, user text and expected sequences are checked against the "
        "oracle. Input hashes and per-run rows are retained in workflow_group_analysis.json. "
        "This does not repeat the full recorded-session integrity audit.",
        "",
        "Reproduce: `prototype/.venv/Scripts/python.exe evaluation/scripts/analyze_workflow_groups.py`",
        "",
    ]
    for label, dataset in result["datasets"].items():
        lines += [
            f"## {label}",
            "",
            f"Repetitions per request: {dataset['repetitions']}.",
            "",
        ]
        for key in ("category", "expected_length", "expected_workflow"):
            lines += [
                f"### {key}",
                "",
                "| Group | Requests | Runs | Validated successes | Exact matches | Numeric matches | Primary errors |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
            for group, s in dataset[key].items():
                errors = (
                    ", ".join(f"{k}: {v}" for k, v in s["primary_errors"].items())
                    or "—"
                )
                lines.append(
                    f"| {group} | {s['requests']} | {s['runs']} | {s['validated_success']} "
                    f"| {s['exact_matches']} | {s['numeric_matches']} | {errors} |"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluation/workflow_group_analysis.json"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "evaluation/workflow_group_analysis.md"
    )
    args = parser.parse_args()
    result = analyze()
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({label: d["overall"] for label, d in result["datasets"].items()}))


if __name__ == "__main__":
    main()
