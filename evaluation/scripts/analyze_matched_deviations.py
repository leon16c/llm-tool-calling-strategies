"""Offline matched-series evidence, reviewer annotations and supplementary traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_recorded_session import analyze, expected_projection, read, require, sha
from audit_measurements import ROOT, canonical, dependency_findings, projection

SESSION = Path("evaluation/experiments/2026-10-09_matched")


def differences(expected: list[dict], actual: list[dict]) -> list[dict]:
    require(
        [c["tool_name"] for c in expected] == [c["tool_name"] for c in actual],
        "Accepted sequence differs; needs a separate review design",
    )
    result = []
    for index, (ref, call) in enumerate(zip(expected, actual, strict=True)):
        for field in sorted(ref["arguments"].keys() | call["arguments"].keys()):
            left, right = ref["arguments"], call["arguments"]
            if (
                field not in left
                or field not in right
                or canonical(left[field]) != canonical(right[field])
            ):
                result.append(
                    {
                        "call_index": index,
                        "tool_name": call["tool_name"],
                        "field": field,
                        "expected_present": field in left,
                        "actual_present": field in right,
                        "expected": left.get(field),
                        "actual": right.get(field),
                    }
                )
    return result


def decision_content(raw: str) -> dict:
    """Normalize JSON layout only; keep IDs, finish, array order and numeric types."""
    stripped = raw.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        return {"kind": "json", "value": json.loads(stripped)}
    except json.JSONDecodeError:
        return {"kind": "invalid_json", "raw": raw}


def e3_cause(log: dict, expected_length: int) -> dict:
    """Classify only a directly evidenced post-completion rejection; retain unknowns."""
    prefix = "Planner selected unnecessary tool after completion: "
    calls = log["tool_calls"]
    for event in log["validation_events"]:
        index = event.get("tool_call_index")
        message = event.get("message", "")
        if (
            message.startswith(prefix)
            and index == expected_length
            and len(calls) == expected_length + 1
            and all(c["status"] == "executed" for c in calls[:-1])
            and calls[-2]["tool_name"] == "format_report"
            and calls[-1]["status"] != "executed"
            and calls[-1]["tool_name"] == message[len(prefix) :]
        ):
            return {
                "cause": "additional_tool_after_oracle_completion",
                "detection_stage": "oracle_next_step_gate_before_argument_validation",
                "proposed_tool": calls[-1]["tool_name"],
            }
    return {"cause": "unclassified_e3", "detection_stage": "requires_manual_review"}


def inventory(analysis: dict, session: Path) -> list[dict]:
    artifacts = session / "artifacts"
    requests = {
        r["id"]: r
        for r in read(artifacts / "evaluation/requests/initial_requests.json")["requests"]
    }
    tools = {
        t["name"]: t["parameters_schema"]
        for t in read(artifacts / "prototype/schemas/tools.json")["tools"]
    }
    grouped = defaultdict(list)
    for row in analysis["runs"]:
        grouped[row["variant"], row["request_id"]].append(row)
    cases = []
    for (variant, rid), rows in sorted(grouped.items()):
        require({r["repetition_index"] for r in rows} == set(range(5)), "Incomplete repetitions")
        selected = [r for r in rows if r["status"] == "success" and not r["exact_projection_match"]]
        if not selected:
            continue
        require(len(selected) == 5, "Mixed acceptance/agreement needs per-trace adjudication")
        require(len({r["legacy_signature"] for r in rows}) == 1, "More than one reviewed trace")
        paths = [
            session / "runs" / r["task_id"] / "run_log.json"
            for r in sorted(rows, key=lambda r: r["repetition_index"])
        ]
        logs = [read(p) for p in paths]
        for log in logs:
            require(not dependency_findings(log), "Unreviewed dependency failure")
            require(
                log["input"]["user_request"] == requests[rid]["user_request"], "Request mismatch"
            )
            require(
                all(c["status"] == "executed" for c in log["tool_calls"]),
                "Unexecuted accepted call",
            )
        expected, actual = expected_projection(requests[rid], tools), projection(logs[0])
        require(canonical(expected) != canonical(actual), "Selection disagrees with evidence")
        cases.append(
            {
                "case_id": f"{variant}_{rid}",
                "variant": variant,
                "request_id": rid,
                "user_request": requests[rid]["user_request"],
                "expected_length": len(expected),
                "category": requests[rid]["category"],
                "log_paths": [p.as_posix() for p in paths],
                "matching_repetitions": 5,
                "dependency_findings": [],
                "expected_projection": expected,
                "actual_projection": actual,
                "differences": differences(expected, actual),
            }
        )
    return cases


def supplementary(analysis: dict, session: Path) -> dict:
    groups = defaultdict(list)
    errors = []
    for row in analysis["runs"]:
        directory = session / "runs" / row["task_id"]
        raw = [
            read(directory / "model_calls" / f"{i:04d}.response.txt")["response"]
            for i in range(row["model_calls"])
        ]
        log = read(directory / "run_log.json")
        full = {
            "decisions": [decision_content(value) for value in raw],
            "calls": [
                {k: c.get(k) for k in ["tool_name", "arguments", "status", "result_ref"]}
                for c in log["tool_calls"]
            ],
            "status": log["status"],
            "errors": log["errors"],
        }
        groups[row["variant"], row["request_id"]].append(
            {
                "repetition_index": row["repetition_index"],
                "raw_model_text_signature": canonical(raw),
                "decision_execution_signature": canonical(full),
            }
        )
        if "E3" in row["error_codes"]:
            errors.append(
                {
                    "task_id": row["task_id"],
                    "request_id": row["request_id"],
                    "variant": row["variant"],
                    "errors": log["errors"],
                    **e3_cause(log, row["expected_length"]),
                    "validation_events": log["validation_events"],
                }
            )
    rows = [
        {
            "variant": v,
            "request_id": rid,
            "repetitions": len(obs),
            "unique_raw_model_texts": len({r["raw_model_text_signature"] for r in obs}),
            "unique_decision_execution_traces": len(
                {r["decision_execution_signature"] for r in obs}
            ),
        }
        for (v, rid), obs in sorted(groups.items())
    ]
    return {
        "requests": rows,
        "variants": {
            v: {
                "requests": sum(r["variant"] == v for r in rows),
                "requests_with_raw_text_variation": sum(
                    r["variant"] == v and r["unique_raw_model_texts"] > 1 for r in rows
                ),
                "requests_with_decision_execution_variation": sum(
                    r["variant"] == v and r["unique_decision_execution_traces"] > 1 for r in rows
                ),
            }
            for v in ["V2", "V3", "V4"]
        },
        "e3_evidence": errors,
        "e3_cause_counts": dict(Counter(e["cause"] for e in errors)),
        "e3_proposed_tool_counts": dict(Counter(e.get("proposed_tool", "unknown") for e in errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/matched_deviation_analysis.json")
    )
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    analysis = analyze(SESSION)
    require(
        analysis == read(Path("evaluation/experiments/2026-10-09_matched_analysis.json")),
        "Recomputed session analysis differs from saved evidence",
    )
    cases = inventory(analysis, SESSION)
    result = {
        "analysis_version": "1",
        "session": SESSION.as_posix(),
        "verified_runs": analysis["verified_runs"],
        "model_calls": analysis["model_calls"],
        "session_analysis_reproduced_exactly": True,
        "input_sha256": analysis["input_sha256"],
        "cases": cases,
        "supplementary": supplementary(analysis, SESSION),
    }
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("analyze_recorded_session.py"),
        Path(__file__).with_name("audit_measurements.py"),
        ROOT / "RESULTS.md",
        ROOT / "evaluation/experiments/2026-10-09_matched_analysis.json",
    ]
    if args.annotations:
        annotations = read(args.annotations)
        require(
            set(annotations["cases"]) == {c["case_id"] for c in cases},
            "Incomplete/extra annotations",
        )
        for case in cases:
            annotation = annotations["cases"][case["case_id"]]
            require(
                annotation["verdict"]
                in {
                    "equivalent",
                    "non_equivalent",
                    "non_equivalent_under_grain_rule",
                    "unresolved",
                },
                "Invalid verdict",
            )
            require(
                bool(annotation["rationale"]) and bool(annotation["rules"]), "Missing rationale"
            )
            case["review"] = annotation
        result["review_metadata"] = {k: v for k, v in annotations.items() if k != "cases"}
        result["verdict_counts"] = {
            v: dict(Counter(c["review"]["verdict"] for c in cases if c["variant"] == v))
            for v in ["V2", "V3", "V4"]
        }
        source_paths.append(args.annotations.resolve())
    result["source_sha256"] = {
        p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in source_paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": dict(Counter(c["variant"] for c in cases)),
                "verdict_counts": result.get("verdict_counts"),
                "supplementary": result["supplementary"]["variants"],
                "e3_runs": len(result["supplementary"]["e3_evidence"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
