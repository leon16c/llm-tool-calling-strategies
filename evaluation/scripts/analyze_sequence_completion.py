"""Separate executed prescribed sequences from runner acceptance in retained logs."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_recorded_session import analyze, read, require, sha
from audit_measurements import ROOT

SESSION = Path("evaluation/experiments/2026-10-09_matched")


def sequence_completed(log: dict, expected: list[str]) -> bool:
    """Count only executed tools, in order; arguments and report meaning are unassessed."""
    require(bool(expected), "Expected sequence must not be empty")
    executed = [
        call["tool_name"] for call in log["tool_calls"] if call["status"] == "executed"
    ]
    return executed[: len(expected)] == expected


def main() -> None:
    analysis = analyze(SESSION)
    saved_path = ROOT / "evaluation/experiments/2026-10-09_matched_analysis.json"
    require(analysis == read(saved_path), "Saved matched analysis not reproduced")
    requests_path = (
        ROOT / SESSION / "artifacts/evaluation/requests/initial_requests.json"
    )
    requests = {r["id"]: r for r in read(requests_path)["requests"]}
    rows, hashes = [], {requests_path.relative_to(ROOT).as_posix(): sha(requests_path)}
    for row in analysis["runs"]:
        log_path = ROOT / SESSION / "runs" / row["task_id"] / "run_log.json"
        log = read(log_path)
        complete = sequence_completed(
            log, requests[row["request_id"]]["expected_tool_sequence"]
        )
        accepted = row["status"] == "success"
        require(not accepted or complete, "Accepted run without prescribed sequence")
        rows.append(
            {
                "task_id": row["task_id"],
                "variant": row["variant"],
                "request_id": row["request_id"],
                "repetition_index": row["repetition_index"],
                "status": row["status"],
                "prescribed_sequence_completed": complete,
            }
        )
        hashes[log_path.relative_to(ROOT).as_posix()] = sha(log_path)
    summaries = {}
    for variant in ("V2", "V3", "V4"):
        selected = [r for r in rows if r["variant"] == variant]
        grouped = {
            rid: [r for r in selected if r["request_id"] == rid] for rid in requests
        }
        require(len(selected) == 150, "Incomplete variant")
        for group in grouped.values():
            require(
                {r["repetition_index"] for r in group} == set(range(5)),
                "Incomplete request",
            )
            require(
                len({(r["status"], r["prescribed_sequence_completed"]) for r in group})
                == 1,
                "Completion or status varies between repetitions",
            )
        summaries[variant] = {
            "runs": len(selected),
            "requests": len(grouped),
            "accepted_runs": sum(r["status"] == "success" for r in selected),
            "completed_sequence_runs": sum(
                r["prescribed_sequence_completed"] for r in selected
            ),
            "completed_sequence_requests": sum(
                g[0]["prescribed_sequence_completed"] for g in grouped.values()
            ),
            "completed_but_rejected_runs": sum(
                r["prescribed_sequence_completed"] and r["status"] == "rejected"
                for r in selected
            ),
            "completed_but_rejected_request_ids": sorted(
                rid
                for rid, g in grouped.items()
                if g[0]["prescribed_sequence_completed"]
                and g[0]["status"] == "rejected"
            ),
        }
    result = {
        "definition": "The executed tool names start with the entire prescribed sequence in order. Proposed or blocked calls do not count. Arguments and business-output correctness are not assessed.",
        "session": SESSION.as_posix(),
        "verified_runs": len(rows),
        "saved_analysis_reproduced": True,
        "saved_analysis_sha256": sha(saved_path),
        "analysis_source_sha256": sha(Path(__file__)),
        "input_sha256": hashes,
        "variants": summaries,
        "runs": rows,
    }
    (ROOT / "evaluation/matched_sequence_completion.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
