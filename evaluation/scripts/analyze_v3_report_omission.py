"""Count report omissions in frozen matched V3 plans; no inference or relabeling."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "evaluation/experiments/2026-10-09_matched"


def main() -> None:
    rows = []
    hashes = {}
    for path in sorted((SESSION / "runs").glob("V3_*/run_log.json")):
        response_path = path.parent / "model_calls/0000.response.txt"
        log = json.loads(path.read_text(encoding="utf-8"))
        envelope = json.loads(response_path.read_text(encoding="utf-8"))
        plan = json.loads(envelope["response"])["tool_calls"]
        names = [call["tool_name"] for call in plan]
        expected = log["input"]["expected_tool_sequence"]
        if not expected or expected[-1] != "format_report":
            raise ValueError(f"Unexpected reference completion: {path}")
        rows.append(
            {
                "task_id": path.parent.name,
                "request_id": log["request_id"],
                "repetition_index": log["repetition_index"],
                "status": log["status"],
                "planned_sequence": names,
                "expected_sequence": expected,
                "report_absent": "format_report" not in names,
                "only_missing_report_in_sequence": names == expected[:-1],
            }
        )
        for source in (path, response_path):
            hashes[source.relative_to(ROOT).as_posix()] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()

    grouped = {}
    for row in rows:
        grouped.setdefault(row["request_id"], []).append(row)
    if len(rows) != 150 or len(grouped) != 30:
        raise ValueError("Expected 150 runs covering thirty requests")
    for request_id, group in grouped.items():
        if len(group) != 5 or {r["repetition_index"] for r in group} != set(range(5)):
            raise ValueError(f"Incomplete repetition coverage: {request_id}")
        signatures = {
            json.dumps(
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("task_id", "repetition_index")
                },
                sort_keys=True,
            )
            for r in group
        }
        if len(signatures) != 1:
            raise ValueError(f"Sequence or outcome variation: {request_id}")

    def counts(selected: list[dict]) -> dict:
        return {
            "total": len(selected),
            "status": dict(Counter(r["status"] for r in selected)),
            "report_absent": sum(r["report_absent"] for r in selected),
            "only_missing_report_in_sequence": sum(
                r["only_missing_report_in_sequence"] for r in selected
            ),
        }

    result = {
        "definition": "Tool-name comparison of complete retained V3 plans. Only missing report means the plan equals the prescribed sequence without its final format_report. Argument validity and task correctness are not assessed.",
        "session": SESSION.relative_to(ROOT).as_posix(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "runs_summary": counts(rows),
        "requests_summary": counts([group[0] for group in grouped.values()]),
        "input_sha256": hashes,
        "runs": rows,
    }
    target = ROOT / "evaluation/matched_v3_report_omission.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("runs_summary", "requests_summary")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
