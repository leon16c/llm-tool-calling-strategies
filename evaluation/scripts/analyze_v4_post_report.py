"""Inspect every matched post-report proposal and retain concrete prompt evidence."""

import json
from collections import Counter
from pathlib import Path

from analyze_recorded_session import read, require, sha

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "evaluation/experiments/2026-10-09_matched"


def main():
    rows, hashes = [], {}
    for path in sorted((SESSION / "runs").glob("V4_*/run_log.json")):
        log = read(path)
        if not any(
            "Planner selected unnecessary tool after completion: " in e["message"]
            for e in log["validation_events"]
        ):
            continue
        index = len(log["tool_calls"]) - 1
        payload_path = path.parent / f"model_calls/{index:04d}.request.json"
        response_path = path.parent / f"model_calls/{index:04d}.response.txt"
        payload, response = read(payload_path), read(response_path)
        decision = json.loads(response["response"])
        history = json.loads(
            payload["prompt"]
            .split("Executed tool calls:\n", 1)[1]
            .split("\n\nTool set:\n", 1)[0]
        )
        require(
            history == log["tool_calls"][:-1], "Prompt history differs from execution"
        )
        require(
            history[-1]["tool_name"] == "format_report",
            "No immediately preceding report",
        )
        require(decision["finish"] is False, "Unexpected finish decision")
        proposed = decision["tool_call"]
        args = proposed["arguments"]
        prior = [c for c in history if c["tool_name"] == proposed["tool_name"]]

        def non_reference(arguments):
            return {k: v for k, v in arguments.items() if k != "data_id"}

        row = {
            "task_id": path.parent.name,
            "request_id": log["request_id"],
            "repetition_index": log["repetition_index"],
            "proposed_tool": proposed["tool_name"],
            "proposed_arguments": args,
            "finish_instruction_in_payload": "If the latest executed tool call is format_report, the request is complete."
            in payload["prompt"],
            "no_repeat_instruction_in_payload": "Do not repeat a tool that has already produced the needed intermediate result."
            in payload["prompt"],
            "same_tool_seen_before": bool(prior),
            "same_non_reference_arguments_seen_before": any(
                non_reference(c["arguments"]) == non_reference(args) for c in prior
            ),
            "reference_kind": "placeholder"
            if args.get("data_id") == "$previous_result"
            else "latest_result"
            if args.get("data_id") == history[-1]["result_ref"]
            else "earlier_result"
            if args.get("data_id") in {c["result_ref"] for c in history[:-1]}
            else "other",
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "raw_response": response["response"],
            "payload_path": payload_path.relative_to(ROOT).as_posix(),
            "response_path": response_path.relative_to(ROOT).as_posix(),
        }
        rows.append(row)
        for source in (path, payload_path, response_path):
            hashes[source.relative_to(ROOT).as_posix()] = sha(source)
    require(
        len(rows) == 70 and len({r["request_id"] for r in rows}) == 14,
        "Unexpected post-report coverage",
    )
    require(
        all(
            r["finish_instruction_in_payload"] and r["no_repeat_instruction_in_payload"]
            for r in rows
        ),
        "Missing original instruction",
    )
    summary = {
        "runs": len(rows),
        "requests": len({r["request_id"] for r in rows}),
        "tools": dict(Counter(r["proposed_tool"] for r in rows)),
        "reference_kind": dict(Counter(r["reference_kind"] for r in rows)),
        "same_tool_seen_before": sum(r["same_tool_seen_before"] for r in rows),
        "same_non_reference_arguments_seen_before": sum(
            r["same_non_reference_arguments_seen_before"] for r in rows
        ),
        "max_prompt_eval_plus_eval_count": max(
            r["prompt_eval_count"] + r["eval_count"] for r in rows
        ),
    }
    result = {
        "summary": summary,
        "source_sha256": sha(Path(__file__)),
        "input_sha256": hashes,
        "runs": rows,
    }
    (ROOT / "evaluation/matched_v4_post_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
