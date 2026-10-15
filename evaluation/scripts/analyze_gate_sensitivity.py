"""Replay retained plans/prefixes without oracle workflow gates, without inference."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from analyze_recorded_session import analyze, read, require, sha
from toolcall_eval.llm_multi_step_planning import LLMMultiStepPlanning
from toolcall_eval.llm_tool_selection import LLMToolSelection, TOOL_NAMES
from toolcall_eval.mock_tools import execute_tool
from toolcall_eval.models import ToolCall
from toolcall_eval.ollama_mapper import OllamaMultiStepPlanner, OllamaToolCallPlanner
from toolcall_eval.oracle import load_requests
from toolcall_eval.validators import ToolSchemaValidator

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "evaluation/experiments/2026-10-09_matched"


class NoSequenceGate(LLMToolSelection):
    def _validate_planned_sequence(self, planned_tool_calls, request):
        return None


def comparable(log):
    return {
        "status": log["status"],
        "errors": log["errors"],
        "calls": [
            {
                k: c[k]
                for k in ("tool_name", "arguments", "status", "result_ref")
                if k in c
            }
            for c in log["tool_calls"]
        ],
    }


def replay_v4(request, generated, payloads, *, max_steps=8):
    """Use recorded decisions only while their original input history still matches."""
    executed = []
    checks = []
    validator = ToolSchemaValidator()
    resolver = LLMMultiStepPlanning()
    for index in range(max_steps):
        if index >= len(generated):
            return outcome("unknown_continuation", executed, checks)
        planner = OllamaMultiStepPlanner(generate=lambda prompt: generated[index])
        if planner._build_prompt(request, executed) != payloads[index]["prompt"]:
            return outcome("unknown_changed_history", executed, checks)
        try:
            decision = planner.plan_next_step(request, executed)
        except (KeyError, TypeError, ValueError) as exc:
            checks.append({"index": index, "failure": "parse", "detail": str(exc)})
            return outcome("rejected_parse", executed, checks)
        if decision.finish:
            return outcome("finished", executed, checks)
        call = decision.tool_call
        if call is None or call.tool_name not in TOOL_NAMES:
            checks.append({"index": index, "failure": "unsupported_or_missing_tool"})
            return outcome("rejected_tool", executed, checks)
        arguments = dict(call.arguments)
        errors = validator.validate_tool_call(call.tool_name, arguments)
        if errors:
            checks.append({"index": index, "failure": "schema", "detail": errors})
            return outcome("rejected_schema", executed, checks)
        error = resolver._resolve_data_reference(
            arguments, executed[-1].result_ref if executed else None
        )
        if error:
            checks.append({"index": index, "failure": "dependency", "detail": error})
            return outcome("rejected_dependency", executed, checks)
        executed.append(
            ToolCall(
                index=index,
                tool_name=call.tool_name,
                arguments=arguments,
                source="llm",
                status="executed",
                result_ref=execute_tool(call.tool_name, arguments),
            )
        )
        checks.append({"index": index, "remaining_checks": "passed"})
    return outcome("rejected_fixed_budget", executed, checks)


def outcome(status, calls, checks):
    names = [c.tool_name for c in calls]
    return {
        "counterfactual_status": status,
        "executed_sequence": names,
        "report_reached": "format_report" in names,
        "last_executed_is_report": bool(names) and names[-1] == "format_report",
        "checks": checks,
    }


def main():
    verified = analyze(SESSION.relative_to(ROOT))
    require(
        verified
        == read(ROOT / "evaluation/experiments/2026-10-09_matched_analysis.json"),
        "Primary analysis no longer reproduces",
    )
    # Reuse the implementation only after checking it against the measured snapshot.
    sources = read(SESSION / "manifest.json")["source_sha256"]
    for name, expected in sources.items():
        if name.startswith(("prototype/src/", "prototype/schemas/")):
            require(
                sha(ROOT / name) == expected, f"Measured implementation changed: {name}"
            )
    requests = {
        r.id: r
        for r in load_requests(
            SESSION / "artifacts/evaluation/requests/initial_requests.json"
        )
    }
    rows, hashes = [], {}
    for row in verified["runs"]:
        if row["variant"] == "V2":
            continue
        directory = SESSION / "runs" / row["task_id"]
        original = read(directory / "run_log.json")
        files = sorted((directory / "model_calls").glob("*.response.txt"))
        generated = [read(path)["response"] for path in files]
        payloads = [
            read(path.with_name(path.name.replace("response.txt", "request.json")))
            for path in files
        ]
        request = requests[row["request_id"]]
        replay_inputs = iter(generated)
        if row["variant"] == "V3":
            planner = OllamaToolCallPlanner(generate=lambda prompt: next(replay_inputs))
            baseline = LLMToolSelection(planner=planner).run(request)
            require(comparable(baseline) == comparable(original), "V3 replay mismatch")
            planner = OllamaToolCallPlanner(generate=lambda prompt: generated[0])
            replay = NoSequenceGate(planner=planner).run(request)
            executed = [
                c["tool_name"]
                for c in replay["tool_calls"]
                if c["status"] == "executed"
            ]
            result = {
                "counterfactual_status": "finished"
                if replay["status"] == "success"
                else "rejected",
                "executed_sequence": executed,
                "report_reached": "format_report" in executed,
                "last_executed_is_report": bool(executed)
                and executed[-1] == "format_report",
                "errors": replay["errors"],
            }
        else:
            planner = OllamaMultiStepPlanner(
                generate=lambda prompt: next(replay_inputs)
            )
            baseline = LLMMultiStepPlanning(planner=planner).run(request)
            require(comparable(baseline) == comparable(original), "V4 replay mismatch")
            result = replay_v4(request, generated, payloads)
        require(next(replay_inputs, None) is None, "Unused baseline generation")
        rows.append(
            {
                "task_id": row["task_id"],
                "variant": row["variant"],
                "request_id": row["request_id"],
                "repetition_index": row["repetition_index"],
                "original_status": original["status"],
                **result,
            }
        )
        for path in [
            directory / "run_log.json",
            *(directory / "model_calls").iterdir(),
        ]:
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    summary = {}
    for variant in ("V3", "V4"):
        group = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "runs": len(group),
            "statuses": dict(Counter(r["counterfactual_status"] for r in group)),
            "report_reached": sum(r["report_reached"] for r in group),
            "finished_with_final_report": sum(
                r["counterfactual_status"] == "finished"
                and r["last_executed_is_report"]
                for r in group
            ),
            "originally_rejected_now_finished": sum(
                r["original_status"] == "rejected"
                and r["counterfactual_status"] == "finished"
                for r in group
            ),
        }
    result = {
        "method": "V3: remove complete-plan sequence gate only. V4: remove oracle next-tool and finish-length gates; fixed eight-decision cap. Retain original parser, support, schema, dependency checks and hashing mocks. Validate every recorded V4 prompt against replay history; never invent continuation. Finished describes technical termination, not semantic success.",
        "baseline_replays_verified": len(rows),
        "primary_series_verified": verified["verified_runs"],
        "source_sha256": {
            Path(__file__).name: sha(Path(__file__)),
            "analyze_recorded_session.py": sha(
                Path(__file__).with_name("analyze_recorded_session.py")
            ),
        },
        "input_sha256": hashes,
        "summary": summary,
        "runs": rows,
    }
    (ROOT / "evaluation/matched_gate_sensitivity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
