import hashlib
import json

import pytest
from analyze_recorded_session import (
    analyze,
    group_results,
    latency_stats,
    render_report,
    verify_ledger,
)
from audit_measurements import canonical, projection


def row(rid, repetition, value, *, variant="V2", status="success"):
    trace = {
        "tool_calls": [{"tool_name": "tool", "status": "executed", "arguments": {"value": value}}]
    }
    return {
        "variant": variant,
        "request_id": rid,
        "repetition_index": repetition,
        "legacy_signature": canonical(projection(trace, statuses=True)),
        "status": status,
        "category": "test",
        "expected_length": 1,
        "exact_projection_match": True,
        "numeric_value_match": True,
        "error_codes": [],
        "model_calls": 1,
        "dependency_findings": [],
        "schema_violations": 0,
        "latency_ms": 10,
    }


def test_repeatability_is_within_request_and_separate_from_success():
    rows = [row("R001", i, 3, status="rejected") for i in range(5)]
    rows += [row("R002", i, 7) for i in range(5)]
    result = group_results(rows)["V2"]
    assert result["requests_with_trace_variation"] == 0
    assert result["validated_success"] == 5
    rows[-1] = row("R002", 4, 8)
    assert group_results(rows)["V2"]["requests_with_trace_variation"] == 1


def test_legacy_signature_keeps_numeric_representation_and_status_variation_separate():
    rows = [row("R001", 0, 3), row("R001", 1, 3.0, status="rejected")]
    result = group_results(rows)["V2"]
    assert result["requests_with_trace_variation"] == 1
    assert result["requests_with_status_variation"] == 1
    with pytest.raises(ValueError, match="Duplicate repetition"):
        group_results([rows[0], rows[0]])


def test_ledger_rejects_missing_duplicate_or_reordered_events():
    task = {
        "position": 0,
        "task_id": "V2_R001_rep00",
        "variant": "V2",
        "request_id": "R001",
        "repetition_index": 0,
    }
    events = [{**task, "state": "started"}, {**task, "state": "completed"}]
    verify_ledger([task], events)
    for bad in [events[:1], events[::-1], events * 2]:
        with pytest.raises(ValueError):
            verify_ledger([task], bad)


def test_latency_empty_group_is_not_zero_and_p95_uses_nearest_rank():
    assert latency_stats([])["mean_ms"] is None
    assert latency_stats(list(range(1, 21)))["p95_nearest_rank_ms"] == 19


def fixture_session(directory):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    req = {
        "id": "R001",
        "category": "fixture",
        "expected_tool_sequence": ["get_sales_data"],
        "expected_parameters": {"timeframe": "last_7_days", "dimensions": ["region"]},
    }
    artifacts = directory / "artifacts"
    names = [
        "evaluation/requests/initial_requests.json",
        "prototype/schemas/tools.json",
        "prototype/schemas/run_log.schema.json",
    ]
    save(artifacts / names[0], {"requests": [req]})
    save(
        artifacts / names[1],
        {
            "tools": [
                {
                    "name": "get_sales_data",
                    "parameters_schema": {"properties": {"timeframe": {}, "dimensions": {}}},
                }
            ]
        },
    )
    save(artifacts / names[2], {"type": "object"})
    save(directory / "hardware.json", {"fixture": True})
    identity = {"version": {"version": "test"}, "digest": "fixture", "show": {}}
    snapshot = {
        "version": identity["version"],
        "model": {"digest": "fixture"},
        "show": {},
        "residency": {"models": []},
    }
    save(directory / "runtime_prepared.json", snapshot)
    tasks, events = [], []
    for variant, full in [
        ("V2", "fixed_sequence_llm_parameters"),
        ("V3", "llm_tool_selection"),
        ("V4", "llm_multi_step_planning"),
    ]:
        for rep in range(5):
            task = {
                "position": len(tasks),
                "task_id": f"{variant}_R001_rep{rep:02d}",
                "request_id": "R001",
                "repetition_index": rep,
                "variant": variant,
            }
            tasks.append(task)
            task_dir = directory / "runs" / task["task_id"]
            call = {
                "tool_name": "get_sales_data",
                "arguments": req["expected_parameters"],
                "status": "executed",
                "index": 0,
                "result_ref": "data",
            }
            log = {
                "run_id": task["task_id"],
                "request_id": "R001",
                "repetition_index": rep,
                "variant": full,
                "status": "success",
                "tool_calls": [call],
                "errors": [],
                "latency_ms": 1,
                "metrics": {"schema_validation_failures": 0},
            }
            save(task_dir / "run_log.json", log)
            save(task_dir / "runtime_before.json", snapshot)
            save(task_dir / "runtime_after.json", snapshot)
            proposed = {"tool_name": call["tool_name"], "arguments": call["arguments"]}
            decisions = (
                [call["arguments"]]
                if variant == "V2"
                else [{"tool_calls": [proposed]}]
                if variant == "V3"
                else [{"finish": False, "tool_call": proposed}, {"finish": True, "tool_call": None}]
            )
            events.extend(
                [
                    {**task, "state": "started"},
                    {
                        **task,
                        "state": "completed",
                        "run_id": log["run_id"],
                        "run_status": "success",
                        "model_calls": len(decisions),
                    },
                ]
            )
            for i, decision in enumerate(decisions):
                folder = task_dir / "model_calls"
                prompt = "fixture"
                save(
                    folder / f"{i:04d}.request.json",
                    {
                        "model": "test",
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.0, "seed": 0},
                    },
                )
                save(folder / f"{i:04d}.response.txt", {"response": json.dumps(decision)})
                save(
                    folder / f"{i:04d}.event.json",
                    {
                        "state": "received",
                        "index": i,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response_sha256": sha(folder / f"{i:04d}.response.txt"),
                        "telemetry": {},
                        "http_response_elapsed_ms": 1,
                    },
                )
    save(
        directory / "manifest.json",
        {
            "schedule": tasks,
            "model": "test",
            "source_sha256": {name: sha(artifacts / name) for name in names},
            "hardware_sha256": sha(directory / "hardware.json"),
            "configuration_identity": identity,
        },
    )
    save(directory / "started.json", {"manifest_sha256": sha(directory / "manifest.json")})
    save(
        directory / "finished.json",
        {
            "state": "completed",
            "error": None,
            "scheduled_runs": len(tasks),
            "completed_runs": len(tasks),
        },
    )
    (directory / "ledger.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return directory


def test_complete_fixture_is_verified_and_unfinished_series_rejected(tmp_path):
    directory = fixture_session(tmp_path / "fixture")
    result = analyze(directory)
    assert result["verified_runs"] == 15
    assert result["model_calls"] == 20
    assert result["h2"]["equal_unique_traces"] == 1
    report = render_report(result, directory / "analysis.json")
    assert "Verified 15 of 15 scheduled runs" in report
    assert "provide no support for H2" in report
    assert "| V4 | 5 | 5 | 5 | 5 | 10 |" in report
    final_path = directory / "finished.json"
    final = json.loads(final_path.read_text())
    final["state"] = "aborted"
    final_path.write_text(json.dumps(final))
    with pytest.raises(ValueError, match="not completed"):
        analyze(directory)


def test_tampered_response_prevents_analysis(tmp_path):
    directory = fixture_session(tmp_path / "fixture")
    path = directory / "runs/V2_R001_rep00/model_calls/0000.response.txt"
    path.write_text('{"response":"changed"}')
    with pytest.raises(ValueError, match="Response hash mismatch"):
        analyze(directory)
