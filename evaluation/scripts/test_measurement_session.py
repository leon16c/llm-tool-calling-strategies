import io
import json
from collections import Counter

import measurement_session as session
import pytest
from toolcall_eval.deterministic_baseline import DeterministicBaseline
from toolcall_eval.ollama_mapper import (
    OllamaMultiStepPlanner,
    OllamaParameterMapper,
    OllamaToolCallPlanner,
)
from toolcall_eval.oracle import get_request


def runtime():
    return {
        "version": {"version": "test"},
        "model": {"digest": "testdigest"},
        "show": {"template": "unchanged"},
        "residency": {"models": []},
    }


def prepare(tmp_path, monkeypatch, variants=None):
    hardware = tmp_path / "hardware.json"
    hardware.write_text('{"source":"test fixture"}')
    monkeypatch.setattr(session, "runtime_snapshot", lambda *args: runtime())
    directory = tmp_path / "experiment"
    session.prepare_session(
        directory,
        request_ids=["R002"],
        repetitions=1,
        variants=variants or ["V2"],
        hardware_path=hardware,
        purpose="smoke",
    )
    return directory


def test_schedule_has_complete_unique_coverage_and_balanced_variant_positions():
    ids = [f"R{i:03d}" for i in range(1, 31)]
    tasks = session.schedule(ids, 5, list(session.VARIANTS))
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 450
    assert Counter(t["variant"] for t in tasks) == {"V2": 150, "V3": 150, "V4": 150}
    for offset in range(3):
        assert Counter(t["variant"] for t in tasks[offset::3]) == {"V2": 50, "V3": 50, "V4": 50}


@pytest.mark.parametrize(
    "bad_args", [([], 5, ["V2"]), (["R002"], 0, ["V2"]), (["R002"], 1, ["V2", "V2"])]
)
def test_schedule_rejects_invalid_coverage(bad_args):
    with pytest.raises(ValueError):
        session.schedule(*bad_args)


@pytest.mark.parametrize(
    "adapter", [OllamaParameterMapper, OllamaToolCallPlanner, OllamaMultiStepPlanner]
)
def test_recorded_payload_matches_existing_adapter_and_keeps_raw_response(
    tmp_path, monkeypatch, adapter
):
    raw = (
        json.dumps(
            {"response": "not valid model JSON", "eval_count": 7, "eval_duration": 100}, indent=2
        )
        + "\n"
    )
    sent = []

    def original_transport(request, timeout):
        assert timeout == 120
        sent.append(json.loads(request.data))
        return io.BytesIO(raw.encode())

    monkeypatch.setattr("urllib.request.urlopen", original_transport)
    assert adapter()._generate_with_ollama("same prompt") == "not valid model JSON"
    recorder = session.RecordedGeneration(
        tmp_path / "calls", transport=lambda payload: sent.append(payload) or raw
    )
    assert recorder("same prompt") == "not valid model JSON"
    assert sent[0] == sent[1]
    assert (tmp_path / "calls/0000.response.txt").read_text() == raw
    assert (tmp_path / "calls/0000.response.txt").read_bytes() == raw.encode("utf-8")
    assert session.read_json(tmp_path / "calls/0000.event.json")["telemetry"]["eval_count"] == 7


def test_transport_error_is_recorded_and_never_retried(tmp_path):
    def fail(payload):
        raise ValueError("transport unavailable")

    recorder = session.RecordedGeneration(tmp_path / "calls", transport=fail)
    with pytest.raises(ValueError, match="transport unavailable"):
        recorder("prompt")
    assert recorder.calls == recorder.infrastructure_errors == 1
    assert session.read_json(tmp_path / "calls/0000.event.json")["state"] == "error"
    assert (tmp_path / "calls/0000.request.json").exists()


def test_session_refuses_existing_directory_and_restart(tmp_path, monkeypatch):
    directory = prepare(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        session.prepare_session(
            directory,
            request_ids=["R002"],
            repetitions=1,
            variants=["V2"],
            hardware_path=tmp_path / "hardware.json",
            purpose="smoke",
        )
    session.write_new(directory / "started.json", {"already": "started"})
    with pytest.raises(FileExistsError):
        session.run_session(directory)
    assert session.read_json(directory / "started.json") == {"already": "started"}


def test_configuration_drift_aborts_without_generation(tmp_path, monkeypatch):
    directory = prepare(tmp_path, monkeypatch)
    changed = runtime()
    changed["model"]["digest"] = "changed"
    monkeypatch.setattr(session, "runtime_snapshot", lambda *args: changed)
    with pytest.raises(RuntimeError, match="drift"):
        session.run_session(directory)
    final = session.read_json(directory / "finished.json")
    assert final["state"] == "aborted"
    assert final["completed_runs"] == 0
    assert "drift" in final["error"]["message"]
    assert not list(directory.rglob("*.request.json"))


def test_infrastructure_failure_stops_schedule_even_when_runner_returns_rejection(
    tmp_path, monkeypatch
):
    directory = prepare(tmp_path, monkeypatch, ["V2", "V4"])

    def fail(*args):
        raise ValueError("connection failed")

    monkeypatch.setattr(session, "api_text", fail)
    with pytest.raises(RuntimeError, match="transport/envelope"):
        session.run_session(directory)
    assert session.read_json(directory / "finished.json")["completed_runs"] == 0
    log = session.read_json(directory / "runs/V2_R002_rep00/run_log.json")
    assert log["status"] == "rejected"
    events = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()]
    assert [e["state"] for e in events] == ["started", "error"]


@pytest.mark.parametrize("variant", ["V2", "V3", "V4"])
def test_recording_preserves_runner_behavior_and_records_finish(tmp_path, monkeypatch, variant):
    request = get_request("R002")
    expected = DeterministicBaseline().run(request)
    calls = [
        {"tool_name": c["tool_name"], "arguments": dict(c["arguments"])}
        for c in expected["tool_calls"]
    ]
    if variant == "V2":
        responses = [json.dumps(c["arguments"]) for c in calls]
    else:
        for call in calls:
            if "data_id" in call["arguments"]:
                call["arguments"]["data_id"] = "$previous_result"
        if variant == "V3":
            responses = [json.dumps({"tool_calls": calls})]
        else:
            responses = [json.dumps({"finish": False, "tool_call": c}) for c in calls]
            responses.append(json.dumps({"finish": True, "tool_call": None}))
    queue = iter(responses)
    directory = prepare(tmp_path, monkeypatch, [variant])
    monkeypatch.setattr(session, "api_text", lambda *args: json.dumps({"response": next(queue)}))
    session.run_session(directory)
    task_dir = directory / "runs" / f"{variant}_R002_rep00"
    log = session.read_json(task_dir / "run_log.json")
    assert log["status"] == "success"
    assert [(c["tool_name"], c["arguments"]) for c in log["tool_calls"]] == [
        (c["tool_name"], c["arguments"]) for c in expected["tool_calls"]
    ]
    assert len(list((task_dir / "model_calls").glob("*.request.json"))) == len(responses)
    assert session.read_json(directory / "finished.json")["state"] == "completed"
    if variant == "V4":
        raw = session.read_json(task_dir / "model_calls/0003.response.txt")
        assert json.loads(raw["response"])["finish"] is True
