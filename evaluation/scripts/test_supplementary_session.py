import json

import pytest
import supplementary_session as session
from analyze_gate_sensitivity import NoSequenceGate, replay_v4
from toolcall_eval.llm_tool_selection import LLMToolSelection
from toolcall_eval.ollama_mapper import OllamaMultiStepPlanner, OllamaToolCallPlanner
from toolcall_eval.oracle import get_request


def test_schedule_pairs_seeds_and_rotates_prompt_arms():
    ids = [f"R{i:03d}" for i in range(1, 31)]
    tasks = session.experiment_schedule(ids, 5, "sampling")
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 300
    for a, b in zip(tasks[::2], tasks[1::2], strict=True):
        assert {a["variant"], b["variant"]} == {"V2", "V4"}
        assert (a["request_id"], a["seed"], a["temperature"]) == (
            b["request_id"],
            b["seed"],
            0.7,
        )
    assert {t["seed"] for t in tasks} == set(session.SEEDS)
    prompt = session.experiment_schedule(ids, 1, "prompt")
    assert len(prompt) == len({t["task_id"] for t in prompt}) == 60
    assert [t["treatment"] for t in prompt[:4]] == [
        "original",
        "completion",
        "completion",
        "original",
    ]


def test_completion_prompt_preserves_original_except_declared_guidance():
    request = get_request("R002")
    original = OllamaToolCallPlanner()._build_prompt(request)
    aligned = session.CompletionToolCallPlanner()._build_prompt(request)
    assert aligned.replace("\n" + session.COMPLETION_GUIDANCE, "", 1) == original
    assert "expected_tool_sequence" not in aligned


def test_recorder_settings_match_adapter_log_settings_and_saved_payload(tmp_path):
    sent = []
    recorder = session.RecordedGeneration(
        tmp_path / "calls",
        temperature=0.7,
        seed=33,
        transport=lambda payload: (
            sent.append(payload) or json.dumps({"response": "{}"})
        ),
    )
    runner = session.build_runner("V4", recorder)
    assert runner._planner.model_info.temperature == 0.7
    assert runner._planner.model_info.seed == 33
    recorder("prompt")
    assert sent[0] == session.read_json(tmp_path / "calls/0000.request.json")
    assert sent[0]["options"] == {"temperature": 0.7, "seed": 33}


def test_removing_v3_gate_exposes_hidden_invalid_arguments():
    request = get_request("R002")
    planner = OllamaToolCallPlanner(
        generate=lambda prompt: json.dumps(
            {
                "tool_calls": [
                    {
                        "tool_name": "get_sales_data",
                        "arguments": {
                            "timeframe": "INVALID",
                            "dimensions": ["category"],
                        },
                    }
                ]
            }
        )
    )
    assert LLMToolSelection(planner).run(request)["errors"][0]["code"] == "E2"
    assert NoSequenceGate(planner).run(request)["errors"][0]["code"] == "E4"


def test_v4_replay_does_not_invent_continuation_and_checks_hidden_schema_errors():
    request = get_request("R002")
    prompt = OllamaMultiStepPlanner()._build_prompt(request, [])
    raw = json.dumps(
        {
            "finish": False,
            "tool_call": {
                "tool_name": "get_sales_data",
                "arguments": {
                    "timeframe": "previous_quarter",
                    "dimensions": ["category"],
                },
            },
        }
    )
    result = replay_v4(request, [raw], [{"prompt": prompt}])
    assert result["counterfactual_status"] == "unknown_continuation"
    assert result["executed_sequence"] == ["get_sales_data"]
    invalid = raw.replace("previous_quarter", "INVALID")
    assert (
        replay_v4(request, [invalid], [{"prompt": prompt}])["counterfactual_status"]
        == "rejected_schema"
    )
    assert (
        replay_v4(request, [raw], [{"prompt": "different history"}])[
            "counterfactual_status"
        ]
        == "unknown_changed_history"
    )


def test_v4_early_finish_removes_oracle_length_check_without_claiming_report():
    request = get_request("R002")
    prompt = OllamaMultiStepPlanner()._build_prompt(request, [])
    result = replay_v4(
        request, [json.dumps({"finish": True, "tool_call": None})], [{"prompt": prompt}]
    )
    assert result["counterfactual_status"] == "finished"
    assert not result["report_reached"]


@pytest.mark.parametrize("repetitions", [0, 6])
def test_sampling_rejects_undeclared_seed_coverage(repetitions):
    with pytest.raises(ValueError):
        session.experiment_schedule(["R002"], repetitions, "sampling")
