import json
import shutil
from pathlib import Path

import pytest
from analyze_supplementary_experiments import analyze

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "evaluation/experiments/2026-10-20_prompt"


@pytest.fixture
def copied_session(tmp_path):
    directory = tmp_path / "session"
    shutil.copytree(SESSION, directory)
    return directory


def test_verifies_frozen_session_and_separates_treatments(copied_session):
    result = analyze(copied_session)
    assert result["verified_runs"] == result["model_calls"] == 60
    assert set(result["variants"]) == {"V3_original", "V3_completion"}
    assert len(result["prompt_pairs"]) == 30


def test_rejects_wrong_seed_in_actual_payload(copied_session):
    path = next((copied_session / "runs").glob("*/model_calls/*.request.json"))
    payload = json.loads(path.read_text())
    payload["options"]["seed"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="payload changed"):
        analyze(copied_session)


def test_rejects_missing_generation_evidence(copied_session):
    # Rename in this test-owned temporary copy to simulate a missing response.
    path = next((copied_session / "runs").glob("*/model_calls/*.response.txt"))
    path.rename(path.with_suffix(".missing"))
    with pytest.raises(ValueError, match="recording"):
        analyze(copied_session)


def test_rejects_incomplete_schedule_even_with_completed_marker(copied_session):
    path = copied_session / "finished.json"
    finished = json.loads(path.read_text())
    finished["completed_runs"] -= 1
    path.write_text(json.dumps(finished))
    with pytest.raises(ValueError, match="Coverage mismatch"):
        analyze(copied_session)
