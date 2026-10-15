import json
import sys
from pathlib import Path

import pytest

from toolcall_eval.cli import main


def test_cli_writes_repeated_run_logs_with_repetition_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toolcall-eval",
            "R001",
            "--repetitions",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    first_path = tmp_path / "R001_rep00.json"
    second_path = tmp_path / "R001_rep01.json"

    assert first_path.exists()
    assert second_path.exists()
    assert json.loads(first_path.read_text(encoding="utf-8"))["repetition_index"] == 0
    assert json.loads(second_path.read_text(encoding="utf-8"))["repetition_index"] == 1


def test_cli_keeps_legacy_filename_for_single_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toolcall-eval",
            "R001",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    assert (tmp_path / "R001.json").exists()
    assert not (tmp_path / "R001_rep00.json").exists()


def test_cli_rejects_non_positive_repetition_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toolcall-eval",
            "R001",
            "--repetitions",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit):
        main()


def test_cli_runs_llm_tool_selection_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toolcall-eval",
            "R001",
            "--variant",
            "llm_tool_selection",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    run_log = json.loads((tmp_path / "R001.json").read_text(encoding="utf-8"))
    assert run_log["variant"] == "llm_tool_selection"


def test_cli_runs_llm_multi_step_planning_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toolcall-eval",
            "R001",
            "--variant",
            "llm_multi_step_planning",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    run_log = json.loads((tmp_path / "R001.json").read_text(encoding="utf-8"))
    assert run_log["variant"] == "llm_multi_step_planning"
