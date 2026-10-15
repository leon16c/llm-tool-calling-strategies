"""Recompute supplementary runner outcomes from exact generations, without inference."""

import argparse
import json
from pathlib import Path

from analyze_gate_sensitivity import comparable
from analyze_recorded_session import read, require, sha
from analyze_supplementary_experiments import analyze
from supplementary_session import build_runner
from toolcall_eval.oracle import load_requests


class ReplayGeneration:
    def __init__(self, directory, manifest, task):
        self.model = manifest["model"]
        self.base_url = manifest["base_url"]
        self.temperature = task["temperature"]
        self.seed = task["seed"]
        self.treatment = task["treatment"]
        self.directory = directory
        self.calls = 0

    def __call__(self, prompt):
        prefix = self.directory / f"{self.calls:04d}"
        payload = read(prefix.with_suffix(".request.json"))
        require(payload["prompt"] == prompt, "Recomputed prompt differs from recording")
        self.calls += 1
        return read(prefix.with_suffix(".response.txt"))["response"]


def verify(directory):
    analysis = analyze(directory)
    manifest = read(directory / "manifest.json")
    requests = {
        r.id: r
        for r in load_requests(
            directory / "artifacts/evaluation/requests/initial_requests.json"
        )
    }
    for task in manifest["schedule"]:
        run = directory / "runs" / task["task_id"]
        recorder = ReplayGeneration(run / "model_calls", manifest, task)
        log = build_runner(task["variant"], recorder).run(
            requests[task["request_id"]], task["repetition_index"]
        )
        saved = read(run / "run_log.json")
        require(comparable(log) == comparable(saved), "Runner replay outcome differs")
        require(log["model"] == saved["model"], "Runner replay model settings differ")
        require(
            recorder.calls == len(list((run / "model_calls").glob("*.response.txt"))),
            "Replay did not consume every recorded generation",
        )
    return {
        "session": directory.as_posix(),
        "replayed_runs": analysis["verified_runs"],
        "model_calls": analysis["model_calls"],
        "all_outcomes_reproduced": True,
        "manifest_sha256": sha(directory / "manifest.json"),
        "verification_source_sha256": sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.session)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
