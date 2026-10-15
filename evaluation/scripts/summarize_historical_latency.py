"""Summarize saved historical runner latencies by outcome without model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_recorded_session import latency_stats

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "V2": "fixed_sequence_llm_parameters_ollama",
    "V3": "llm_tool_selection_ollama",
    "V4": "llm_multi_step_planning_ollama",
    "V2_repeated": "fixed_sequence_llm_parameters_ollama_repeated",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    for label, directory in DATASETS.items():
        paths = sorted((ROOT / "prototype/logs" / directory).glob("*.json"))
        expected = 150 if label == "V2_repeated" else 30
        if len(paths) != expected:
            raise ValueError(f"{label}: expected {expected} logs, found {len(paths)}")
        logs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if any(log["status"] not in {"success", "rejected"} for log in logs):
            raise ValueError(f"Unexpected status in {label}")
        result[label] = {
            "runs": len(logs),
            "by_status": {
                status: latency_stats(
                    [log["latency_ms"] for log in logs if log["status"] == status]
                )
                for status in ("success", "rejected")
            },
            "input_sha256": {
                path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
