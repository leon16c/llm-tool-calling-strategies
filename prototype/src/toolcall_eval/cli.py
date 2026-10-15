import argparse
import json
from pathlib import Path

from toolcall_eval.deterministic_baseline import DeterministicBaseline
from toolcall_eval.fixed_sequence_llm_parameters import FixedSequenceLLMParameters
from toolcall_eval.llm_multi_step_planning import LLMMultiStepPlanning
from toolcall_eval.llm_tool_selection import LLMToolSelection
from toolcall_eval.models import Variant
from toolcall_eval.ollama_mapper import (
    OllamaMultiStepPlanner,
    OllamaParameterMapper,
    OllamaToolCallPlanner,
)
from toolcall_eval.oracle import get_request, load_requests
from toolcall_eval.validators import RunLogValidator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request_id", nargs="?", help="Evaluation request id, e.g. R001.")
    parser.add_argument("--all", action="store_true", help="Run all requests in the oracle.")
    parser.add_argument(
        "--variant",
        choices=[
            "deterministic_workflow",
            "fixed_sequence_llm_parameters",
            "llm_tool_selection",
            "llm_multi_step_planning",
        ],
        default="deterministic_workflow",
        help="Evaluation variant to execute.",
    )
    parser.add_argument(
        "--parameter-mapper",
        choices=["oracle", "ollama"],
        default="oracle",
        help="LLM backend for LLM-based variants.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repeated runs per request.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSON run logs.")
    args = parser.parse_args()

    if not args.all and args.request_id is None:
        parser.error("provide a request_id or use --all")
    if args.variant == "deterministic_workflow" and args.parameter_mapper != "oracle":
        parser.error("--parameter-mapper is only supported for LLM-based variants")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    requests = load_requests() if args.all else [get_request(args.request_id)]
    runner = _build_runner(args.variant, args.parameter_mapper)
    run_log_validator = RunLogValidator()

    for request in requests:
        for repetition_index in range(args.repetitions):
            run_log = runner.run(request, repetition_index=repetition_index)
            validation_errors = run_log_validator.validate(run_log)
            if validation_errors:
                raise SystemExit(f"Invalid run log for {request.id}: {validation_errors}")

            serialized = json.dumps(run_log, indent=2, sort_keys=True)
            if args.output_dir is None:
                print(serialized)
            else:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / _run_log_filename(
                    request.id,
                    repetition_index,
                    args.repetitions,
                )
                output_path.write_text(serialized + "\n", encoding="utf-8")


def _build_runner(
    variant: Variant,
    parameter_mapper: str = "oracle",
) -> DeterministicBaseline | FixedSequenceLLMParameters | LLMToolSelection | LLMMultiStepPlanning:
    if variant == "deterministic_workflow":
        return DeterministicBaseline()
    if variant == "fixed_sequence_llm_parameters":
        if parameter_mapper == "oracle":
            return FixedSequenceLLMParameters()
        if parameter_mapper == "ollama":
            return FixedSequenceLLMParameters(mapper=OllamaParameterMapper())
        raise ValueError(f"Unsupported parameter mapper: {parameter_mapper}")
    if variant == "llm_tool_selection":
        if parameter_mapper == "oracle":
            return LLMToolSelection()
        if parameter_mapper == "ollama":
            return LLMToolSelection(planner=OllamaToolCallPlanner())
        raise ValueError(f"Unsupported parameter mapper: {parameter_mapper}")
    if variant == "llm_multi_step_planning":
        if parameter_mapper == "oracle":
            return LLMMultiStepPlanning()
        if parameter_mapper == "ollama":
            return LLMMultiStepPlanning(planner=OllamaMultiStepPlanner())
        raise ValueError(f"Unsupported parameter mapper: {parameter_mapper}")
    raise ValueError(f"Unsupported variant: {variant}")


def _run_log_filename(
    request_id: str,
    repetition_index: int,
    repetition_count: int,
) -> str:
    if repetition_count == 1:
        return f"{request_id}.json"
    return f"{request_id}_rep{repetition_index:02d}.json"


if __name__ == "__main__":
    main()
