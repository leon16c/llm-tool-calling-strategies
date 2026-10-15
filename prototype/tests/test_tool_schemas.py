from toolcall_eval.deterministic_baseline import DeterministicBaseline
from toolcall_eval.oracle import load_requests
from toolcall_eval.validators import RunLogValidator, ToolSchemaValidator


def test_expected_tool_calls_are_valid_against_tool_schemas() -> None:
    runner = DeterministicBaseline()
    validator = ToolSchemaValidator()

    for request in load_requests():
        run_log = runner.run(request)
        for tool_call in run_log["tool_calls"]:
            errors = validator.validate_tool_call(tool_call["tool_name"], tool_call["arguments"])
            assert errors == []


def test_deterministic_run_logs_match_run_log_schema() -> None:
    runner = DeterministicBaseline()
    validator = RunLogValidator()

    for request in load_requests():
        run_log = runner.run(request)
        assert validator.validate(run_log) == []

