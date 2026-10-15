import json

from jsonschema import Draft202012Validator

from toolcall_eval.oracle import load_requests
from toolcall_eval.paths import REQUEST_ORACLE_PATH, REQUEST_ORACLE_SCHEMA_PATH


def test_initial_request_oracle_contains_30_unique_requests() -> None:
    requests = load_requests()

    assert len(requests) == 30
    assert {request.id for request in requests} == {f"R{index:03d}" for index in range(1, 31)}


def test_all_requests_define_required_oracle_fields() -> None:
    for request in load_requests():
        assert request.category
        assert request.user_request
        assert request.expected_tool_sequence
        assert request.expected_parameters["timeframe"]
        assert request.expected_parameters["dimensions"]
        assert request.expected_parameters["group_by"]
        assert request.expected_parameters["metrics"]
        assert request.expected_parameters["output_format"]


def test_initial_request_oracle_matches_json_schema() -> None:
    schema = json.loads(REQUEST_ORACLE_SCHEMA_PATH.read_text(encoding="utf-8"))
    data = json.loads(REQUEST_ORACLE_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    errors = sorted(validator.iter_errors(data), key=str)

    assert errors == []
