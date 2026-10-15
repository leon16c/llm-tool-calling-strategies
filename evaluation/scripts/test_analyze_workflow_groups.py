import pytest

from analyze_workflow_groups import analyze, group_rows, summarize, validate_coverage


def test_grouping_uses_expected_length_and_keeps_request_denominators():
    rows = [
        {
            "request_id": "A",
            "expected_length": 4,
            "status": "rejected",
            "exact_match": False,
            "numeric_match": True,
            "primary_error": "E4",
        }
        for _ in range(5)
    ] + [
        {
            "request_id": "B",
            "expected_length": 3,
            "status": "success",
            "exact_match": True,
            "numeric_match": True,
            "primary_error": None,
        }
    ]
    groups = group_rows(rows, "expected_length")
    assert groups["4"]["requests"] == 1
    assert groups["4"]["runs"] == 5
    assert groups["4"]["validated_success"] == 0
    assert groups["4"]["exact_matches"] == 0
    assert groups["4"]["numeric_matches"] == 5
    for key in ("runs", "validated_success", "exact_matches", "numeric_matches"):
        assert sum(g[key] for g in groups.values()) == summarize(rows)[key]


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "variant", "sequence", "text"]
)
def test_coverage_rejects_missing_duplicate_or_misidentified_evidence(defect):
    requests = {"A": {"expected_tool_sequence": ["tool"], "user_request": "request"}}
    logs = [
        {
            "request_id": "A",
            "repetition_index": 0,
            "variant": "v",
            "status": "success",
            "input": {"expected_tool_sequence": ["tool"], "user_request": "request"},
        }
    ]
    validate_coverage(logs, requests, 1, "v")
    if defect == "missing":
        logs.clear()
    elif defect == "duplicate":
        logs.append(logs[0])
    elif defect == "variant":
        logs[0]["variant"] = "other"
    elif defect == "sequence":
        logs[0]["input"]["expected_tool_sequence"] = []
    else:
        logs[0]["input"]["user_request"] = "changed"
    with pytest.raises(ValueError):
        validate_coverage(logs, requests, 1, "v")


def test_saved_evidence_reproduces_audited_totals_and_partitions():
    expected = {
        "historical_V1": (30, 30, 30, 30),
        "historical_V2": (30, 27, 16, 18),
        "historical_V3": (30, 3, 2, 3),
        "historical_V4": (30, 1, 1, 1),
        "historical_V2_repeated": (150, 135, 80, 90),
        "matched_V2": (150, 145, 55, 65),
        "matched_V3": (150, 15, 5, 10),
        "matched_V4": (150, 5, 0, 0),
    }
    measures = ("runs", "validated_success", "exact_matches", "numeric_matches")
    for label, dataset in analyze()["datasets"].items():
        total = dataset["overall"]
        assert tuple(total[key] for key in measures) == expected[label]
        for dimension in ("category", "expected_length", "expected_workflow"):
            groups = dataset[dimension].values()
            assert sum(group["requests"] for group in groups) == 30
            for key in measures:
                assert sum(group[key] for group in groups) == total[key]
        assert dataset["expected_length"]["3"]["requests"] == 16
        assert dataset["expected_length"]["4"]["requests"] == 14
