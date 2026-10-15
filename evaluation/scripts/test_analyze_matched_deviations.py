"""Check evidence boundaries without executing a model or changing archived artifacts."""

import copy
import unittest
from pathlib import Path

from analyze_matched_deviations import decision_content, differences, e3_cause, inventory
from analyze_recorded_session import read
from audit_measurements import canonical


class EvidenceTests(unittest.TestCase):
    def test_layout_normalization_preserves_decision_values(self):
        self.assertEqual(
            decision_content('{"finish": true}'),
            decision_content('```json\n{ "finish" : true }\n```'),
        )
        for other in ['{"finish": false}', '{"finish": true, "tool_call": null}']:
            self.assertNotEqual(decision_content('{"finish": true}'), decision_content(other))
        self.assertNotEqual(
            canonical(decision_content('{"threshold": 3}')),
            canonical(decision_content('{"threshold": 3.0}')),
        )
        self.assertNotEqual(
            decision_content('{"data_id": "a"}'), decision_content('{"data_id": "b"}')
        )
        self.assertNotEqual(
            decision_content('{"metrics": ["a", "b"]}'), decision_content('{"metrics": ["b", "a"]}')
        )

    def test_invalid_json_is_preserved(self):
        self.assertEqual(decision_content("not json"), {"kind": "invalid_json", "raw": "not json"})

    def test_projection_difference_keeps_numeric_and_missing_fields(self):
        ref = [{"tool_name": "detect_anomalies", "arguments": {"threshold": 3.0, "method": None}}]
        actual = [{"tool_name": "detect_anomalies", "arguments": {"threshold": 3}}]
        result = differences(ref, actual)
        self.assertEqual([r["field"] for r in result], ["method", "threshold"])
        self.assertFalse(result[0]["actual_present"])
        with self.assertRaises(ValueError):
            differences(ref, [])

    def test_e3_requires_diagnostic_and_execution_evidence(self):
        log = {
            "tool_calls": [
                {"tool_name": "get_sales_data", "status": "executed"},
                {"tool_name": "format_report", "status": "executed"},
                {"tool_name": "aggregate_data", "status": "rejected"},
            ],
            "validation_events": [
                {
                    "tool_call_index": 2,
                    "message": "Planner selected unnecessary tool after completion: aggregate_data",
                }
            ],
        }
        self.assertEqual(e3_cause(log, 2)["cause"], "additional_tool_after_oracle_completion")
        log["tool_calls"][-1]["status"] = "executed"
        self.assertEqual(e3_cause(log, 2)["cause"], "unclassified_e3")
        log["validation_events"][0]["message"] = "Dependency mapping failed"
        self.assertEqual(e3_cause(log, 2)["cause"], "unclassified_e3")

    def test_inventory_and_repetition_guards(self):
        session = Path("evaluation/experiments/2026-10-09_matched")
        analysis = read(Path("evaluation/experiments/2026-10-09_matched_analysis.json"))
        cases = inventory(analysis, session)
        self.assertEqual(len(cases), 21)
        self.assertEqual(sum(c["variant"] == "V2" for c in cases), 18)
        self.assertEqual(
            {c["case_id"] for c in cases if c["variant"] == "V3"}, {"V3_R005", "V3_R029"}
        )
        self.assertEqual({c["case_id"] for c in cases if c["variant"] == "V4"}, {"V4_R023"})
        broken = copy.deepcopy(analysis)
        next(r for r in broken["runs"] if r["variant"] == "V2" and r["request_id"] == "R001")[
            "legacy_signature"
        ] = "changed"
        with self.assertRaisesRegex(ValueError, "More than one"):
            inventory(broken, session)
        broken = copy.deepcopy(analysis)
        next(r for r in broken["runs"] if r["variant"] == "V2" and r["request_id"] == "R001")[
            "status"
        ] = "rejected"
        with self.assertRaisesRegex(ValueError, "Mixed acceptance"):
            inventory(broken, session)


if __name__ == "__main__":
    unittest.main()
