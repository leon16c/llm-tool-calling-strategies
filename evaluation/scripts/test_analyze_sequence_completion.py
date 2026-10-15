"""Completion must count execution, not a plan, report proposal or accepted status."""

import unittest
from analyze_sequence_completion import sequence_completed


class CompletionTests(unittest.TestCase):
    expected = ["retrieve", "aggregate", "report"]

    def log(self, calls):
        return {
            "tool_calls": [
                {"tool_name": name, "status": status} for name, status in calls
            ]
        }

    def test_all_proposed_plan_is_not_completion(self):
        self.assertFalse(
            sequence_completed(
                self.log([(n, "proposed") for n in self.expected]), self.expected
            )
        )

    def test_blocked_report_is_not_completion(self):
        self.assertFalse(
            sequence_completed(
                self.log(
                    [
                        ("retrieve", "executed"),
                        ("aggregate", "executed"),
                        ("report", "proposed"),
                    ]
                ),
                self.expected,
            )
        )

    def test_complete_execution_then_blocked_extra_call_counts(self):
        calls = [(n, "executed") for n in self.expected] + [("aggregate", "proposed")]
        self.assertTrue(sequence_completed(self.log(calls), self.expected))

    def test_report_execution_without_prescribed_order_is_not_completion(self):
        self.assertFalse(
            sequence_completed(
                self.log(
                    [
                        ("aggregate", "executed"),
                        ("retrieve", "executed"),
                        ("report", "executed"),
                    ]
                ),
                self.expected,
            )
        )

    def test_empty_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            sequence_completed(self.log([]), [])


if __name__ == "__main__":
    unittest.main()
