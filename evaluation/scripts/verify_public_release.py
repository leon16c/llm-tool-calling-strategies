"""Verify the selected public evidence without private drafts or model calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from analyze_matched_deviations import differences, inventory, supplementary
from analyze_recorded_session import analyze, read, require, sha
from analyze_sequence_completion import sequence_completed
from analyze_workflow_groups import analyze as workflow_analysis
from audit_measurements import ROOT, audit, canonical, dependency_findings, projection

SESSION = Path("evaluation/experiments/2026-10-09_matched")


def verify_inventory() -> dict:
    manifest = read(ROOT / "RELEASE_MANIFEST.json")
    for name, expected in manifest["sha256"].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "Invalid manifest path")
        require(sha(ROOT / name) == expected, f"Export checksum mismatch: {name}")
    archive = ROOT / manifest["matched_archive"]["path"]
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        require(len(entries) == manifest["matched_archive"]["file_count"], "Archive count mismatch")
        names = {entry.filename for entry in entries}
        require(len(names) == len(entries), "Duplicate archive path")
        destination = ROOT / SESSION
        for entry in entries:
            relative = PurePosixPath(entry.filename)
            require(not relative.is_absolute() and ".." not in relative.parts
                    and "\\" not in entry.filename and ":" not in entry.filename,
                    "Unsafe archive member")
            target = destination.joinpath(*relative.parts)
            require(target.resolve().is_relative_to(destination.resolve()), "Archive path escaped")
            content = source.read(entry)
            if target.exists():
                require(target.read_bytes() == content, f"Extracted evidence changed: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(content)
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*")
                  if p.is_file()}
        require(actual == names, "Unexpected extracted evidence files")
    baseline = read(ROOT / "evaluation/audits/2026-10-09/baseline_manifest.json")
    checked = 0
    for name, expected in baseline["sha256"].items():
        if name.startswith(("prototype/", "evaluation/requests/")) and (ROOT / name).is_file():
            require(sha(ROOT / name) == expected, f"Historical baseline mismatch: {name}")
            checked += 1
    require(checked > 300, "Historical experimental evidence missing")
    return {"release_files": len(manifest["sha256"]), "archive_files": len(entries),
            "historical_baseline_hashes_checked": checked}


def verify_historical_semantics() -> dict:
    saved = read(ROOT / "evaluation/audits/2026-10-09/fixed_sequence_semantic_review.json")
    annotations = read(ROOT / "evaluation/fixed_sequence_semantic_annotations.json")
    requests = {r["id"]: r for r in read(ROOT / "evaluation/requests/initial_requests.json")["requests"]}
    observed = set()
    records = {c["request_id"]: c for c in saved["cases"]}
    require(len(records) == len(saved["cases"]), "Duplicate historical semantic case")
    for path in sorted((ROOT / "prototype/logs/fixed_sequence_llm_parameters_ollama").glob("R*.json")):
        log = read(path)
        rid = log["request_id"]
        reference = ROOT / f"prototype/logs/fixed_sequence_llm_parameters/{rid}.json"
        actual, expected = projection(log), projection(read(reference))
        if log["status"] != "success" or canonical(actual) == canonical(expected):
            continue
        observed.add(rid)
        require(not dependency_findings(log), f"Historical dependency failure: {rid}")
        repeats = sorted((ROOT / "prototype/logs/fixed_sequence_llm_parameters_ollama_repeated").glob(f"{rid}_rep*.json"))
        require(len(repeats) == 5, f"Incomplete historical repeats: {rid}")
        for repeated_path in repeats:
            repeated = read(repeated_path)
            require(repeated["status"] == log["status"] and not dependency_findings(repeated)
                    and canonical(projection(repeated, statuses=True)) == canonical(projection(log, statuses=True)),
                    f"Historical reviewed trace differs: {rid}")
        case = records[rid]
        require(case["review"] == annotations["cases"][rid], f"Historical annotation mismatch: {rid}")
        require(case["differences"] == differences(expected, actual), f"Historical differences mismatch: {rid}")
        require(case["user_request"] == requests[rid]["user_request"]
                and case["log_path"] == path.relative_to(ROOT).as_posix()
                and case["reference_path"] == reference.relative_to(ROOT).as_posix()
                and case["matching_repetitions"] == 5,
                f"Historical evidence identity mismatch: {rid}")
    require(observed == set(records) == set(annotations["cases"]), "Historical annotation coverage mismatch")
    counts = dict(Counter(c["review"]["verdict"] for c in saved["cases"]))
    require(counts == saved["verdict_counts"] and saved["reviewed_cases"] == len(observed), "Historical verdict counts mismatch")
    return {"cases": len(observed), "verdicts": counts}


def verify_completion(analysis: dict) -> dict:
    saved = read(ROOT / "evaluation/matched_sequence_completion.json")
    requests = {r["id"]: r for r in read(ROOT / SESSION / "artifacts/evaluation/requests/initial_requests.json")["requests"]}
    rows = []
    for row in analysis["runs"]:
        log = read(ROOT / SESSION / "runs" / row["task_id"] / "run_log.json")
        complete = sequence_completed(log, requests[row["request_id"]]["expected_tool_sequence"])
        rows.append({**{k: row[k] for k in ("task_id", "variant", "request_id", "repetition_index", "status")},
                     "prescribed_sequence_completed": complete})
    require(rows == saved["runs"], "Sequence completion rows differ")
    summaries = {}
    for variant in ("V2", "V3", "V4"):
        selected = [r for r in rows if r["variant"] == variant]
        groups = {rid: [r for r in selected if r["request_id"] == rid] for rid in requests}
        summaries[variant] = {
            "runs": len(selected), "requests": len(groups),
            "accepted_runs": sum(r["status"] == "success" for r in selected),
            "completed_sequence_runs": sum(r["prescribed_sequence_completed"] for r in selected),
            "completed_sequence_requests": sum(g[0]["prescribed_sequence_completed"] for g in groups.values()),
            "completed_but_rejected_runs": sum(r["prescribed_sequence_completed"] and r["status"] == "rejected" for r in selected),
            "completed_but_rejected_request_ids": sorted(rid for rid, g in groups.items() if g[0]["prescribed_sequence_completed"] and g[0]["status"] == "rejected"),
        }
    require(summaries == saved["variants"], "Sequence completion summaries differ")
    return summaries


def main() -> None:
    os.chdir(ROOT)
    result = {"inventory": verify_inventory()}
    historical = audit()
    require(historical == read(ROOT / "evaluation/audits/2026-10-09/measurement_audit.json")["sets"], "Historical numerical audit differs")
    result["historical_audit"] = {k: v["summary"] for k, v in historical.items()}
    analysis = analyze(SESSION)
    require(analysis == read(ROOT / "evaluation/experiments/2026-10-09_matched_analysis.json"), "Primary matched analysis differs")
    result["matched"] = {"runs": analysis["verified_runs"], "model_calls": analysis["model_calls"], "reproduced_exactly": True}
    workflow = workflow_analysis()
    require(workflow == read(ROOT / "evaluation/workflow_group_analysis.json"), "Workflow group analysis differs")
    result["workflow_datasets"] = len(workflow["datasets"])
    result["completion"] = verify_completion(analysis)
    saved = read(ROOT / "evaluation/matched_deviation_analysis.json")
    annotations = read(ROOT / "evaluation/matched_semantic_annotations.json")
    cases = inventory(analysis, SESSION)
    require(set(annotations["cases"]) == {c["case_id"] for c in cases}, "Matched annotation coverage mismatch")
    for case in cases:
        case["review"] = annotations["cases"][case["case_id"]]
    require(cases == saved["cases"], "Matched semantic case evidence differs")
    supplement = supplementary(analysis, SESSION)
    require(supplement == saved["supplementary"], "Supplementary signatures or E3 diagnostics differ")
    verdicts = {v: dict(Counter(c["review"]["verdict"] for c in cases if c["variant"] == v)) for v in ("V2", "V3", "V4")}
    require(verdicts == saved["verdict_counts"], "Matched semantic verdict counts differ")
    result["matched_semantics"] = {"cases": len(cases), "verdicts": verdicts}
    result["historical_semantics"] = verify_historical_semantics()
    latency = json.loads(subprocess.check_output([sys.executable, "evaluation/scripts/summarize_historical_latency.py"], cwd=ROOT))
    require(latency == read(ROOT / "evaluation/historical_latency_by_outcome.json"), "Historical latencies differ")
    result["historical_latency_reproduced"] = True
    result["all_checks_passed"] = True
    output = ROOT / "local-output/public-verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": True, "matched_runs": analysis["verified_runs"],
                      "model_calls": analysis["model_calls"], "workflow_datasets": len(workflow["datasets"]),
                      "historical_semantic_cases": result["historical_semantics"]["cases"],
                      "matched_semantic_cases": len(cases), "report": output.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
