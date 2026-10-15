"""Verify the public supplementary archive and reproduce every new reported outcome."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from analyze_recorded_session import read, require
from analyze_supplementary_experiments import analyze
from verify_recorded_replays import verify

ROOT = Path(__file__).resolve().parents[2]


def main():
    subprocess.run(
        [sys.executable, "evaluation/scripts/verify_public_release.py"],
        cwd=ROOT,
        check=True,
    )
    manifest = read(ROOT / "RELEASE_MANIFEST.json")
    info = manifest["supplementary_archive"]
    with zipfile.ZipFile(ROOT / info["path"]) as archive:
        entries = archive.infolist()
        require(
            len(entries) == info["file_count"] == len({e.filename for e in entries}),
            "Invalid supplementary archive count",
        )
        for entry in entries:
            relative = PurePosixPath(entry.filename)
            require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and "\\" not in entry.filename
                and ":" not in entry.filename,
                "Unsafe archive member",
            )
            allowed = (
                entry.filename == "evaluation/supplementary_protocol.md"
                or any(
                    entry.filename.startswith(session + "/")
                    for session in info["sessions"]
                )
            )
            require(allowed, "Unexpected archive destination")
            target = ROOT.joinpath(*relative.parts)
            require(
                target.resolve().is_relative_to(ROOT.resolve()), "Archive path escaped"
            )
            content = archive.read(entry)
            if target.exists():
                require(target.read_bytes() == content, "Extracted evidence changed")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(content)
        names = {e.filename for e in entries}
        for session in info["sessions"]:
            actual = {
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / session).rglob("*")
                if p.is_file()
            }
            require(
                actual == {n for n in names if n.startswith(session + "/")},
                "Unexpected session evidence",
            )
    results = {}
    for session in info["sessions"]:
        path = Path(session)
        calculated = analyze(path)
        require(
            calculated == read(ROOT / (session + "_analysis.json")),
            "Supplementary analysis differs",
        )
        results[path.name] = verify(path)
    for name, script in [
        ("matched_gate_sensitivity", "analyze_gate_sensitivity.py"),
        ("matched_v4_post_report", "analyze_v4_post_report.py"),
        ("matched_v3_report_omission", "analyze_v3_report_omission.py"),
    ]:
        path = ROOT / f"evaluation/{name}.json"
        expected = read(path)
        subprocess.run(
            [sys.executable, f"evaluation/scripts/{script}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        require(read(path) == expected, f"Offline diagnostic differs: {name}")
    result = {
        "all_checks_passed": True,
        "archive_files": len(entries),
        "supplementary_runs": sum(r["replayed_runs"] for r in results.values()),
        "supplementary_model_calls": sum(r["model_calls"] for r in results.values()),
        "all_runner_outcomes_reproduced": True,
        "sessions": results,
    }
    output = ROOT / "local-output/supplementary-verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
