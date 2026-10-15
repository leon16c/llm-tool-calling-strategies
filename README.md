# Backend Tool Calling Strategies

Research artifacts for the bachelor thesis *Integration of Large Language Models
into Backend Systems: A Comparative Study of Tool Calling Strategies*.
Code, request definitions, recorded logs and analyses accompany the results in
[RESULTS.md](RESULTS.md).

## Run and verify

Requirements: Python 3.12+, Git and `uv`. Run from the repository root:

```powershell
uv sync --project prototype --frozen
uv run --project prototype --frozen toolcall-eval R002 --variant deterministic_workflow --output-dir local-output/reference
uv run --project prototype --frozen python evaluation/scripts/verify_supplementary_release.py
uv run --project prototype --frozen python -m pytest prototype/tests evaluation/scripts
```

These commands run a deterministic example, verify the saved results and run the
tests. Verification extracts the matched-session and supplementary archives before the saved-data
tests need it. Recorder tests require a Git commit, available in a normal clone.
Verification and tests require no model service. The verification report is
written to `local-output/public-verification.json` and `local-output/supplementary-verification.json`.

For a new model run, install `qwen2.5:7b-instruct` in a local Ollama service and
select `--parameter-mapper ollama`; otherwise the CLI uses an oracle control
planner. See `toolcall-eval --help` for variants and options. The default endpoint
is `http://localhost:11434`; `TOOLCALL_EVAL_LLM_BASE_URL` and
`TOOLCALL_EVAL_LLM_MODEL` override endpoint and model. New runs use the current
installation. Keep their outputs in `local-output/`.

## Files

| Path | Contents |
| --- | --- |
| `prototype/` | Implementation, JSON schemas, tests, locked dependencies and historical logs |
| `evaluation/requests/` | Thirty requests and their reference sequences/arguments |
| `evaluation/experiments/` | Matched and supplementary measurement ZIPs, saved analyses and hardware inventory |
| `evaluation/scripts/` | Recording, offline analyses, public verification and analysis tests |
| `evaluation/*.json` | Final results and semantic case assessments |
| `evaluation/audits/` | Historical numerical audit, case evidence and original provenance manifest |

`RELEASE_MANIFEST.json` records export checksums. The verifier checks those hashes,
the archived session and the saved numerical results and case assessments. The
measurement archives are extracted to ignored directories under
`evaluation/experiments/`. The supplementary protocol is extracted to
`evaluation/supplementary_protocol.md`. The primary session's provenance retains
its recorded references and Git identifiers. Supplementary export metadata
records source-copy normalization separately from measurement records. Cite this repository's commit ID as the
public artifact version.

## Supplementary experiments (2026-10-20)

The retained 60-run paired V3 prompt comparison and 300-run V2/V4 sampling comparison use Ollama 0.40.2 with the original model digest. Offline gate replay and post-report diagnostics are reported separately in [RESULTS.md](RESULTS.md). The verifier reproduces every new runner outcome from its exact recorded model generations without calling a model. The protocol was specified after inspecting the baseline results and before the new generations. It is extracted to `evaluation/supplementary_protocol.md`; source and protocol snapshots are included in each session. The prompt comparison keeps the original workflow gate, and the sampling comparison keeps original prompts and gates. The primary matched series and the supplementary experiments are reported separately; the supplementary runs have no author semantic adjudication.
