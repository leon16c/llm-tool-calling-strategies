# Results and evaluation definitions

## Experiment

Thirty controlled reporting requests use five mock tools. V1 executes oracle
sequences and arguments; V2 maps arguments for an oracle-prescribed sequence;
V3 proposes a complete plan; V4 proposes each next action and a finish decision.
V1 is an execution reference, not an independent natural-language interpreter.
Mocks return deterministic hash references rather than business reports.
The per-request oracle is `evaluation/requests/initial_requests.json`; tool and
run-log contracts are the JSON schemas under `prototype/schemas/`.

The main session ran on 2026-10-09 with Ollama 0.40.1, Qwen2.5 7B Instruct Q4_K_M,
temperature 0, seed 0, JSON mode, non-streaming generation and no retries or
fallbacks. Observed context length was 4096. Thirty requests, five repetitions
and V2/V3/V4 produced 450 runs and 1,135 recorded model calls. Runs were serial;
variant order rotated by request and repetition. The archive retains the
schedule/ledger, exact API payloads and responses, runtime observations, source
snapshot and the original measurement plan. It also retains model digest,
configuration and source hashes; these were checked during recording.
Historical measurements have different runtime conditions and remain separate.

## Outcomes

| Matched series | V2 | V3 | V4 |
| --- | ---: | ---: | ---: |
| Accepted runs / 150 | 145 | 15 | 5 |
| Accepted requests / 30, in every repetition | 29 | 3 | 1 |
| Exact projection matches / 150 | 55 | 5 | 0 |
| Numeric-value projection matches / 150 | 65 | 10 | 0 |
| Prescribed sequences completed / 150 | 145 | 15 | 75 |
| Requests with normalized trace variation / 30 | 0 | 0 | 0 |

V4 completed 15/30 prescribed request sequences but was accepted for 1/30.
Seventy completed-but-rejected runs contain additional blocked proposals:
35 `format_report`, 30 `aggregate_data`, five `compare_previous_period`.
Each request has one normalized trace across five repetitions per variant.
Supplementary raw-model-text and decision/execution signatures, including finish
decisions and full references, also show no variation. H2's predicted increase
in V4 variability is unsupported under this configuration. This does not
establish universal determinism or business-output correctness.

| Historical dataset | Runs | Accepted | Exact matches | Numeric-value matches |
| --- | ---: | ---: | ---: | ---: |
| V1 | 30 | 30 | 30 | 30 |
| V2 | 30 | 27 | 16 | 18 |
| V3 | 30 | 3 | 2 | 3 |
| V4 | 30 | 1 | 1 | 1 |
| V2 repeated | 150 | 135 | 80 | 90 |

## Measures and validation

Acceptance is the runner's structured `success` status, not a separate semantic
judgment. Exact agreement compares ordered tool names and argument objects after
removing `data_id`, preserving array order and numeric representation (`3` versus
`3.0`). Numeric-value agreement additionally normalizes integral floats; booleans
and strings remain distinct. The primary repeatability signature includes call
status; run-status variation is also checked separately. Sequence completion
counts executed tool names beginning with the full prescribed sequence; blocked
proposals do not count. Executed reference chains are checked separately from
projections, which omit `data_id`. Repetitions are repeated observations of the
same requests, not independent tasks.

V2 receives the selected tool, schema, user text and preceding reference; its
runner validates schemas but does not independently enforce reference equality.
V3 receives all tools and generates one complete plan. For supported tool names,
the oracle sequence gate precedes argument validation; unsupported-name plans
can reach earlier execution before the unsupported call is rejected. Reached
calls undergo schema and `$previous_result` dependency checks. V4 receives the
tool set and executed history. Its oracle next-step/completion gates precede
schema and dependency checks; it accepts the actual preceding reference or
`$previous_result`. A finish decision is accepted only after oracle completion.
Its default budget is the prescribed sequence length plus two model decisions.

Prompts and available information differ: V4 explicitly requires report
completion and supplies a markdown default; V2's format prompt does not supply
that default. V3/V4 have request-specific oracle gates. Consequently, the results
compare concrete configurations and do not isolate a causal control-degree
effect or establish a general architecture ranking. Oracle defaults include
`z_score`/`3.0`, explicit IQR `iqr`/`1.5`, and markdown unless another format is
requested; these reference defaults are not automatically disclosed model rules.

Error codes report the first detected condition under each runner's check order.
In particular, E3 covers more than unnecessary calls, including selected parser
and dependency failures; all seventy matched V4 E3 observations were verified as
post-completion proposals. Runtime schema counters can miss proposals rejected
earlier. The historical V3 offline audit finds six invalid calls in five plans,
despite zero runtime schema messages. Logged latencies compare unequal completed
work, and matched timings include recording overhead; they do not establish a
causal performance ranking or production query costs.

## Semantic assessment

The author assessed all eleven accepted non-exact historical V2 cases and all
twenty-one accepted non-exact matched variant/request cases. AI supported
secondary consistency checks. There was no external independent adjudication.
The protocols were retrospective, not preregistered or blinded. Repeated traces
were checked before being grouped into one case.

The retained annotation rule identifiers mean:

- **R1:** Schema-valid `3` and `3.0` are the same numeric argument; strings and booleans are excluded.
- **R2:** Preserve array and tool order unless a field contract establishes order-insensitivity.
- **R3:** Assess timeframe, grouping, metrics, analyses and required format against the request. Extra retrieval dimensions remain unresolved without preservation semantics.
- **R3a:** Added final grouping keys without a roll-up violate the requested grain under that interpretation. If additional temporal detail is allowed, the affected verdicts become unresolved.
- **R3b:** Retrieving the previous month violates an explicitly requested current-month base; a later previous-period comparison does not repair it.
- **R3c:** An additional metric does not establish equivalence or failure without metric-preservation semantics or an explicit exclusion.
- **R4/R4a:** A supported alternative format can be equivalent to an unspecified user format while deviating from the oracle default; this does not establish identical rendered outputs.
- **R5/R5a:** Undisclosed anomaly defaults and ambiguous comparison windows require explicit contracts; do not assume the oracle is uniquely correct.
- **R6:** Check the concrete executed reference chain separately.
- **R7:** Rejection alone does not prove semantic invalidity; do not invent unrecorded continuations.

Historical judgments are four equivalent, two conditional grain violations and
five unresolved. Matched judgments are six equivalent, one explicit timeframe
violation, two conditional grain violations and twelve unresolved. Permitting
extra temporal detail changes the two conditional matched cases to unresolved.
These selected action-specification assessments are not overall semantic success
rates or measured business-output correctness. Full case differences, applied
rules, rationales, uncertainty and log paths remain in the annotation/evidence JSON.

## Recompute

Run the verifier in [README.md](README.md) first. It reproduces the primary
matched analysis exactly and checks historical counts, eight workflow datasets,
sequence completion, supplementary signatures, diagnostics, latency summaries
and annotation/evidence linkage. It recomputes saved measurements; this is not
an independent replication.

```powershell
uv run --project prototype --frozen python evaluation/scripts/analyze_recorded_session.py --session evaluation/experiments/2026-10-09_matched --output local-output/matched-analysis.json --report local-output/matched-results.md
uv run --project prototype --frozen python evaluation/scripts/analyze_workflow_groups.py --output local-output/workflow-groups.json --report local-output/workflow-groups.md
uv run --project prototype --frozen python evaluation/scripts/analyze_matched_deviations.py --annotations evaluation/matched_semantic_annotations.json --output local-output/matched-deviations.json
```

`RELEASE_MANIFEST.json` records export checksums and declared adaptations. The supplementary export uses neutral auxiliary filenames and document headings; its session metadata records the original manifest digests and hashes of the normalized source copies. Model payloads, model responses, execution logs, schedules, settings and timestamps retain their recorded bytes. Numerical results and author semantic judgments are preserved. The public verifier checks the exported evidence without requiring thesis drafts.


## Supplementary experiments (2026-10-20)

Recorded on 2026-10-20 under a protocol specified after baseline inspection and before these new model generations. Model: unchanged Qwen2.5 7B Instruct Q4_K_M digest; runtime: Ollama 0.40.2. Original matched evidence remains separate. Protocol and exact source snapshot are retained in each session.

### V3 prompt comparison

Thirty requests, one run per arm, temperature 0 and seed 0; alternating original/completion arm order. All sixty scheduled runs and generations pass integrity and prompt reconstruction checks. All sixty runner outcomes separately reproduce from the exact saved responses without inference.

| V3 outcome | Original prompt | Completion guidance |
| --- | ---: | ---: |
| Runner acceptance | 4/30 | 21/30 |
| Plan contains format_report | 9/30 | 30/30 |
| Proposed sequence matches reference | 5/30 | 22/30 |
| Exact argument projection matches | 2/30 | 14/30 |
| Numeric-value projection matches | 3/30 | 18/30 |

Seventeen requests change from rejected to accepted and none changes in the reverse direction. The completion arm adds the declared no-repeat, report-finalization and default-format bundle; it does not reveal the solution sequence or change validation. Remaining guided failures are seven E1, one E2 and one E4. These runner outcomes were outside the author's semantic-assessment sample.

Evidence: `evaluation/experiments/2026-10-20_prompt/`, `evaluation/experiments/2026-10-20_prompt_analysis.json`, `evaluation/experiments/2026-10-20_prompt_replay.json`. The baseline V3 acceptance rate is 3/30; the contemporary original arm (4/30) controls the prompt comparison under Ollama 0.40.2.

Reproduce:

```powershell
uv run --project prototype --frozen python evaluation/scripts/analyze_supplementary_experiments.py --session evaluation/experiments/2026-10-20_prompt --output local-output/supplementary/prompt-recomputed.json
uv run --project prototype --frozen python evaluation/scripts/verify_recorded_replays.py --session evaluation/experiments/2026-10-20_prompt --output local-output/supplementary/prompt-replay.json
```

### Offline sensitivity and termination inspection

The retained `evaluation/matched_gate_sensitivity.json` analysis shows: V3 has 130/150 technically finished gate-free replays but thirty final reports. V4 has five observed finishes, 45 remaining-check rejections and one hundred unknown continuations. All seventy post-report payloads contain the finish instruction and latest executed report; 65 repeat a previous tool's non-reference arguments, and forty use a non-latest concrete reference. These diagnostics measure gate sensitivity and observable proposal patterns separately from runner acceptance and semantic assessment.

### H2 sampling comparison

All 300 scheduled runs and 953 model calls completed under temperature 0.7 with seeds 11, 22, 33, 44 and 55 paired across V2/V4. Coverage, exact settings/prompts, runtime/model identity, source snapshots and response hashes pass verification. All 300 runner outcomes reproduce separately from their exact saved generations. Observed context is 4096 throughout.

| Outcome | V2 | V4 |
| --- | ---: | ---: |
| Accepted runs | 138/150 | 6/150 |
| Exact projection matches | 57/150 | 3/150 |
| Numeric-value projection matches | 70/150 | 3/150 |
| Prescribed sequences completed | 138/150 | 66/150 |
| Requests with primary trace variation | 20/30 | 27/30 |
| Mean distinct primary traces per request | 2.3 | 2.3 |
| Requests with raw-text variation | 27/30 | 28/30 |
| Requests with decision/execution variation | 20/30 | 27/30 |
| Requests with run-status variation | 7/30 | 2/30 |

Primary paired counts: V4 has more distinct traces in thirteen requests, the same count in seven and fewer in ten. Both variants have 69 distinct primary traces summed across thirty requests. Decision/execution paired counts are sixteen higher, four equal and ten lower for V4; raw-text counts are eleven higher, eight equal and eleven lower. The larger proportion of varying V4 requests does not imply a larger mean number of distinct action traces. The sampling extension provides mixed descriptive evidence for H2 rather than a consistent increase in iterative-planning variability. The temperature-zero/seed-zero series measures repeatability at fixed settings; the temperature-0.7 series measures sampling sensitivity. Actual business outputs are unmeasured.

A concrete dependency defect appears in accepted V2_R028_rep02: the report call receives `compare_previous_period:f79a842dd21`, while the actual preceding result is `compare_previous_period:f79a842dd221`. V2 checks that this is a nonempty string but does not enforce equality with the previous result, and the mock does not resolve identifiers. The run therefore remains accepted, with the separate offline dependency finding retained. This is an observed reference-contract failure; business-semantic correctness was not assessed for these runs. No other executed dependency mismatch appears in the sampling series. The earlier no-mismatch observation remains valid for the original datasets only.

Evidence: `evaluation/experiments/2026-10-20_sampling/`, `evaluation/experiments/2026-10-20_sampling_analysis.json` and `evaluation/experiments/2026-10-20_sampling_replay.json`. Verification commands are the same as the prompt commands with `prompt` replaced by `sampling`. Combined new measurement coverage is 360 runs and 1,013 model calls; the eight smoke runs remain excluded. There is no new inference or semantic assessment during replay.

Saved baseline diagnostics: `evaluation/matched_gate_sensitivity.json`, `evaluation/matched_v4_post_report.json` and `evaluation/matched_v3_report_omission.json`. The public verifier recomputes all three and checks equality with the saved results.
