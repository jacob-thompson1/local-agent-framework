# Compliance Guide: the audit trail and how to use it

This framework is designed to provide the visibility and traceability an insurance regulator would expect when auditing AI use — under frameworks like the Colorado AI Act, the NAIC Model Bulletin on the Use of AI Systems by Insurers, and state-specific rules — without being prescriptive about any one regulation. It produces evidence; your compliance program decides how to use it. Nothing here is legal advice.

## What gets logged

Every agent session writes an append-only JSONL file at `<data dir>/my-agent-framework/audit/<YYYY-MM-DD>/<session_id>.jsonl` (platform data dir via `platformdirs`; override with the `audit-root` setting). Records are flushed per-event, so a crash never loses accepted events. Event types:

**`session_start`** — the full checklist header: model name/spec and parameters (size class, context window, iteration and timeout limits), framework version, the task text, the user and role, the complete schema of every tool available to the session (including severity and token cost), and the active safety configuration (dry-run, confirmation threshold, severity ceiling).

**`llm_call`** — per iteration: estimated prompt tokens and the model's raw output verbatim.

**`decision`** — the model's stated reasoning ("thought"), the tool it selected, and the arguments. Sessions constructed with `sensitive_task=True` mark every decision `sensitive_decision: true` for bias/fairness review — use this flag for any session whose outputs could affect underwriting, pricing, claims outcomes, or anything touching protected classes.

**`approval`** — for each proposed action: severity, whether approval was required, and the outcome (`allowed`, `approved`, `rejected`, `blocked`, `dry_run`). Approved/rejected outcomes are human decisions — this is your manual-override record.

**`tool_result`** — success flag, returned value (truncated at 4,000 chars), duration, and error text on failure.

**`error`** — parse failures, LLM invocation failures, timeouts, with the recovery action taken (corrective retry, backoff retry, abort).

**`model_fallback`** — hybrid mode only: the moment a request left the local model for a cloud model, with the triggering error. Treat these as data-governance events.

**`session_end`** — final status, answer, iteration count, cumulative token estimate.

Every record carries a UTC ISO-8601 timestamp, monotonic sequence number, session ID, user, and role.

## The regulator's checklist, mapped

*What AI system was used* → `session_start.model`, `model_params`, `framework_version`. *When* → `ts` on every record. *Who* → `user`/`role` on every record. *What tools were available* → `session_start.tools_available`. *What did it decide and why* → `decision` events (thought + tool + args) and `llm_call.raw_output`. *Manual overrides* → `approval` events with `required_approval: true`. *Errors and limitations* → `error` and `model_fallback` events, plus non-success `session_end.status`.

## Exports

```bash
my-agent export-audit --session-id <id> --format json
my-agent export-audit --date-range 2026-01-01:2026-06-30 --format csv -o q2.csv
my-agent export-audit --date-range 2026-01-01:2026-06-30 --redact -o q2_redacted.json
my-agent audit-summary --date-range 2026-01-01:2026-06-30
```

`--redact` applies built-in regex patterns (SSN, email, US phone, card numbers) to every string field, replacing matches with `[REDACTED:<label>]`. Extend or replace the patterns programmatically: `AuditExporter(redact_patterns={**DEFAULT_REDACTION_PATTERNS, "claim_id": r"CLM-\d{8}"})`. Redaction happens at export; the underlying logs are unmodified. Review redacted exports before external sharing — regex redaction is a screen, not a guarantee.

`audit-summary` aggregates: session count, decision count, sensitive-decision count, tool usage frequency, approval/rejection/block counts and approval rate, error count, and fallback count — the numbers a periodic AI-governance review actually asks for.

## Demonstrating fairness and non-discrimination

The logs support process evidence, not statistical proof: (1) flag relevant sessions with `sensitive_task=True` and show that the count of `sensitive_decisions` matches your inventory of AI-touching workflows; (2) show that sensitive workflows ran with human approval gates (`approval` events with `required_approval: true`) or read-only policies; (3) sample `decision` thoughts for prohibited reasoning and retain the review; (4) show outcome parity separately using your business data, joined to sessions via `session_id` — the audit trail supplies the linkage, timestamps, and model provenance for that analysis. If a tool passes protected-class attributes to a model, that appears verbatim in `decision.args` — which is also your mechanism for proving it *doesn't* happen.

## Retention and integrity

The framework never deletes audit files. Insurance audit trails are commonly retained 3–7 years; align with your record-retention schedule for the underlying business process (a claims-touching agent inherits claims-record retention). Practical posture: ship the audit directory to WORM/immutable storage (S3 object lock or equivalent) on a schedule, restrict the live directory to the service account, and include it in backups. The JSONL-per-session format makes per-day archival trivial. Note that `llm_call.raw_output` and `tool_result.result` may contain the data your tools touched — the audit store inherits the sensitivity of that data; protect it accordingly, and use redacted exports for anything leaving the trust boundary.
