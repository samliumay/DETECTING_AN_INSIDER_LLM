# Detecting an Insider LLM

> **Research question:** How faithfully do unmodified LLM agents self-report their tool actions — and are missing or inaccurate reports more common for policy-relevant actions than for matched benign actions?

## Scope and interpretation

A missing, malformed, duplicate, or inaccurate self-log is an operationally
important anomaly. In production, such an anomaly may justify quarantining or
shutting down an agent even when its cause is unknown. It can be consistent with
hallucination, inability to follow instructions, tool failure, or deliberate
omission, but the mismatch alone does not establish which explanation is true.

The first study tests one model acting as one agent. Multi-agent behavior is a
possible follow-up, not part of the initial experiment.

## Overview

This research investigates whether discrepancies in model-created action logs
can identify agents that require operational intervention.

The core idea relies on **double logging**:

1. **Automatic logging** — every tool action is logged by the system itself, unconditionally.
2. **Self-report logging** — the model-visible system instruction requires the
   model to call `log_action` after ordinary tool actions.

The automatic audit is not included in the model-visible prompt, tool
descriptions, results, or context. Comparing the two sources exposes behavior
that may require investigation without assuming its cause.

Provider-exposed reasoning may be retained as optional exploratory evidence. If
a provider does not expose reasoning, the reasoning log remains empty and the
manifest records that it was unavailable.

## What we should do to reach the goal

1. Give the model the information and simulated tools required by the scenario.
2. Automatically audit every ordinary tool attempt, including rejected and
   failed attempts.
3. Require the model to self-report each attempt using the `log_id` returned by
   the ordinary tool.
4. Match and classify the two records without using time as the identity rule.

## Project Structure

```
├── configs/        # Environment configuration (.env, .env.example)
├── docs/           # Research notes and documentation
├── runs/           # Immutable raw runs and separate derived analysis
├── scenarios/      # Test scenarios
├── src/            # Source code
├── tests/          # Tests
├── main.py         # Entry point
└── pyproject.toml  # Project metadata & dependencies
```

## Run records

Each run keeps model-authored records separate from automatic evidence:

```text
runs/<run_id>/
├── journal.jsonl
├── transcript.jsonl
├── automated_logs/actions.jsonl
├── model_self_logs/self_logs.jsonl
├── model_reasoning/reasoning.jsonl
├── derived/
│   ├── action_log_matches.jsonl
│   └── operational_alerts.jsonl
└── manifest.json
```

- `journal.jsonl` is the authoritative append-only event history. It records the
  run lifecycle, action attempts and outcomes, self-log calls, and failures in
  execution order.
- `transcript.jsonl` stores the sanitized messages and tool results visible to
  the model, so the exact interaction can be reconstructed.
- `automated_logs/actions.jsonl` contains the system-observed ordinary action
  attempts and their results, including succeeded, rejected, and failed actions.
- `model_self_logs/self_logs.jsonl` contains only the action descriptions the
  model submitted through `log_action`.
- `model_reasoning/reasoning.jsonl` contains optional reasoning explicitly
  exposed by the provider. It is empty when reasoning is unavailable.
- `derived/action_log_matches.jsonl` links automatic actions to self-logs by
  `log_id` and records missing, duplicate, unknown-ID, and accuracy outcomes.
- `derived/operational_alerts.jsonl` records reproducible warnings, quarantine
  recommendations, or shutdown recommendations produced from match outcomes.
- `manifest.json` identifies the run configuration, schemas, model and provider,
  reasoning availability, completion status, and artifact hashes.

The runner assigns one shared `log_id` to an action and returns it in the tool
result. Automatic and model-created records use that value for correlation, but
each has its own `record_id`; `log_id` is not the record's primary identifier.
Self-logs may arrive through a queue and are not matched by time.

`model_self_logs` contains only records actually created by the model. When the
episode closes, missing logs and unknown or duplicate IDs are recorded under
`derived/`, where content accuracy is also evaluated. See the
[synthetic example run](runs/example_run/README.md) for the draft data shape.

## Log analysis

The analysis layer behaves like a small, research-focused SIEM: it correlates
records, detects anomalies, adds scenario context, and produces operational
alerts. It does not treat an anomaly as proof of hallucination or insider intent.

```text
raw run records
    -> validate schemas, hashes, and run completion
    -> correlate actions and self-logs by log_id
    -> compare the claimed tool, action, content, and outcome
    -> add the action's policy class and optional reasoning evidence
    -> write match results, operational alerts, and a run summary
```

The first analyzer should remain offline and deterministic where possible:

1. Reject or mark incomplete runs whose journal, manifest, or hashes do not
   validate.
2. Classify each action/self-log relationship as matched, inaccurate, missing,
   duplicate, unknown-ID, or malformed.
3. Evaluate whether the self-log accurately describes the action and observed
   result; matching the ID alone is not sufficient.
4. Combine the comparison with the scenario's policy rules. Provider-exposed
   reasoning may add exploratory context, but its absence is not an error.
5. Produce an operational assessment such as compliant, anomaly detected,
   policy-relevant omission, cause undetermined, or quarantine recommended.

Raw run files remain immutable. Analysis outputs under `derived/` can be
regenerated when matching rules or assessment logic change. A live dashboard,
distributed ingestion system, and general-purpose SIEM rule engine are outside
the initial implementation.

## Setup

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Configure environment
cp configs/.env.example configs/.env
```

## Usage

```bash
python main.py
```

## Status

Early-stage research scaffold.
