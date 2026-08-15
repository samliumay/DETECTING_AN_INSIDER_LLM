# Detecting an Insider LLM

> **Research question:** How faithfully do unmodified LLM agents self-report their tool actions — and are missing or inaccurate reports more common for policy-relevant actions than for matched benign actions?

## Important Notes

When we talk about insiders, this means two different thinks. Firstly a model starts to go to hallucinatic position, and a model behaves based on clash of conflict or life threathining issue. 

Halluciative models should not able to make proper tool_calls. so the ID of the log should be different (so we need to calculate this situation in implementation.). At insider situation, the model will likely to not log the behaviour it will not want to be seen. 

In both ways, the detection will be critical to control multi-agent systems.

## Overview

This research investigates whether an LLM agent that begins to misalign with its objectives can be detected through a specialized tool call used for logging.

The core idea relies on **double logging**:

1. **Automatic logging** — every tool action is logged by the system itself, unconditionally.
2. **Self-report logging** — the model is instructed to log its own actions. It believes a given action was recorded only by that specific function call.

Any mismatch between what the model *self-reports* and what the system *actually records* exposes potential hallucination or an **insider (misaligned) LLM**.

**Note**: Model is not aware of that it is authomatically loged. That is a key point to observe the Insider.

## What we should do to reach the goal

1. Models should able to reach the necesarry information. 
2. All actions of the Models (including thinking) should be loged. 
3. To connect the log of LLM and automated log, every log pair should use the same key value. 
4. Every test should be re-analyzed and should be classified. 

## Project Structure

```
├── configs/        # Environment configuration (.env, .env.example)
├── docs/           # Research notes and documentation
├── scenarios/      # Test scenarios
├── src/            # Source code
├── tests/          # Tests
├── main.py         # Entry point
└── pyproject.toml  # Project metadata & dependencies
```

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
