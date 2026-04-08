---
title: Network Rca Env
emoji: 🏆
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: NOC RCA env, real data, dense rewards, LangGraph+OpenAI.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Network Root Cause Analysis Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://github.com/open-env/openenv)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Space-deployed-ffcc00)](https://huggingface.co/spaces)

## Overview

This environment simulates the **real‑world task of network root cause analysis (RCA)**. Built from actual incidents extracted from a production network database, it provides authentic alarms, topologies, and fault propagation patterns. The environment is designed to train and evaluate AI agents that can assist NOC engineers in diagnosing network faults efficiently and accurately.

## Why Network RCA?

In a live network, a single failure can trigger dozens of alarms. Engineers must correlate these alarms, gather evidence (metrics, logs), and identify the root cause – a time‑consuming, error‑prone process. Automating this with AI agents has the potential to dramatically reduce downtime. Our environment is a first step toward training such agents in a safe, reproducible, and realistic setting.

## Environment Features

- **Real‑data tasks**: 3+ incidents per difficulty level extracted from your own database.
- **Dynamic observation**: initial alarms are short; investigating reveals detailed descriptions.
- **Hidden information**: metrics (latency, packet loss) and logs (syslog‑like messages) are hidden until the agent queries them.
- **Dense reward function**: rewards each useful action, penalises inefficiency, guessing, and insufficient evidence.
- **OpenEnv compliant**: full `step()`/`reset()`/`state()` API with Pydantic models.
- **Deployable**: containerised Docker image, ready for Hugging Face Spaces.

## Action & Observation Spaces

| Space | Description |
|-------|-------------|
| **Observation** | `alarms` (list of `Alarm`), `topology_edges` (list of (src,dst)), `step_count`, `metrics` (revealed after `query_metrics`), `logs` (revealed after `check_logs`) |
| **Action** | `investigate <alarm_id\|device_name>` – reveals detailed description; <br/> `correlate` – small reward; <br/> `query_metrics <device>` – reveals latency/packet loss/CPU; <br/> `check_logs <device>` – reveals syslog‑like entries; <br/> `conclude <root_cause>` – ends episode and triggers grading. |

## Tasks (Easy → Medium → Hard)

Tasks are generated offline from the production database using `extract_tasks.py`. They reflect real incidents with increasing complexity:

- **Easy**: 1–2 alarms, single device, root cause obvious from alarms.
- **Medium**: 3–5 alarms, up to 2 devices, clear cascade, but still inferable from alarms alone.
- **Hard**: >5 alarms, multiple devices, overlapping symptoms, missing data, and **hidden root cause** – the agent must explicitly query metrics and logs on the affected devices to discover the truth.

Each task includes a ground truth root cause, relevant alarm IDs, dependency alarms, and a maximum number of steps.

## Reward Function

The reward is dense and shaped to encourage efficient, evidence‑based diagnosis:

| Action | Reward |
|--------|--------|
| Investigate relevant alarm | +0.25 |
| Discover dependency | +0.2 (extra) |
| Correlate | +0.1 |
| Query metrics | +0.15 |
| Check logs | +0.15 |
| Investigate irrelevant alarm | –0.1 |
| Redundant investigation | –0.05 |
| Step penalty (each step) | –0.2 |
| Wrong conclusion (score < 0.5) | –0.3 |
| Conclude without evidence (no metrics/logs on root device) | –0.3 |
| Timeout | –0.5 |
| Final grader score (0.0–1.0) | added as reward |

The grader uses rule‑based + semantic similarity (sentence‑transformers) to compare the agent’s conclusion to the ground truth, returning a score in [0.0, 1.0].

## Baseline Agent (Reproducible)

The official baseline for evaluation is `baseline.py`. It is deterministic and reproducible:

- fixed seed (`--seed`, default `42`)
- fixed policy flow (investigate -> query metrics -> check logs -> correlate -> conclude)
- structured JSON output per difficulty and aggregate
- configurable via CLI flags

### Baseline Commands

Run all difficulties:

```bash
python baseline.py --difficulty all --seed 42 --episodes 1
```

Run one difficulty:

```bash
python baseline.py --difficulty hard --seed 42 --episodes 1
```

Save machine-readable report:

```bash
python baseline.py --difficulty all --seed 42 --episodes 1 --output-json baseline_report.json
```

The report contains:

- `results.<difficulty>.average_total_reward`
- `results.<difficulty>.episodes[].trajectory`
- `results.<difficulty>.episodes[].termination_reason`
- `aggregate.overall_average_total_reward`

## API Endpoints

- `POST /reset` -> reset episode by difficulty
- `POST /step` -> apply one action
- `GET /state` -> current environment state
- `GET /tasks` -> task list + action schema
- `POST /grader` -> deterministic score, feedback, and breakdown
- `GET /baseline` -> run baseline and return results

`/grader` returns:

- `score` in `[0.0, 1.0]`
- `feedback`
- `breakdown` with `root_cause_score`, `evidence_score`, `efficiency_score`, `missing_evidence`
- `task_id`, `required_evidence`, `gathered_evidence`

## Setup and Usage

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Generate tasks from your DB:

```bash
python extract_tasks.py
```

4. Set provider credentials (OpenAI required for baseline):

> `OPENAI_API_KEY` must be your real OpenAI secret key (starts with `sk-...`).
> Replace `your_key_here` / `your_openai_key_here` with your actual key value.

```bash
# Linux/macOS (bash/zsh)
# OpenAI
export OPENAI_API_KEY=your_key_here
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini

# Optional: fallback model through local/cloud Ollama
export OLLAMA_FALLBACK_MODEL=deepseek-v3.1:671b-cloud
```

```powershell
# Windows PowerShell
# OpenAI
$env:OPENAI_API_KEY="your_key_here"
$env:LLM_PROVIDER="openai"
$env:LLM_MODEL="gpt-4o-mini"

# Optional: fallback model through local/cloud Ollama
$env:OLLAMA_FALLBACK_MODEL="deepseek-v3.1:671b-cloud"
```

5. Validate spec:

```bash
openenv validate
```

6. Run server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Step-by-Step Verification

Use this exact order before final submission.

1) Environment setup

```bash
python --version
pip install -r requirements.txt
```

Expected:
- Python is `3.10+`
- All dependencies install without errors

2) OpenEnv validation

```bash
openenv validate
```

Expected:
- Output includes `[OK] ... Ready for multi-mode deployment`

3) Baseline run (OpenAI first, Ollama fallback available)

```bash
python baseline.py --difficulty all --episodes 1
```

Expected:
- Command exits successfully
- JSON output includes `results.easy`, `results.medium`, `results.hard`
- `aggregate.overall_average_total_reward` is present

4) API smoke test (local server)

Start server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

In another terminal:

```bash
curl http://127.0.0.1:7860/
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d "{\"difficulty\":\"easy\"}"
curl http://127.0.0.1:7860/tasks
curl http://127.0.0.1:7860/baseline
```

Expected:
- Root endpoint returns running message
- `/reset` returns observation payload
- `/tasks` returns task list + action schema
- `/baseline` returns baseline report JSON

5) Docker verification

```bash
docker build -t network-rca-env .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 7860:7860 network-rca-env
```

PowerShell:

```powershell
docker run --rm -e OPENAI_API_KEY=$env:OPENAI_API_KEY -p 7860:7860 network-rca-env
```

Expected:
- Build completes successfully
- Container starts and serves API on port `7860`

6) Final pre-submit checks

- `openenv validate` passes
- Baseline runs end-to-end
- All required endpoints respond
- Docker build/run works
- README is up to date with setup + verification flow

## Latest Verification Output

The following outputs were captured from a full local module sweep.

1) OpenEnv validator

Command:

```bash
myenv3\Scripts\openenv.exe validate
```

Output:

```text
[OK] network-rca-env: Ready for multi-mode deployment
```

2) Baseline module

Command:

```bash
python baseline.py --difficulty easy --episodes 1
```

Output summary:

```text
{
  "difficulty": "easy",
  "steps_taken": 4,
  "done": true,
  "inferred_root_cause": "High CPU utilization (93%) causing interface flapping and BGP session failure"
}
```

3) Random agent module

Command:

```bash
python random_agent.py
```

Output:

```text
easy total: 0.25
medium total: 0.20
hard total: 0.35
```

4) Task extraction module

Command:

```bash
python extract_tasks.py
```

Output:

```text
Extracted 3 easy, 3 medium, 3 hard tasks
```

Note: pandas emits SQLAlchemy-related warnings in this script, but extraction completed successfully.

5) API endpoint smoke test (running server on `127.0.0.1:7860`)

Output:

```text
ROOT=Network RCA Environment is running
RESET_ALARMS=1
STEP_DONE=False
STATE_STEP_COUNT=1
TASKS_COUNT=9
GRADER_SCORE=0.2928871540228526
BASELINE_KEYS=easy,medium,hard
BASELINE_OVERALL=0.350125
```

## Docker Deployment

Build image:

```bash
docker build -t network-rca-env .
```

Run container:

```bash
# Linux/macOS
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 7860:7860 network-rca-env

# Windows PowerShell
docker run --rm -e OPENAI_API_KEY=$env:OPENAI_API_KEY -p 7860:7860 network-rca-env
```

PowerShell quick example (copy/paste):

```powershell
$env:OPENAI_API_KEY="sk-...your-real-openai-key..."
docker run --rm -e OPENAI_API_KEY=$env:OPENAI_API_KEY -p 7860:7860 network-rca-env
```

