---
title: Prana Chain Environment
emoji: 🚑
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# Prana-Chain: The Oxygen War Room Simulation

Autonomous dynamic triage simulation designed for reinforcement-learning and LLM-agents, built on the OpenEnv framework.

Prana-Chain models a real emergency-operations workflow: dispatch teams deciding how to route oxygen tankers between production plants and hospitals with continuously draining oxygen buffers.

## Table of Contents
- [Why Prana-Chain?](#why-prana-chain)
- [Quick Start](#quick-start)
- [Environment Overview](#environment-overview)
- [Actions](#actions)
- [Observations](#observations)
- [Reward Structure](#reward-structure)

## Why Prana-Chain?

Resource constraints during public health crises form devastating bottlenecks. Logistics networks break down due to shifting consumption rates and delayed dispatching. "Prana-Chain" forces AI systems into extreme constraints: determining who gets oxygen and when.

## Quick Start

### Using Docker
```bash
docker build -t prana_chain-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 prana_chain-env:latest
```

### Basic Python Client
```python
import asyncio
from prana_chain import PranaChainEnv, OxygenAction

async def main():
    async with PranaChainEnv(base_url="http://127.0.0.1:8000") as env:
        result = await env.reset(task="easy")
        print(f"Hospitals: {len(result.observation.Hospitals)}")

        action = OxygenAction(
            action_type="DELIVER_TO_HOSPITAL", 
            target_id="AIIMS_Delhi", 
            priority_level=5
        )
        result = await env.step(action)
        print(f"Reward: {result.reward:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Environment Overview

This environment models logistics physics tied explicitly to patient consumption:
- Hospitals: Drain oxygen continuously based on dynamic consumption rates.
- Time-To-Zero (TTZ): If TTZ hits 0, it triggers a Zero-Oxygen event.
- Fleet Dynamics: Trucks carry and deliver oxygen.
- Episode horizon: max 30 steps.

## OpenEnv API

- `reset(task=...) -> OxygenObservation`
- `step(OxygenAction) -> OxygenObservation`
- `state -> OxygenState`

`OxygenObservation` carries OpenEnv-standard fields (`reward`, `done`, `metadata`) and domain fields (`Hospitals`, `Fleet`, `Suppliers`, `message`).

## Actions

### OxygenAction
1. DELIVER_TO_HOSPITAL - Route truck to a hospital.
2. DISPATCH_TO_PLANT - Route a truck back to refilling nodes.
3. DIVERT_IN_TRANSIT - Rationing.

Action schema fields:
- `action_type`: one of `DELIVER_TO_HOSPITAL`, `DISPATCH_TO_PLANT`, `DIVERT_IN_TRANSIT`
- `truck_id`: target truck id (default `Truck_1`)
- `target_id`: hospital or plant id
- `priority_level`: integer 1-10

## Observations

### OxygenObservation
- Hospitals: List of hospital statuses (TTZ, SOS, etc.)
- Fleet: truck positions and loads.
- Suppliers: Plant stock levels.
- message: textual status message
- done: whether episode terminated
- reward: normalized step reward in `[0, 1]`

## Reward Structure

| Event | Reward |
| --- | --- |
| Step safety ratio (hospitals > 100L) | +0.0 to +0.7 |
| Average stock health ratio | +0.0 to +0.3 |
| Repeated same action loop | -0.03 each repeated turn (capped) |
| Any casualty event | immediate collapse to 0.0 |

## Tasks and Graders

Difficulty progression and deterministic graders are implemented in `graders.py`.

- `easy`: keep 2 hospitals alive; lower required throughput.
- `medium`: keep 4 hospitals alive and deliver more total oxygen.
- `hard`: keep 6 hospitals alive with strict throughput requirement.

Each task returns a final score in `[0.0, 1.0]` based on:
- hospital survival ratio,
- horizon durability (`step_count / max_steps`),
- normalized delivery throughput (`total_delivered / target_liters`),
- zero score on casualties.

## Baseline Inference

Root script: `../inference.py`

Required env vars:
- `HF_TOKEN`
- `API_BASE_URL` (default `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default `Qwen/Qwen2.5-72B-Instruct`)

Run:
```bash
python ../inference.py
```

The script emits strict structured logs:
- `[START]`
- `[STEP]`
- `[END]`

## Hugging Face Spaces

This repo is compatible with Docker Spaces and includes `server/Dockerfile`.
Use Space SDK `docker`, expose port `8000`, and include the `openenv` tag.
