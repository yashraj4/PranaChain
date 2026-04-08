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

## Actions

### OxygenAction
1. DELIVER_TO_HOSPITAL - Route truck to a hospital.
2. DISPATCH_TO_PLANT - Route a truck back to refilling nodes.
3. DIVERT_IN_TRANSIT - Rationing.

## Observations

### OxygenObservation
- Hospitals: List of hospital statuses (TTZ, SOS, etc.)
- Fleet: truck positions and loads.
- Suppliers: Plant stock levels.

## Reward Structure

| Event | Reward |
| --- | --- |
| Supply Injection | +0.1 per liter |
| Zero-Oxygen Event | -1000.0 |
