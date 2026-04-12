import os
import sys
import time
import asyncio
import subprocess
import json
from typing import List, Optional
import requests

from openai import OpenAI  # Fix 3: synchronous client, top-level import

from prana_chain.client import PranaChainEnv
from prana_chain.models import OxygenAction
from prana_chain.graders import SCORE_MAX, SCORE_MIN

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Fix 2: mandatory HF_TOKEN validation
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Fix 3: initialize OpenAI client once at top level using HF_TOKEN
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

_INFERENCE_OFFLINE = os.getenv("PRANA_INFERENCE_OFFLINE", "").lower() in ("1", "true", "yes")


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Use 3 decimals for score so grader floor (e.g. 0.001) is not shown as 0.00
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _fallback_dispatch(obs) -> OxygenAction:
    """Deterministic policy when LLM is unavailable or PRANA_INFERENCE_OFFLINE=1."""
    try:
        idle = [t for t in obs.Fleet if t.status == "IDLE"]
        truck = idle[0] if idle else obs.Fleet[0]
        tid = truck.id
        if truck.status != "IDLE":
            return OxygenAction(
                action_type="DELIVER_TO_HOSPITAL",
                truck_id=tid,
                target_id="Hospital_1",
            )
        if truck.current_load < 1200:
            plant_id = obs.Suppliers[0].id if obs.Suppliers else "Plant_1"
            return OxygenAction(action_type="DISPATCH_TO_PLANT", truck_id=tid, target_id=plant_id)

        active_hs = [h for h in obs.Hospitals if getattr(h, "active", True)]
        if not active_hs:
            plant_id = obs.Suppliers[0].id if obs.Suppliers else "Plant_1"
            return OxygenAction(action_type="DISPATCH_TO_PLANT", truck_id=tid, target_id=plant_id)
        critical = min(active_hs, key=lambda h: (h.time_to_zero, h.current_o2_liters))
        return OxygenAction(
            action_type="DELIVER_TO_HOSPITAL",
            truck_id=tid,
            target_id=critical.id,
        )
    except Exception:
        return OxygenAction(action_type="DELIVER_TO_HOSPITAL", target_id="Hospital_1")


# Fix 3: synchronous call_llm using top-level `client`, not AsyncOpenAI inside function
def call_llm(obs) -> OxygenAction:
    if _INFERENCE_OFFLINE:
        return _fallback_dispatch(obs)

    # Build structured context so the LLM makes smart decisions
    active_hospitals = [h for h in obs.Hospitals if getattr(h, "active", True)]
    active_hospitals_sorted = sorted(active_hospitals, key=lambda h: getattr(h, "time_to_zero", 999))

    truck = obs.Fleet[0] if obs.Fleet else None
    truck_load = getattr(truck, "current_load", 0)
    truck_cap = getattr(truck, "capacity", 10000)
    truck_pct = int(100 * truck_load / truck_cap) if truck_cap else 0
    truck_status = getattr(truck, "status", "IDLE")
    truck_id = getattr(truck, "id", "Truck_1")

    supplier_id = obs.Suppliers[0].id if obs.Suppliers else "Plant_1"
    most_critical = active_hospitals_sorted[0] if active_hospitals_sorted else None
    critical_id = getattr(most_critical, "id", "Hospital_1") if most_critical else "Hospital_1"
    critical_ttz = getattr(most_critical, "time_to_zero", 999) if most_critical else 999

    hospital_summary = "\n".join(
        f"  {getattr(h,'id','')} | O2={getattr(h,'current_o2_liters',0):.0f}L | TTZ={getattr(h,'time_to_zero',999):.1f}h ← MOST CRITICAL" 
        if i == 0 else
        f"  {getattr(h,'id','')} | O2={getattr(h,'current_o2_liters',0):.0f}L | TTZ={getattr(h,'time_to_zero',999):.1f}h"
        for i, h in enumerate(active_hospitals_sorted)
    )

    refuel_warning = ""
    if truck_pct < 30:
        refuel_warning = f"\n⚠️  TRUCK LOAD CRITICAL ({truck_pct}%). You MUST dispatch to {supplier_id} to refuel before hospitals run out."

    prompt = f"""You are an emergency oxygen dispatcher. Make the single best dispatch decision.

TRUCK STATUS:
  ID: {truck_id} | Load: {truck_load:.0f}L / {truck_cap:.0f}L ({truck_pct}%) | Status: {truck_status}{refuel_warning}

ACTIVE HOSPITALS (sorted by urgency — lowest TTZ = most critical):
{hospital_summary}

SUPPLIER: {supplier_id}

DECISION RULES (follow in order):
1. If truck load < 30% OR truck is empty → DISPATCH_TO_PLANT to {supplier_id}
2. If truck is IDLE and a hospital has TTZ < 15h → DELIVER_TO_HOSPITAL to the most critical one
3. Otherwise → DELIVER_TO_HOSPITAL to {critical_id} (TTZ={critical_ttz:.1f}h)

Return ONLY valid JSON (no explanation):
{{"action_type": "DELIVER_TO_HOSPITAL or DISPATCH_TO_PLANT", "target_id": "<hospital_or_plant_id>", "truck_id": "{truck_id}"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return OxygenAction(**data)
    except Exception:
        return _fallback_dispatch(obs)


async def run_evaluation(task: str):
    done = False
    rewards: List[float] = []
    success = False
    score = 0.0
    env = None

    try:
        # Fix 4: log_start is now INSIDE the try block so [END] is always paired with it
        log_start(task=task, env="prana_chain", model=MODEL_NAME)

        env = PranaChainEnv(base_url="http://127.0.0.1:8000")
        result = None
        last_err = None
        for _ in range(6):
            try:
                result = await env.reset(task=task)
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.0)
        if result is None:
            raise RuntimeError(str(last_err) if last_err else "reset failed")
        obs = result.observation

        # Episode Loop — one [STEP] line per rewards entry so [END] steps == len(rewards)
        while not done and len(rewards) < 20:
            action = call_llm(obs)  # synchronous call — no await needed

            try:
                result = await env.step(action)
                obs = result.observation
                done = result.done
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                action_text = f"{action.action_type}:{action.target_id}"
                log_step(step=len(rewards), action=action_text, reward=reward, done=done)
            except Exception as e:
                rewards.append(0.0)
                log_step(
                    step=len(rewards),
                    action="ERROR",
                    reward=0.0,
                    done=True,
                    error=str(e),
                )
                done = True

        # Final Grading (no extra [STEP] on failure — keep steps == len(rewards))
        try:
            state = await env.state()
            score = max(SCORE_MIN, min(SCORE_MAX, float(state.score)))
            success = score >= 0.8
        except Exception:
            score = SCORE_MIN
            success = False
    except Exception as e:
        rewards.append(0.0)
        log_step(step=len(rewards), action="ERROR", reward=0.0, done=True, error=str(e))
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)


async def main():
    for t in ["easy", "medium", "hard"]:
        await run_evaluation(t)


if __name__ == "__main__":
    server_process = None
    try:
        # Start server as a separate process for reliability
        env_vars = os.environ.copy()
        env_vars["PYTHONUTF8"] = "1"
        env_vars["PYTHONPATH"] = os.getcwd()

        server_log = open("server_stdout.log", "w")
        server_err = open("server_stderr.log", "w")

        server_process = subprocess.Popen(
            [sys.executable, "-m", "prana_chain.server.app", "--port", "8000"],
            cwd=os.getcwd(),
            env=env_vars,
            stdout=server_log,
            stderr=server_err
        )
        ready = False
        for _ in range(20):
            try:
                r = requests.post("http://127.0.0.1:8000/reset", json={}, timeout=1.5)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            print("[DEBUG] Server warmup did not reach /reset=200 before run", flush=True)

        asyncio.run(main())
    finally:
        if server_process:
            server_process.terminate()
