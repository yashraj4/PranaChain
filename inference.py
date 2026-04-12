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


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


# Fix 1: removed `score` parameter — [END] format now matches spec exactly
def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# Fix 3: synchronous call_llm using top-level `client`, not AsyncOpenAI inside function
def call_llm(obs) -> OxygenAction:
    prompt = f"""
    You are an emergency oxygen dispatcher.
    CURRENT STATE: {obs.message}
    HOSPITALS: {obs.Hospitals}
    FLEET: {obs.Fleet}
    SUPPLIERS: {obs.Suppliers}

    Choose ONE action for one truck: include truck_id (e.g. Truck_1) and target_id.
    Only hospitals with active=true receive oxygen; inactive slots are closed.
    Delivery takes multiple steps in transit (see Fleet eta_steps).
    Available actions: DELIVER_TO_HOSPITAL, DISPATCH_TO_PLANT.

    Return ONLY a JSON object like: {{"action_type": "...", "target_id": "...", "truck_id": "Truck_1"}}
    """

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
        # Deterministic fallback policy:
        # 1) Refill when truck is empty/low.
        # 2) Otherwise deliver to the most critical hospital.
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


async def run_evaluation(task: str):
    step_count = 0
    done = False
    rewards: List[float] = []
    success = False
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

        # Episode Loop
        while not done and step_count < 20:
            step_count += 1
            action = call_llm(obs)  # synchronous call — no await needed

            try:
                result = await env.step(action)
                obs = result.observation
                done = result.done
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                action_text = f"{action.action_type}:{action.target_id}"
                log_step(step=step_count, action=action_text, reward=reward, done=done)
            except Exception as e:
                log_step(step=step_count, action="ERROR", reward=0.0, done=True, error=str(e))
                done = True

        # Final Grading
        state = await env.state()
        score = max(SCORE_MIN, min(SCORE_MAX, float(state.score)))
        success = score >= 0.8
    except Exception as e:
        log_step(step=max(1, step_count + 1), action="ERROR", reward=0.0, done=True, error=str(e))
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        # Fix 1 + Fix 4: always emitted, no score field
        log_end(success=success, steps=step_count, rewards=rewards)


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
