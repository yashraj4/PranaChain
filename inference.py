import os
import sys
import time
import asyncio
import subprocess
from typing import List, Optional

from prana_chain.client import PranaChainEnv
from prana_chain.models import OxygenAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

def log_start(task_id: str):
    print(f"[START] Task ID: {task_id}")

def log_step(step: int, action: str, observation: str, score: float):
    print(f"[STEP] Step {step}")
    print(f"Action: {action}")
    print(f"Observation: {observation}")
    print(f"Score: {score}")

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] Success: {success}")
    print(f"Total Steps: {steps}")
    print(f"Final Score: {score}")
    print(f"Rewards: {rewards}")

async def call_llm(prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content or ""

async def run_evaluation(task: str):
    log_start(task)
    
    # Connect to the local server
    try:
        env = PranaChainEnv(base_url="http://127.0.0.1:8000")
        result = await env.reset(task=task)
        obs = result.observation
    except Exception as e:
        print(f"Failed to connect to env: {e}")
        return

    step = 0
    done = False
    rewards = []
    success = False
    score = 0.0

    while not done and step < 10:
        step += 1
        
        prompt = f"State: {obs.message}\nHospitals: {obs.Hospitals}\nSelect OxygenAction (action_type, target_id)."
        action_str = await call_llm(prompt)
        
        # Simple heuristic mapping for baseline
        if "AIIMS" in action_str:
            action = OxygenAction(action_type="DELIVER_TO_HOSPITAL", target_id="AIIMS_Delhi")
        else:
            action = OxygenAction(action_type="DELIVER_TO_HOSPITAL", target_id="Max_Saket")
            
        result = await env.step(action)
        obs = result.observation
        done = result.done
        rewards.append(result.reward or 0.0)
        
        log_step(step, str(action), obs.message, score)

    state = await env.state()
    score = state.score
    success = score > 0.5
    
    log_end(success=success, steps=step, score=score, rewards=rewards)
    await env.close()

async def main():
    for t in ["easy", "medium", "hard"]:
        await run_evaluation(t)

if __name__ == "__main__":
    process = None
    try:
        # Start server as a separate process
        env_vars = os.environ.copy()
        env_vars["PYTHONPATH"] = os.getcwd()
        process = subprocess.Popen(
            [sys.executable, "-m", "prana_chain.server.app", "--port", "8000"],
            cwd=os.getcwd(),
            env=env_vars,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(5)
    except Exception as e:
        print(f"[DEBUG] Error starting local server: {e}")
    
    try:
        asyncio.run(main())
    finally:
        if process:
            process.terminate()
