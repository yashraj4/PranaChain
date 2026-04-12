#!/usr/bin/env python3
"""
Run before Hugging Face Space upload: openenv validate + inference.py with log validation.

Validates stdout lines match the competition-style [START] / [STEP] / [END] contract.

Usage (from repo root):
  python scripts/pre_deploy_check.py

Environment:
  HF_TOKEN            Required for inference.py import (use any non-empty string for offline mode).
  PRANA_INFERENCE_OFFLINE=1  Set automatically by this script for the inference subprocess.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RE_START = re.compile(
    r"^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>.+)$"
)
# Match visualize_inference.py STEP_RE (error may be null or a message)
RE_STEP = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>.+?) reward=(?P<reward>-?\d+\.\d{2}) "
    r"done=(?P<done>true|false) error=(?P<error>.*)$"
)
RE_END = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) score=(?P<score>-?\d+\.\d{2,3}) "
    r"rewards=(?P<rewards>.*)$"
)
RE_REWARD_TOKEN = re.compile(r"^-?\d+\.\d{2}$")


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )


def validate_inference_stdout(stdout: str) -> list[str]:
    errors: list[str] = []
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("[")]

    i = 0
    episode = 0
    expected_tasks = ["easy", "medium", "hard"]

    while i < len(lines):
        m_start = RE_START.match(lines[i])
        if not m_start:
            errors.append(f"Line {i + 1}: expected [START], got: {lines[i][:120]}")
            return errors
        task = m_start.group("task")
        if episode < len(expected_tasks) and task != expected_tasks[episode]:
            errors.append(
                f"Episode {episode + 1}: expected task={expected_tasks[episode]}, got task={task}"
            )
        i += 1
        steps_seen = 0
        while i < len(lines) and lines[i].startswith("[STEP]"):
            m_step = RE_STEP.match(lines[i])
            if not m_step:
                errors.append(f"Bad [STEP] format: {lines[i][:200]}")
                return errors
            exp_step = int(m_step.group("step"))
            if exp_step != steps_seen + 1:
                errors.append(
                    f"[STEP] step numbering: expected {steps_seen + 1}, got {exp_step} in {lines[i]}"
                )
            steps_seen += 1
            i += 1
        if i >= len(lines) or not lines[i].startswith("[END]"):
            errors.append(f"Episode {episode + 1}: missing [END] after [START] task={task}")
            return errors
        m_end = RE_END.match(lines[i])
        if not m_end:
            errors.append(f"Bad [END] format: {lines[i][:200]}")
            return errors
        end_steps = int(m_end.group("steps"))
        if end_steps != steps_seen:
            errors.append(
                f"[END] steps={end_steps} does not match number of [STEP] lines ({steps_seen})"
            )
        rewards_raw = m_end.group("rewards").strip()
        reward_parts = (
            [p.strip() for p in rewards_raw.split(",") if p.strip()] if rewards_raw else []
        )
        if len(reward_parts) != end_steps:
            errors.append(
                f"[END] rewards count {len(reward_parts)} != steps={end_steps}"
            )
        for j, p in enumerate(reward_parts):
            if not RE_REWARD_TOKEN.match(p.strip()):
                errors.append(f"[END] rewards[{j}] not dd.dd format: {p!r}")
        i += 1
        episode += 1

    if episode != 3:
        errors.append(f"Expected 3 episodes (easy/medium/hard), got {episode}")
    return errors


def main() -> int:
    print("== 1/2: openenv validate ==", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    r = run([sys.executable, "-m", "openenv.cli", "validate"], env)
    if r.returncode != 0:
        print(r.stdout or "", file=sys.stderr)
        print(r.stderr or "", file=sys.stderr)
        print("FAILED: openenv validate", file=sys.stderr)
        return 1
    print((r.stdout or "").strip() or "[OK] openenv validate", flush=True)

    print("== 2/2: inference.py (offline deterministic policy) ==", flush=True)
    inf_env = env.copy()
    inf_env["PYTHONPATH"] = str(ROOT)
    inf_env["PRANA_INFERENCE_OFFLINE"] = "1"
    inf_env.setdefault("HF_TOKEN", "offline-local-check")
    r2 = run([sys.executable, "inference.py"], inf_env)
    out = (r2.stdout or "") + "\n" + (r2.stderr or "")
    print(r2.stdout or "", end="" if r2.stdout else "", flush=True)
    print(r2.stderr or "", end="" if r2.stderr else "", file=sys.stderr, flush=True)

    if r2.returncode != 0:
        print("FAILED: inference.py exit code", r2.returncode, file=sys.stderr)
        return 1

    log_errors = validate_inference_stdout(r2.stdout or "")
    if log_errors:
        for e in log_errors:
            print(f"LOG CHECK: {e}", file=sys.stderr)
        print("FAILED: [START]/[STEP]/[END] contract", file=sys.stderr)
        return 1

    print("OK: All checks passed (validate + inference logs). Safe to deploy.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
