import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


START_RE = re.compile(r"^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>.+)$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>.+?) reward=(?P<reward>-?\d+\.\d+) "
    r"done=(?P<done>true|false) error=(?P<error>.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) score=(?P<score>-?\d+\.\d{2,3}) rewards=(?P<rewards>.*)$"
)


@dataclass
class TaskState:
    task: str
    model: str = ""
    env: str = ""
    current_step: int = 0
    last_action: str = "-"
    last_reward: float = 0.0
    total_reward: float = 0.0
    done: bool = False
    success: Optional[bool] = None
    score: float = 0.0
    last_error: str = "null"
    actions: List[str] = field(default_factory=list)


def reward_bar(value: float, width: int = 28) -> str:
    v = max(0.0, min(1.0, value))
    filled = int(round(v * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def clear_screen() -> None:
    # ANSI clear (works in modern terminals); harmless if unsupported.
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def render(tasks: Dict[str, TaskState], active_task: Optional[str]) -> None:
    clear_screen()
    print("Prana-Chain Live Inference Monitor")
    print("=" * 72)
    if active_task:
        print(f"Active Task: {active_task}")
    print()

    for task_name in ["easy", "medium", "hard"]:
        if task_name not in tasks:
            continue
        t = tasks[task_name]
        status = "RUNNING"
        if t.done:
            if t.success is True:
                status = "DONE (SUCCESS)"
            elif t.success is False:
                status = "DONE (FAIL)"
            else:
                status = "DONE"

        avg_reward = (t.total_reward / t.current_step) if t.current_step > 0 else 0.0
        print(f"[{task_name.upper()}] {status}")
        print(f"  model={t.model}  env={t.env}")
        print(f"  step={t.current_step:>2}  last_reward={t.last_reward:.2f}  avg_reward={avg_reward:.2f}")
        print(f"  total_reward={t.total_reward:.2f}  score={t.score:.2f}  success={t.success}")
        print(f"  reward_bar {reward_bar(avg_reward)}")
        print(f"  last_action: {t.last_action}")
        if t.last_error and t.last_error != "null":
            print(f"  last_error: {t.last_error}")
        if t.actions:
            preview = " | ".join(t.actions[-5:])
            print(f"  recent_actions: {preview}")
        print("-" * 72)


def parse_line(line: str, tasks: Dict[str, TaskState], active_task: Optional[str]) -> Optional[str]:
    m = START_RE.match(line)
    if m:
        task = m.group("task")
        state = tasks.get(task, TaskState(task=task))
        state.env = m.group("env")
        state.model = m.group("model")
        tasks[task] = state
        return task

    m = STEP_RE.match(line)
    if m and active_task and active_task in tasks:
        t = tasks[active_task]
        t.current_step = int(m.group("step"))
        t.last_action = m.group("action")
        t.last_reward = float(m.group("reward"))
        t.total_reward += t.last_reward
        t.done = m.group("done") == "true"
        t.last_error = m.group("error")
        t.actions.append(t.last_action)
        return active_task

    m = END_RE.match(line)
    if m and active_task and active_task in tasks:
        t = tasks[active_task]
        t.done = True
        t.success = m.group("success") == "true"
        t.score = float(m.group("score"))
        return active_task

    return active_task


def run_live(inference_cmd: str) -> int:
    tasks: Dict[str, TaskState] = {}
    active_task: Optional[str] = None

    proc = subprocess.Popen(
        inference_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        active_task = parse_line(line, tasks, active_task)
        render(tasks, active_task)
        # Keep raw line visible for auditability.
        print(f"raw: {line}")

    rc = proc.wait()
    render(tasks, active_task)
    print(f"\nProcess finished with exit code {rc}")
    return rc


def replay_file(path: str) -> int:
    tasks: Dict[str, TaskState] = {}
    active_task: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            active_task = parse_line(line, tasks, active_task)
            render(tasks, active_task)
            print(f"raw: {line}")
    print("\nReplay complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time monitor for inference.py action/reward output."
    )
    parser.add_argument(
        "--cmd",
        default="python inference.py",
        help="Command used to run inference (default: 'python inference.py').",
    )
    parser.add_argument(
        "--from-file",
        default="",
        help="Replay an existing inference log file instead of running a command.",
    )
    args = parser.parse_args()

    if args.from_file:
        return replay_file(args.from_file)
    return run_live(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
