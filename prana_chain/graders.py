from typing import Dict


def _survival_ratio(hospitals_state: Dict[str, Dict]) -> float:
    if not hospitals_state:
        return 0.0
    alive = sum(1 for h in hospitals_state.values() if h.get("o2", 0.0) > 0.0)
    return alive / len(hospitals_state)


def grade_episode(
    task_id: str,
    step_count: int,
    max_steps: int,
    hospitals_state: Dict[str, Dict],
    casualties: int,
    total_delivered: float,
) -> float:
    """
    Deterministic grader that returns score in [0, 1].
    easy   : prioritize survival, modest delivery requirement
    medium : balanced survival + sustained logistics throughput
    hard   : stricter throughput and no-casualty expectation
    """
    if casualties > 0:
        return 0.0

    survival = _survival_ratio(hospitals_state)
    horizon = min(step_count / max_steps, 1.0) if max_steps > 0 else 0.0

    if task_id == "easy":
        delivery = min(total_delivered / 6000.0, 1.0)
        score = (0.6 * survival) + (0.2 * horizon) + (0.2 * delivery)
    elif task_id == "medium":
        delivery = min(total_delivered / 12000.0, 1.0)
        score = (0.5 * survival) + (0.2 * horizon) + (0.3 * delivery)
    else:
        delivery = min(total_delivered / 20000.0, 1.0)
        score = (0.45 * survival) + (0.15 * horizon) + (0.4 * delivery)

    return round(max(0.0, min(1.0, score)), 3)
