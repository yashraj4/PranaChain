from typing import Dict

# Submission validators require task scores strictly in (0, 1), not 0.0 nor 1.0.
SCORE_MIN = 0.001
SCORE_MAX = 0.999


def clamp_task_score(raw: float) -> float:
    """Map raw score into (SCORE_MIN, SCORE_MAX) with stable rounding."""
    return round(max(SCORE_MIN, min(SCORE_MAX, raw)), 3)


def _survival_ratio(hospitals_state: Dict[str, Dict]) -> float:
    """Fraction of *active* hospital slots with oxygen remaining."""
    active = [h for h in hospitals_state.values() if h.get("active")]
    if not active:
        return 0.0
    alive = sum(1 for h in active if h.get("o2", 0.0) > 0.0)
    return alive / len(active)


def grade_episode(
    task_id: str,
    step_count: int,
    max_steps: int,
    hospitals_state: Dict[str, Dict],
    casualties: int,
    total_delivered: float,
) -> float:
    """
    Deterministic grader; returns score strictly in (0, 1).
    Uses active hospitals only for survival; delivery/horizon unchanged by task tier.
    """
    if casualties > 0:
        return SCORE_MIN

    survival = _survival_ratio(hospitals_state)
    horizon = min(step_count / max_steps, 1.0) if max_steps > 0 else 0.0

    if task_id == "easy":
        delivery = min(total_delivered / 6000.0, 1.0)
        raw = (0.6 * survival) + (0.2 * horizon) + (0.2 * delivery)
    elif task_id == "medium":
        delivery = min(total_delivered / 12000.0, 1.0)
        raw = (0.5 * survival) + (0.2 * horizon) + (0.3 * delivery)
    else:
        delivery = min(total_delivered / 20000.0, 1.0)
        raw = (0.45 * survival) + (0.15 * horizon) + (0.4 * delivery)

    return clamp_task_score(raw)
