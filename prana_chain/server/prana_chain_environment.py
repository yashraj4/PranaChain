import random
from uuid import uuid4
from typing import Dict, Any, List, Tuple, Optional
from openenv.core.env_server import Environment

try:
    from prana_chain.models import (
        OxygenAction,
        OxygenObservation,
        OxygenState,
        HospitalStatus,
        TruckStatus,
        SupplyNode,
    )
    from prana_chain.graders import grade_episode
except ImportError:
    from models import (
        OxygenAction,
        OxygenObservation,
        OxygenState,
        HospitalStatus,
        TruckStatus,
        SupplyNode,
    )
    from graders import grade_episode

MAX_HOSPITAL_SLOTS = 12
DELIVERY_CHUNK_L = 800.0
PLANT_CHUNK_L = 2000.0


class PranaChainEnvironment(Environment):
    """
    Oxygen crisis logistics with fixed hospital slots (add/deactivate without variable schema),
    delivery lead time (ETA), dynamic bed-driven consumption, and progressive rewards.
    """

    def __init__(self):
        self.max_steps = 30
        self._set_initial_state()

    def _lead_time(self) -> int:
        if self._task == "easy":
            return 2
        if self._task == "medium":
            return 3
        return 4

    def _slot_ids(self) -> List[str]:
        return [f"Hospital_{i}" for i in range(1, MAX_HOSPITAL_SLOTS + 1)]

    def _set_initial_state(self, task: str = "easy"):
        seed_map = {"easy": 11, "medium": 23, "hard": 37}
        random.seed(seed_map.get(task, 11))
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._casualties = 0
        self._total_delivered = 0.0
        self._done = False
        self._task = task
        self._custom_layout = False
        self._h_count = 0
        self._t_count = 0
        self._loop_penalty_count = 0
        self._last_action_signature = None
        self._wasted_arrival_step = False
        self._ignored_busy_dispatch = False

        if task == "easy":
            initial_active, t_count, s_count = 2, 1, 1
            load_mult = 1.0
        elif task == "medium":
            initial_active, t_count, s_count = 4, 2, 2
            load_mult = 0.8
        else:
            initial_active, t_count, s_count = 6, 2, 2
            load_mult = 0.5

        self._h_count = initial_active
        self._t_count = t_count

        self._hospitals: Dict[str, Dict[str, Any]] = {}
        for idx, h_id in enumerate(self._slot_ids()):
            slot_index = idx + 1
            active = slot_index <= initial_active
            if active:
                base = random.uniform(18, 42) + (8 if task != "easy" else 0)
                beds = random.randint(45, 120)
                self._hospitals[h_id] = {
                    "active": True,
                    "o2": random.uniform(500, 1000) * load_mult,
                    "base_rate": base,
                    "rate": base * (beds / 80.0),
                    "beds": beds,
                    "patients": random.randint(50, 200),
                }
            else:
                self._hospitals[h_id] = {
                    "active": False,
                    "o2": 0.0,
                    "base_rate": 0.0,
                    "rate": 0.0,
                    "beds": 0,
                    "patients": 0,
                }

        self._fleet = {}
        for i in range(t_count):
            t_id = f"Truck_{i+1}"
            self._fleet[t_id] = {
                "load": 5000.0 if i == 0 else 0.0,
                "cap": 10000.0,
                "loc": "Plant_1",
                "target": None,
                "status": "IDLE",
                "eta": 0,
                "pending_liters": 0.0,
                "lead": 0,
            }

        self._suppliers = {}
        for i in range(s_count):
            s_id = f"Plant_{i+1}"
            self._suppliers[s_id] = {"stock": 50000.0, "rate": 1000.0}

        self._slot_events = self._build_slot_events()

    def _build_slot_events(self) -> List[Tuple[int, int, str]]:
        """Deterministic (step, hospital_number 1-12, 'on'|'off') from episode id."""
        rng = random.Random(hash(self._episode_id) % (2**31))
        events: List[Tuple[int, int, str]] = []
        for step in range(5, self.max_steps, 5):
            hid = rng.randint(1, MAX_HOSPITAL_SLOTS)
            kind = "off" if rng.random() < 0.45 else "on"
            events.append((step, hid, kind))
        return events

    def _apply_slot_events(self) -> None:
        for step, hid, kind in self._slot_events:
            if step != self._step_count:
                continue
            h_id = f"Hospital_{hid}"
            if h_id not in self._hospitals:
                continue
            data = self._hospitals[h_id]
            active_count = sum(1 for h in self._hospitals.values() if h["active"])
            if kind == "off" and data["active"]:
                if active_count <= 1:
                    continue
                data["active"] = False
                data["o2"] = 0.0
                data["base_rate"] = 0.0
                data["rate"] = 0.0
                data["beds"] = 0
                data["patients"] = 0
            elif kind == "on" and not data["active"]:
                rng = random.Random((hash(self._episode_id) + hid * 997 + self._step_count) % (2**31))
                beds = rng.randint(55, 140)
                base = rng.uniform(22, 48)
                data["active"] = True
                data["o2"] = rng.uniform(350, 700)
                data["base_rate"] = base
                data["rate"] = base * (beds / 80.0)
                data["beds"] = beds
                data["patients"] = rng.randint(60, 180)

    def reset(self, seed=None, episode_id=None, task: str = "easy", **kwargs) -> OxygenObservation:
        self._set_initial_state(task)
        dh, dt, ds = self._h_count, self._t_count, len(self._suppliers)

        layout_keys = ("hospitals", "trucks", "suppliers")
        has_any_layout_key = any(k in kwargs for k in layout_keys)
        req_h = int(kwargs["hospitals"]) if "hospitals" in kwargs else dh
        req_t = int(kwargs["trucks"]) if "trucks" in kwargs else dt
        req_s = int(kwargs["suppliers"]) if "suppliers" in kwargs else ds
        req_h = max(2, min(MAX_HOSPITAL_SLOTS, req_h))
        req_t = max(1, min(6, req_t))
        req_s = max(1, min(4, req_s))

        custom_needed = has_any_layout_key and (req_h, req_t, req_s) != (dh, dt, ds)

        if custom_needed:
            initial_active = req_h
            t_count = req_t
            s_count = req_s
            self._custom_layout = True
            self._h_count = initial_active
            self._t_count = t_count

            self._hospitals = {}
            for idx, h_id in enumerate(self._slot_ids()):
                slot_index = idx + 1
                if slot_index <= initial_active:
                    rng = random.Random((hash(self._episode_id) + slot_index * 131) % (2**31))
                    beds = rng.randint(50, 130)
                    base = rng.uniform(24, 52)
                    self._hospitals[h_id] = {
                        "active": True,
                        "o2": rng.uniform(400, 900),
                        "base_rate": base,
                        "rate": base * (beds / 80.0),
                        "beds": beds,
                        "patients": rng.randint(45, 190),
                    }
                else:
                    self._hospitals[h_id] = {
                        "active": False,
                        "o2": 0.0,
                        "base_rate": 0.0,
                        "rate": 0.0,
                        "beds": 0,
                        "patients": 0,
                    }

            self._fleet = {}
            for i in range(t_count):
                t_id = f"Truck_{i+1}"
                self._fleet[t_id] = {
                    "load": 5000.0 if i == 0 else 0.0,
                    "cap": 10000.0,
                    "loc": "Plant_1",
                    "target": None,
                    "status": "IDLE",
                    "eta": 0,
                    "pending_liters": 0.0,
                    "lead": 0,
                }
            self._suppliers = {}
            for i in range(s_count):
                s_id = f"Plant_{i+1}"
                self._suppliers[s_id] = {"stock": 50000.0, "rate": 1000.0}
            self._slot_events = self._build_slot_events()

        layout = "custom" if self._custom_layout else "task_default"
        obs = self._make_observation(
            f"Emergency Dispatch Protocol Active. Difficulty: {task.upper()}",
            env_layout=layout,
            reward_components=None,
        )
        obs.reward = 0.0
        obs.done = False
        obs.metadata = {
            "task": task,
            "grader_score": self._calculate_final_score(),
            "layout": layout,
            "counts": {
                "hospitals": self._h_count,
                "trucks": self._t_count,
                "suppliers": len(self._suppliers),
            },
            "slots": MAX_HOSPITAL_SLOTS,
        }
        return obs

    def _resolve_hospital_arrival(self, truck: Dict[str, Any], target: str) -> None:
        pending = truck["pending_liters"]
        h = self._hospitals.get(target)
        if h and h["active"]:
            h["o2"] += pending
            truck["load"] -= pending
            self._total_delivered += pending
            truck["loc"] = target
        else:
            self._wasted_arrival_step = True
            truck["loc"] = "Idle (delivery target inactive — load retained)"
        truck["status"] = "IDLE"
        truck["target"] = None
        truck["eta"] = 0
        truck["pending_liters"] = 0.0
        truck["lead"] = 0

    def _resolve_plant_arrival(self, truck: Dict[str, Any], target: str) -> None:
        plant = self._suppliers.get(target)
        pending = truck["pending_liters"]
        if plant:
            truck["load"] += pending
            plant["stock"] -= pending
        truck["status"] = "IDLE"
        truck["target"] = None
        truck["eta"] = 0
        truck["pending_liters"] = 0.0
        truck["lead"] = 0
        truck["loc"] = target

    def step(self, action: OxygenAction, timeout_s=None, **kwargs) -> OxygenObservation:
        if self._done:
            layout = "custom" if self._custom_layout else "task_default"
            obs = self._make_observation(
                "Episode finished.",
                env_layout=layout,
                reward_components={},
            )
            obs.reward = 0.0
            obs.done = True
            obs.metadata = {
                "task": self._task,
                "grader_score": self._calculate_final_score(),
                "layout": layout,
                "reward_components": {},
            }
            return obs

        self._step_count += 1
        self._wasted_arrival_step = False
        self._ignored_busy_dispatch = False

        action_signature = f"{action.action_type}:{action.target_id}:{action.truck_id}"
        if action_signature == self._last_action_signature:
            self._loop_penalty_count += 1
        else:
            self._loop_penalty_count = 0
        self._last_action_signature = action_signature

        for _, truck in self._fleet.items():
            if truck["status"] == "TRANSIT_TO_HOSPITAL" and truck["eta"] > 0:
                truck["eta"] -= 1
                if truck["eta"] == 0 and truck.get("target"):
                    self._resolve_hospital_arrival(truck, truck["target"])
            elif truck["status"] == "TRANSIT_TO_PLANT" and truck["eta"] > 0:
                truck["eta"] -= 1
                if truck["eta"] == 0 and truck.get("target"):
                    self._resolve_plant_arrival(truck, truck["target"])

        t_id = action.truck_id
        if t_id in self._fleet:
            truck = self._fleet[t_id]
            if truck["status"] == "IDLE":
                if action.action_type == "DELIVER_TO_HOSPITAL" and action.target_id in self._hospitals:
                    tgt = self._hospitals[action.target_id]
                    if tgt["active"] and truck["load"] > 0:
                        lead = self._lead_time()
                        truck["pending_liters"] = min(truck["load"], DELIVERY_CHUNK_L)
                        truck["target"] = action.target_id
                        truck["status"] = "TRANSIT_TO_HOSPITAL"
                        truck["eta"] = lead
                        truck["lead"] = lead
                        truck["loc"] = f"In-Transit to {action.target_id}"
                elif action.action_type == "DISPATCH_TO_PLANT" and action.target_id in self._suppliers:
                    plant = self._suppliers[action.target_id]
                    space = truck["cap"] - truck["load"]
                    if space > 0:
                        lead = self._lead_time()
                        truck["pending_liters"] = min(plant["stock"], space, PLANT_CHUNK_L)
                        if truck["pending_liters"] > 0:
                            truck["target"] = action.target_id
                            truck["status"] = "TRANSIT_TO_PLANT"
                            truck["eta"] = lead
                            truck["lead"] = lead
                            truck["loc"] = f"In-Transit to {action.target_id}"
            else:
                self._ignored_busy_dispatch = True

        for h_id, data in list(self._hospitals.items()):
            if not data["active"] or data["rate"] <= 0:
                continue
            data["o2"] -= data["rate"]
            if data["o2"] <= 0:
                data["o2"] = 0
                self._casualties += int(data["patients"])
                self._done = True

        if self._step_count >= self.max_steps:
            self._done = True

        self._apply_slot_events()

        for data in self._hospitals.values():
            if not data["active"]:
                continue
            delta = random.randint(-4, 6)
            data["beds"] = int(max(25, min(220, data["beds"] + delta)))
            data["rate"] = data["base_rate"] * (data["beds"] / 80.0)

        reward, reward_components = self._progressive_reward_detail()
        layout = "custom" if self._custom_layout else "task_default"
        obs = self._make_observation(
            "Simulation in progress..." if not self._done else "Simulation Terminated.",
            env_layout=layout,
            reward_components=reward_components,
        )
        obs.reward = reward
        obs.done = self._done
        obs.metadata = {
            "task": self._task,
            "grader_score": self._calculate_final_score(),
            "layout": layout,
            "counts": {
                "hospitals": self._h_count,
                "trucks": self._t_count,
                "suppliers": len(self._suppliers),
            },
            "slots": MAX_HOSPITAL_SLOTS,
            "reward_components": reward_components,
        }
        return obs

    def _active_hospitals(self) -> List[Dict[str, Any]]:
        return [h for h in self._hospitals.values() if h["active"]]

    def _progressive_reward_detail(self) -> Tuple[float, Dict[str, float]]:
        """
        Dense step reward (rounded to 2 dp for API) plus unrounded / term breakdown in metadata.
        """
        if self._casualties > 0:
            return 0.0, {
                "safety_ratio": 0.0,
                "stock": 0.0,
                "transit_progress": 0.0,
                "loop_penalty": 0.0,
                "waste_penalty": 0.0,
                "busy_penalty": 0.0,
                "term_safety": 0.0,
                "term_stock": 0.0,
                "term_transit": 0.0,
                "pre_clamp": 0.0,
                "clamped_unrounded": 0.0,
            }
        active = self._active_hospitals()
        if not active:
            pre = 0.05
            clamped = max(0.0, min(1.0, pre))
            return round(clamped, 2), {
                "safety_ratio": 0.0,
                "stock": 0.0,
                "transit_progress": 0.0,
                "loop_penalty": 0.0,
                "waste_penalty": 0.0,
                "busy_penalty": 0.0,
                "term_safety": 0.0,
                "term_stock": 0.0,
                "term_transit": 0.0,
                "pre_clamp": round(pre, 6),
                "clamped_unrounded": round(clamped, 6),
            }

        critical = 120.0
        safe = sum(1 for h in active if h["o2"] > critical)
        safety_ratio = safe / len(active)
        stock = sum(min(h["o2"] / 1000.0, 1.0) for h in active) / len(active)

        transit_scores = []
        for t in self._fleet.values():
            if t["status"] in ("TRANSIT_TO_HOSPITAL", "TRANSIT_TO_PLANT") and t.get("lead", 0) > 0:
                lead = max(1, int(t["lead"]))
                eta = max(0, int(t["eta"]))
                transit_scores.append((lead - eta) / lead)
        transit_progress = sum(transit_scores) / max(1, len(self._fleet)) if transit_scores else 0.0

        loop_penalty = min(self._loop_penalty_count * 0.025, 0.25)
        waste_penalty = 0.12 if self._wasted_arrival_step else 0.0
        busy_penalty = 0.06 if self._ignored_busy_dispatch else 0.0

        term_safety = 0.48 * safety_ratio
        term_stock = 0.27 * stock
        term_transit = 0.20 * transit_progress
        pre_clamp = term_safety + term_stock + term_transit - loop_penalty - waste_penalty - busy_penalty
        clamped = max(0.0, min(1.0, pre_clamp))
        rounded = round(clamped, 2)

        components: Dict[str, float] = {
            "safety_ratio": round(safety_ratio, 4),
            "stock": round(stock, 4),
            "transit_progress": round(transit_progress, 4),
            "loop_penalty": round(loop_penalty, 4),
            "waste_penalty": round(waste_penalty, 4),
            "busy_penalty": round(busy_penalty, 4),
            "term_safety": round(term_safety, 4),
            "term_stock": round(term_stock, 4),
            "term_transit": round(term_transit, 4),
            "pre_clamp": round(pre_clamp, 6),
            "clamped_unrounded": round(clamped, 6),
        }
        return rounded, components

    def _calculate_final_score(self) -> float:
        return grade_episode(
            task_id=self._task,
            step_count=self._step_count,
            max_steps=self.max_steps,
            hospitals_state=self._hospitals,
            casualties=self._casualties,
            total_delivered=self._total_delivered,
        )

    @property
    def state(self) -> OxygenState:
        final_score = self._calculate_final_score()
        return OxygenState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            hospitals_state=self._hospitals,
            fleet_state=self._fleet,
            total_delivered=self._total_delivered,
            casualties=self._casualties,
            score=round(final_score, 3),
        )

    def _make_observation(
        self,
        message: str,
        env_layout: Optional[str] = None,
        reward_components: Optional[Dict[str, float]] = None,
    ) -> OxygenObservation:
        h_list = []
        for k in self._slot_ids():
            v = self._hospitals[k]
            rate = v["rate"]
            o2 = v["o2"]
            ttz = round(o2 / rate, 1) if v["active"] and rate > 0 else 999.0
            sos = (
                "INACTIVE"
                if not v["active"]
                else ("URGENT: Stock Low!" if o2 < 220 else "Stable.")
            )
            h_list.append(
                HospitalStatus(
                    id=k,
                    active=bool(v["active"]),
                    beds=int(v["beds"]),
                    current_o2_liters=round(o2, 1),
                    consumption_rate=round(rate, 2) if v["active"] else 0.0,
                    time_to_zero=ttz,
                    critical_patients=int(v["patients"]),
                    sos_alert=sos,
                )
            )
        t_list = [
            TruckStatus(
                id=k,
                current_load=v["load"],
                capacity=v["cap"],
                location=v["loc"],
                target_destination=v["target"],
                status=v["status"],
                eta_steps=int(v["eta"]) if v.get("eta") else None,
                in_transit_liters=float(v["pending_liters"]) if v.get("pending_liters") else None,
            )
            for k, v in self._fleet.items()
        ]
        s_list = [
            SupplyNode(id=k, available_stock=v["stock"], replenishment_rate=v["rate"])
            for k, v in self._suppliers.items()
        ]
        return OxygenObservation(
            Hospitals=h_list,
            Fleet=t_list,
            Suppliers=s_list,
            message=message,
            env_layout=env_layout,
            reward_components=reward_components,
        )
