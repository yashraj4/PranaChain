import random
from uuid import uuid4
from typing import Dict, Any
from openenv.core.env_server import Environment

try:
    from prana_chain.models import (
        OxygenAction,
        OxygenObservation,
        OxygenState,
        HospitalStatus, TruckStatus, SupplyNode
    )
    from prana_chain.graders import grade_episode
except ImportError:
    from models import (
        OxygenAction,
        OxygenObservation,
        OxygenState,
        HospitalStatus, TruckStatus, SupplyNode
    )
    from graders import grade_episode

class PranaChainEnvironment(Environment):
    """
    Overhauled Simulation of the 2021 Indian Oxygen Crisis.
    Features: Refill logic, Difficulty scaling, Normalized rewards, Partial credit.
    """
    
    def __init__(self):
        self.max_steps = 30
        self._set_initial_state()

    def _set_initial_state(self, task: str = "easy"):
        # Deterministic scenarios per task so grader output is reproducible.
        seed_map = {"easy": 11, "medium": 23, "hard": 37}
        random.seed(seed_map.get(task, 11))
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._casualties = 0
        self._total_delivered = 0.0
        self._done = False
        self._task = task
        self._loop_penalty_count = 0
        self._last_action_signature = None
        
        # Scenario configuration based on difficulty
        if task == "easy":
            h_count, t_count, s_count = 2, 1, 1
            load_mult = 1.0
        elif task == "medium":
            h_count, t_count, s_count = 4, 2, 2
            load_mult = 0.8
        else: # hard
            h_count, t_count, s_count = 6, 2, 2
            load_mult = 0.5

        # Initialize Hospitals
        self._hospitals = {}
        for i in range(h_count):
            h_id = f"Hospital_{i+1}"
            rate = random.uniform(20, 50) + (10 if task != "easy" else 0)
            self._hospitals[h_id] = {
                "o2": random.uniform(500, 1000) * load_mult,
                "rate": rate,
                "patients": random.randint(50, 200),
                "sos": "Stock dipping."
            }

        # Initialize Fleet
        self._fleet = {}
        for i in range(t_count):
            t_id = f"Truck_{i+1}"
            self._fleet[t_id] = {
                "load": 5000.0 if i == 0 else 0.0,
                "cap": 10000.0,
                "loc": "Plant_1",
                "target": None,
                "status": "IDLE"
            }

        # Initialize Suppliers
        self._suppliers = {}
        for i in range(s_count):
            s_id = f"Plant_{i+1}"
            self._suppliers[s_id] = {
                "stock": 50000.0,
                "rate": 1000.0
            }

    def reset(self, seed=None, episode_id=None, task: str = "easy", **kwargs) -> OxygenObservation:
        self._set_initial_state(task)
        obs = self._make_observation(f"Emergency Dispatch Protocol Active. Difficulty: {task.upper()}")
        obs.reward = 0.0
        obs.done = False
        obs.metadata = {"task": task, "grader_score": 0.0}
        return obs

    def step(self, action: OxygenAction, timeout_s=None, **kwargs) -> OxygenObservation:
        if self._done:
            obs = self._make_observation("Episode finished.")
            obs.reward = 0.0
            obs.done = True
            obs.metadata = {"task": self._task, "grader_score": self._calculate_final_score()}
            return obs
            
        self._step_count += 1
        
        # 1. Dispatcher Logic
        t_id = action.truck_id
        action_signature = f"{action.action_type}:{action.target_id}:{action.truck_id}"
        if action_signature == self._last_action_signature:
            self._loop_penalty_count += 1
        else:
            self._loop_penalty_count = 0
        self._last_action_signature = action_signature

        if t_id in self._fleet:
            truck = self._fleet[t_id]
            if action.action_type == "DELIVER_TO_HOSPITAL":
                if action.target_id in self._hospitals:
                    truck["target"] = action.target_id
                    truck["status"] = "TRANSIT_TO_HOSPITAL"
            elif action.action_type == "DISPATCH_TO_PLANT":
                if action.target_id in self._suppliers:
                    truck["target"] = action.target_id
                    truck["status"] = "TRANSIT_TO_PLANT"

        # 2. Physics Simulation Tick
        # 2.1 Update Fleet Movement & Loading/Unloading
        for t_id, truck in self._fleet.items():
            if truck["status"] == "TRANSIT_TO_HOSPITAL":
                # Move to hospital and dump 1000L per tick
                target = truck["target"]
                truck["loc"] = f"In-Transit to {target}"
                # Simplification: reach in 1 tick, then start offloading
                amount = min(truck["load"], 1000.0)
                self._hospitals[target]["o2"] += amount
                truck["load"] -= amount
                self._total_delivered += amount
                if truck["load"] <= 0:
                    truck["status"] = "IDLE"
                    truck["loc"] = target
                    truck["target"] = None
                    
            elif truck["status"] == "TRANSIT_TO_PLANT":
                # Move to plant and load 2000L per tick
                target = truck["target"]
                truck["loc"] = f"In-Transit to {target}"
                plant = self._suppliers[target]
                amount = min(plant["stock"], truck["cap"] - truck["load"], 2000.0)
                truck["load"] += amount
                plant["stock"] -= amount
                if truck["load"] >= truck["cap"]:
                    truck["status"] = "IDLE"
                    truck["loc"] = target
                    truck["target"] = None

        # 2.2 Update Hospital Decay & Casualties
        for h_id, data in self._hospitals.items():
            data["o2"] -= data["rate"]
            if data["o2"] <= 0:
                data["o2"] = 0
                self._casualties += data["patients"]
                self._done = True # Any zero-oxygen event ends the simulation in failure

        if self._step_count >= self.max_steps:
            self._done = True

        # 3. Reward / Scoring (Normalized to [0, 1])
        reward = self._calculate_normalized_reward()
        obs = self._make_observation("Simulation in progress..." if not self._done else "Simulation Terminated.")
        obs.reward = reward
        obs.done = self._done
        obs.metadata = {"task": self._task, "grader_score": self._calculate_final_score()}
        return obs

    def _calculate_normalized_reward(self) -> float:
        if self._casualties > 0:
            return 0.0
        safe_hospitals = sum(1 for h in self._hospitals.values() if h["o2"] > 100)
        safety_ratio = safe_hospitals / len(self._hospitals)
        avg_stock_ratio = sum(
            min(h["o2"] / 1000.0, 1.0) for h in self._hospitals.values()
        ) / len(self._hospitals)
        loop_penalty = min(self._loop_penalty_count * 0.03, 0.3)
        reward = (0.7 * safety_ratio) + (0.3 * avg_stock_ratio) - loop_penalty
        return round(max(0.0, min(1.0, reward)), 2)

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
            score=round(final_score, 2)
        )

    def _make_observation(self, message: str) -> OxygenObservation:
        h_list = [
            HospitalStatus(
                id=k, 
                current_o2_liters=round(v["o2"], 1), 
                consumption_rate=round(v["rate"], 1), 
                time_to_zero=round(v["o2"]/v["rate"], 1) if v["rate"] > 0 else 999.0,
                critical_patients=v["patients"],
                sos_alert="URGENT: Stock Low!" if v["o2"] < 200 else "Stable."
            ) for k, v in self._hospitals.items()
        ]
        t_list = [
            TruckStatus(
                id=k, current_load=v["load"], capacity=v["cap"],
                location=v["loc"], target_destination=v["target"], status=v["status"]
            ) for k, v in self._fleet.items()
        ]
        s_list = [
            SupplyNode(id=k, available_stock=v["stock"], replenishment_rate=v["rate"])
            for k, v in self._suppliers.items()
        ]
        return OxygenObservation(Hospitals=h_list, Fleet=t_list, Suppliers=s_list, message=message)
