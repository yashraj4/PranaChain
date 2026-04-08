import random
from uuid import uuid4
from typing import Dict, Any
from openenv.core.env_server import Environment
from openenv.core.client_types import StepResult
try:
    from ..models import OxygenAction, OxygenObservation, OxygenState, HospitalStatus, TruckStatus, SupplyNode
except (ImportError, ValueError):
    from models import OxygenAction, OxygenObservation, OxygenState, HospitalStatus, TruckStatus, SupplyNode

class PranaChainEnvironment(Environment):
    """
    Simulation of the 2021 Indian Oxygen Crisis triage dispatcher.
    """
    
    def __init__(self):
        self.max_steps = 50
        self._set_initial_state()

    def _set_initial_state(self):
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._hospitals = {
            "AIIMS_Delhi": {"o2": 1000.0, "rate": 50.0, "patients": 200, "sos": "Low stocks!"},
            "Max_Saket": {"o2": 500.0, "rate": 30.0, "patients": 100, "sos": "Critical levels!"},
        }
        self._fleet = {
            "Truck_1": {"load": 5000.0, "cap": 10000.0, "loc": "Plant_A", "target": None, "status": "IDLE"},
        }
        self._suppliers = {
            "Plant_A": {"stock": 50000.0, "rate": 500.0}
        }
        self._total_delivered = 0.0
        self._casualties = 0

    def reset(self, task: str = "easy") -> OxygenObservation:
        self._set_initial_state()
        if task == "medium":
            self._hospitals["Max_Saket"]["rate"] = 60.0
        elif task == "hard":
            self._hospitals["Max_Saket"]["rate"] = 100.0
            self._hospitals["AIIMS_Delhi"]["rate"] = 80.0
            
        return self._make_observation("Environment reset to task: " + task)

    def step(self, action: OxygenAction) -> StepResult[OxygenObservation]:
        self._step_count += 1
        
        # 1. Update Truck Logic
        if action.action_type == "DELIVER_TO_HOSPITAL":
            if action.target_id in self._hospitals:
                truck = self._fleet["Truck_1"]
                truck["target"] = action.target_id
                truck["status"] = "TRANSIT"
        
        # Simulating movement and delivery in 1 tick for simplicity
        truck = self._fleet["Truck_1"]
        if truck["status"] == "TRANSIT" and truck["target"] in self._hospitals:
            h_id = truck["target"]
            amount = min(truck["load"], 1000.0) # Move 1000L per step
            self._hospitals[h_id]["o2"] += amount
            truck["load"] -= amount
            self._total_delivered += amount
            if truck["load"] <= 0:
                truck["status"] = "IDLE"
                truck["loc"] = h_id

        # 2. Update Hospital Decay
        done = False
        message = "Crisis ongoing..."
        for h_name, data in self._hospitals.items():
            data["o2"] -= data["rate"]
            if data["o2"] <= 0:
                data["o2"] = 0
                self._casualties += data["patients"]
                done = True
                message = f"CRITICAL FAILURE: {h_name} ran out of oxygen!"

        if self._step_count >= self.max_steps:
            done = True
            message = "Shift completed."

        # 3. Reward / Scoring
        reward = self._calculate_reward(done)
        obs = self._make_observation(message)
        
        return StepResult(observation=obs, reward=reward, done=done)

    def _calculate_reward(self, done: bool) -> float:
        if self._casualties > 0:
            return -1000.0
        return self._total_delivered * 0.1

    def state(self) -> OxygenState:
        score = 1.0 if (self._step_count >= self.max_steps and self._casualties == 0) else 0.0
        return OxygenState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            hospitals_state=self._hospitals,
            fleet_state=self._fleet,
            total_delivered=self._total_delivered,
            casualties=self._casualties,
            score=score
        )

    def _make_observation(self, message: str) -> OxygenObservation:
        h_list = [
            HospitalStatus(
                id=k, 
                current_o2_liters=v["o2"], 
                consumption_rate=v["rate"], 
                time_to_zero=v["o2"]/v["rate"] if v["rate"] > 0 else 999,
                critical_patients=v["patients"],
                sos_alert=v["sos"]
            ) for k, v in self._hospitals.items()
        ]
        t_list = [
            TruckStatus(
                id=k,
                current_load=v["load"],
                capacity=v["cap"],
                location=v["loc"],
                target_destination=v["target"],
                status=v["status"]
            ) for k, v in self._fleet.items()
        ]
        s_list = [
            SupplyNode(id=k, available_stock=v["stock"], replenishment_rate=v["rate"])
            for k, v in self._suppliers.items()
        ]
        return OxygenObservation(Hospitals=h_list, Fleet=t_list, Suppliers=s_list, message=message)
