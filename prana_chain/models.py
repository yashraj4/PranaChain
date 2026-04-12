from typing import List, Dict, Optional, Literal
from pydantic import Field, BaseModel
from openenv.core.env_server import Action, Observation, State


class HospitalStatus(BaseModel):
    id: str
    active: bool = True
    beds: int = 80
    current_o2_liters: float
    consumption_rate: float
    time_to_zero: float
    critical_patients: int
    sos_alert: str


class TruckStatus(BaseModel):
    id: str
    current_load: float
    capacity: float
    location: str
    target_destination: Optional[str] = None
    status: str
    eta_steps: Optional[int] = None
    in_transit_liters: Optional[float] = None


class SupplyNode(BaseModel):
    id: str
    available_stock: float
    replenishment_rate: float


class OxygenObservation(Observation):
    Hospitals: List[HospitalStatus]
    Fleet: List[TruckStatus]
    Suppliers: List[SupplyNode]
    message: str
    # Included in HTTP JSON (OpenEnv strips observation.metadata from wire format).
    env_layout: Optional[str] = None  # "task_default" | "custom"
    reward_components: Optional[Dict[str, float]] = None


class OxygenAction(Action):
    action_type: Literal["DELIVER_TO_HOSPITAL", "DISPATCH_TO_PLANT", "DIVERT_IN_TRANSIT"]
    truck_id: str = "Truck_1"
    target_id: str
    priority_level: int = Field(default=5, ge=1, le=10)


class OxygenState(State):
    hospitals_state: Dict[str, Dict]
    fleet_state: Dict[str, Dict]
    total_delivered: float = 0.0
    casualties: int = 0
    score: float = 0.0
    episode_id: str
    step_count: int
