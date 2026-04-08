from typing import List, Dict, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State

class HospitalStatus(Observation):
    id: str
    current_o2_liters: float
    consumption_rate: float
    time_to_zero: float  # Minutes until empty
    critical_patients: int
    sos_alert: str

class TruckStatus(Observation):
    id: str
    current_load: float
    capacity: float
    location: str # "Plant_A" or "Hospital_B" or "In-Transit"
    target_destination: Optional[str]
    status: str # "OFFLOADING", "LOADING", "IDLE", "TRANSIT"

class SupplyNode(Observation):
    id: str
    available_stock: float
    replenishment_rate: float

class OxygenObservation(Observation):
    Hospitals: List[HospitalStatus]
    Fleet: List[TruckStatus]
    Suppliers: List[SupplyNode]
    message: str

class OxygenAction(Action):
    action_type: str  # "DELIVER_TO_HOSPITAL", "DISPATCH_TO_PLANT", "DIVERT_IN_TRANSIT"
    target_id: str
    priority_level: int = Field(default=5, ge=1, le=10)

class OxygenState(State):
    hospitals_state: Dict[str, Dict]
    fleet_state: Dict[str, Dict]
    total_delivered: float = 0.0
    casualties: int = 0
    score: float = 0.0
