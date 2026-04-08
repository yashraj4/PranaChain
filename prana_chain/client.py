from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import OxygenAction, OxygenObservation, OxygenState

class PranaChainEnv(EnvClient[OxygenAction, OxygenObservation, OxygenState]):
    """
    Client for oxygen triage dispatcher.
    """
    
    def _step_payload(self, action: OxygenAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[OxygenObservation]:
        obs_data = payload.get("observation", {})
        observation = OxygenObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> OxygenState:
        return OxygenState(**payload)
