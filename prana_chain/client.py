from typing import Dict, Any, Optional
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import OxygenAction, OxygenObservation, OxygenState

class PranaChainEnv(EnvClient[OxygenAction, OxygenObservation, OxygenState]):
    """
    Client for the Prana Chain Oxygen Crisis Simulator.
    """
    
    def _step_payload(self, action: OxygenAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[OxygenObservation]:
        # The base EnvClient expects we return a StepResult wrapping our Observation type
        obs_data = payload.get("observation")
        if not obs_data:
            # Fallback for unexpected payload formats
            return StepResult(
                observation=OxygenObservation(**payload),
                reward=0.0,
                done=False
            )

        observation = OxygenObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> OxygenState:
        return OxygenState(**payload)
