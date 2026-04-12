from openenv.core.env_server import create_app
import openenv.core.env_server.http_server as _openenv_http
import openenv.core.env_server.serialization as _openenv_serialization
from fastapi.responses import PlainTextResponse

# OpenEnv's serialize_observation drops observation.metadata from JSON. The dashboard and
# clients need layout + reward_components on the wire; http_server holds a reference to
# serialize_observation imported at load time, so patch both modules after import.
_original_serialize_observation = _openenv_serialization.serialize_observation


def _serialize_observation_with_wire_extras(observation):
    payload = _original_serialize_observation(observation)
    obs_dict = payload.get("observation")
    if isinstance(obs_dict, dict):
        meta = getattr(observation, "metadata", None) or {}
        if isinstance(meta, dict):
            if "layout" in meta:
                obs_dict["env_layout"] = meta["layout"]
            rc = meta.get("reward_components")
            if rc is not None:
                obs_dict["reward_components"] = rc
    return payload


_openenv_serialization.serialize_observation = _serialize_observation_with_wire_extras
_openenv_http.serialize_observation = _serialize_observation_with_wire_extras

try:
    from prana_chain.models import OxygenAction, OxygenObservation
    from prana_chain.server.prana_chain_environment import PranaChainEnvironment
except (ImportError, ModuleNotFoundError):
    from models import OxygenAction, OxygenObservation
    from prana_chain_environment import PranaChainEnvironment

app = create_app(
    PranaChainEnvironment,
    OxygenAction,
    OxygenObservation,
    env_name="prana_chain",
    # factory=True, # Optional: Create new environment per session
)


_ROOT_TEXT = """Prana Chain — OpenEnv HTTP API

This Space exposes the standard OpenEnv endpoints only (no browser dashboard).

  POST /reset     JSON: {"task":"easy"|"medium"|"hard"} optional hospitals, trucks, suppliers
  POST /step      JSON: {"action": { "action_type", "target_id", "truck_id", ... }}
  GET  /state     Current episode state

Interactive schema: /docs
Health: GET /health (if enabled by OpenEnv)

Local CLI monitor (optional): python visualize_inference.py
"""


@app.get("/", response_class=PlainTextResponse)
def root():
    return _ROOT_TEXT

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
