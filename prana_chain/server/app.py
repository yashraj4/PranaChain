from openenv.core.env_server import create_app
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


@app.get("/")
def root():
    return {
        "name": "prana_chain",
        "status": "ok",
        "hint": "Use POST /reset, WebSocket /ws, and GET /health for OpenEnv interactions.",
    }

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
