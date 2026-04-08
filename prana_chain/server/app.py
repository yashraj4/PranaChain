from openenv.core.env_server import create_app
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prana-Chain Live Dashboard</title>
  <style>
    :root { --bg:#0b1020; --panel:#131a2f; --muted:#9aa6c4; --text:#eaf0ff; --good:#34d399; --warn:#f59e0b; --bad:#ef4444; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; background:var(--bg); color:var(--text); }
    .wrap { max-width:1100px; margin:24px auto; padding:0 16px; }
    .title { font-size:28px; font-weight:700; margin-bottom:8px; }
    .sub { color:var(--muted); margin-bottom:18px; }
    .row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
    .panel { background:var(--panel); border-radius:14px; padding:14px; box-shadow:0 6px 20px rgba(0,0,0,.25); }
    .panel h3 { margin:0 0 10px 0; font-size:16px; }
    .grow { flex:1 1 320px; }
    button, select { background:#1e2a4a; color:var(--text); border:1px solid #2d3a62; border-radius:10px; padding:8px 12px; }
    button { cursor:pointer; }
    .metric { font-size:22px; font-weight:700; }
    .muted { color:var(--muted); }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid #263255; padding:6px 4px; text-align:left; }
    .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
    .log { height:170px; overflow:auto; background:#0f1530; border-radius:10px; padding:8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">Prana-Chain Live RL Visualization</div>
    <div class="sub">Interactive real-time simulation view of actions, rewards, and hospital oxygen risk.</div>

    <div class="row">
      <div class="panel grow">
        <h3>Controls</h3>
        <label class="muted">Task:</label>
        <select id="task">
          <option value="easy">easy</option>
          <option value="medium">medium</option>
          <option value="hard">hard</option>
        </select>
        <button onclick="resetEnv()">Reset</button>
        <button onclick="stepPolicy()">Step Heuristic</button>
        <button onclick="toggleAuto()" id="autoBtn">Start Auto</button>
      </div>
      <div class="panel grow">
        <h3>Episode Metrics</h3>
        <div>Step: <span class="metric" id="step">0</span></div>
        <div>Reward: <span class="metric" id="reward">0.00</span></div>
        <div>Done: <span id="done" class="muted">false</span></div>
        <div>Last Action: <span id="action" class="muted">-</span></div>
      </div>
    </div>

    <div class="row">
      <div class="panel grow">
        <h3>Hospitals</h3>
        <table id="hospTable">
          <thead><tr><th>ID</th><th>O2</th><th>Rate</th><th>TTZ</th><th>SOS</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="panel grow">
        <h3>Fleet</h3>
        <table id="fleetTable">
          <thead><tr><th>ID</th><th>Load</th><th>Cap</th><th>Status</th><th>Target</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <h3>Live Event Log</h3>
      <div id="log" class="log"></div>
    </div>
  </div>

  <script>
    let obs = null;
    let stepN = 0;
    let auto = null;
    let isDone = false;
    let lastReward = 0;

    function addLog(msg) {
      const el = document.getElementById('log');
      const row = document.createElement('div');
      row.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      el.appendChild(row);
      el.scrollTop = el.scrollHeight;
    }

    function render() {
      if (!obs) return;
      document.getElementById('step').textContent = String(stepN);
      document.getElementById('reward').textContent = Number(lastReward || 0).toFixed(2);
      document.getElementById('done').textContent = String(Boolean(isDone));

      const hb = document.querySelector('#hospTable tbody');
      hb.innerHTML = '';
      (obs.Hospitals || []).forEach(h => {
        const tr = document.createElement('tr');
        const ttzClass = h.time_to_zero < 6 ? 'bad' : (h.time_to_zero < 12 ? 'warn' : 'good');
        tr.innerHTML = `<td>${h.id}</td><td>${h.current_o2_liters.toFixed(1)}</td><td>${h.consumption_rate.toFixed(1)}</td><td class="${ttzClass}">${h.time_to_zero.toFixed(1)}</td><td>${h.sos_alert}</td>`;
        hb.appendChild(tr);
      });

      const fb = document.querySelector('#fleetTable tbody');
      fb.innerHTML = '';
      (obs.Fleet || []).forEach(f => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${f.id}</td><td>${Number(f.current_load).toFixed(1)}</td><td>${Number(f.capacity).toFixed(1)}</td><td>${f.status}</td><td>${f.target_destination || '-'}</td>`;
        fb.appendChild(tr);
      });
    }

    async function resetEnv() {
      const task = document.getElementById('task').value;
      const res = await fetch('/reset', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ task }) });
      const data = await res.json();
      obs = data.observation;
      stepN = 0;
      isDone = false;
      lastReward = 0;
      document.getElementById('action').textContent = '-';
      addLog(`Reset task=${task}`);
      render();
    }

    function chooseAction() {
      const hs = (obs && obs.Hospitals) ? obs.Hospitals.slice() : [];
      const fleet = (obs && obs.Fleet && obs.Fleet[0]) ? obs.Fleet[0] : null;
      if (!fleet || hs.length === 0) {
        return { action_type: "DELIVER_TO_HOSPITAL", truck_id: "Truck_1", target_id: "Hospital_1", priority_level: 5 };
      }
      if (Number(fleet.current_load || 0) < 1000) {
        return { action_type: "DISPATCH_TO_PLANT", truck_id: fleet.id || "Truck_1", target_id: "Plant_1", priority_level: 5 };
      }
      hs.sort((a,b) => (a.time_to_zero - b.time_to_zero) || (a.current_o2_liters - b.current_o2_liters));
      return { action_type: "DELIVER_TO_HOSPITAL", truck_id: fleet.id || "Truck_1", target_id: hs[0].id, priority_level: 8 };
    }

    async function stepPolicy() {
      if (!obs || isDone) return;
      const action = chooseAction();
      const res = await fetch('/step', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ action }) });
      const data = await res.json();
      obs = data.observation;
      lastReward = Number(data.reward || 0);
      isDone = Boolean(data.done);
      stepN += 1;
      document.getElementById('action').textContent = `${action.action_type}:${action.target_id}`;
      addLog(`step=${stepN} action=${action.action_type}:${action.target_id} reward=${lastReward.toFixed(2)} done=${isDone}`);
      render();
      if (isDone && auto) toggleAuto();
    }

    function toggleAuto() {
      const btn = document.getElementById('autoBtn');
      if (auto) {
        clearInterval(auto); auto = null; btn.textContent = 'Start Auto'; return;
      }
      auto = setInterval(() => { stepPolicy(); }, 900);
      btn.textContent = 'Stop Auto';
    }

    resetEnv();
  </script>
</body>
</html>
"""

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
