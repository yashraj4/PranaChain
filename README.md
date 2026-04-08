---
title: Prana Chain OpenEnv
emoji: 🚑
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# Prana-Chain OpenEnv Submission

This repository contains a real-world OpenEnv environment for emergency oxygen dispatch logistics.

- Environment package: `prana_chain/`
- OpenEnv config: `openenv.yaml`
- Baseline inference script: `inference.py`
- Real-time monitor script: `visualize_inference.py`
- Docker deployment file: `Dockerfile`
- Requirements file: `requirements.txt`

See full environment documentation in `prana_chain/README.md`.

## Optional Live Visualization

To visualize model actions/rewards live during baseline execution:

```bash
python visualize_inference.py
```

To replay a previous structured inference log:

```bash
python visualize_inference.py --from-file inference_output.log
```
