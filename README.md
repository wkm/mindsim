# MindSim

2-wheeler robot simulation in MuJoCo for training simple neural networks.

## Usage

**This project uses `uv`:**

```bash
uv run python visualize.py
```

See [CLAUDE.md](CLAUDE.md) for complete documentation.

## Remote Training

Training runs on GCP spot VMs with Docker for cheap, hands-off execution.

### Architecture

```
local machine                    GCP
─────────────                    ───
docker build + push ──────► Artifact Registry
                                    │
./remote_train.sh ──────► COS spot VM (c3d-standard-16)
                                    │ startup script
                                    ├─ docker pull
                                    ├─ docker run (training)
                                    ├─ logs → Cloud Logging
                                    └─ shutdown -h now
```

**Why spot VMs + Docker (not Cloud Run, GKE, or bare VMs)?**

- **Spot VMs** are ~$0.06-0.12/hr for 16 vCPUs. Cloud Run charges ~$0.35/hr for the same. GKE adds cluster overhead. Spot gives raw compute at 80-84% discount.
- **Docker** eliminates the 3-5 min dependency install that bare VMs need on every launch. The image is pre-built with all deps; the VM just pulls and runs.
- **COS (Container-Optimized OS)** has Docker pre-installed, boots fast, and sends container stdout to Cloud Logging automatically.
- **c3d-standard-16** (AMD EPYC Genoa) was chosen for best spot price/performance for CPU-bound MuJoCo: 3.3 GHz sustained all-core, cheapest spot price of all compute-optimized families.

### Usage

```bash
# Prerequisites: gcloud CLI, Docker, WANDB_API_KEY set

./remote_train.sh --smoketest              # Quick validation
./remote_train.sh --bot simple2wheeler     # Full training run
./remote_train.sh --bot simplebiped --num-workers 16

# Skip rebuild if image hasn't changed
SKIP_BUILD=1 ./remote_train.sh --smoketest

# Override machine type
MACHINE_TYPE=c3d-standard-32 ./remote_train.sh --bot simple2wheeler
```

The script handles everything: creates Artifact Registry repo, builds/pushes the Docker image, launches a spot VM with a startup script, and prints commands to monitor the run. The VM shuts itself down when training finishes (stopped VMs cost nothing).

### Cost

| Machine | Spot $/hr | Spot $/day | vCPUs |
|---------|-----------|------------|-------|
| c3d-standard-8 | ~$0.058 | ~$1.39 | 8 |
| c3d-standard-16 (default) | ~$0.116 | ~$2.78 | 16 |
| c3d-standard-32 | ~$0.232 | ~$5.57 | 32 |

## Future Work

- Try better models (larger networks, different architectures)
- Once the robot consistently learns to navigate to a static target:
  1. Randomize the target location ✓ (curriculum learning, stage 1)
  2. Randomize starting location
  3. Move the target during episodes ✓ (curriculum stage 2)
  4. Add visual distractors ✓ (curriculum stage 3)
  5. Add physical obstacles (walls, ramps)
  6. Multi-target sequences (visit targets in order)
