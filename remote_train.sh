#!/usr/bin/env bash
#
# Run mindsim training on a GCP spot VM using Docker.
#
# Usage:
#   ./remote_train.sh --smoketest
#   ./remote_train.sh --bot simple2wheeler
#   ./remote_train.sh --bot simplebiped --num-workers 4
#   ./remote_train.sh --resume wandb://run_id/checkpoint:latest
#
# Environment:
#   WANDB_API_KEY       Required. Your Weights & Biases API key.
#   ANTHROPIC_API_KEY   Optional. For auto-generating run notes via Claude.
#   MACHINE_TYPE        Optional. GCP machine type (default: c3d-standard-16).
#   ZONE                Optional. GCP zone (default: us-central1-a).
#   SKIP_BUILD          Optional. Set to 1 to skip Docker build & push.
#
# The container runs directly on a COS (Container-Optimized OS) spot VM.
# When training finishes, the VM shuts itself down (stopped VMs cost nothing).
#
# Cost: ~$0.50-3.00 per 24h run depending on machine type and spot pricing.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MACHINE_TYPE="${MACHINE_TYPE:-c3d-standard-16}"  # c3d: AMD Genoa, best spot price/perf for CPU-bound MuJoCo
ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"  # e.g. us-central1 from us-central1-a
MAX_HOURS="${MAX_HOURS:-24}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
GCP_PROJECT="$(gcloud config get-value project 2>/dev/null)"
REPO_NAME="mindsim"
IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO_NAME}/mindsim-train"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if ! command -v gcloud &>/dev/null; then
    echo "Error: gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install" >&2
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "Error: docker not found." >&2
    exit 1
fi

if [[ -z "$GCP_PROJECT" ]]; then
    echo "Error: No GCP project configured. Run: gcloud config set project <PROJECT_ID>" >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Warning: WANDB_API_KEY not set. W&B logging will be disabled." >&2
fi

# All remaining args are passed to main.py train
TRAIN_ARGS=("$@")
if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
    echo "Usage: ./remote_train.sh [main.py train args]" >&2
    echo "  e.g. ./remote_train.sh --smoketest" >&2
    echo "       ./remote_train.sh --bot simple2wheeler" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Infrastructure (idempotent)
# ---------------------------------------------------------------------------
echo "==> Ensuring Artifact Registry repo exists..."
if ! gcloud artifacts repositories describe "$REPO_NAME" \
        --location="$REGION" --format="value(name)" &>/dev/null; then
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="MindSim training images" \
        --quiet
    echo "    Created repo: $REPO_NAME"
else
    echo "    Repo exists: $REPO_NAME"
fi

echo "==> Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---------------------------------------------------------------------------
# Phase 2: Build & push Docker image
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD:-}" == "1" ]]; then
    echo "==> Skipping Docker build (SKIP_BUILD=1)"
else
    echo "==> Building Docker image..."
    docker build --platform linux/amd64 -t "${IMAGE}:latest" "$PROJECT_DIR"

    echo "==> Pushing Docker image..."
    docker push "${IMAGE}:latest"
fi

# ---------------------------------------------------------------------------
# Phase 3: Create spot VM with container
# ---------------------------------------------------------------------------
# Instance name = W&B run name (one name everywhere)
# Extract --bot name from args, default to simple2wheeler
BOT_NAME="simple2wheeler"
for i in "${!TRAIN_ARGS[@]}"; do
    if [[ "${TRAIN_ARGS[$i]}" == "--bot" && -n "${TRAIN_ARGS[$((i+1))]:-}" ]]; then
        BOT_NAME="${TRAIN_ARGS[$((i+1))]}"
        break
    fi
done
INSTANCE="${BOT_NAME}-$(date +%m%d-%H%M)"

echo "==> Creating spot VM: $INSTANCE ($MACHINE_TYPE in $ZONE)"

# Build labels for tracking what's running on each instance
GIT_BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
# Sanitize for GCP labels: lowercase, alphanumeric/hyphens/underscores, max 63 chars
sanitize_label() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g' | cut -c1-63; }
LABEL_BRANCH="$(sanitize_label "$GIT_BRANCH")"
LABEL_ARGS="$(sanitize_label "${TRAIN_ARGS[*]}")"
LABELS="mindsim=true,mindsim-branch=${LABEL_BRANCH},mindsim-args=${LABEL_ARGS}"

# Build the docker run env flags
DOCKER_ENV_FLAGS="-e MUJOCO_GL=egl -e PYTHONUNBUFFERED=1 -e RUN_NAME=${INSTANCE}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    DOCKER_ENV_FLAGS+=" -e WANDB_API_KEY=${WANDB_API_KEY}"
else
    DOCKER_ENV_FLAGS+=" -e WANDB_MODE=disabled"
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    DOCKER_ENV_FLAGS+=" -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
fi

# Build quoted train args for the startup script
QUOTED_TRAIN_ARGS=""
for arg in "${TRAIN_ARGS[@]}"; do
    QUOTED_TRAIN_ARGS+=" $(printf '%q' "$arg")"
done

# Generate startup script that pulls and runs the container on COS
# COS has Docker pre-installed; we just need to configure registry auth
STARTUP_SCRIPT_FILE="$(mktemp)"
trap "rm -f $STARTUP_SCRIPT_FILE" EXIT
cat > "$STARTUP_SCRIPT_FILE" <<STARTUP_EOF
#!/bin/bash
set -euo pipefail
logger 'mindsim: startup script starting'

# Safety: auto-shutdown after ${MAX_HOURS}h no matter what
shutdown -h +$((MAX_HOURS * 60))

# COS has a read-only root filesystem; point HOME to a writable dir
export HOME=/var/tmp

# Configure Docker to pull from Artifact Registry
docker-credential-gcr configure-docker --registries=${REGION}-docker.pkg.dev

# Pull and run the training container
logger 'mindsim: pulling image ${IMAGE}:latest'
docker pull ${IMAGE}:latest

logger 'mindsim: starting training container'
docker run --rm ${DOCKER_ENV_FLAGS} ${IMAGE}:latest${QUOTED_TRAIN_ARGS} 2>&1 | logger -t mindsim-train

# Shut down when training completes
logger 'mindsim: training finished, shutting down'
shutdown -h now
STARTUP_EOF

gcloud compute instances create "$INSTANCE" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --zone="$ZONE" \
    --boot-disk-size=30GB \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --scopes=default,logging-write \
    --labels="$LABELS" \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT_FILE" \
    --quiet

echo ""
echo "=========================================="
echo "  Training started on $INSTANCE"
echo "=========================================="
echo ""
echo "  Run:   ${INSTANCE}"
echo "  Image: ${IMAGE}:latest"
echo "  Args:  ${TRAIN_ARGS[*]}"
echo ""
echo "  SSH into VM:"
echo "    gcloud compute ssh $INSTANCE --zone=$ZONE"
echo ""
echo "  Stream container logs (via SSH):"
echo "    gcloud compute ssh $INSTANCE --zone=$ZONE -- docker logs -f \$(docker ps -q)"
echo ""
echo "  Stream logs (Cloud Logging):"
echo "    gcloud logging read 'resource.type=\"gce_instance\" resource.labels.instance_id=\"'\"$INSTANCE\"'\" jsonPayload.message=~\"mindsim\"' --freshness=1h --order=asc --format='value(jsonPayload.message)'"
echo ""
echo "  VM self-stops when training finishes."
echo "  Delete when done:"
echo "    gcloud compute instances delete $INSTANCE --zone=$ZONE -q"
echo ""
