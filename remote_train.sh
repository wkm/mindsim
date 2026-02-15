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
#   MACHINE_TYPE        Optional. GCP machine type (default: e2-standard-8).
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
INSTANCE="mindsim-$(date +%m%d-%H%M%S)"
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
    docker build -t "${IMAGE}:latest" "$PROJECT_DIR"

    echo "==> Pushing Docker image..."
    docker push "${IMAGE}:latest"
fi

# ---------------------------------------------------------------------------
# Phase 3: Create spot VM with container
# ---------------------------------------------------------------------------
echo "==> Creating spot VM: $INSTANCE ($MACHINE_TYPE in $ZONE)"

# Generate run name (same format as train.py) so instance ↔ W&B run map 1:1
# Extract --bot name from args, default to simple2wheeler
BOT_NAME="simple2wheeler"
for i in "${!TRAIN_ARGS[@]}"; do
    if [[ "${TRAIN_ARGS[$i]}" == "--bot" && -n "${TRAIN_ARGS[$((i+1))]:-}" ]]; then
        BOT_NAME="${TRAIN_ARGS[$((i+1))]}"
        break
    fi
done
RUN_NAME="${BOT_NAME}-$(date +%m%d-%H%M)"

# Build container env vars
CONTAINER_ENV="MUJOCO_GL=egl,PYTHONUNBUFFERED=1,RUN_NAME=${RUN_NAME}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    CONTAINER_ENV+=",WANDB_API_KEY=${WANDB_API_KEY}"
else
    CONTAINER_ENV+=",WANDB_MODE=disabled"
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    CONTAINER_ENV+=",ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
fi

# Build container args array
CONTAINER_ARGS=()
for arg in "${TRAIN_ARGS[@]}"; do
    CONTAINER_ARGS+=(--container-arg="$arg")
done

# Build labels for tracking what's running on each instance
GIT_BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
# Sanitize for GCP labels: lowercase, alphanumeric/hyphens/underscores, max 63 chars
sanitize_label() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g' | cut -c1-63; }
LABEL_BRANCH="$(sanitize_label "$GIT_BRANCH")"
LABEL_ARGS="$(sanitize_label "${TRAIN_ARGS[*]}")"
LABEL_RUN="$(sanitize_label "$RUN_NAME")"
LABELS="mindsim=true,mindsim-run=${LABEL_RUN},mindsim-branch=${LABEL_BRANCH},mindsim-args=${LABEL_ARGS}"

gcloud compute instances create-with-container "$INSTANCE" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --zone="$ZONE" \
    --boot-disk-size=30GB \
    --scopes=default,logging-write \
    --labels="$LABELS" \
    --container-image="${IMAGE}:latest" \
    --container-env="$CONTAINER_ENV" \
    "${CONTAINER_ARGS[@]}" \
    --container-restart-policy=never \
    --metadata=shutdown-script='#!/bin/bash
# Auto-shutdown is handled by the container exiting + restart-policy=never
# This is a fallback safety net
logger "VM shutting down via metadata shutdown script"' \
    --quiet

# Safety: schedule auto-shutdown after MAX_HOURS
echo "==> Setting ${MAX_HOURS}h auto-shutdown watchdog..."
# Wait for SSH to be ready before scheduling shutdown
for i in $(seq 1 30); do
    if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="true" 2>/dev/null; then
        break
    fi
    sleep 2
done
gcloud compute ssh "$INSTANCE" --zone="$ZONE" -- \
    "sudo shutdown -h +$((MAX_HOURS * 60))" 2>/dev/null || true

# Schedule VM self-shutdown when container exits
# COS runs the container via konlet; we watch for it to exit
gcloud compute ssh "$INSTANCE" --zone="$ZONE" -- bash -s <<'WATCH_SCRIPT'
nohup bash -c '
    # Wait for the training container to start
    sleep 30
    # Poll until no container is running
    while docker ps --format "{{.Names}}" 2>/dev/null | grep -q .; do
        sleep 60
    done
    echo "Container exited. Shutting down VM."
    sudo shutdown -h now
' &>/dev/null &
WATCH_SCRIPT

echo ""
echo "=========================================="
echo "  Training started on $INSTANCE"
echo "=========================================="
echo ""
echo "  Run:   ${RUN_NAME}"
echo "  Image: ${IMAGE}:latest"
echo "  Args:  ${TRAIN_ARGS[*]}"
echo ""
echo "  Stream container logs:"
echo "    gcloud compute ssh $INSTANCE --zone=$ZONE -- docker logs -f \$(docker ps -q)"
echo ""
echo "  Stream logs (Cloud Logging):"
echo "    gcloud logging read 'resource.type=\"gce_instance\" labels.\"container-name\"=\"mindsim-train\" resource.labels.instance_id=\"'\"$INSTANCE\"'\"' --freshness=1h --order=asc --format='value(textPayload)'"
echo ""
echo "  VM self-stops when training finishes."
echo "  Delete when done:"
echo "    gcloud compute instances delete $INSTANCE --zone=$ZONE -q"
echo ""
