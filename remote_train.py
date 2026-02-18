#!/usr/bin/env python3
"""
Run mindsim training on a GCP spot VM using Docker.

Tries multiple US zones automatically when spot capacity is unavailable.

Usage:
    python remote_train.py --smoketest
    python remote_train.py --bot simple2wheeler
    python remote_train.py --bot simplebiped --num-workers 4
    python remote_train.py --resume wandb://run_id/checkpoint:latest

Environment:
    WANDB_API_KEY       Required. Your Weights & Biases API key.
    ANTHROPIC_API_KEY   Optional. For auto-generating run notes via Claude.
    MACHINE_TYPE        Optional. GCP machine type (default: c3d-standard-16).
    ZONE                Optional. GCP zone (skips multi-zone search).
    SKIP_BUILD          Optional. Set to 1 to skip Docker build & push.
    MAX_HOURS           Optional. Auto-shutdown timeout in hours (default: 24).

The container runs on a COS (Container-Optimized OS) spot VM. When training
finishes, the VM shuts itself down (stopped VMs cost nothing).

Cost: ~$0.50-3.00 per 24h run depending on machine type and spot pricing.
"""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# US zones to try for spot VMs, roughly ordered by typical availability.
# When ZONE is not set explicitly, we try each in order until one works.
US_ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-east4-a",
    "us-east4-b",
    "us-east4-c",
    "us-east5-a",
    "us-east5-b",
    "us-east5-c",
    "us-west1-a",
    "us-west1-b",
    "us-west4-a",
    "us-west4-b",
    "us-south1-a",
    "us-south1-b",
]

# Use multi-region US Artifact Registry so the image is fast to pull from any US zone.
REGISTRY_LOCATION = "us"


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command."""
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def sanitize_label(s: str) -> str:
    """Sanitize a string for use as a GCP label value (lowercase, alphanumeric/hyphens, max 63 chars)."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s.lower())[:63]


def get_gcp_project() -> str:
    result = run(["gcloud", "config", "get-value", "project"], capture=True, check=False)
    return result.stdout.strip()


def get_git_branch(project_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def extract_bot_name(train_args: list[str]) -> str:
    """Extract --bot NAME from training args, default to simple2wheeler."""
    for i, arg in enumerate(train_args):
        if arg == "--bot" and i + 1 < len(train_args):
            return train_args[i + 1]
    return "simple2wheeler"


def build_docker_env_flags(instance: str) -> str:
    """Build the -e flags for docker run."""
    flags = f"-e MUJOCO_GL=egl -e PYTHONUNBUFFERED=1 -e RUN_NAME={instance}"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        flags += f" -e WANDB_API_KEY={wandb_key}"
    else:
        flags += " -e WANDB_MODE=disabled"
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        flags += f" -e ANTHROPIC_API_KEY={anthropic_key}"
    return flags


def generate_startup_script(
    image: str, docker_env: str, quoted_args: str, max_hours: int
) -> str:
    """Generate the COS VM startup script that pulls and runs the training container."""
    shutdown_minutes = max_hours * 60
    return f"""#!/bin/bash
set -euo pipefail
logger 'mindsim: startup script starting'

# Safety: auto-shutdown after {max_hours}h no matter what
shutdown -h +{shutdown_minutes}

# COS has a read-only root filesystem; point HOME to a writable dir
export HOME=/var/tmp

# Configure Docker to pull from Artifact Registry
docker-credential-gcr configure-docker --registries={REGISTRY_LOCATION}-docker.pkg.dev

# Pull and run the training container
logger 'mindsim: pulling image {image}:latest'
docker pull {image}:latest

logger 'mindsim: starting training container'
set +eo pipefail
docker run --rm {docker_env} {image}:latest {quoted_args} 2>&1 | logger -t mindsim-train
EXIT_CODE=${{PIPESTATUS[0]}}
set -eo pipefail

if [ $EXIT_CODE -eq 0 ]; then
    logger 'mindsim: === TRAINING COMPLETED SUCCESSFULLY ==='
else
    logger "mindsim: === TRAINING FAILED (exit code $EXIT_CODE) ==="
fi

shutdown -h now
"""


def try_create_instance(
    instance: str,
    machine_type: str,
    zone: str,
    labels: str,
    startup_file: str,
) -> tuple[bool, str]:
    """Try to create a spot VM in the given zone. Returns (success, reason)."""
    result = subprocess.run(
        [
            "gcloud", "compute", "instances", "create", instance,
            f"--machine-type={machine_type}",
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            f"--zone={zone}",
            "--boot-disk-size=30GB",
            "--image-family=cos-stable",
            "--image-project=cos-cloud",
            "--scopes=default,logging-write",
            f"--labels={labels}",
            f"--metadata-from-file=startup-script={startup_file}",
            "--quiet",
        ],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        return True, ""

    stderr = result.stderr.lower()

    # Capacity/availability errors → try next zone
    capacity_indicators = [
        "zone_resource_pool_exhausted",
        "stockout",
        "does not have enough resources",
        "unsupported machine type",
        "is not available in zone",
        "not found in zone",
        "resource pool exhausted",
    ]
    quota_indicators = ["quota"]

    for indicator in capacity_indicators:
        if indicator in stderr:
            return False, "no spot capacity"
    for indicator in quota_indicators:
        if indicator in stderr:
            return False, "quota exceeded"

    # Unknown error → print full stderr and bail
    print(f"FAILED\n{result.stderr}", file=sys.stderr)
    sys.exit(1)


def main():
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    machine_type = os.environ.get("MACHINE_TYPE", "c3d-standard-16")
    explicit_zone = os.environ.get("ZONE")
    max_hours = int(os.environ.get("MAX_HOURS", "24"))
    skip_build = os.environ.get("SKIP_BUILD") == "1"
    project_dir = Path(__file__).resolve().parent

    # -------------------------------------------------------------------------
    # Preflight checks
    # -------------------------------------------------------------------------
    for tool in ("gcloud", "docker"):
        if not shutil.which(tool):
            print(f"Error: {tool} not found.", file=sys.stderr)
            sys.exit(1)

    gcp_project = get_gcp_project()
    if not gcp_project:
        print(
            "Error: No GCP project configured. Run: gcloud config set project <PROJECT_ID>",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set. W&B logging will be disabled.", file=sys.stderr)

    train_args = sys.argv[1:]
    if not train_args:
        print(
            "Usage: python remote_train.py [main.py train args]\n"
            "  e.g. python remote_train.py --smoketest\n"
            "       python remote_train.py --bot simple2wheeler",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_name = "mindsim"
    image = f"{REGISTRY_LOCATION}-docker.pkg.dev/{gcp_project}/{repo_name}/mindsim-train"

    # -------------------------------------------------------------------------
    # Phase 1: Infrastructure (idempotent)
    # -------------------------------------------------------------------------
    print("==> Ensuring Artifact Registry repo exists...")
    result = run(
        ["gcloud", "artifacts", "repositories", "describe", repo_name,
         "--location", REGISTRY_LOCATION, "--format", "value(name)"],
        check=False, capture=True,
    )
    if result.returncode != 0:
        run([
            "gcloud", "artifacts", "repositories", "create", repo_name,
            "--repository-format=docker", f"--location={REGISTRY_LOCATION}",
            "--description=MindSim training images", "--quiet",
        ])
        print(f"    Created repo: {repo_name}")
    else:
        print(f"    Repo exists: {repo_name}")

    print("==> Configuring Docker auth for Artifact Registry...")
    run(["gcloud", "auth", "configure-docker", f"{REGISTRY_LOCATION}-docker.pkg.dev", "--quiet"])

    # -------------------------------------------------------------------------
    # Phase 2: Build & push Docker image
    # -------------------------------------------------------------------------
    if skip_build:
        print("==> Skipping Docker build (SKIP_BUILD=1)")
    else:
        print("==> Building Docker image...")
        run(["docker", "build", "--platform", "linux/amd64", "-t", f"{image}:latest", str(project_dir)])
        print("==> Pushing Docker image...")
        run(["docker", "push", f"{image}:latest"])

    # -------------------------------------------------------------------------
    # Phase 3: Create spot VM
    # -------------------------------------------------------------------------
    bot_name = extract_bot_name(train_args)
    instance = f"{bot_name}-{datetime.now().strftime('%m%d-%H%M')}"

    # Labels for tracking
    branch = get_git_branch(project_dir)
    labels = (
        f"mindsim=true,"
        f"mindsim-branch={sanitize_label(branch)},"
        f"mindsim-args={sanitize_label(' '.join(train_args))}"
    )

    docker_env = build_docker_env_flags(instance)
    quoted_args = " ".join(shlex.quote(a) for a in train_args)

    # Determine zones to try
    if explicit_zone:
        zones = [explicit_zone]
        print(f"==> Using specified zone: {explicit_zone}")
    else:
        zones = US_ZONES
        print(f"==> Will try {len(zones)} US zones for spot availability")

    # Write startup script to temp file
    startup_script = generate_startup_script(image, docker_env, quoted_args, max_hours)
    startup_fd, startup_file = tempfile.mkstemp(suffix=".sh")
    try:
        with os.fdopen(startup_fd, "w") as f:
            f.write(startup_script)

        # Try each zone until one works
        created_zone = None
        for zone in zones:
            print(f"    {zone}... ", end="", flush=True)
            success, reason = try_create_instance(
                instance, machine_type, zone, labels, startup_file,
            )
            if success:
                print("OK")
                created_zone = zone
                break
            else:
                print(reason)
    finally:
        os.unlink(startup_file)

    if not created_zone:
        print(
            f"\nError: Could not find spot capacity for {machine_type} in any US zone.",
            file=sys.stderr,
        )
        print("Try a different machine type or try again later.", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Success
    # -------------------------------------------------------------------------
    print(f"""
==========================================
  Training started on {instance}
==========================================

  Run:   {instance}
  Zone:  {created_zone}
  Image: {image}:latest
  Args:  {' '.join(train_args)}

  SSH into VM:
    gcloud compute ssh {instance} --zone={created_zone}

  Stream container logs (via SSH):
    gcloud compute ssh {instance} --zone={created_zone} -- 'docker logs -f $(docker ps -q)'

  Stream logs (Cloud Logging):
    gcloud logging read 'resource.type="gce_instance" jsonPayload.message=~"mindsim"' \\
      --freshness=1h --order=asc --format='value(jsonPayload.message)'

  VM self-stops when training finishes.
  Delete when done:
    gcloud compute instances delete {instance} --zone={created_zone} -q
""")


if __name__ == "__main__":
    main()
