#!/usr/bin/env python3
"""
Spin up a ComfyUI GPU instance on GCP for concept-to-3D generation.

Creates a spot GPU VM with ComfyUI + TRELLIS2 + FLUX pre-installed.
First boot takes ~10-15 min (downloads models). Subsequent starts are fast.

Usage:
    python comfyui/remote_comfyui.py              # Create a new instance
    python comfyui/remote_comfyui.py --list        # List running instances
    python comfyui/remote_comfyui.py --stop NAME   # Stop an instance
    python comfyui/remote_comfyui.py --delete NAME # Delete an instance

Environment:
    MACHINE_TYPE    GPU machine type (default: g2-standard-8, L4 24GB)
    ZONE            GCP zone (default: auto-search GPU zones)
    MAX_HOURS       Auto-shutdown timeout in hours (default: 8)

Once running:
    export COMFY_URL=http://<EXTERNAL_IP>:8188
    python comfyui/run.py table "dining table"

    Or open COMFY_URL in browser → drag in workflow_concept_to_3d_ui.json
    for visual debugging.

GPU options:
    g2-standard-8   L4 24GB   ~$0.80/hr spot  Two-pass pipeline required
    a2-highgpu-1g   A100 40GB ~$1.20/hr spot  Can run single-pass

The L4 can't hold FLUX (~17GB) and TRELLIS2 (~10-13GB) simultaneously,
so run.py uses a two-pass pipeline. An A100 could run single-pass but
the workflow_concept_to_3d.json hasn't been tested on it yet.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

# GPU zones to try (L4/g2 instances have broad availability in these).
GPU_ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-d",
    "us-east4-a",
    "us-east4-b",
    "us-east4-c",
    "us-west1-a",
    "us-west1-b",
    "us-west4-a",
    "us-west4-b",
]

FIREWALL_RULE = "allow-comfyui-8188"


def run(
    cmd: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def get_gcp_project() -> str:
    result = run(
        ["gcloud", "config", "get-value", "project"], capture=True, check=False
    )
    return result.stdout.strip()


def ensure_firewall_rule(project: str):
    """Create firewall rule to allow port 8188 if it doesn't exist."""
    result = run(
        [
            "gcloud",
            "compute",
            "firewall-rules",
            "describe",
            FIREWALL_RULE,
            "--project",
            project,
            "--format",
            "value(name)",
        ],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        print(f"  Creating firewall rule {FIREWALL_RULE}...")
        run(
            [
                "gcloud",
                "compute",
                "firewall-rules",
                "create",
                FIREWALL_RULE,
                "--project",
                project,
                "--allow",
                "tcp:8188",
                "--target-tags",
                "comfyui",
                "--description",
                "Allow ComfyUI web UI access on port 8188",
                "--quiet",
            ]
        )
    else:
        print(f"  Firewall rule exists: {FIREWALL_RULE}")


def generate_startup_script(max_hours: int) -> str:
    """Startup script that installs ComfyUI + TRELLIS2 + FLUX on a Deep Learning VM."""
    shutdown_minutes = max_hours * 60
    return f"""#!/bin/bash
set -euo pipefail

# Safety: auto-shutdown after {max_hours}h
shutdown -h +{shutdown_minutes}

LOG=/var/log/comfyui-setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== ComfyUI setup starting at $(date) ==="

# Deep Learning VMs have CUDA + Python pre-installed.
# Install ComfyUI into /opt/comfyui.
INSTALL_DIR=/opt/comfyui

if [ ! -d "$INSTALL_DIR/ComfyUI" ]; then
    echo "=== Installing ComfyUI ==="
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # Install system deps
    apt-get update -qq
    apt-get install -y -qq git python3-venv python3-pip

    # Clone ComfyUI
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI

    # Create venv and install
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

    # Install ComfyUI Manager (for visual node installation)
    cd custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ..

    echo "=== Installing TRELLIS2 custom nodes ==="
    cd custom_nodes
    git clone https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git
    cd ComfyUI-TRELLIS2
    ../../venv/bin/pip install -r requirements.txt || true
    python ../../venv/bin/python install.py || true
    cd ../..

    echo "=== Downloading FLUX models ==="
    mkdir -p models/unet models/clip models/vae

    # FLUX.1-dev UNet (~24GB) — use fp8 quantized version for faster download + less VRAM
    if [ ! -f models/unet/flux1-dev-fp8.safetensors ]; then
        echo "Downloading FLUX.1-dev fp8..."
        cd models/unet
        wget -q --show-progress "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" || true
        cd ../..
    fi

    # CLIP models
    if [ ! -f models/clip/clip_l.safetensors ]; then
        echo "Downloading CLIP-L..."
        cd models/clip
        wget -q --show-progress "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" || true
        cd ../..
    fi

    if [ ! -f models/clip/t5xxl_fp16.safetensors ]; then
        echo "Downloading T5-XXL fp16..."
        cd models/clip
        wget -q --show-progress "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" || true
        cd ../..
    fi

    # FLUX VAE
    if [ ! -f models/vae/ae.safetensors ]; then
        echo "Downloading FLUX VAE..."
        cd models/vae
        wget -q --show-progress "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" || true
        cd ../..
    fi

    deactivate
    echo "=== Installation complete ==="
else
    echo "=== ComfyUI already installed, skipping ==="
fi

# Install Google Cloud Ops Agent for GPU metrics in GCP console
if ! systemctl is-active --quiet google-cloud-ops-agent; then
    echo "=== Installing Ops Agent for GPU monitoring ==="
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    bash add-google-cloud-ops-agent-repo.sh --also-install
    rm -f add-google-cloud-ops-agent-repo.sh
fi

# Start ComfyUI server
echo "=== Starting ComfyUI server ==="
cd "$INSTALL_DIR/ComfyUI"
source venv/bin/activate

# --lowvram: offload models to CPU between nodes (FLUX 17GB + TRELLIS2 don't fit in 24GB together)
exec python main.py --listen 0.0.0.0 --port 8188 --lowvram
"""


def try_create_instance(
    instance: str,
    machine_type: str,
    zone: str,
    project: str,
    startup_file: str,
    accelerator: str,
) -> tuple[bool, str]:
    """Try to create a GPU spot VM. Returns (success, reason)."""
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance,
            f"--project={project}",
            f"--machine-type={machine_type}",
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            f"--zone={zone}",
            "--boot-disk-size=100GB",
            "--boot-disk-type=pd-ssd",
            "--image-family=common-cu128-ubuntu-2204-nvidia-570",
            "--image-project=deeplearning-platform-release",
            f"--accelerator=type={accelerator},count=1",
            "--maintenance-policy=TERMINATE",
            "--scopes=default,logging-write",
            "--tags=comfyui",
            "--labels=mindsim=true,mindsim-type=comfyui",
            f"--metadata-from-file=startup-script={startup_file}",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return True, ""

    stderr = result.stderr.lower()
    capacity_indicators = [
        "zone_resource_pool_exhausted",
        "stockout",
        "does not have enough resources",
        "unsupported machine type",
        "is not available",
        "resource pool exhausted",
    ]
    for indicator in capacity_indicators:
        if indicator in stderr:
            return False, "no spot capacity"
    if "quota" in stderr:
        return False, "quota exceeded"

    print(f"FAILED\n{result.stderr}", file=sys.stderr)
    sys.exit(1)


def get_instance_ip(instance: str, zone: str, project: str) -> str:
    """Get the external IP of a running instance."""
    result = run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            instance,
            f"--zone={zone}",
            f"--project={project}",
            "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
        ],
        capture=True,
    )
    return result.stdout.strip()


def list_instances(project: str):
    """List all mindsim ComfyUI instances."""
    result = run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            "--filter=labels.mindsim-type=comfyui",
            "--format=json",
        ],
        capture=True,
    )
    instances = json.loads(result.stdout) if result.stdout.strip() else []
    if not instances:
        print("No ComfyUI instances found.")
        return

    print(
        f"{'NAME':<30} {'ZONE':<22} {'STATUS':<12} {'MACHINE':<18} {'EXTERNAL_IP':<16}"
    )
    print("-" * 100)
    for inst in instances:
        name = inst["name"]
        zone = inst["zone"].split("/")[-1]
        status = inst["status"]
        machine = inst["machineType"].split("/")[-1]
        ip = ""
        for iface in inst.get("networkInterfaces", []):
            for ac in iface.get("accessConfigs", []):
                ip = ac.get("natIP", "")
        url = f"http://{ip}:8188" if ip and status == "RUNNING" else ""
        print(f"{name:<30} {zone:<22} {status:<12} {machine:<18} {url}")


def stop_instance(name: str, project: str):
    """Stop a ComfyUI instance (keeps disk, stops billing for compute)."""
    # Find the zone
    result = run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--filter=name={name}",
            "--format=value(zone)",
        ],
        capture=True,
    )
    zone = result.stdout.strip().split("/")[-1]
    if not zone:
        print(f"Instance not found: {name}", file=sys.stderr)
        sys.exit(1)
    print(f"Stopping {name} in {zone}...")
    run(
        [
            "gcloud",
            "compute",
            "instances",
            "stop",
            name,
            f"--zone={zone}",
            f"--project={project}",
            "--quiet",
        ]
    )
    print(f"Stopped. Restart with: gcloud compute instances start {name} --zone={zone}")


def delete_instance(name: str, project: str):
    """Delete a ComfyUI instance."""
    result = run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--filter=name={name}",
            "--format=value(zone)",
        ],
        capture=True,
    )
    zone = result.stdout.strip().split("/")[-1]
    if not zone:
        print(f"Instance not found: {name}", file=sys.stderr)
        sys.exit(1)
    print(f"Deleting {name} in {zone}...")
    run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            name,
            f"--zone={zone}",
            f"--project={project}",
            "--quiet",
        ]
    )
    print("Deleted.")


# ---------------------------------------------------------------------------
# GPU machine type → accelerator mapping
# ---------------------------------------------------------------------------

ACCELERATOR_MAP = {
    "g2-standard-8": "nvidia-l4",
    "g2-standard-16": "nvidia-l4",
    "g2-standard-32": "nvidia-l4",
    "a2-highgpu-1g": "nvidia-tesla-a100",
    "a2-ultragpu-1g": "nvidia-a100-80gb",
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage ComfyUI GPU instances on GCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="List running instances")
    parser.add_argument("--stop", metavar="NAME", help="Stop an instance")
    parser.add_argument("--delete", metavar="NAME", help="Delete an instance")
    args = parser.parse_args()

    if not shutil.which("gcloud"):
        print(
            "Error: gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install",
            file=sys.stderr,
        )
        sys.exit(1)

    project = get_gcp_project()
    if not project:
        print(
            "Error: No GCP project. Run: gcloud config set project <PROJECT_ID>",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list:
        list_instances(project)
        return
    if args.stop:
        stop_instance(args.stop, project)
        return
    if args.delete:
        delete_instance(args.delete, project)
        return

    # --- Create new instance ---
    machine_type = os.environ.get("MACHINE_TYPE", "g2-standard-8")
    explicit_zone = os.environ.get("ZONE")
    max_hours = int(os.environ.get("MAX_HOURS", "8"))

    accelerator = ACCELERATOR_MAP.get(machine_type)
    if not accelerator:
        print(f"Unknown machine type: {machine_type}", file=sys.stderr)
        print(f"Supported: {', '.join(ACCELERATOR_MAP)}", file=sys.stderr)
        sys.exit(1)

    instance = f"comfyui-{datetime.now().strftime('%m%d-%H%M')}"

    print(f"==> Creating ComfyUI instance: {instance}")
    print(f"    Machine: {machine_type} ({accelerator})")
    print(f"    Auto-shutdown: {max_hours}h")

    # Ensure firewall rule exists
    ensure_firewall_rule(project)

    # Write startup script
    startup_script = generate_startup_script(max_hours)
    fd, startup_file = tempfile.mkstemp(suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(startup_script)

        zones = [explicit_zone] if explicit_zone else GPU_ZONES
        if not explicit_zone:
            print(f"==> Searching {len(zones)} zones for GPU spot capacity...")

        created_zone = None
        for zone in zones:
            print(f"    {zone}... ", end="", flush=True)
            success, reason = try_create_instance(
                instance,
                machine_type,
                zone,
                project,
                startup_file,
                accelerator,
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
            f"\nError: No spot GPU capacity for {machine_type} in any zone.",
            file=sys.stderr,
        )
        print(
            "Try: MACHINE_TYPE=a2-highgpu-1g python comfyui/remote_comfyui.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get external IP
    print("==> Waiting for external IP...")
    import time

    ip = ""
    for _ in range(30):
        ip = get_instance_ip(instance, created_zone, project)
        if ip:
            break
        time.sleep(2)

    if not ip:
        print("Warning: Could not get external IP. Check GCP console.", file=sys.stderr)

    print(f"""
==========================================
  ComfyUI instance created: {instance}
==========================================

  Zone:     {created_zone}
  Machine:  {machine_type} ({accelerator})
  IP:       {ip or "(pending)"}

  The startup script is installing ComfyUI + TRELLIS2 + FLUX models.
  This takes ~10-15 min on first boot. Monitor progress:

    gcloud compute ssh {instance} --zone={created_zone} -- tail -f /var/log/comfyui-setup.log

  Once ready:

    Visual editor:  http://{ip}:8188
    API access:     export COMFY_URL=http://{ip}:8188
                    python comfyui/run.py table "dining table"

  Management:

    python comfyui/remote_comfyui.py --list
    python comfyui/remote_comfyui.py --stop {instance}
    python comfyui/remote_comfyui.py --delete {instance}

    SSH:  gcloud compute ssh {instance} --zone={created_zone}

  Cost: ~$0.80/hr spot. Auto-shuts down after {max_hours}h.
""")


if __name__ == "__main__":
    main()
