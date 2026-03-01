"""Submit concept-to-3D workflows to a ComfyUI instance and download results.

Uses a two-pass pipeline to fit within 24GB VRAM (L4 GPU):
  Pass 1: FLUX text → image (renders a concept product shot)
  Pass 2: TRELLIS2 image → 3D (converts the render to a GLB mesh)

FLUX (~17GB) and TRELLIS2 (~10-13GB) can't coexist in 24GB VRAM, so they
run as separate job submissions. ComfyUI unloads pass 1 models before
pass 2 loads.

Setup (GCP — recommended):
    python comfyui/remote_comfyui.py           # Spin up a GPU instance
    export COMFY_URL=http://<EXTERNAL_IP>:8188  # Set the URL

Setup (local ComfyUI):
    export COMFY_URL=http://127.0.0.1:8188

Usage:
    # Run a single concept → 3D model (~6 min on L4 with --lowvram)
    python comfyui/run.py table "dining table"

    # Run with custom seed
    python comfyui/run.py chair "armchair" --seed 123

    # Use the image-to-3D workflow with an existing image
    python comfyui/run.py --from-image path/to/render.png

    # Check status of a running job
    python comfyui/run.py --status <prompt_id>

    # Dry run: print the workflow JSON without submitting
    python comfyui/run.py table "coffee table" --dry-run

Output goes to comfyui/output/<concept>_<variation>_<seed>/

Known issues:
    - BiRefNet (background removal) has a PyTorch inference_mode bug that
      causes "Inference tensors do not track version counter" on the second
      run. Workaround: patched nodes_inference.py on the server to force-
      reload the model each time (see session notes).
    - With --lowvram, model loading makes the HTTP API unresponsive for
      1-2 min. The client retries up to 30 times with 120s timeouts.
    - GLB files from Trellis2ExportGLB don't appear in ComfyUI's history
      API. The client falls back to SSH/SCP to download them from remote
      servers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import namedtuple
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests", file=sys.stderr)
    sys.exit(1)

# Add project root for scene_gen imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_URL = "http://127.0.0.1:8188"
POLL_INTERVAL = 3  # seconds
WORKFLOW_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# ComfyUI API client (works with any ComfyUI instance)
# ---------------------------------------------------------------------------


class ComfyClient:
    """Thin client for the standard ComfyUI HTTP API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex[:8]

    def submit(self, workflow: dict) -> str:
        """Queue a workflow for execution. Returns prompt_id."""
        resp = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        """Get execution history for a prompt. Returns None if not found."""
        resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get(prompt_id)

    def wait(self, prompt_id: str) -> dict:
        """Poll until the prompt finishes. Returns the history entry."""
        print(f"  Job queued: {prompt_id}")
        retries = 0
        while True:
            try:
                entry = self.history(prompt_id)
                retries = 0  # reset on success
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > 30:
                    raise
                print(f"  (poll retry {retries}/30: {e.__class__.__name__})")
                time.sleep(POLL_INTERVAL * 2)
                continue
            if entry and entry.get("status", {}).get("completed", False):
                print("  Status: completed")
                return entry
            if entry and entry.get("status", {}).get("status_str") == "error":
                print("  Status: error")
                messages = entry.get("status", {}).get("messages", [])
                for msg in messages:
                    if isinstance(msg, list) and len(msg) >= 2:
                        print(f"    {msg[0]}: {msg[1]}")
                return entry
            time.sleep(POLL_INTERVAL)

    def upload_image(self, local_path: Path) -> str:
        """Upload an image to ComfyUI's input directory. Returns the server filename."""
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (local_path.name, f, "image/png")},
                data={"type": "input"},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["name"]

    def download_file(
        self, filename: str, subfolder: str, filetype: str, dest: Path
    ) -> Path:
        """Download an output file."""
        resp = requests.get(
            f"{self.base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": filetype},
            timeout=120,
        )
        resp.raise_for_status()
        out_path = dest / filename
        out_path.write_bytes(resp.content)
        return out_path

    def download_outputs(
        self, prompt_id: str, output_dir: Path, glb_prefix: str | None = None
    ) -> list[Path]:
        """Download all output files from a completed job."""
        output_dir.mkdir(parents=True, exist_ok=True)
        entry = self.history(prompt_id)
        if not entry:
            print("  No history found for this job.", file=sys.stderr)
            return []

        outputs = entry.get("outputs", {})
        downloaded = []

        for _node_id, node_out in outputs.items():
            # Standard output categories (images, video, etc.)
            for category in ("images", "video", "audio", "gltf", "meshes"):
                for item in node_out.get(category, []):
                    fname = item.get("filename", "")
                    subfolder = item.get("subfolder", "")
                    ftype = item.get("type", "output")
                    if fname:
                        print(f"  Downloading: {fname}")
                        path = self.download_file(fname, subfolder, ftype, output_dir)
                        downloaded.append(path)

            # Text outputs may contain GLB file paths (Trellis2ExportGLB)
            for item in node_out.get("text", []):
                if isinstance(item, str) and item.endswith(".glb"):
                    fname = Path(item).name
                    print(f"  Downloading GLB: {fname}")
                    path = self.download_file(fname, "", "output", output_dir)
                    downloaded.append(path)

        # Trellis2ExportGLB saves to output/ but doesn't register in history.
        # Use SSH to list and download GLB files from remote servers.
        if glb_prefix and not any(str(p).endswith(".glb") for p in downloaded):
            self._download_glb_via_ssh(glb_prefix, output_dir, downloaded)

        return downloaded

    def _download_glb_via_ssh(
        self, prefix: str, output_dir: Path, downloaded: list[Path]
    ):
        """Find and download GLB files from the remote ComfyUI output directory via SSH."""
        import subprocess
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        host = parsed.hostname
        if not host or host in ("127.0.0.1", "localhost"):
            return

        ssh_key = Path.home() / ".ssh" / "google_compute_engine"
        ssh_opts = [
            "-i",
            str(ssh_key),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
        ]

        # List GLB files matching prefix on the remote server
        try:
            result = subprocess.run(
                [
                    "ssh",
                    *ssh_opts,
                    f"wkm@{host}",
                    f"ls -t /opt/comfyui/ComfyUI/output/{prefix}*.glb 2>/dev/null | head -1",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            remote_path = result.stdout.strip()
            if not remote_path:
                return

            fname = Path(remote_path).name
            local_path = output_dir / fname
            print(f"  Downloading GLB via SCP: {fname}")
            subprocess.run(
                ["scp", *ssh_opts, f"wkm@{host}:{remote_path}", str(local_path)],
                capture_output=True,
                timeout=120,
            )
            if local_path.exists():
                downloaded.append(local_path)
        except Exception as e:
            print(f"  Warning: GLB download via SSH failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Workflow loading + prompt injection
# ---------------------------------------------------------------------------


def load_workflow(name: str) -> dict:
    path = WORKFLOW_DIR / f"{name}.json"
    with open(path) as f:
        wf = json.load(f)
    wf.pop("_comment", None)
    return wf


def make_flux_render_workflow(prompt: str, seed: int = 42) -> dict:
    """FLUX text → image (pass 1 of two-pass pipeline)."""
    wf = load_workflow("workflow_flux_render")
    wf["3"]["inputs"]["text"] = prompt
    wf["5"]["inputs"]["noise_seed"] = seed
    return wf


def make_text_to_3d_workflow(prompt: str, seed: int = 42) -> dict:
    """Full pipeline: FLUX text → image → TRELLIS2 → GLB (single-pass, needs >24GB VRAM)."""
    wf = load_workflow("workflow_concept_to_3d")
    wf["3"]["inputs"]["text"] = prompt
    wf["5"]["inputs"]["noise_seed"] = seed
    wf["25"]["inputs"]["seed"] = seed
    wf["26"]["inputs"]["seed"] = seed
    return wf


def make_image_to_3d_workflow(image_filename: str, seed: int = 42) -> dict:
    """Image → TRELLIS2 → GLB (pass 2 of two-pass pipeline, or standalone)."""
    wf = load_workflow("workflow_image_to_3d")
    wf["1"]["inputs"]["image"] = image_filename
    wf["15"]["inputs"]["seed"] = seed
    wf["16"]["inputs"]["seed"] = seed
    return wf


# ---------------------------------------------------------------------------
# Resolve ComfyUI URL
# ---------------------------------------------------------------------------


def get_comfy_url(cli_url: str | None = None) -> str:
    if cli_url:
        return cli_url
    if url := os.environ.get("COMFY_URL"):
        return url
    # Check dotfile
    urlfile = Path.home() / ".comfy_url"
    if urlfile.exists():
        return urlfile.read_text().strip()
    return DEFAULT_URL


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

BatchJob = namedtuple(
    "BatchJob", ["concept", "variation", "slug", "prompt", "seed", "out_dir"]
)


def build_batch_jobs(
    concept_names: list[str] | None,
    all_variations: bool,
    seed: int,
    force: bool,
) -> tuple[list[BatchJob], list[BatchJob]]:
    """Build lists of batch jobs, split by what work remains.

    Returns (flux_jobs, trellis_jobs):
      - flux_jobs: need FLUX render (phase 1) then TRELLIS2 (phase 2)
      - trellis_jobs: already have PNG, only need TRELLIS2 (phase 2)
    """
    from comfyui.concept_prompt import concept_prompt
    from scene_gen.concepts import all_concepts

    concepts = all_concepts()

    # Filter to requested concepts or all
    if concept_names:
        missing = [c for c in concept_names if c not in concepts]
        if missing:
            print(f"Unknown concepts: {', '.join(missing)}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(concepts))}", file=sys.stderr)
            sys.exit(1)
        names = concept_names
    else:
        names = sorted(concepts)

    flux_jobs = []
    trellis_jobs = []

    for name in names:
        mod = concepts[name]
        variations = getattr(mod, "VARIATIONS", {})

        if all_variations and variations:
            items = [(vname, variations[vname]) for vname in sorted(variations)]
        elif variations:
            # First variation only
            first_key = sorted(variations)[0]
            items = [(first_key, variations[first_key])]
        else:
            items = [("default", mod.Params())]

        for vname, params in items:
            slug = vname.replace(" ", "_").replace("/", "")
            out_dir = WORKFLOW_DIR / "output" / f"{name}_{slug}_{seed}"
            prompt = concept_prompt(name, params)

            if not force and out_dir.exists():
                has_glb = any(out_dir.glob("*.glb"))
                has_png = any(out_dir.glob("*.png"))
                if has_glb:
                    continue  # fully complete
                if has_png:
                    trellis_jobs.append(
                        BatchJob(name, vname, slug, prompt, seed, out_dir)
                    )
                    continue  # skip phase 1

            flux_jobs.append(BatchJob(name, vname, slug, prompt, seed, out_dir))

    return flux_jobs, trellis_jobs


def run_batch(
    client: ComfyClient,
    flux_jobs: list[BatchJob],
    trellis_jobs: list[BatchJob],
    dry_run: bool = False,
):
    """Run the two-phase batch pipeline: all FLUX renders, then all TRELLIS2 conversions."""
    total = len(flux_jobs) + len(trellis_jobs)
    if total == 0:
        print("Nothing to do — all outputs already exist. Use --force to regenerate.")
        return

    print(
        f"Batch: {len(flux_jobs)} need FLUX+TRELLIS2, {len(trellis_jobs)} need TRELLIS2 only"
    )
    print(f"Total jobs: {total}")
    print()

    if dry_run:
        print("=== Phase 1: FLUX renders ===")
        for job in flux_jobs:
            print(f"  {job.concept}/{job.variation} → {job.out_dir}")
            print(f"    Prompt: {job.prompt[:80]}...")
        print()
        print("=== Phase 2: TRELLIS2 3D ===")
        for job in flux_jobs + trellis_jobs:
            print(f"  {job.concept}/{job.variation} → {job.out_dir}")
        return

    succeeded = []
    failed = []

    # Phase 1: FLUX renders
    if flux_jobs:
        print(f"=== Phase 1: FLUX renders ({len(flux_jobs)} jobs) ===")
        print()

    for i, job in enumerate(flux_jobs, 1):
        print(f"[{i}/{len(flux_jobs)}] FLUX: {job.concept}/{job.variation}")
        print(f"  Prompt: {job.prompt[:80]}...")
        try:
            wf = make_flux_render_workflow(job.prompt, seed=job.seed)
            prompt_id = client.submit(wf)
            entry = client.wait(prompt_id)

            if not entry.get("status", {}).get("completed"):
                print("  FAILED: FLUX render did not complete")
                failed.append((job, "FLUX render failed"))
                continue

            job.out_dir.mkdir(parents=True, exist_ok=True)
            render_files = client.download_outputs(prompt_id, job.out_dir)
            png_files = [f for f in render_files if str(f).endswith(".png")]
            if not png_files:
                print("  FAILED: No PNG in output")
                failed.append((job, "No PNG produced"))
                continue

            print(f"  OK: {png_files[0].name}")
            # This job now needs TRELLIS2 in phase 2
            trellis_jobs.append(job)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((job, str(e)))

    print()

    # Phase 2: TRELLIS2 conversions
    if trellis_jobs:
        print(f"=== Phase 2: TRELLIS2 3D ({len(trellis_jobs)} jobs) ===")
        print()

    for i, job in enumerate(trellis_jobs, 1):
        print(f"[{i}/{len(trellis_jobs)}] TRELLIS2: {job.concept}/{job.variation}")
        try:
            # Find the PNG in the output dir
            pngs = sorted(job.out_dir.glob("*.png"))
            if not pngs:
                print(f"  FAILED: No PNG found in {job.out_dir}")
                failed.append((job, "No PNG for TRELLIS2"))
                continue

            local_png = pngs[0]
            server_name = client.upload_image(local_png)
            print(f"  Uploaded: {server_name}")

            wf = make_image_to_3d_workflow(server_name, seed=job.seed)
            prompt_id = client.submit(wf)
            entry = client.wait(prompt_id)

            if not entry.get("status", {}).get("completed"):
                print("  FAILED: TRELLIS2 did not complete")
                failed.append((job, "TRELLIS2 conversion failed"))
                continue

            files = client.download_outputs(
                prompt_id, job.out_dir, glb_prefix="concept_3d"
            )
            glb_files = [f for f in files if str(f).endswith(".glb")]
            if glb_files:
                print(f"  OK: {glb_files[0].name}")
            else:
                print(f"  OK (no GLB in download list — check {job.out_dir})")
            succeeded.append(job)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((job, str(e)))

    # Summary
    print()
    print("=" * 60)
    print(f"Batch complete: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print()
        print("Failures:")
        for job, reason in failed:
            print(f"  {job.concept}/{job.variation}: {reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run concept-to-3D pipeline on ComfyUI (RunPod, local, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("concept", nargs="*", help="Concept name(s) (e.g. table chair)")
    parser.add_argument(
        "--variation",
        help="Variation name (e.g. 'dining table') — single-concept mode only",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--url", help="ComfyUI URL (or set COMFY_URL env var)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print workflow JSON, don't submit"
    )
    parser.add_argument(
        "--status", metavar="PROMPT_ID", help="Check status of existing job"
    )
    parser.add_argument(
        "--from-image",
        metavar="PATH",
        help="Use image-to-3D workflow with existing image",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: run all FLUX first, then all TRELLIS2",
    )
    parser.add_argument(
        "--all-variations",
        action="store_true",
        help="Process all variations (batch mode)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if output exists (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: comfyui/output/<concept>/)",
    )

    args = parser.parse_args()
    comfy_url = get_comfy_url(args.url)

    # Status check mode
    if args.status:
        client = ComfyClient(comfy_url)
        entry = client.history(args.status)
        if entry:
            status = entry.get("status", {})
            print(json.dumps(status, indent=2))
        else:
            print(f"No history found for {args.status}")
        return

    # Batch mode
    if args.batch:
        concept_names = args.concept if args.concept else None
        flux_jobs, trellis_jobs = build_batch_jobs(
            concept_names,
            args.all_variations,
            args.seed,
            args.force,
        )
        if args.dry_run:
            run_batch(None, flux_jobs, trellis_jobs, dry_run=True)
        else:
            client = ComfyClient(comfy_url)
            print(f"Batch mode on {comfy_url}")
            print()
            run_batch(client, flux_jobs, trellis_jobs)
        return

    # Image-to-3D mode
    if args.from_image:
        image_path = Path(args.from_image)
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        workflow = make_image_to_3d_workflow(image_path.name, seed=args.seed)
        out_dir = args.output_dir or WORKFLOW_DIR / "output" / image_path.stem
        if args.dry_run:
            json.dump(workflow, sys.stdout, indent=2)
            print()
            return
        client = ComfyClient(comfy_url)
        print(f"Running image-to-3D on {comfy_url}")
        print(f"  Image: {image_path.name}")
        prompt_id = client.submit(workflow)
        entry = client.wait(prompt_id)
        if entry.get("status", {}).get("completed"):
            files = client.download_outputs(prompt_id, out_dir)
            print(f"\nDone! {len(files)} file(s) in {out_dir}/")
        else:
            print("Job failed.", file=sys.stderr)
            sys.exit(1)
        return

    # Text-to-3D mode (single concept)
    # concept is now a list; support: `run.py table "dining table"` (positional variation)
    # or `run.py table --variation "dining table"`
    if not args.concept:
        parser.print_help()
        return

    # Parse concept + optional positional variation for backwards compat
    concept_name = args.concept[0]
    variation_name = args.variation
    if not variation_name and len(args.concept) > 1:
        variation_name = " ".join(args.concept[1:])

    from comfyui.concept_prompt import concept_prompt
    from scene_gen.concepts import all_concepts

    concepts = all_concepts()
    if concept_name not in concepts:
        print(f"Unknown concept: {concept_name}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(concepts))}", file=sys.stderr)
        sys.exit(1)

    mod = concepts[concept_name]
    variations = getattr(mod, "VARIATIONS", {})

    if variation_name:
        if variation_name not in variations:
            print(f"Unknown variation: {variation_name}", file=sys.stderr)
            sys.exit(1)
        params = variations[variation_name]
        slug = variation_name.replace(" ", "_").replace("/", "")
    else:
        params = mod.Params()
        slug = "default"

    prompt = concept_prompt(concept_name, params)
    out_dir = (
        args.output_dir
        or WORKFLOW_DIR / "output" / f"{concept_name}_{slug}_{args.seed}"
    )

    if args.dry_run:
        print(f"# Prompt: {prompt}\n")
        json.dump(
            make_flux_render_workflow(prompt, seed=args.seed), sys.stdout, indent=2
        )
        print()
        return

    client = ComfyClient(comfy_url)
    print(f"Running text-to-3D pipeline on {comfy_url}")
    print(f"  Concept: {concept_name} / {variation_name or 'default'}")
    print(f"  Seed: {args.seed}")
    print(f"  Prompt: {prompt[:80]}...")
    print()

    # Two-pass pipeline: FLUX and TRELLIS2 don't fit in VRAM together on 24GB GPUs.
    # Pass 1: FLUX renders the concept image
    print("  Pass 1/2: FLUX text → image")
    flux_wf = make_flux_render_workflow(prompt, seed=args.seed)
    prompt_id = client.submit(flux_wf)
    entry = client.wait(prompt_id)

    if not entry.get("status", {}).get("completed"):
        print("FLUX render failed.", file=sys.stderr)
        sys.exit(1)

    # Download the rendered image and find its filename on the server
    out_dir.mkdir(parents=True, exist_ok=True)
    render_files = client.download_outputs(prompt_id, out_dir)
    png_files = [f for f in render_files if str(f).endswith(".png")]
    if not png_files:
        print("No rendered image found.", file=sys.stderr)
        sys.exit(1)

    # Upload the rendered image to ComfyUI's input/ directory for pass 2
    local_png = png_files[0]
    server_image_name = client.upload_image(local_png)

    # Pass 2: TRELLIS2 converts image → 3D (FLUX is now unloaded from VRAM)
    print(f"  Pass 2/2: TRELLIS2 image → 3D ({server_image_name})")
    t3d_wf = make_image_to_3d_workflow(server_image_name, seed=args.seed)
    prompt_id = client.submit(t3d_wf)
    entry = client.wait(prompt_id)

    if entry.get("status", {}).get("completed"):
        files = client.download_outputs(prompt_id, out_dir, glb_prefix="concept_3d")
        all_files = png_files + files
        print(f"\nDone! {len(all_files)} file(s) in {out_dir}/")
        for f in all_files:
            print(f"  {f}")
    else:
        print("TRELLIS2 3D conversion failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
