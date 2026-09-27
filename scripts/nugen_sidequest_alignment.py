"""Create, inspect, and deploy SideQuest's Nugen domain alignment.

Only the synthetic corpus in backend/nugen_sidequest_alignment.md is uploaded.
The script stores provider IDs under ignored data/app, never in frontend code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from backend import nugen  # noqa: E402

STATE_PATH = ROOT / "data" / "app" / "nugen_sidequest_digital_twin.json"
CORPUS_PATH = ROOT / "backend" / "nugen_digital_twin_training.md"
ALIGNMENT_NAME = "SideQuest Weather Impact Digital Twin"
BASE_MODEL_ID = "qwen2-vl-2b-instruct"


def read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def set_local_model_id(model_id: str) -> None:
    """Persist the non-secret deployment identifier to the ignored local env file."""
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated = False
    for index, line in enumerate(lines):
        if line.strip().startswith("NUGEN_ALIGNED_MODEL_ID="):
            lines[index] = f'NUGEN_ALIGNED_MODEL_ID="{model_id}"'
            updated = True
            break
    if not updated:
        lines.append(f'NUGEN_ALIGNED_MODEL_ID="{model_id}"')
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def start() -> dict:
    state = read_state()
    aligned, projects, base_models = nugen._provider_state()
    existing = next((p for p in projects if p.get("alignment_name") == ALIGNMENT_NAME), None)
    if existing and existing.get("alignment_id"):
        state["alignment_id"] = existing["alignment_id"]
        save_state(state)
        return existing

    eligible = next((m for m in base_models if m.get("model_id") == BASE_MODEL_ID and m.get("alignment_ready") is True), None)
    if not eligible:
        raise nugen.NugenError("Nugen no longer marks the selected SideQuest base model as alignment-ready.")

    document_id = state.get("document_id")
    if not document_id:
        with CORPUS_PATH.open("rb") as corpus:
            uploaded = nugen._request(
                "POST",
                "/api/v3/documents/create",
                files={"files": (CORPUS_PATH.name, corpus, "text/markdown")},
                data={"names": "SideQuest synthetic weather impact examples", "categories": "sidequest,weather_impact"},
            )
        document_ids = uploaded.get("document_ids", []) if isinstance(uploaded, dict) else []
        if not document_ids or not isinstance(document_ids[0], str):
            raise nugen.NugenError("Nugen accepted no SideQuest training document.")
        document_id = document_ids[0]
        state["document_id"] = document_id
        save_state(state)

    created = nugen._request("POST", "/api/v3/alignment-projects/create", json_body={
        "alignment_name": ALIGNMENT_NAME,
        "base_model_id": BASE_MODEL_ID,
        "document_ids": [document_id],
        "description": "Synthetic SideQuest travel-impact examples for weather and itinerary what-if guidance. No traveller records included.",
    })
    alignment_id = created.get("alignment_id") if isinstance(created, dict) else None
    if not isinstance(alignment_id, str) or not alignment_id:
        raise nugen.NugenError("Nugen did not return an alignment project identifier.")
    state["alignment_id"] = alignment_id
    save_state(state)
    return created


def poll_and_deploy(wait_minutes: int) -> None:
    state = read_state()
    alignment_id = state.get("alignment_id")
    if not alignment_id:
        raise nugen.NugenError("No SideQuest alignment job is saved. Run with --start first.")
    deadline = time.monotonic() + max(wait_minutes, 0) * 60
    while True:
        project = nugen._request("GET", f"/api/v3/alignment-projects/{alignment_id}")
        status = str(project.get("status", "unknown")).upper() if isinstance(project, dict) else "unknown"
        print(f"Nugen alignment status: {status}")
        if status == "READY":
            model_id = project.get("model_id")
            if not model_id:
                models = nugen._list(nugen._request("GET", "/api/v3/models/aligned"), "domain_aligned_models")
                model = next((m for m in models if m.get("alignment_id") == alignment_id), None)
                model_id = model.get("model_id") if model else None
            if not model_id:
                raise nugen.NugenError("Training completed, but Nugen did not expose the aligned model ID.")
            models = nugen._list(nugen._request("GET", "/api/v3/models/aligned"), "domain_aligned_models")
            deployed = next((m for m in models if m.get("model_id") == model_id), None)
            deployment = nugen._request("GET", f"/api/v3/models/{model_id}/deployment/status")
            deployment_status = str(deployment.get("status", (deployed or {}).get("deployment_status", ""))).upper()
            if deployment_status == "UNDEPLOYED":
                if deployment.get("error"):
                    raise nugen.NugenError("Nugen deployment failed; inspect the provider's deployment status for the reason.")
                if state.get("deployment_requested_for_model") != model_id:
                    nugen._request("POST", f"/api/v3/models/{model_id}/deployment")
                    state["deployment_requested_for_model"] = model_id
                    save_state(state)
            set_local_model_id(str(model_id))
            state["model_id"] = model_id
            save_state(state)
            deployment = nugen._request("GET", f"/api/v3/models/{model_id}/deployment/status")
            deployment_status = str(deployment.get("status", "unknown")).upper()
            print(f"Nugen deployment status: {deployment_status}")
            if deployment_status == "DEPLOYED":
                print("Nugen SideQuest model is serving inference.")
                return
            if deployment_status == "UNDEPLOYED" and deployment.get("error"):
                raise nugen.NugenError("Nugen deployment failed; inspect the provider's deployment status for the reason.")
            if time.monotonic() >= deadline:
                print("Nugen accepted deployment, but it is not serving yet. Resume with --wait-minutes 10.")
                return
            time.sleep(15)
            continue
        if status in {"FAILED", "STOPPED"}:
            raise nugen.NugenError(f"Nugen alignment ended with status {status}.")
        if time.monotonic() >= deadline:
            print("Alignment is still processing. Resume with --wait-minutes 10.")
            return
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Upload synthetic examples and start alignment")
    parser.add_argument("--wait-minutes", type=int, default=0, help="Poll alignment, then request deployment when ready")
    args = parser.parse_args()
    try:
        if args.start:
            result = start()
            print("SideQuest Nugen alignment job:", result.get("status", "accepted"))
        if args.wait_minutes >= 0 and read_state().get("alignment_id"):
            poll_and_deploy(args.wait_minutes)
        elif not args.start:
            print("No SideQuest alignment job has been started. Use --start.")
    except nugen.NugenError as exc:
        print(f"Nugen: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
