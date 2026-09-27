"""Server-only client for Nugen's current public API.

Credentials, provider IDs and raw provider errors never leave this module.
"""
from __future__ import annotations

import os
import json
import time
import re
from typing import Any

import requests

BASE_URL = "https://api.nugen.in"
TIMEOUT_SECONDS = 25


class NugenError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _key() -> str:
    value = os.environ.get("NUGEN_API_KEY", "").strip()
    if not value:
        raise NugenError("Nugen is not configured")
    return value


def _request(method: str, path: str, *, json_body: dict[str, Any] | None = None,
             files: dict[str, Any] | None = None, data: dict[str, Any] | None = None,
             timeout: int = TIMEOUT_SECONDS) -> Any:
    if not path.startswith("/api/v3/") or "://" in path:
        raise NugenError("Invalid Nugen API path")
    try:
        response = requests.request(
            method,
            BASE_URL + path,
            headers={"Authorization": f"Bearer {_key()}", "Accept": "application/json"},
            json=json_body,
            files=files,
            data=data,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise NugenError(f"Nugen request failed ({type(exc).__name__})") from None
    if not response.ok:
        messages = {
            401: "Nugen rejected the configured credentials.",
            403: "Nugen denied access to this model or endpoint.",
            404: "Nugen could not find the configured model or endpoint.",
            413: "The Nugen upload is too large.",
            422: "Nugen rejected the supplied request or training data.",
            429: "Nugen rate limit reached; try again shortly.",
            502: "Nugen inference is temporarily unavailable.",
            503: "Nugen inference is temporarily unavailable.",
        }
        raise NugenError(messages.get(response.status_code, f"Nugen request failed (HTTP {response.status_code})."), response.status_code)
    try:
        return response.json()
    except ValueError:
        raise NugenError("Nugen returned an unreadable response", response.status_code) from None


def _list(payload: Any, key: str) -> list[dict[str, Any]]:
    value = payload.get(key, []) if isinstance(payload, dict) else []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _provider_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch aligned models, projects, and base model metadata from documented routes."""
    aligned = _list(_request("GET", "/api/v3/models/aligned"), "domain_aligned_models")
    projects = _list(_request("GET", "/api/v3/alignment-projects/list"), "alignment_projects")
    base = _list(_request("GET", "/api/v3/models/base"), "models")
    return aligned, projects, base


def status() -> dict[str, Any]:
    """Return safe diagnostics; never return credentials or provider object IDs."""
    if not os.environ.get("NUGEN_API_KEY", "").strip():
        return {"configured": False, "reachable": False, "aligned_model_count": 0,
                "aligned_model_ready": False, "message": "Nugen credentials are not configured."}
    try:
        aligned, projects, base = _provider_state()
        deployed = [item for item in aligned if str(item.get("deployment_status", "")).upper() == "DEPLOYED"]
        expected_model = os.environ.get("NUGEN_ALIGNED_MODEL_ID", "").strip()
        ready_ids = {str(item.get("model_id", "")) for item in deployed}
        selected = expected_model if expected_model in ready_ids else (str(deployed[0].get("model_id", "")) if len(deployed) == 1 else "")
        eligible_base_models = [item for item in base if item.get("alignment_ready") is True]
        processing = sum(str(item.get("status", "")).upper() in {"PROCESSING", "QUEUED"} for item in projects)
        if selected:
            message = "A deployed Nugen aligned model is ready for SideQuest inference."
        elif processing:
            message = "Nugen is reachable; SideQuest model alignment is still processing."
        elif deployed:
            message = "Nugen has deployed models, but select one with NUGEN_ALIGNED_MODEL_ID before use."
        elif eligible_base_models:
            message = "Nugen is reachable; no deployed SideQuest-aligned model was found. Alignment-ready base models are available."
        else:
            message = "Nugen is reachable; no deployed SideQuest-aligned model or eligible base model was found."
        return {
            "configured": True,
            "reachable": True,
            "aligned_model_count": len(aligned),
            "aligned_model_ready": bool(selected),
            "alignment_project_count": len(projects),
            "alignment_processing_count": processing,
            "base_model_count": len(base),
            "alignment_ready_base_model_count": len(eligible_base_models),
            "message": message,
        }
    except NugenError as exc:
        return {"configured": True, "reachable": False, "aligned_model_count": 0,
                "aligned_model_ready": False, "message": str(exc)}


def selected_model_id() -> str:
    """Resolve the configured aligned deployment; infer only on deployed aligned models."""
    aligned = _list(_request("GET", "/api/v3/models/aligned"), "domain_aligned_models")
    deployed = [item for item in aligned if str(item.get("deployment_status", "")).upper() == "DEPLOYED"]
    expected_model = os.environ.get("NUGEN_ALIGNED_MODEL_ID", "").strip()
    for item in deployed:
        if str(item.get("model_id", "")) == expected_model and expected_model:
            return expected_model
    if len(deployed) == 1:
        return str(deployed[0].get("model_id", ""))
    raise NugenError("No unambiguous deployed SideQuest-aligned Nugen model is available.")


def predict(prompt: str, model_id: str | None = None) -> dict[str, str]:
    """Run one bounded streaming request against a deployed aligned model.

    Nugen's chat-completions endpoint serves this aligned deployment. Consume
    token events as they arrive rather than waiting on non-streaming output.
    """
    clean = prompt.strip()
    if not clean or len(clean) > 3000:
        raise NugenError("Prompt must contain 1 to 3000 characters")
    model = (model_id or selected_model_id()).strip()
    if len(model) > 120 or any(ch.isspace() for ch in model):
        raise NugenError("Nugen model configuration is invalid")
    try:
        response = requests.post(
            BASE_URL + "/api/v3/inference/chat/completions",
            headers={"Authorization": f"Bearer {_key()}", "Accept": "text/event-stream"},
            json={"model": model,
                  "messages": [{"role": "user", "content": clean}],
                  "max_tokens": 320, "temperature": 0.1, "stream": True,
                  "reasoning": {"effort": "none", "exclude": True}},
            timeout=(10, 75),
            stream=True,
        )
    except requests.RequestException as exc:
        raise NugenError(f"Nugen request failed ({type(exc).__name__})") from None
    if not response.ok:
        messages = {
            401: "Nugen rejected the configured credentials.",
            403: "Nugen denied access to this model or endpoint.",
            404: "Nugen could not find the configured model or endpoint.",
            429: "Nugen rate limit reached; try again shortly.",
            502: "Nugen inference is temporarily unavailable.",
            503: "Nugen inference is temporarily unavailable.",
            504: "Nugen inference timed out before returning an answer.",
        }
        raise NugenError(messages.get(response.status_code, f"Nugen request failed (HTTP {response.status_code})."), response.status_code)

    chunks: list[str] = []
    started = time.monotonic()
    try:
        for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if time.monotonic() - started > 90:
                raise NugenError("Nugen inference exceeded the 90-second response limit.")
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            choices = event.get("choices", []) if isinstance(event, dict) else []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            delta = choice.get("delta", {})
            message = choice.get("message", {})
            complete_message = message.get("content") if isinstance(message, dict) else None
            content = (choice.get("text") or complete_message
                       or (delta.get("content") if isinstance(delta, dict) else None)
                       or event.get("text") or event.get("content"))
            if isinstance(content, str) and content:
                chunks.append(content)
                if sum(map(len, chunks)) > 6000:
                    break
            if complete_message or choice.get("finish_reason"):
                break
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
        raise NugenError("Nugen closed the response before returning a complete answer.") from None
    except requests.exceptions.ReadTimeout:
        raise NugenError("Nugen took too long to return an answer.") from None
    except requests.RequestException as exc:
        raise NugenError(f"Nugen request failed ({type(exc).__name__})") from None
    finally:
        response.close()

    output = "".join(chunks).strip()
    if not output:
        raise NugenError("Nugen returned an empty model answer")
    return {"output": output[:6000]}


_IMPACT_ENUMS = {
    "accessibility": {"likely_open", "limited", "unknown"},
    "safety_impact": {"low", "moderate", "high", "unknown"},
    "recommendation": {"continue_with_caution", "wait_for_update", "consider_alternative", "insufficient_data"},
    "confidence": {"low", "medium", "high"},
}


def predict_impact(prompt: str, model_id: str | None = None) -> dict[str, Any]:
    """Return a validated, bounded Digital Twin response; reject prose or malformed JSON."""
    raw = predict(prompt, model_id=model_id)["output"].strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise NugenError("Nugen returned an answer without the required impact fields.")
    try:
        value = json.loads(raw[start:end + 1])
    except ValueError:
        raise NugenError("Nugen returned unreadable impact fields.") from None
    if not isinstance(value, dict):
        raise NugenError("Nugen returned an invalid impact result.")
    for field, allowed in _IMPACT_ENUMS.items():
        if value.get(field) not in allowed:
            raise NugenError("Nugen returned an incomplete impact result.")
    for field in ("delay_prob", "demand_shift_pct"):
        number = value.get(field)
        if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float))):
            raise NugenError("Nugen returned an invalid estimate.")
    probability = value.get("delay_prob")
    if probability is not None and not 0 <= probability <= 1:
        raise NugenError("Nugen returned an out-of-range delay estimate.")
    demand = value.get("demand_shift_pct")
    if demand is not None and not -100 <= demand <= 100:
        raise NugenError("Nugen returned an out-of-range demand estimate.")
    cascades = value.get("cascades")
    if not isinstance(cascades, list) or any(not isinstance(item, str) for item in cascades):
        raise NugenError("Nugen returned invalid itinerary effects.")
    guidance = value.get("guidance")
    if not isinstance(guidance, str) or not guidance.strip():
        raise NugenError("Nugen returned no traveller guidance.")
    return {
        "accessibility": value["accessibility"],
        "delay_prob": probability,
        "demand_shift_pct": demand,
        "safety_impact": value["safety_impact"],
        "cascades": [item.strip()[:240] for item in cascades[:3] if item.strip()],
        "recommendation": value["recommendation"],
        "confidence": value["confidence"],
        "guidance": guidance.strip()[:700],
    }
