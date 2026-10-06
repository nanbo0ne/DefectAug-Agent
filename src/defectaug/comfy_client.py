"""Minimal client for ComfyUI's built-in local API.

It does not execute arbitrary commands.  The caller supplies a previously reviewed
API-format workflow and this client only uploads inputs, queues it and downloads the
recorded outputs.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/system_stats", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def upload_image(self, path: Path, subfolder: str = "defectaug") -> dict[str, Any]:
        with path.open("rb") as stream:
            response = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (path.name, stream)},
                data={"subfolder": subfolder, "type": "input", "overwrite": "true"},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def queue(self, workflow: dict[str, Any]) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        response = requests.post(
            f"{self.base_url}/prompt", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        if result.get("node_errors"):
            raise RuntimeError(json.dumps(result["node_errors"], ensure_ascii=False))
        return str(result["prompt_id"])

    def wait_history(
        self, prompt_id: str, poll_seconds: float = 1.0, timeout_seconds: float = 1800.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = requests.get(
                f"{self.base_url}/history/{prompt_id}", timeout=self.timeout
            )
            response.raise_for_status()
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(poll_seconds)
        raise TimeoutError(f"ComfyUI prompt did not finish within {timeout_seconds}s")

    def download_image(self, metadata: dict[str, Any], destination: Path) -> Path:
        response = requests.get(
            f"{self.base_url}/view",
            params={
                "filename": metadata["filename"],
                "subfolder": metadata.get("subfolder", ""),
                "type": metadata.get("type", "output"),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination

