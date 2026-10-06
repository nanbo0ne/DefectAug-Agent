"""Audited, fail-closed authorization for the one-time final test evaluation."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TestAccessAuthorization:
    protocol_path: Path
    protocol_sha256: str
    audit_log: Path
    event_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorized": True,
            "protocol_path": str(self.protocol_path),
            "protocol_sha256": self.protocol_sha256,
            "audit_log": str(self.audit_log),
            "event_id": self.event_id,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _append_event(audit_log: Path, event: Mapping[str, Any]) -> None:
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def authorize_final_test_access(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str | None,
    supplied_token: str | None,
    audit_log: Path,
    context: Mapping[str, Any] | None = None,
    required_protocol_path: Path | None = None,
) -> TestAccessAuthorization:
    """Authorize test access without reading any test manifest or test artifact."""

    protocol_path = protocol_path.resolve()
    audit_log = audit_log.resolve()
    timestamp = datetime.now(timezone.utc).isoformat()
    event_id = hashlib.sha256(
        f"{timestamp}|{protocol_path}|{audit_log}".encode("utf-8")
    ).hexdigest()[:16]
    event: dict[str, Any] = {
        "schema_version": "1.0",
        "event_id": event_id,
        "timestamp": timestamp,
        "event": "final_test_access_attempt",
        "authorized": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": None,
        "expected_protocol_sha256": expected_protocol_sha256,
        "token_supplied": supplied_token is not None,
        "context": dict(context or {}),
    }

    def deny(reason: str, message: str) -> None:
        event["reason"] = reason
        _append_event(audit_log, event)
        raise PermissionError(message)

    if (
        required_protocol_path is not None
        and protocol_path != required_protocol_path.resolve()
    ):
        deny(
            "nonstandard_protocol_path",
            f"test access must use frozen protocol path: {required_protocol_path.resolve()}",
        )
    if expected_protocol_sha256 is None or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_protocol_sha256
    ):
        deny(
            "missing_or_invalid_expected_protocol_sha256",
            "test access requires --expected-protocol-sha256 with a 64-character digest",
        )
    expected_hash = expected_protocol_sha256.lower()
    if not protocol_path.is_file():
        deny("protocol_not_found", f"test access protocol not found: {protocol_path}")
    protocol_hash = sha256_file(protocol_path)
    event["protocol_sha256"] = protocol_hash
    if not hmac.compare_digest(protocol_hash, expected_hash):
        deny(
            "protocol_sha256_mismatch",
            f"test access protocol hash mismatch: expected {expected_hash}, found {protocol_hash}",
        )

    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        deny("invalid_protocol", "test access protocol root must be a mapping")
    gate = document.get("final_test_access")
    experiment = document.get("experiment")
    if not isinstance(gate, dict) or not isinstance(experiment, dict):
        deny("missing_gate_configuration", "protocol has no final_test_access gate")
    if gate.get("enabled") is not True:
        deny("gate_disabled", "final test access remains disabled in the frozen protocol")
    required_status = gate.get("required_experiment_status")
    actual_status = experiment.get("status")
    if not isinstance(required_status, str) or actual_status != required_status:
        deny(
            "experiment_not_final",
            f"experiment status must be {required_status!r}; found {actual_status!r}",
        )
    expected_token = gate.get("unlock_token")
    if not isinstance(expected_token, str) or not expected_token:
        deny("invalid_gate_token", "protocol final-test token is not configured")
    if supplied_token is None or not hmac.compare_digest(supplied_token, expected_token):
        deny("unlock_token_mismatch", "final test unlock token is missing or incorrect")

    event["authorized"] = True
    event["reason"] = "all_final_test_gates_passed"
    _append_event(audit_log, event)
    return TestAccessAuthorization(
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        audit_log=audit_log,
        event_id=event_id,
    )


__all__ = ["TestAccessAuthorization", "authorize_final_test_access"]
