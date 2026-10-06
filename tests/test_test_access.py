import hashlib
import json
from pathlib import Path

import pytest

from defectaug.test_access import authorize_final_test_access


def _protocol(tmp_path: Path, *, enabled: bool, status: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "experiment:\n"
        f"  status: {status}\n"
        "final_test_access:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        "  required_experiment_status: frozen_final\n"
        "  unlock_token: FINAL_TEST_EVALUATION\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_test_access_is_denied_and_audited_while_protocol_disabled(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path, enabled=False, status="pilot_not_frozen")
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(PermissionError, match="remains disabled"):
        authorize_final_test_access(
            protocol_path=protocol,
            expected_protocol_sha256=_sha256(protocol),
            supplied_token="FINAL_TEST_EVALUATION",
            audit_log=audit,
        )
    event = json.loads(audit.read_text(encoding="utf-8").splitlines()[0])
    assert event["authorized"] is False
    assert event["reason"] == "gate_disabled"
    assert "FINAL_TEST_EVALUATION" not in audit.read_text(encoding="utf-8")


def test_test_access_requires_exact_hash_and_token_then_audits_success(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path, enabled=True, status="frozen_final")
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(PermissionError, match="hash mismatch"):
        authorize_final_test_access(
            protocol_path=protocol,
            expected_protocol_sha256="0" * 64,
            supplied_token="FINAL_TEST_EVALUATION",
            audit_log=audit,
        )
    result = authorize_final_test_access(
        protocol_path=protocol,
        expected_protocol_sha256=_sha256(protocol),
        supplied_token="FINAL_TEST_EVALUATION",
        audit_log=audit,
        context={"weights_sha256": "a" * 64},
    )
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert result.protocol_sha256 == _sha256(protocol)
    assert events[-1]["authorized"] is True
    assert events[-1]["context"]["weights_sha256"] == "a" * 64
