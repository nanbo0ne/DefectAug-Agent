from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/download_sdxl_inpaint.py"
SPEC = importlib.util.spec_from_file_location("download_sdxl_inpaint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dry_run_uses_explicit_official_endpoint(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://stale-mirror.invalid")

    assert MODULE.main(["--dry-run"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["endpoint"] == "https://huggingface.co"
    assert plan["requested_revision"] == MODULE.DEFAULT_REVISION
    assert plan["allow_patterns"] == list(MODULE.FP16_ALLOW_PATTERNS)


def test_rejects_non_http_endpoint() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        MODULE.main(["--dry-run", "--endpoint", "not-a-url"])


def test_snapshot_manifest_excludes_huggingface_transfer_cache(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (cache / "model_index.json.metadata").write_text("transport", encoding="utf-8")

    files = MODULE.snapshot_files(tmp_path, hash_files=True)

    assert [record["path"] for record in files] == ["model_index.json"]
