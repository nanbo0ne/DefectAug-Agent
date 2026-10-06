"""Weight-free Qwen batch and model-manifest tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_comfy_qwen_inpaint.py"
SPEC = importlib.util.spec_from_file_location("run_comfy_qwen_inpaint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_verified_model_manifest_requires_four_matching_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "relative_path": relative_path,
                        "sha256": digest,
                        "expected_sha256": digest,
                    }
                    for relative_path, digest in MODULE.EXPECTED_MODEL_HASHES.items()
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = MODULE.validate_model_manifest(manifest, allow_unverified=False)

    assert loaded is not None and len(loaded["files"]) == 4


def test_formal_mode_rejects_missing_model_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    try:
        MODULE.validate_model_manifest(missing, allow_unverified=False)
    except FileNotFoundError as exc:
        assert "formal Qwen runs require" in str(exc)
    else:
        raise AssertionError("missing formal model manifest was accepted")


def test_live_qwen_files_are_rehashed_from_comfy_model_root(tmp_path: Path) -> None:
    comfy_root = tmp_path / "ComfyUI"
    entries = []
    for index, (relative_path, _) in enumerate(MODULE.EXPECTED_MODEL_HASHES.items()):
        actual = comfy_root / "models" / relative_path
        actual.parent.mkdir(parents=True, exist_ok=True)
        actual.write_bytes(f"weight-{index}".encode())
        digest = MODULE.sha256_file(actual)
        entries.append(
            {
                "relative_path": relative_path,
                "bytes": actual.stat().st_size,
                "sha256": digest,
                "expected_sha256": digest,
            }
        )

    original = MODULE.EXPECTED_MODEL_HASHES
    MODULE.EXPECTED_MODEL_HASHES = {
        entry["relative_path"]: entry["sha256"] for entry in entries
    }
    try:
        manifest = tmp_path / "models.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "comfy_root": str(comfy_root),
                    "files": entries,
                }
            ),
            encoding="utf-8",
        )
        loaded = MODULE.validate_model_manifest(
            manifest, allow_unverified=False, verify_files=True
        )
        assert loaded is not None

        (comfy_root / "models" / entries[0]["relative_path"]).write_bytes(b"tampered!")
        try:
            MODULE.validate_model_manifest(
                manifest, allow_unverified=False, verify_files=True
            )
        except ValueError as exc:
            assert "changed in size" in str(exc) or "SHA-256 mismatch" in str(exc)
        else:
            raise AssertionError("tampered live Qwen weight was accepted")
    finally:
        MODULE.EXPECTED_MODEL_HASHES = original


def test_qwen_agent_parameter_ranges_match_installed_nodes() -> None:
    valid = SimpleNamespace(
        steps=20,
        seed=7,
        cfg=2.5,
        control_strength=1.0,
        mask_expand=0,
        mask_blur=1,
        shift=3.1,
        mask_threshold=127,
        pixel_threshold=8,
        boundary_ring_width=2,
    )
    MODULE.validate_runtime_parameters(valid)

    for name, value in (
        ("cfg", 100.1),
        ("control_strength", 10.1),
        ("mask_blur", 0),
        ("shift", -0.1),
    ):
        invalid = SimpleNamespace(**vars(valid))
        setattr(invalid, name, value)
        try:
            MODULE.validate_runtime_parameters(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {name}={value} was accepted")
