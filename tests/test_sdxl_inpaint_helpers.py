"""Weight-free tests for the SDXL inpainting pilot helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_sdxl_inpaint.py"
SPEC = importlib.util.spec_from_file_location("run_sdxl_inpaint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_batch_csv_accepts_utf8_bom(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot.csv"
    manifest.write_text(
        "source_id,image,mask,prompt\ncase_1,image.png,mask.png,small scratch\n",
        encoding="utf-8-sig",
    )

    rows = MODULE.load_batch_rows(manifest)

    assert rows == [
        {
            "source_id": "case_1",
            "image": "image.png",
            "mask": "mask.png",
            "prompt": "small scratch",
        }
    ]


def test_snapshot_commit_is_detected_from_hf_cache_path(tmp_path: Path) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "models--example" / "snapshots" / commit
    snapshot.mkdir(parents=True)

    assert MODULE.commit_from_snapshot_path(snapshot) == commit


def test_verified_sdxl_manifest_requires_pinned_commit_hashes_and_local_files(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    records = []
    for index, relative in enumerate(sorted(MODULE.REQUIRED_MODEL_FILES), start=1):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]))
        records.append(
            {
                "path": relative,
                "size_bytes": 1,
                "sha256": f"{index:064x}",
            }
        )
    manifest = tmp_path / "download.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "downloaded",
                "model_id": MODULE.DEFAULT_MODEL_ID,
                "resolved_commit": MODULE.DEFAULT_REVISION,
                "variant": "fp16",
                "snapshot_path": str(model),
                "files": records,
            }
        ),
        encoding="utf-8",
    )

    loaded = MODULE.validate_model_manifest(
        manifest, model_path=model, allow_unverified=False
    )

    assert loaded is not None and loaded["resolved_commit"] == MODULE.DEFAULT_REVISION


def test_formal_sdxl_mode_rejects_missing_model_manifest(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    try:
        MODULE.validate_model_manifest(
            tmp_path / "missing.json", model_path=model, allow_unverified=False
        )
    except FileNotFoundError as exc:
        assert "formal SDXL runs require" in str(exc)
    else:
        raise AssertionError("missing formal SDXL model manifest was accepted")


def test_run_job_keeps_successful_candidate_pending_blind_review(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    label = tmp_path / "source.txt"
    output = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), color=(64, 64, 64)).save(image)
    mask_image = Image.new("L", (8, 8), color=0)
    for x in range(2, 6):
        for y in range(2, 6):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask)
    label.write_text("0 0.5 0.5 0.75 0.75\n", encoding="utf-8")

    job = MODULE.Job(
        index=0,
        job_id="case_1__s7",
        source_id="case_1",
        image=image,
        mask=mask,
        label=label,
        output=output,
        prompt="one realistic scratch",
        negative_prompt="text",
        seed=7,
        steps=1,
        guidance=1.0,
        strength=1.0,
        width=8,
        height=8,
        padding_mask_crop=None,
        expected_image_sha256=None,
        expected_mask_sha256=None,
        expected_label_sha256=None,
        extra={},
    )

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> "FakeGenerator":
            self.seed = seed
            return self

    class FakeCuda:
        @staticmethod
        def reset_peak_memory_stats(index: int) -> None:
            return None

        @staticmethod
        def synchronize(index: int) -> None:
            return None

        @staticmethod
        def max_memory_allocated(index: int) -> int:
            return 0

    torch_module = SimpleNamespace(
        Generator=FakeGenerator,
        cuda=FakeCuda(),
        inference_mode=nullcontext,
    )

    def pipeline(**_: object) -> SimpleNamespace:
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), color=(160, 160, 160))])

    result = MODULE.run_job(
        pipeline,
        torch_module,
        job,
        SimpleNamespace(
            mask_threshold=127,
            device="cuda:0",
            pixel_threshold=1,
            boundary_ring_width=1,
        ),
    )

    assert result["review_status"] == "pending_blind_human_review"
    assert result["quality"]["composite_hard_invariant"] is True
