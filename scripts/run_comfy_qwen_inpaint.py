"""Run pinned Qwen-Image mask inpainting for one job or a frozen batch table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from defectaug.comfy_client import ComfyClient
from defectaug.quality import (
    save_and_verify_candidate,
    sha256_file,
    write_png,
)

DEFAULT_MODEL_MANIFEST = PROJECT / "reports/qwen_inpaint_models_manifest.json"
EXPECTED_MODEL_HASHES = {
    "diffusion_models/qwen_image_fp8_e4m3fn.safetensors": "98763a127701eb6fb59096f7742cb3aa7d64ed510b9f4e882d8351f8176e3ce3",
    "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors": "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4",
    "vae/qwen_image_vae.safetensors": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    "controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors": "49c01aafe6545c1f6e7627a724c8fb13357fe74efee235622158f0a8f30e5458",
}
DEFAULT_PROMPT = (
    "A monochrome close-up industrial inspection image of a hot-rolled steel surface. "
    "Inside the supplied mask, resynthesize one realistic narrow elongated scratch at "
    "the existing location and scale; preserve steel grain, viewpoint and illumination."
)
DEFAULT_NEGATIVE = (
    "text, watermark, frame, border, seam, repeated pattern, color, foreign object, "
    "multiple defects, concrete, fabric, wood, plastic"
)


@dataclass(frozen=True)
class Job:
    index: int
    job_id: str
    source_id: str
    image: Path
    mask: Path
    label: Path
    prompt: str
    negative_prompt: str
    seed: int
    expected_image_sha256: str | None
    expected_mask_sha256: str | None
    expected_label_sha256: str | None


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_input_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError("shared generation job table must be CSV")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def validate_job(job: Job) -> None:
    if not job.job_id or not job.source_id or not job.prompt.strip():
        raise ValueError("job_id, source_id and prompt must be non-empty")
    if job.seed < 0:
        raise ValueError(f"seed must be non-negative for {job.job_id}")
    for path, expected, label in (
        (job.image, job.expected_image_sha256, "image"),
        (job.mask, job.expected_mask_sha256, "mask"),
        (job.label, job.expected_label_sha256, "label"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist for {job.job_id}: {path}")
        if expected and sha256_file(path) != expected.lower():
            raise ValueError(f"{label} hash mismatch for {job.job_id}: {path}")


def build_jobs(args: argparse.Namespace) -> list[Job]:
    if args.batch_manifest:
        manifest_path = project_path(args.batch_manifest).resolve(strict=True)
        rows = load_rows(manifest_path)
        jobs: list[Job] = []
        for index, row in enumerate(rows):
            missing = [name for name in ("image", "mask", "label") if not row.get(name)]
            if missing:
                raise ValueError(f"batch row {index + 1} lacks: {', '.join(missing)}")
            image = resolve_input_path(row["image"], manifest_path.parent).resolve()
            seed = int(row.get("seed") or args.seed + index)
            source_id = str(row.get("source_id") or image.stem)
            job = Job(
                index=index,
                job_id=str(row.get("job_id") or f"{source_id}__s{seed}"),
                source_id=source_id,
                image=image,
                mask=resolve_input_path(row["mask"], manifest_path.parent).resolve(),
                label=resolve_input_path(row["label"], manifest_path.parent).resolve(),
                prompt=str(row.get("prompt") or args.prompt),
                negative_prompt=str(row.get("negative_prompt") or args.negative_prompt),
                seed=seed,
                expected_image_sha256=str(row.get("image_sha256") or "").lower() or None,
                expected_mask_sha256=str(row.get("mask_sha256") or "").lower() or None,
                expected_label_sha256=str(row.get("label_sha256") or "").lower() or None,
            )
            validate_job(job)
            jobs.append(job)
    else:
        if not args.image or not args.mask or not args.label:
            raise ValueError("single mode requires --image, --mask and --label")
        image = project_path(args.image).resolve()
        job = Job(
            index=0,
            job_id=f"{image.stem}__s{args.seed}",
            source_id=image.stem,
            image=image,
            mask=project_path(args.mask).resolve(),
            label=project_path(args.label).resolve(),
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            expected_image_sha256=None,
            expected_mask_sha256=None,
            expected_label_sha256=None,
        )
        validate_job(job)
        jobs = [job]
    if not jobs:
        raise ValueError("job table is empty")
    ids = [job.job_id for job in jobs]
    pairs = [(job.source_id, job.seed) for job in jobs]
    if len(set(ids)) != len(ids) or len(set(pairs)) != len(pairs):
        raise ValueError("duplicate job_id or source_id/seed pair in job table")
    return jobs


def validate_model_manifest(
    path: Path,
    allow_unverified: bool,
    *,
    verify_files: bool = False,
) -> dict[str, Any] | None:
    if not path.is_file():
        if allow_unverified:
            return None
        raise FileNotFoundError(
            f"formal Qwen runs require the verified model manifest: {path}; "
            "use --allow-unverified-models only for an explicitly non-formal pilot"
        )
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, list) or len(files) != 4:
        raise ValueError(f"invalid Qwen model manifest: {path}")
    actual_entries: dict[str, str] = {}
    for entry in files:
        relative_path = str(entry.get("relative_path") or "").replace("\\", "/")
        digest = str(entry.get("sha256") or "").lower()
        expected_digest = str(entry.get("expected_sha256") or "").lower()
        if len(digest) != 64 or digest != expected_digest:
            raise ValueError(f"model manifest has an unverified entry: {entry}")
        actual_entries[relative_path] = digest
    if actual_entries != EXPECTED_MODEL_HASHES:
        raise ValueError("model manifest does not match the four pinned Qwen inpainting weights")
    if verify_files:
        comfy_root_value = manifest.get("comfy_root")
        if not isinstance(comfy_root_value, str) or not comfy_root_value.strip():
            raise ValueError("Qwen model manifest lacks comfy_root for live file verification")
        model_root = (Path(comfy_root_value).expanduser().resolve() / "models").resolve()
        for entry in files:
            relative_path = str(entry["relative_path"]).replace("\\", "/")
            actual = (model_root / Path(relative_path)).resolve()
            try:
                actual.relative_to(model_root)
            except ValueError as exc:
                raise ValueError(
                    f"Qwen manifest path escapes the ComfyUI model root: {relative_path}"
                ) from exc
            expected_bytes = entry.get("bytes")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes <= 0
            ):
                raise ValueError(f"Qwen model manifest has invalid byte count: {entry}")
            if not actual.is_file() or actual.stat().st_size != expected_bytes:
                raise ValueError(f"Qwen model file is missing or changed in size: {actual}")
            if sha256_file(actual) != str(entry["sha256"]).lower():
                raise ValueError(f"Qwen model file SHA-256 mismatch: {actual}")
    return manifest


def validate_runtime_parameters(args: argparse.Namespace) -> None:
    """Keep Agent-supplied values inside the installed ComfyUI node ranges."""

    if not 1 <= args.steps <= 10000 or args.seed < 0:
        raise ValueError("steps must be in [1, 10000] and seed must be non-negative")
    if not 0.0 <= args.cfg <= 100.0:
        raise ValueError("cfg must be in [0, 100]")
    if not 0.0 <= args.control_strength <= 10.0:
        raise ValueError("control_strength must be in [0, 10]")
    if args.mask_expand < 0 or not 1 <= args.mask_blur <= 31:
        raise ValueError("mask_expand must be non-negative and mask_blur must be in [1, 31]")
    if not 0.0 <= args.shift <= 100.0:
        raise ValueError("shift must be in [0, 100]")
    if not 0 <= args.mask_threshold <= 255 or args.pixel_threshold <= 0:
        raise ValueError("mask_threshold must be in [0, 255] and pixel_threshold positive")
    if args.boundary_ring_width <= 0:
        raise ValueError("boundary_ring_width must be positive")


def uploaded_name(value: dict[str, Any]) -> str:
    return "/".join(part for part in (value.get("subfolder", ""), value["name"]) if part)


def first_saved_image(history: dict[str, Any]) -> dict[str, Any]:
    for node_output in history.get("outputs", {}).values():
        images = node_output.get("images", [])
        if images:
            return images[0]
    raise RuntimeError(f"ComfyUI returned no image; status={history.get('status', {})}")


def job_record(job: Job) -> dict[str, Any]:
    value = asdict(job)
    for key in ("image", "mask", "label"):
        value[key] = str(value[key])
    return value


def output_paths(output_dir: Path, job: Job) -> dict[str, Path]:
    prefix = output_dir / f"{job.job_id}__qwen"
    return {
        "native": prefix.with_name(f"{prefix.name}_native.png"),
        "raw": prefix.with_name(f"{prefix.name}_raw.png"),
        "composite": prefix.with_name(f"{prefix.name}_composite.png"),
        "mask": prefix.with_name(f"{prefix.name}_mask.png"),
    }


def assert_outputs_absent(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")


def run_job(
    *,
    client: ComfyClient,
    template: dict[str, Any],
    job: Job,
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = output_paths(args.output_dir, job)
    assert_outputs_absent(paths)
    with Image.open(job.image) as stream:
        source_image = ImageOps.exif_transpose(stream).convert("RGB")
    with Image.open(job.mask) as stream:
        mask_image = ImageOps.exif_transpose(stream).convert("L")
    source = np.asarray(source_image, dtype=np.uint8)
    mask = np.asarray(mask_image, dtype=np.uint8)
    if mask.shape != source.shape[:2] or not (mask >= args.mask_threshold).any():
        raise ValueError("mask must be non-empty after thresholding and match the source")

    image_upload = client.upload_image(job.image, "defectaug/qwen_inpaint")
    mask_upload = client.upload_image(job.mask, "defectaug/qwen_inpaint")
    workflow = deepcopy(template)
    workflow["5"]["inputs"]["image"] = uploaded_name(image_upload)
    workflow["6"]["inputs"]["image"] = uploaded_name(mask_upload)
    workflow["8"]["inputs"]["expand"] = args.mask_expand
    workflow["10"]["inputs"]["blur_radius"] = args.mask_blur
    workflow["13"]["inputs"]["text"] = job.prompt
    workflow["14"]["inputs"]["text"] = job.negative_prompt
    workflow["15"]["inputs"]["shift"] = args.shift
    workflow["16"]["inputs"]["strength"] = args.control_strength
    workflow["19"]["inputs"].update({"seed": job.seed, "steps": args.steps, "cfg": args.cfg})

    started = time.perf_counter()
    prompt_id = client.queue(workflow)
    history = client.wait_history(prompt_id, timeout_seconds=args.timeout_seconds)
    elapsed = time.perf_counter() - started
    client.download_image(first_saved_image(history), paths["native"])
    with Image.open(paths["native"]) as stream:
        native_image = stream.convert("RGB")
    if native_image.size != source_image.size:
        native_image = native_image.resize(source_image.size, Image.Resampling.LANCZOS)
    raw = np.asarray(native_image, dtype=np.uint8)
    raw_sha256 = write_png(paths["raw"], raw)
    with Image.open(paths["raw"]) as stream:
        raw_reloaded = np.asarray(stream.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(raw, raw_reloaded):
        raise RuntimeError(f"raw PNG changed after round-trip: {paths['raw']}")
    qc = save_and_verify_candidate(
        original=source,
        raw_candidate=raw_reloaded,
        mask=mask,
        composite_path=paths["composite"],
        final_mask_path=paths["mask"],
        label_path=job.label,
        pixel_threshold=args.pixel_threshold,
        boundary_ring_width=args.boundary_ring_width,
        mask_threshold=args.mask_threshold,
    )
    return {
        "prompt_id": prompt_id,
        "elapsed_seconds": elapsed,
        "native_output": str(paths["native"]),
        "native_output_sha256": sha256_file(paths["native"]),
        "raw_output": str(paths["raw"]),
        "raw_output_sha256": raw_sha256,
        "composite_output": str(paths["composite"]),
        "final_binary_mask": str(paths["mask"]),
        "input_image_sha256": sha256_file(job.image),
        "input_mask_sha256": sha256_file(job.mask),
        "input_label_sha256": sha256_file(job.label),
        **qc,
        "review_status": "pending_blind_human_review",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--image", type=Path)
    mode.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--workflow", type=Path, default=PROJECT / "workflows/qwen_inpaint_api_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data/synthetic/pilot/qwen_inpaint")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--allow-unverified-models", action="store_true")
    parser.add_argument(
        "--skip-model-file-hash",
        action="store_true",
        help=(
            "Skip live SHA-256 of the four ComfyUI weights; allowed only together "
            "with --allow-unverified-models for an explicitly non-formal run."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--mask-expand", type=int, default=0)
    parser.add_argument("--mask-blur", type=int, default=1)
    parser.add_argument("--shift", type=float, default=3.1)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--pixel-threshold", type=int, default=8)
    parser.add_argument("--boundary-ring-width", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = project_path(args.output_dir).resolve()
    args.workflow = project_path(args.workflow).resolve(strict=True)
    args.model_manifest = project_path(args.model_manifest).resolve()
    args.manifest = (
        project_path(args.manifest).resolve()
        if args.manifest
        else args.output_dir / "manifest.jsonl"
    )
    validate_runtime_parameters(args)
    if args.skip_model_file_hash and not args.allow_unverified_models:
        raise ValueError(
            "--skip-model-file-hash requires --allow-unverified-models; "
            "formal Qwen runs must verify the live model files"
        )

    run_id = str(uuid.uuid4())
    try:
        jobs = build_jobs(args)
        model_manifest = validate_model_manifest(
            args.model_manifest,
            args.allow_unverified_models,
            verify_files=not args.skip_model_file_hash,
        )
        template = json.loads(args.workflow.read_text(encoding="utf-8"))
        planned = [path for job in jobs for path in output_paths(args.output_dir, job).values()]
        if len(set(planned)) != len(planned):
            raise ValueError("job table maps multiple jobs to the same output artifact")
        for job in jobs:
            assert_outputs_absent(output_paths(args.output_dir, job))
    except Exception as exc:
        append_jsonl(
            args.manifest,
            {
                "record_type": "preflight_error",
                "run_id": run_id,
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            },
        )
        raise
    if args.validate_only:
        batch_manifest = (
            project_path(args.batch_manifest).resolve() if args.batch_manifest else None
        )
        preflight = {
            "record_type": "preflight_ok",
            "run_id": run_id,
            "finished_at": datetime.now().astimezone().isoformat(),
            "job_count": len(jobs),
            "batch_manifest": str(batch_manifest) if batch_manifest else None,
            "batch_manifest_sha256": (
                sha256_file(batch_manifest) if batch_manifest is not None else None
            ),
            "workflow": str(args.workflow),
            "workflow_sha256": sha256_file(args.workflow),
            "model_manifest": str(args.model_manifest) if model_manifest else None,
            "model_manifest_sha256": (
                sha256_file(args.model_manifest) if model_manifest else None
            ),
            "live_model_files_sha256_verified": not args.skip_model_file_hash,
            "unverified_model_pilot": model_manifest is None,
        }
        append_jsonl(args.manifest, preflight)
        print(json.dumps({"status": "inputs_valid", **preflight}, indent=2))
        return 0

    client = ComfyClient(args.base_url)
    try:
        health = client.health()
        append_jsonl(
            args.manifest,
            {
                "record_type": "run",
                "run_id": run_id,
                "started_at": datetime.now().astimezone().isoformat(),
                "job_count": len(jobs),
                "workflow": str(args.workflow),
                "workflow_sha256": sha256_file(args.workflow),
                "model_manifest": str(args.model_manifest) if model_manifest else None,
                "model_manifest_sha256": sha256_file(args.model_manifest) if model_manifest else None,
                "unverified_model_pilot": model_manifest is None,
                "live_model_files_sha256_verified": not args.skip_model_file_hash,
                "parameters": {
                    key: getattr(args, key)
                    for key in (
                        "steps", "cfg", "control_strength", "mask_expand", "mask_blur",
                        "shift", "mask_threshold", "pixel_threshold", "boundary_ring_width",
                    )
                },
                "comfyui": health.get("system", {}),
            },
        )
    except Exception as exc:
        append_jsonl(
            args.manifest,
            {
                "record_type": "run_error",
                "run_id": run_id,
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            },
        )
        raise

    failed = 0
    for job in jobs:
        started_at = datetime.now().astimezone()
        caught: Exception | None = None
        try:
            result = run_job(client=client, template=template, job=job, args=args)
            record = {
                "record_type": "result", "run_id": run_id, "status": "ok",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "job": job_record(job), "result": result,
            }
            print(result["composite_output"])
        except Exception as exc:
            caught = exc
            failed += 1
            record = {
                "record_type": "result", "run_id": run_id, "status": "error",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "job": job_record(job),
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
        append_jsonl(args.manifest, record)
        if record["status"] == "error" and not args.continue_on_error:
            append_jsonl(
                args.manifest,
                {
                    "record_type": "run_end", "run_id": run_id,
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "succeeded": job.index, "failed": failed, "aborted": True,
                },
            )
            raise RuntimeError(f"generation failed for {job.job_id}; see {args.manifest}") from caught

    append_jsonl(
        args.manifest,
        {
            "record_type": "run_end", "run_id": run_id,
            "finished_at": datetime.now().astimezone().isoformat(),
            "succeeded": len(jobs) - failed, "failed": failed,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
