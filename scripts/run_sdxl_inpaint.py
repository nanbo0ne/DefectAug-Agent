"""Run reproducible single-image or batch SDXL mask inpainting.

The model output is resized to the source resolution and then hard-composited
with a binary mask.  Pixels outside the mask are copied from the decoded source
image verbatim and the saved PNG is reloaded to verify that invariant.

Heavy ML imports are lazy so ``--help`` and ``--dry-run`` work without model
weights or a generation environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
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

from defectaug.quality import save_and_verify_candidate, sha256_file, write_png

DEFAULT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
DEFAULT_MODEL_MANIFEST = PROJECT / "runs/generation/models/sdxl_inpaint_download.json"
REQUIRED_MODEL_FILES = frozenset(
    {
        "model_index.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    }
)


@dataclass(frozen=True)
class Job:
    index: int
    job_id: str
    source_id: str
    image: Path
    mask: Path
    label: Path
    output: Path
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance: float
    strength: float
    width: int
    height: int
    padding_mask_crop: int | None
    expected_image_sha256: str | None
    expected_mask_sha256: str | None
    expected_label_sha256: str | None
    extra: dict[str, Any]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_commit(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 40
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def commit_from_snapshot_path(path: Path) -> str | None:
    parts = path.resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    candidate = parts[index + 1] if index + 1 < len(parts) else None
    return candidate.lower() if is_commit(candidate) else None


def parse_scalar(row: dict[str, Any], key: str, default: Any, cast: Any) -> Any:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return default
    return cast(value)


def load_batch_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(value)
        return rows
    raise ValueError("Batch manifest must be .csv, .jsonl, or .ndjson")


def resolve_input_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def validate_job(job: Job) -> None:
    if not job.image.is_file():
        raise FileNotFoundError(f"Image does not exist: {job.image}")
    if not job.mask.is_file():
        raise FileNotFoundError(f"Mask does not exist: {job.mask}")
    if not job.label.is_file():
        raise FileNotFoundError(f"YOLO label does not exist: {job.label}")
    if not job.prompt.strip():
        raise ValueError(f"Prompt is empty for {job.source_id}")
    if job.output.suffix.lower() != ".png":
        raise ValueError(
            f"Output must be PNG to preserve exact outside-mask pixels: {job.output}"
        )
    if job.steps <= 0:
        raise ValueError("steps must be positive")
    if job.guidance < 0:
        raise ValueError("guidance must be non-negative")
    if not 0.0 <= job.strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if job.width <= 0 or job.height <= 0 or job.width % 8 or job.height % 8:
        raise ValueError("width and height must be positive multiples of 8")
    if job.padding_mask_crop is not None and job.padding_mask_crop < 0:
        raise ValueError("padding_mask_crop cannot be negative")
    for path, expected, name in (
        (job.image, job.expected_image_sha256, "image"),
        (job.mask, job.expected_mask_sha256, "mask"),
        (job.label, job.expected_label_sha256, "label"),
    ):
        if expected and sha256_file(path) != expected.lower():
            raise ValueError(f"{name} hash mismatch for job {job.job_id}: {path}")


def build_jobs(args: argparse.Namespace) -> list[Job]:
    if args.batch_manifest:
        manifest_path = project_path(args.batch_manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Batch manifest does not exist: {manifest_path}")
        rows = load_batch_rows(manifest_path)
        output_root = project_path(args.output).resolve()
        jobs: list[Job] = []
        for index, row in enumerate(rows):
            if not row.get("image") or not row.get("mask") or not row.get("label"):
                raise ValueError(f"Batch row {index + 1} requires image, mask and label")
            image = resolve_input_path(str(row["image"]), manifest_path.parent).resolve()
            mask = resolve_input_path(str(row["mask"]), manifest_path.parent).resolve()
            label = resolve_input_path(str(row["label"]), manifest_path.parent).resolve()
            seed = parse_scalar(row, "seed", args.seed + index, int)
            source_id = str(row.get("source_id") or image.stem)
            job_id = str(row.get("job_id") or f"{source_id}__s{seed}")
            row_output = str(row.get("output") or "").strip()
            output = (
                resolve_input_path(row_output, output_root)
                if row_output
                else output_root / f"{job_id}__sdxl_composite.png"
            ).resolve()
            prompt = str(row.get("prompt") or args.prompt or "")
            negative_prompt = str(
                row.get("negative_prompt")
                if row.get("negative_prompt") not in {None, ""}
                else args.negative_prompt
            )
            known = {
                "source_id",
                "job_id",
                "image",
                "mask",
                "label",
                "image_sha256",
                "mask_sha256",
                "label_sha256",
                "output",
                "prompt",
                "negative_prompt",
                "seed",
                "steps",
                "guidance",
                "strength",
                "width",
                "height",
                "padding_mask_crop",
            }
            job = Job(
                index=index,
                job_id=job_id,
                source_id=source_id,
                image=image,
                mask=mask,
                label=label,
                output=output,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=parse_scalar(row, "steps", args.steps, int),
                guidance=parse_scalar(row, "guidance", args.guidance, float),
                strength=parse_scalar(row, "strength", args.strength, float),
                width=parse_scalar(row, "width", args.width, int),
                height=parse_scalar(row, "height", args.height, int),
                padding_mask_crop=parse_scalar(
                    row, "padding_mask_crop", args.padding_mask_crop, int
                ),
                expected_image_sha256=str(row.get("image_sha256") or "").lower() or None,
                expected_mask_sha256=str(row.get("mask_sha256") or "").lower() or None,
                expected_label_sha256=str(row.get("label_sha256") or "").lower() or None,
                extra={key: value for key, value in row.items() if key not in known},
            )
            validate_job(job)
            jobs.append(job)
        if not jobs:
            raise ValueError(f"Batch manifest is empty: {manifest_path}")
        return jobs

    if not args.image or not args.mask or not args.label:
        raise ValueError("Single mode requires --image, --mask and --label")
    image = project_path(args.image).resolve()
    seed = args.seed
    job = Job(
        index=0,
        job_id=f"{image.stem}__s{seed}",
        source_id=image.stem,
        image=image,
        mask=project_path(args.mask).resolve(),
        label=project_path(args.label).resolve(),
        output=project_path(args.output).resolve(),
        prompt=args.prompt or "",
        negative_prompt=args.negative_prompt,
        seed=seed,
        steps=args.steps,
        guidance=args.guidance,
        strength=args.strength,
        width=args.width,
        height=args.height,
        padding_mask_crop=args.padding_mask_crop,
        expected_image_sha256=None,
        expected_mask_sha256=None,
        expected_label_sha256=None,
        extra={},
    )
    validate_job(job)
    return [job]


def load_source_and_mask(image_path: Path, mask_path: Path) -> tuple[Image.Image, Image.Image]:
    with Image.open(image_path) as image_stream:
        source = ImageOps.exif_transpose(image_stream).convert("RGB")
    with Image.open(mask_path) as mask_stream:
        mask = ImageOps.exif_transpose(mask_stream).convert("L")
    if source.size != mask.size:
        raise ValueError(
            f"Image/mask dimensions differ: {source.size} versus {mask.size} "
            f"for {image_path.name}"
        )
    return source, mask


def binary_mask(mask: Image.Image, threshold: int) -> Image.Image:
    if not 0 <= threshold <= 255:
        raise ValueError("mask threshold must be in [0, 255]")
    return mask.convert("L").point(lambda value: 255 if value >= threshold else 0)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def validate_model_manifest(
    path: Path,
    *,
    model_path: Path,
    allow_unverified: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        if allow_unverified:
            return None
        raise FileNotFoundError(
            f"formal SDXL runs require the verified download manifest: {path}; "
            "use --allow-unverified-models only for an explicitly non-formal pilot"
        )
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("status") not in {"downloaded", "resolved_from_cache"}:
        raise ValueError("SDXL model manifest does not record a completed download")
    if document.get("model_id") != DEFAULT_MODEL_ID:
        raise ValueError("SDXL model manifest has an unexpected model_id")
    if str(document.get("resolved_commit") or "").lower() != DEFAULT_REVISION:
        raise ValueError("SDXL model manifest does not match the pinned commit")
    if document.get("variant") != "fp16":
        raise ValueError("SDXL model manifest is not the fp16 variant")
    snapshot = Path(str(document.get("snapshot_path") or "")).resolve()
    if snapshot != model_path.resolve():
        raise ValueError(
            f"SDXL model manifest snapshot mismatch: {snapshot} != {model_path.resolve()}"
        )
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("SDXL model manifest has no file records")
    records: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("SDXL model manifest contains a malformed file record")
        relative = str(entry.get("path") or "").replace("\\", "/")
        digest = str(entry.get("sha256") or "").lower()
        size = entry.get("size_bytes")
        if not relative or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"SDXL file is not hash-verified in the manifest: {entry}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"SDXL file has an invalid recorded size: {entry}")
        records[relative] = entry
    missing = REQUIRED_MODEL_FILES - set(records)
    if missing:
        raise ValueError(f"SDXL model manifest lacks required files: {sorted(missing)}")
    for relative, entry in records.items():
        actual = (model_path / Path(relative)).resolve()
        try:
            actual.relative_to(model_path.resolve())
        except ValueError as exc:
            raise ValueError(f"SDXL manifest path escapes model root: {relative}") from exc
        if not actual.is_file() or actual.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"SDXL model file is missing or changed in size: {actual}")
    return document


def resolve_model_identity(
    args: argparse.Namespace,
    model_manifest: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    if args.model_path:
        model_path = project_path(args.model_path).resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model snapshot does not exist: {model_path}")
        resolved = (
            str(model_manifest["resolved_commit"]).lower()
            if model_manifest is not None
            else commit_from_snapshot_path(model_path)
        )
        return str(model_path), resolved
    if is_commit(args.revision):
        return args.model_id, args.revision.lower()
    if args.local_files_only:
        return args.model_id, None
    try:
        from huggingface_hub import HfApi

        commit = HfApi().model_info(args.model_id, revision=args.revision).sha
    except Exception as exc:  # identity lookup is recorded but not a generation blocker
        print(f"Warning: could not resolve Hub commit: {exc}", file=sys.stderr)
        commit = None
    return args.model_id, commit


def device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", maxsplit=1)[1])
    raise ValueError("This fp16 pilot requires an NVIDIA CUDA device, e.g. cuda:0")


def load_pipeline(args: argparse.Namespace, source: str) -> tuple[Any, Any]:
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as exc:
        raise RuntimeError(
            "Generation dependencies are unavailable. Create .venv-generation and "
            "install requirements-generation.txt."
        ) from exc

    gpu_index = device_index(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an impractical fp16 CPU run")

    source_path = Path(source)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = str(project_path(args.cache_dir).resolve())
    if not source_path.is_dir():
        kwargs["revision"] = args.revision

    pipeline = AutoPipelineForInpainting.from_pretrained(source, **kwargs)
    if args.xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            raise RuntimeError(
                "--xformers was requested but a compatible xformers installation "
                "could not be enabled"
            ) from exc
    if args.vae_tiling:
        pipeline.enable_vae_tiling()
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=gpu_index)
    else:
        pipeline.to(f"cuda:{gpu_index}")
    pipeline.set_progress_bar_config(disable=args.quiet)
    return pipeline, torch


def environment_record(torch_module: Any | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "project_git_commit": git_commit(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "safetensors",
                "numpy",
                "Pillow",
                "xformers",
            )
        },
    }
    if torch_module is not None:
        record["cuda"] = {
            "available": torch_module.cuda.is_available(),
            "torch_cuda": torch_module.version.cuda,
            "cudnn": torch_module.backends.cudnn.version(),
            "device_count": torch_module.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch_module.cuda.get_device_name(index),
                    "total_memory_bytes": torch_module.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch_module.cuda.device_count())
            ],
        }
    return record


def job_record(job: Job) -> dict[str, Any]:
    value = asdict(job)
    for key in ("image", "mask", "label", "output"):
        value[key] = str(value[key])
    return value


def inspect_job_input(job: Job, mask_threshold: int) -> dict[str, Any]:
    source, mask = load_source_and_mask(job.image, job.mask)
    mask_binary = binary_mask(mask, mask_threshold)
    mask_pixels = int(np.count_nonzero(np.asarray(mask_binary, dtype=np.uint8)))
    if mask_pixels == 0:
        raise ValueError(f"Mask is empty after thresholding: {job.mask}")
    return {
        "source_width": source.width,
        "source_height": source.height,
        "mask_pixels": mask_pixels,
        "mask_fraction": mask_pixels / (source.width * source.height),
        "image_sha256": sha256_file(job.image),
        "mask_sha256": sha256_file(job.mask),
        "label_sha256": sha256_file(job.label),
    }


def run_job(
    pipeline: Any,
    torch_module: Any,
    job: Job,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if job.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {job.output}")
    planned_raw_output = job.output.with_name(f"{job.output.stem}.raw.png")
    final_mask_output = job.output.with_name(f"{job.output.stem}.mask.png")
    for artifact in (planned_raw_output, final_mask_output):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite output: {artifact}")

    source, mask = load_source_and_mask(job.image, job.mask)
    mask_binary = binary_mask(mask, args.mask_threshold)
    mask_pixels = int(np.count_nonzero(np.asarray(mask_binary, dtype=np.uint8)))
    if mask_pixels == 0:
        raise ValueError(f"Mask is empty after thresholding: {job.mask}")

    working_image = source.resize((job.width, job.height), Image.Resampling.LANCZOS)
    working_mask = mask_binary.resize((job.width, job.height), Image.Resampling.NEAREST)
    generator = torch_module.Generator(device="cpu").manual_seed(job.seed)

    gpu_index = device_index(args.device)
    torch_module.cuda.reset_peak_memory_stats(gpu_index)
    torch_module.cuda.synchronize(gpu_index)
    started = time.perf_counter()
    call_kwargs: dict[str, Any] = {
        "prompt": job.prompt,
        "negative_prompt": job.negative_prompt,
        "image": working_image,
        "mask_image": working_mask,
        "generator": generator,
        "num_inference_steps": job.steps,
        "guidance_scale": job.guidance,
        "strength": job.strength,
        "width": job.width,
        "height": job.height,
    }
    if job.padding_mask_crop:
        call_kwargs["padding_mask_crop"] = job.padding_mask_crop
    with torch_module.inference_mode():
        generated = pipeline(**call_kwargs).images[0].convert("RGB")
    torch_module.cuda.synchronize(gpu_index)
    duration = time.perf_counter() - started

    generated_at_source_size = generated.resize(source.size, Image.Resampling.LANCZOS)
    source_array = np.asarray(source, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)
    raw_array = np.asarray(generated_at_source_size, dtype=np.uint8)
    write_png(planned_raw_output, raw_array)
    with Image.open(planned_raw_output) as stream:
        raw_reloaded = np.asarray(stream.convert("RGB"), dtype=np.uint8)
    if raw_reloaded is None or not np.array_equal(raw_array, raw_reloaded):
        raise RuntimeError(f"raw PNG changed after round-trip: {planned_raw_output}")
    qc = save_and_verify_candidate(
        original=source_array,
        raw_candidate=raw_reloaded,
        mask=mask_array,
        composite_path=job.output,
        final_mask_path=final_mask_output,
        label_path=job.label,
        pixel_threshold=args.pixel_threshold,
        boundary_ring_width=args.boundary_ring_width,
        mask_threshold=args.mask_threshold,
    )

    return {
        "duration_seconds": duration,
        "peak_cuda_memory_bytes": torch_module.cuda.max_memory_allocated(gpu_index),
        "input_image_sha256": sha256_file(job.image),
        "input_mask_sha256": sha256_file(job.mask),
        "input_label_sha256": sha256_file(job.label),
        "output_sha256": sha256_file(job.output),
        "raw_output": str(planned_raw_output),
        "raw_output_sha256": sha256_file(planned_raw_output),
        "final_binary_mask": str(final_mask_output),
        **qc,
        "review_status": "pending_blind_human_review",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Diffusers SDXL single/batch mask-inpainting pilot."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--image", help="Single-mode source image.")
    mode.add_argument(
        "--batch-manifest",
        help="CSV/JSONL with image, mask and optional per-row parameters.",
    )
    parser.add_argument("--mask", help="Single-mode mask; white is generated.")
    parser.add_argument("--label", help="Single-mode YOLO label constraint.")
    parser.add_argument("--prompt", help="Global prompt; batch rows may override it.")
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=8.0)
    parser.add_argument("--strength", type=float, default=0.99)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--padding-mask-crop",
        type=int,
        default=None,
        help="Optional official Diffusers mask-focused crop padding.",
    )
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--pixel-threshold", type=int, default=8)
    parser.add_argument("--boundary-ring-width", type=int, default=2)
    parser.add_argument(
        "--output",
        required=True,
        help="Single-mode PNG path or batch-mode output directory.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Append-only JSONL run/result manifest.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-path", help="Optional downloaded local snapshot path.")
    parser.add_argument(
        "--model-manifest",
        default=str(DEFAULT_MODEL_MANIFEST),
        help="Hash-bearing manifest produced by download_sdxl_inpaint.py.",
    )
    parser.add_argument("--allow-unverified-models", action="store_true")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Accelerate model CPU offload (default: enabled).",
    )
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print jobs without importing Torch/Diffusers or writing files.",
    )
    return parser.parse_args(argv)


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    value = vars(args).copy()
    # Authentication is intentionally read only from standard HF environment or
    # credential storage; no token argument can leak into this record.
    value["hf_token_present"] = bool(os.environ.get("HF_TOKEN"))
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = build_jobs(args)
    outputs = [job.output for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("multiple jobs map to the same composite output")
    planned = [
        artifact
        for job in jobs
        for artifact in (
            job.output,
            job.output.with_name(f"{job.output.stem}.raw.png"),
            job.output.with_name(f"{job.output.stem}.mask.png"),
        )
    ]
    if len(set(planned)) != len(planned):
        raise ValueError("multiple jobs map to the same persisted artifact")
    if args.pixel_threshold <= 0 or args.boundary_ring_width <= 0:
        raise ValueError("pixel threshold and boundary ring width must be positive")
    if args.dry_run:
        inspected = [
            {"job": job_record(job), "input": inspect_job_input(job, args.mask_threshold)}
            for job in jobs
        ]
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "jobs": inspected,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    manifest = project_path(args.manifest).resolve()
    verified_model_manifest: dict[str, Any] | None = None
    model_manifest_path = project_path(args.model_manifest).resolve()
    if args.model_path:
        verified_model_manifest = validate_model_manifest(
            model_manifest_path,
            model_path=project_path(args.model_path).resolve(),
            allow_unverified=args.allow_unverified_models,
        )
    elif args.local_files_only and not args.allow_unverified_models:
        raise ValueError(
            "formal local SDXL runs require --model-path plus its verified model manifest"
        )
    model_source, resolved_commit = resolve_model_identity(args, verified_model_manifest)
    pipeline, torch_module = load_pipeline(args, model_source)
    run_id = str(uuid.uuid4())
    append_jsonl(
        manifest,
        {
            "record_type": "run",
            "run_id": run_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "command": sys.argv,
            "arguments": serializable_arguments(args),
            "model": {
                "model_id": args.model_id,
                "model_source": model_source,
                "requested_revision": args.revision,
                "resolved_commit": resolved_commit,
                "variant": "fp16",
                "pipeline": "AutoPipelineForInpainting",
                "download_manifest": (
                    str(model_manifest_path) if verified_model_manifest is not None else None
                ),
                "download_manifest_sha256": (
                    sha256_file(model_manifest_path)
                    if verified_model_manifest is not None
                    else None
                ),
                "unverified_model_pilot": verified_model_manifest is None,
            },
            "environment": environment_record(torch_module),
            "job_count": len(jobs),
        },
    )

    failed = 0
    for job in jobs:
        started_at = datetime.now().astimezone()
        caught: Exception | None = None
        try:
            result = run_job(pipeline, torch_module, job, args)
            record = {
                "record_type": "result",
                "run_id": run_id,
                "status": "ok",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "job": job_record(job),
                "result": result,
            }
            print(job.output)
        except Exception as exc:
            caught = exc
            failed += 1
            record = {
                "record_type": "result",
                "run_id": run_id,
                "status": "error",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "job": job_record(job),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        append_jsonl(manifest, record)
        if record["status"] == "error" and not args.continue_on_error:
            append_jsonl(
                manifest,
                {
                    "record_type": "run_end",
                    "run_id": run_id,
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "succeeded": job.index,
                    "failed": failed,
                    "aborted": True,
                },
            )
            raise RuntimeError(
                f"Generation failed for {job.source_id}; see {manifest}"
            ) from caught

    append_jsonl(
        manifest,
        {
            "record_type": "run_end",
            "run_id": run_id,
            "finished_at": datetime.now().astimezone().isoformat(),
            "succeeded": len(jobs) - failed,
            "failed": failed,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
