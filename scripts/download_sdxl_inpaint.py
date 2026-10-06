"""Download the pinned official SDXL 1.0 inpainting fp16 snapshot.

This script uses only ``huggingface_hub`` APIs.  By default it downloads the
minimal fp16 Diffusers layout rather than the duplicate fp32 weights stored in
the same repository, and records the requested revision and resolved commit.
Use ``--dry-run`` to inspect the plan without importing Hugging Face libraries
or downloading model data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_ENDPOINT = "https://huggingface.co"
# Resolved from the official Hugging Face repository on 2026-08-20.
DEFAULT_REVISION = "115134f363124c53c7d878647567d04daf26e41e"

# Files required by AutoPipelineForInpainting(..., variant="fp16").
FP16_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)


def project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(snapshot_path: Path, hash_files: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot_path)
        # local_dir downloads may create Hugging Face transfer metadata under
        # .cache.  It is not part of the model snapshot and must not enter the
        # immutable model-file manifest.
        if relative.parts and relative.parts[0] == ".cache":
            continue
        record: dict[str, Any] = {
            "path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if hash_files:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def commit_from_snapshot_path(path: Path) -> str | None:
    parts = path.resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    candidate = parts[index + 1] if index + 1 < len(parts) else None
    if candidate and len(candidate) == 40 and all(
        char in "0123456789abcdefABCDEF" for char in candidate
    ):
        return candidate.lower()
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned official SDXL inpainting fp16 snapshot."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Hugging Face API/download endpoint; defaults to the official service.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face branch, tag, or preferably immutable commit SHA.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional Hugging Face cache directory (relative paths use project root).",
    )
    parser.add_argument(
        "--local-dir",
        help="Optional explicit snapshot directory instead of the HF cache layout.",
    )
    parser.add_argument(
        "--manifest",
        default="runs/generation/models/sdxl_inpaint_download.json",
        help="JSON download record (relative paths use project root).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--full-snapshot",
        action="store_true",
        help="Also download duplicate fp32 weights and repository documentation.",
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="SHA-256 every downloaded file (slow for multi-GB weights).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve only an already cached snapshot; never contact the Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without importing HF libraries or downloading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    endpoint = args.endpoint.rstrip("/")
    if not endpoint.startswith(("https://", "http://")):
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    # huggingface_hub reads HF_ENDPOINT when it is imported below.  Make the
    # selected endpoint explicit instead of silently inheriting a stale global
    # mirror setting from the user's shell.
    os.environ["HF_ENDPOINT"] = endpoint
    cache_dir = project_path(args.cache_dir)
    local_dir = project_path(args.local_dir)
    manifest = project_path(args.manifest)
    assert manifest is not None

    plan = {
        "model_id": args.model_id,
        "endpoint": endpoint,
        "requested_revision": args.revision,
        "variant": "fp16",
        "cache_dir": str(cache_dir.resolve()) if cache_dir else None,
        "local_dir": str(local_dir.resolve()) if local_dir else None,
        "manifest": str(manifest.resolve()),
        "allow_patterns": None if args.full_snapshot else list(FP16_ALLOW_PATTERNS),
        "local_files_only": args.local_files_only,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is unavailable. Create .venv-generation and install "
            "requirements-generation.txt first."
        ) from exc

    started_at = datetime.now().astimezone()
    resolved_commit: str | None = None
    if not args.local_files_only:
        resolved_commit = HfApi().model_info(
            repo_id=args.model_id, revision=args.revision
        ).sha
    elif len(args.revision) == 40 and all(char in "0123456789abcdefABCDEF" for char in args.revision):
        resolved_commit = args.revision.lower()

    snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=resolved_commit or args.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_dir=str(local_dir) if local_dir else None,
            allow_patterns=None if args.full_snapshot else list(FP16_ALLOW_PATTERNS),
            max_workers=args.max_workers,
            local_files_only=args.local_files_only,
        )
    ).resolve()
    resolved_commit = resolved_commit or commit_from_snapshot_path(snapshot)

    finished_at = datetime.now().astimezone()
    record = {
        "status": "downloaded" if not args.local_files_only else "resolved_from_cache",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        **plan,
        "resolved_commit": resolved_commit,
        "snapshot_path": str(snapshot),
        "files": snapshot_files(snapshot, args.hash_files),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "huggingface_hub": package_version("huggingface-hub"),
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(snapshot)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
