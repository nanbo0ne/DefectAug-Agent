"""Content-level dataset freezing without decoding image or mask media.

The lock records SHA-256 digests for every image, YOLO label and mask in the
active train/validation/test manifests.  Verification can be limited to a
selected split, so ordinary training and validation never need to touch test
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = "1.0"
ACTIVE_SPLITS: dict[str, str] = {
    "train_fewshot": "train",
    "val": "val",
    "test": "test",
}
MEDIA_HASH_ALGORITHM = "sha256-bytes-no-decode"


@dataclass(frozen=True)
class ContentLockVerification:
    lock_path: Path
    lock_sha256: str
    content_digest: str
    verified_splits: tuple[str, ...]
    verified_items: int
    verified_files: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.lock_path),
            "sha256": self.lock_sha256,
            "content_digest": self.content_digest,
            "verified_splits": list(self.verified_splits),
            "verified_items": self.verified_items,
            "verified_files": self.verified_files,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash file bytes only; image and mask files are never decoded."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_portable(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve_dataset_source(
    data_yaml: Path, dataset: Mapping[str, Any], dataset_key: str
) -> Path:
    raw = dataset.get(dataset_key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"dataset YAML must define one .txt manifest for '{dataset_key}'")
    path = Path(raw).expanduser()
    raw_root = dataset.get("path")
    if not path.is_absolute():
        if raw_root:
            root = Path(str(raw_root)).expanduser()
            if not root.is_absolute():
                root = data_yaml.parent / root
        else:
            root = data_yaml.parent
        path = root / path
    path = path.resolve()
    if path.suffix.lower() != ".txt" or not path.is_file():
        raise FileNotFoundError(f"frozen split manifest not found: {path}")
    return path


def _manifest_entries(manifest: Path, *, allow_duplicates: bool = False) -> list[Path]:
    entries: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if path in seen and not allow_duplicates:
            raise ValueError(f"duplicate image at {manifest}:{line_number}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"image listed at {manifest}:{line_number} not found: {path}")
        seen.add(path)
        entries.append(path)
    if not entries:
        raise ValueError(f"split manifest is empty: {manifest}")
    return entries


def _unique_companion(root: Path, stem: str, extensions: Sequence[str], kind: str) -> Path:
    matches = [root / f"{stem}{suffix}" for suffix in extensions]
    existing = [path.resolve() for path in matches if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"expected exactly one {kind} for {stem!r} under {root}; found {len(existing)}"
        )
    return existing[0]


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _portable_path(path, project_root),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def _content_digest(document: Mapping[str, Any]) -> str:
    material = {
        "schema_version": document.get("schema_version"),
        "hash_algorithm": document.get("hash_algorithm"),
        "dataset_yaml": document.get("dataset_yaml"),
        "splits": document.get("splits"),
    }
    payload = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_content_lock(
    *,
    project_root: Path,
    data_yaml: Path,
    labels_root: Path,
    masks_root: Path,
    allow_duplicate_train_entries: bool = False,
    active_splits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create an in-memory lock document for selected split artifacts.

    ``active_splits`` defaults to the full train/validation/test protocol.  A
    transfer package may explicitly freeze only train and validation so the
    locked final-test media never needs to leave the primary machine.
    """

    project_root = project_root.resolve()
    data_yaml = data_yaml.resolve()
    labels_root = labels_root.resolve()
    masks_root = masks_root.resolve()
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("dataset YAML root must be a mapping")

    selected_splits = dict(ACTIVE_SPLITS if active_splits is None else active_splits)
    if not selected_splits:
        raise ValueError("active_splits cannot be empty")
    unknown = set(selected_splits) - set(ACTIVE_SPLITS)
    if unknown:
        raise ValueError(f"unknown active split(s): {sorted(unknown)}")
    for logical_split, dataset_key in selected_splits.items():
        if ACTIVE_SPLITS[logical_split] != dataset_key:
            raise ValueError(
                f"active split {logical_split!r} must map to "
                f"{ACTIVE_SPLITS[logical_split]!r}, got {dataset_key!r}"
            )

    splits: dict[str, Any] = {}
    stems_by_split: dict[str, set[str]] = {}
    for logical_split, dataset_key in selected_splits.items():
        split_manifest = _resolve_dataset_source(data_yaml, document, dataset_key)
        items: list[dict[str, Any]] = []
        split_stems: set[str] = set()
        manifest_entries = _manifest_entries(
            split_manifest,
            allow_duplicates=allow_duplicate_train_entries and logical_split == "train_fewshot",
        )
        for image in manifest_entries:
            stem = image.stem
            if stem in split_stems:
                if allow_duplicate_train_entries and logical_split == "train_fewshot":
                    continue
                raise ValueError(f"duplicate image stem in {logical_split}: {stem}")
            split_stems.add(stem)
            label = _unique_companion(labels_root, stem, (".txt",), "label")
            mask = _unique_companion(
                masks_root, stem, (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"), "mask"
            )
            items.append(
                {
                    "id": stem,
                    "image": _file_record(image, project_root),
                    "label": _file_record(label, project_root),
                    "mask": _file_record(mask, project_root),
                }
            )
        stems_by_split[logical_split] = split_stems
        splits[logical_split] = {
            "dataset_key": dataset_key,
            "manifest": _file_record(split_manifest, project_root),
            "manifest_entry_count": len(manifest_entries),
            "unique_item_count": len(items),
            "items": sorted(items, key=lambda item: item["id"]),
        }

    split_names = tuple(selected_splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = stems_by_split[left] & stems_by_split[right]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"image stems cross {left}/{right}: {sample}")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": MEDIA_HASH_ALGORITHM,
        "path_base": "project_root",
        "project_root_hint": str(project_root),
        "dataset_yaml": _file_record(data_yaml, project_root),
        "splits": splits,
    }
    result["content_digest"] = _content_digest(result)
    return result


def write_content_lock(document: Mapping[str, Any], output: Path, *, force: bool = False) -> None:
    """Write a lock atomically; refuse replacement unless force is explicit."""

    output = output.resolve()
    if output.exists() and not force:
        raise FileExistsError(f"content lock already exists (not overwritten): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, output)


def _require_file_record(record: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    if not isinstance(record.get("path"), str):
        raise ValueError(f"{label}.path must be a string")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label}.sha256 must be a SHA-256 digest")
    if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
        raise ValueError(f"{label}.bytes must be a non-negative integer")
    return record


def _verify_file_record(record: Mapping[str, Any], project_root: Path, label: str) -> None:
    path = _resolve_portable(str(record["path"]), project_root)
    if not path.is_file():
        raise FileNotFoundError(f"locked {label} not found: {path}")
    size = path.stat().st_size
    if size != record["bytes"]:
        raise RuntimeError(f"locked {label} size mismatch: {path} ({record['bytes']} -> {size})")
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise RuntimeError(
            f"locked {label} SHA-256 mismatch: {path} ({record['sha256']} -> {digest})"
        )


def verify_content_lock(
    lock_path: Path,
    *,
    project_root: Path,
    splits: Iterable[str],
    expected_data_yaml: Path | None = None,
) -> ContentLockVerification:
    """Verify only requested splits against a frozen content lock."""

    lock_path = lock_path.resolve()
    project_root = project_root.resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"dataset content lock not found: {lock_path}")
    document = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset content lock schema: {document!r}")
    if document.get("hash_algorithm") != MEDIA_HASH_ALGORITHM:
        raise ValueError("dataset content lock uses an unsupported hash algorithm")
    expected_content_digest = document.get("content_digest")
    actual_content_digest = _content_digest(document)
    if expected_content_digest != actual_content_digest:
        raise RuntimeError(
            "dataset content lock document digest mismatch; the lock itself was modified"
        )

    requested = tuple(dict.fromkeys(splits))
    unknown = [split for split in requested if split not in ACTIVE_SPLITS]
    if unknown:
        raise ValueError(f"unknown content-lock split(s): {unknown}")
    all_splits = document.get("splits")
    if not isinstance(all_splits, dict):
        raise ValueError("dataset content lock has no split mapping")

    dataset_record = _require_file_record(document.get("dataset_yaml"), "dataset_yaml")
    locked_data_yaml = _resolve_portable(str(dataset_record["path"]), project_root)
    if expected_data_yaml is not None and locked_data_yaml != expected_data_yaml.resolve():
        raise RuntimeError(
            f"content lock belongs to {locked_data_yaml}, not requested dataset "
            f"{expected_data_yaml.resolve()}"
        )
    _verify_file_record(dataset_record, project_root, "dataset YAML")
    verified_items = 0
    verified_files = 1
    for split in requested:
        split_record = all_splits.get(split)
        if not isinstance(split_record, dict):
            raise ValueError(f"dataset content lock has no '{split}' split")
        manifest_record = _require_file_record(split_record.get("manifest"), f"{split}.manifest")
        _verify_file_record(manifest_record, project_root, f"{split} manifest")
        verified_files += 1
        items = split_record.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"dataset content lock '{split}' items must be a non-empty list")
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError(f"invalid item in locked split '{split}'")
            if item["id"] in seen:
                raise ValueError(f"duplicate item ID in locked split '{split}': {item['id']}")
            seen.add(item["id"])
            for kind in ("image", "label", "mask"):
                record = _require_file_record(item.get(kind), f"{split}.{item['id']}.{kind}")
                _verify_file_record(record, project_root, f"{split} {item['id']} {kind}")
                verified_files += 1
            verified_items += 1

    return ContentLockVerification(
        lock_path=lock_path,
        lock_sha256=sha256_file(lock_path),
        content_digest=actual_content_digest,
        verified_splits=requested,
        verified_items=verified_items,
        verified_files=verified_files,
    )


def maybe_verify_content_lock(
    lock_path: Path,
    *,
    project_root: Path,
    splits: Iterable[str],
    mode: str,
    expected_data_yaml: Path | None = None,
) -> ContentLockVerification | None:
    """Implement required/optional/off CLI semantics consistently."""

    if mode not in {"required", "optional", "off"}:
        raise ValueError("content lock mode must be required, optional, or off")
    if mode == "off":
        return None
    if not lock_path.is_file():
        if mode == "optional":
            return None
        raise FileNotFoundError(f"required dataset content lock not found: {lock_path}")
    return verify_content_lock(
        lock_path,
        project_root=project_root,
        splits=splits,
        expected_data_yaml=expected_data_yaml,
    )
