"""Evaluate an Ultralytics detector on one explicitly selected locked split.

The default and safe path is validation.  The original dataset YAML is used as
the source of truth, but Ultralytics receives a generated runtime YAML that
contains only the selected manifest.  Consequently a validation run cannot
open the locked test manifest through Ultralytics as a side effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from defectaug.data_lock import maybe_verify_content_lock
from defectaug.test_access import authorize_final_test_access


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
ALLOWED_SPLITS = ("val", "test")
METRIC_FIELDS = ("precision", "recall", "ap50", "ap50_95")


@dataclass(frozen=True)
class ManifestLock:
    path: Path
    sha256: str
    entries: int


@dataclass(frozen=True)
class DatasetSplitLock:
    data_yaml: Path
    data_yaml_sha256: str
    logical_split: str
    split_sha256: str
    manifests: tuple[ManifestLock, ...]
    class_names: dict[int, str]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expected_hash(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return normalized


def _project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


# Only these protocol files may open a test split; each must also pass the
# frozen_final status and sha256 checks inside authorize_final_test_access.
FROZEN_PROTOCOLS = ("configs/experiment_v2.yaml", "configs/experiment_v3.yaml", "configs/experiment_v31.yaml")


def _frozen_protocol_path(value: str | os.PathLike[str]) -> Path:
    resolved = _project_path(value).resolve()
    allowed = {(PROJECT / p).resolve() for p in FROZEN_PROTOCOLS}
    # Anything else is compared against the v2 path, so the gate denies it and
    # writes the refusal to the audit log.
    return resolved if resolved in allowed else (PROJECT / FROZEN_PROTOCOLS[0]).resolve()


def _normalize_class_names(dataset: Mapping[str, Any]) -> dict[int, str]:
    raw_names = dataset.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset 'names' keys must be integer class IDs") from exc
    elif raw_names is None and isinstance(dataset.get("nc"), int):
        names = {index: f"class_{index}" for index in range(dataset["nc"])}
    else:
        raise ValueError("dataset YAML must define 'names' or an integer 'nc'")

    expected = list(range(len(names)))
    if sorted(names) != expected:
        raise ValueError(f"dataset class IDs must be contiguous from 0; found {sorted(names)}")
    if len(set(names.values())) != len(names):
        raise ValueError("dataset class names must be unique")
    return dict(sorted(names.items()))


def _resolve_manifest_paths(
    data_yaml: Path, dataset: Mapping[str, Any], split: str
) -> tuple[Path, ...]:
    selected = dataset.get(split)
    if selected is None or selected == "":
        raise ValueError(f"dataset YAML has no non-empty '{split}' entry")
    sources: Sequence[Any] = selected if isinstance(selected, list) else [selected]
    if not sources:
        raise ValueError(f"dataset YAML '{split}' entry is empty")

    raw_root = dataset.get("path")
    if raw_root:
        dataset_root = Path(str(raw_root)).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = data_yaml.parent / dataset_root
    else:
        dataset_root = data_yaml.parent
    dataset_root = dataset_root.resolve()

    manifests: list[Path] = []
    for source in sources:
        if not isinstance(source, (str, os.PathLike)):
            raise ValueError(f"dataset '{split}' entries must be filesystem paths")
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"selected {split} manifest not found: {path}")
        if path.suffix.lower() != ".txt":
            raise ValueError(
                f"selected {split} source must be a frozen .txt manifest, got: {path}"
            )
        manifests.append(path)
    return tuple(manifests)


def _manifest_entry_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def lock_dataset_split(
    data_yaml: Path,
    split: str = "val",
    *,
    expected_data_sha256: str | None = None,
    expected_split_sha256: str | None = None,
) -> DatasetSplitLock:
    """Hash the source YAML and only the explicitly selected split manifest.

    Unselected split files are neither opened nor checked for existence here.
    """

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}; got {split!r}")
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")

    wanted_data_hash = _validate_expected_hash(expected_data_sha256, "expected data hash")
    wanted_split_hash = _validate_expected_hash(expected_split_sha256, "expected split hash")
    data_hash = sha256_file(data_yaml)
    if wanted_data_hash is not None and data_hash != wanted_data_hash:
        raise RuntimeError(
            f"dataset YAML hash mismatch: expected {wanted_data_hash}, found {data_hash}"
        )

    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("dataset YAML root must be a mapping")
    class_names = _normalize_class_names(document)
    manifest_paths = _resolve_manifest_paths(data_yaml, document, split)
    manifests = tuple(
        ManifestLock(path=path, sha256=sha256_file(path), entries=_manifest_entry_count(path))
        for path in manifest_paths
    )
    if len(manifests) == 1:
        split_hash = manifests[0].sha256
    else:
        digest = hashlib.sha256()
        for lock in manifests:
            digest.update(lock.sha256.encode("ascii"))
            digest.update(b"\n")
        split_hash = digest.hexdigest()
    if wanted_split_hash is not None and split_hash != wanted_split_hash:
        raise RuntimeError(
            f"{split} manifest hash mismatch: expected {wanted_split_hash}, found {split_hash}"
        )

    return DatasetSplitLock(
        data_yaml=data_yaml,
        data_yaml_sha256=data_hash,
        logical_split=split,
        split_sha256=split_hash,
        manifests=manifests,
        class_names=class_names,
    )


def verify_dataset_lock_unchanged(lock: DatasetSplitLock) -> None:
    """Fail a run if its YAML or selected manifest changed during evaluation."""

    current_data_hash = sha256_file(lock.data_yaml)
    if current_data_hash != lock.data_yaml_sha256:
        raise RuntimeError(
            f"dataset YAML changed during evaluation: {lock.data_yaml_sha256} -> {current_data_hash}"
        )
    for manifest in lock.manifests:
        current_hash = sha256_file(manifest.path)
        if current_hash != manifest.sha256:
            raise RuntimeError(
                f"selected {lock.logical_split} manifest changed during evaluation: "
                f"{manifest.path} ({manifest.sha256} -> {current_hash})"
            )


def runtime_dataset_document(lock: DatasetSplitLock) -> dict[str, Any]:
    """Return a minimal Ultralytics YAML document exposing one split only."""

    paths = [str(item.path) for item in lock.manifests]
    selected: str | list[str] = paths[0] if len(paths) == 1 else paths
    document: dict[str, Any] = {
        # Ultralytics requires train and val keys even for evaluation.  Both are
        # deliberately pointed at the selected, already-locked manifest.
        "train": selected,
        "val": selected,
        "names": lock.class_names,
    }
    if lock.logical_split == "test":
        document["test"] = selected
    return document


def write_runtime_dataset_yaml(lock: DatasetSplitLock, destination: Path) -> str:
    destination.write_text(
        yaml.safe_dump(runtime_dataset_document(lock), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return sha256_file(destination)


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (str, bytes)):
        raise TypeError("numeric metric arrays cannot be strings")
    try:
        return list(value)
    except TypeError:
        return [value]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _indexed_count(values: Any, class_id: int) -> int | None:
    sequence = _to_list(values)
    if class_id >= len(sequence):
        return None
    value = _finite_float(sequence[class_id])
    return int(value) if value is not None else None


def extract_detection_metrics(metrics: Any, class_names: Mapping[int, str]) -> dict[str, Any]:
    """Convert an Ultralytics DetMetrics-like object to a stable plain schema."""

    box = getattr(metrics, "box", None)
    if box is None:
        raise TypeError("validation result has no detection 'box' metrics")

    class_ids = [int(value) for value in _to_list(getattr(box, "ap_class_index", []))]
    arrays = {
        "precision": [_finite_float(value) for value in _to_list(getattr(box, "p", []))],
        "recall": [_finite_float(value) for value in _to_list(getattr(box, "r", []))],
        "ap50": [_finite_float(value) for value in _to_list(getattr(box, "ap50", []))],
        "ap50_95": [_finite_float(value) for value in _to_list(getattr(box, "ap", []))],
    }
    for field, values in arrays.items():
        if len(values) != len(class_ids):
            raise ValueError(
                f"Ultralytics metric length mismatch for {field}: "
                f"{len(values)} values for {len(class_ids)} class IDs"
            )
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("Ultralytics returned duplicate class IDs")
    unknown = sorted(set(class_ids) - set(class_names))
    if unknown:
        raise ValueError(f"Ultralytics returned undeclared class IDs: {unknown}")

    position_by_class = {class_id: position for position, class_id in enumerate(class_ids)}
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in sorted(class_names.items()):
        position = position_by_class.get(class_id)
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "evaluated": position is not None,
            "images": _indexed_count(getattr(metrics, "nt_per_image", None), class_id),
            "instances": _indexed_count(getattr(metrics, "nt_per_class", None), class_id),
        }
        for field in METRIC_FIELDS:
            row[field] = arrays[field][position] if position is not None else None
        per_class.append(row)

    overall = {
        "precision": _finite_float(getattr(box, "mp", None)),
        "recall": _finite_float(getattr(box, "mr", None)),
        "ap50": _finite_float(getattr(box, "map50", None)),
        "ap50_95": _finite_float(getattr(box, "map", None)),
        "declared_classes": len(class_names),
        "evaluated_classes": len(class_ids),
    }

    candidates = [row for row in per_class if row["evaluated"] and row["ap50"] is not None]
    worst_class = None
    if candidates:
        worst = min(candidates, key=lambda row: (row["ap50"], row["class_id"]))
        worst_class = {
            "criterion": "ap50",
            **{key: worst[key] for key in ("class_id", "class_name", *METRIC_FIELDS)},
        }

    metric_std: dict[str, Any] = {
        "definition": "population standard deviation across evaluated classes (ddof=0)",
        "evaluated_classes": len(class_ids),
    }
    for field in METRIC_FIELDS:
        values = [row[field] for row in per_class if row["evaluated"] and row[field] is not None]
        metric_std[field] = statistics.pstdev(values) if values else None

    speed = {
        str(key): _finite_float(value)
        for key, value in (getattr(metrics, "speed", {}) or {}).items()
    }
    return {
        "overall": overall,
        "per_class": per_class,
        "worst_class": worst_class,
        "class_metric_std": metric_std,
        "speed_ms_per_image": speed,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_devices": devices,
        "ultralytics": ultralytics.__version__,
        "numpy": _package_version("numpy"),
        "pillow": _package_version("Pillow"),
        "pyyaml": _package_version("PyYAML"),
        "git_commit": git_commit(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def write_json_report(report: Mapping[str, Any], path: Path) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
    _atomic_write_text(path, payload + "\n")


def _csv_rows(report: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    common = {
        "schema_version": report["schema_version"],
        "evaluated_at": report["completed_at"],
        "split": report["evaluation"]["split"],
        "weights_sha256": report["inputs"]["weights"]["sha256"],
        "data_yaml_sha256": report["inputs"]["data_yaml"]["sha256"],
        "split_sha256": report["inputs"]["selected_split"]["sha256"],
    }
    metric_data = report["metrics"]
    yield {**common, "record_type": "overall", **metric_data["overall"]}
    for row in metric_data["per_class"]:
        yield {**common, "record_type": "per_class", **row}
    if metric_data["worst_class"] is not None:
        yield {**common, "record_type": "worst_class", **metric_data["worst_class"]}
    yield {**common, "record_type": "class_std", **metric_data["class_metric_std"]}


def write_csv_report(report: Mapping[str, Any], path: Path) -> None:
    rows = list(_csv_rows(report))
    fieldnames = [
        "schema_version",
        "evaluated_at",
        "split",
        "weights_sha256",
        "data_yaml_sha256",
        "split_sha256",
        "record_type",
        "class_id",
        "class_name",
        "criterion",
        "evaluated",
        "images",
        "instances",
        "precision",
        "recall",
        "ap50",
        "ap50_95",
        "declared_classes",
        "evaluated_classes",
        "definition",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _safe_run_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    if not name:
        raise ValueError("evaluation run name is empty after sanitization")
    return name


def _default_run_name(weights: Path, split: str) -> str:
    training_run = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    return _safe_run_name(f"{training_run}_{weights.stem}_{split}")


def _prepare_run_directory(root: Path, name: str, exist_ok: bool) -> Path:
    run_dir = root / name
    if run_dir.exists() and any(run_dir.iterdir()) and not exist_ok:
        raise FileExistsError(
            f"evaluation output already exists: {run_dir}; choose --name or pass --exist-ok"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an Ultralytics detector on a locked validation or test manifest."
    )
    parser.add_argument("--weights", required=True, help="Ultralytics .pt weights")
    parser.add_argument("--data", default="configs/dataset_sd900_3class_v2.yaml")
    parser.add_argument(
        "--split",
        choices=ALLOWED_SPLITS,
        default="val",
        help="Safe default is val; test additionally requires all final-unlock gates.",
    )
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument(
        "--content-lock",
        default="data/manifests/dataset_content_lock_v2.json",
        help="Byte-level image/label/mask lock generated by freeze_dataset_content.py.",
    )
    parser.add_argument(
        "--content-lock-mode",
        choices=("required", "optional", "off"),
        default="required",
        help="Required by default. Use optional only for legacy reports; off is explicit opt-out.",
    )
    parser.add_argument(
        "--protocol",
        default="configs/experiment_v2.yaml",
        help="Frozen experiment protocol used only by the final test gate.",
    )
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--final-test-unlock-token")
    parser.add_argument("--test-audit-log", default="reports/test_access_audit_v2.jsonl")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--quantize",
        type=int,
        choices=(16, 32),
        default=None,
        help="Inference precision: 16 for FP16, 32/omitted for FP32.",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-root", default="reports/yolo_eval")
    parser.add_argument("--name")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def _validate_numeric_arguments(args: argparse.Namespace) -> None:
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0 or args.max_det <= 0:
        raise ValueError("imgsz, batch, and max-det must be positive; workers cannot be negative")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("conf and iou must be within [0, 1]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_numeric_arguments(args)
    weights = _project_path(args.weights)
    data_yaml = _project_path(args.data)
    output_root = _project_path(args.output_root)
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    expected_weights_hash = _validate_expected_hash(
        args.expected_weights_sha256, "expected weights hash"
    )
    weights_hash = sha256_file(weights)
    if expected_weights_hash is not None and weights_hash != expected_weights_hash:
        raise RuntimeError(
            f"weights hash mismatch: expected {expected_weights_hash}, found {weights_hash}"
        )

    test_authorization = None
    if args.split == "test":
        # This gate deliberately runs before lock_dataset_split and before the
        # content-lock verifier, so a denied request never opens test.txt or any
        # test image/label/mask.
        test_authorization = authorize_final_test_access(
            protocol_path=_project_path(args.protocol),
            expected_protocol_sha256=args.expected_protocol_sha256,
            supplied_token=args.final_test_unlock_token,
            audit_log=_project_path(args.test_audit_log),
            context={
                "weights_path": str(weights),
                "weights_sha256": weights_hash,
                "data_yaml": str(data_yaml),
            },
            required_protocol_path=_frozen_protocol_path(args.protocol),
        )

    content_verification = maybe_verify_content_lock(
        _project_path(args.content_lock),
        project_root=PROJECT,
        splits=(args.split,),
        mode=args.content_lock_mode,
        expected_data_yaml=data_yaml,
    )

    lock = lock_dataset_split(
        data_yaml,
        args.split,
        expected_data_sha256=args.expected_data_sha256,
        expected_split_sha256=args.expected_split_sha256,
    )
    run_name = _safe_run_name(args.name) if args.name else _default_run_name(weights, args.split)
    run_dir = _prepare_run_directory(output_root, run_name, args.exist_ok)
    runtime_yaml = run_dir / "selected_split.runtime.yaml"
    runtime_yaml_hash = write_runtime_dataset_yaml(lock, runtime_yaml)

    validation_parameters = {
        "data": str(runtime_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "quantize": None if args.quantize in (None, 32) else args.quantize,
        "augment": False,
        "rect": True,
        "plots": args.plots,
        "save_json": False,
        "save_txt": False,
        "project": str(run_dir),
        "name": "ultralytics",
        "exist_ok": True,
        "verbose": args.verbose,
    }

    from ultralytics import YOLO

    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    model = YOLO(str(weights), task="detect")
    raw_metrics = model.val(**validation_parameters)
    elapsed = time.perf_counter() - started
    completed_at = datetime.now().astimezone().isoformat()
    current_weights_hash = sha256_file(weights)
    if current_weights_hash != weights_hash:
        raise RuntimeError(
            f"weights changed during evaluation: {weights_hash} -> {current_weights_hash}"
        )
    verify_dataset_lock_unchanged(lock)
    final_content_verification = maybe_verify_content_lock(
        _project_path(args.content_lock),
        project_root=PROJECT,
        splits=(args.split,),
        mode=args.content_lock_mode,
        expected_data_yaml=data_yaml,
    )
    if content_verification != final_content_verification:
        raise RuntimeError("dataset content-lock verification changed during evaluation")

    metrics = extract_detection_metrics(raw_metrics, lock.class_names)
    report = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed,
        "evaluation": {
            "split": lock.logical_split,
            "test_access_authorized": test_authorization is not None,
            "test_access_authorization": (
                test_authorization.as_dict() if test_authorization is not None else None
            ),
            "runtime_dataset_yaml": str(runtime_yaml),
            "runtime_dataset_yaml_sha256": runtime_yaml_hash,
            "ultralytics_save_dir": str(getattr(raw_metrics, "save_dir", "")) or None,
        },
        "inputs": {
            "weights": {"path": str(weights), "sha256": weights_hash},
            "data_yaml": {"path": str(lock.data_yaml), "sha256": lock.data_yaml_sha256},
            "selected_split": {
                "name": lock.logical_split,
                "sha256": lock.split_sha256,
                "entries": sum(item.entries for item in lock.manifests),
                "manifests": [
                    {"path": str(item.path), "sha256": item.sha256, "entries": item.entries}
                    for item in lock.manifests
                ],
            },
            "class_names": lock.class_names,
            "content_lock": (
                content_verification.as_dict() if content_verification is not None else None
            ),
        },
        "parameters": validation_parameters,
        "environment": collect_environment(),
        "metrics": metrics,
    }
    json_path = run_dir / "metrics.json"
    csv_path = run_dir / "metrics.csv"
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)
    print(json.dumps({"metrics_json": str(json_path), "metrics_csv": str(csv_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
