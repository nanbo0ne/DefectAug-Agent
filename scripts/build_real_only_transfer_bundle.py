"""Build and self-check the train/validation-only Real-only transfer ZIP."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from defectaug.data_lock import verify_content_lock


PROJECT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = PROJECT.parent / "exports"
BUNDLE_NAME = "DefectAug-RealOnly-Transfer-v1"
EXPECTED_COUNTS = {"train_fewshot": 100, "val": 150}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_relative(source_relative: str, bundle: Path, destination_relative: str | None = None) -> None:
    source = PROJECT / source_relative
    destination = bundle / (destination_relative or source_relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def manifest_ids(split: str) -> list[str]:
    source = PROJECT / "data" / "splits" / "v1" / f"{split}.txt"
    paths = [Path(line.strip()) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(paths) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"unexpected {split} count: {len(paths)}")
    ids = [path.stem for path in paths]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate IDs in {split}")
    return ids


def write_bundle_manifest(bundle: Path) -> tuple[int, int]:
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    lines = [f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}" for path in files]
    output = bundle / "bundle_manifest.sha256"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return len(files), sum(path.stat().st_size for path in files)


def validate_relocated_bundle(zip_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="realonly_bundle_verify_") as temporary:
        temporary_root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temporary_root)
        root = temporary_root / BUNDLE_NAME
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        commands = (
            [sys.executable, "scripts/verify_transfer_bundle.py"],
            [sys.executable, "scripts/prepare_real_only_machine.py"],
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "scripts/train_yolo.py", "--help"],
            [sys.executable, "scripts/evaluate_yolo.py", "--help"],
        )
        outputs: list[str] = []
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            outputs.append((completed.stdout + completed.stderr).strip()[-1000:])
        setup = json.loads((root / "reports" / "real_only_machine_setup.json").read_text(encoding="utf-8"))
        if setup["test_in_bundle"] or setup["test_accessed"]:
            raise RuntimeError("relocated bundle unexpectedly includes/accesses test")
        if setup["content_lock"]["verified_items"] != 250:
            raise RuntimeError("relocated content lock did not verify 250 train/val items")
        return {"commands": len(commands), "verified_items": 250, "test_in_bundle": False, "tails": outputs}


def main() -> int:
    verify_content_lock(
        PROJECT / "data" / "manifests" / "dataset_content_lock_v1.json",
        project_root=PROJECT,
        splits=("train_fewshot", "val"),
        expected_data_yaml=PROJECT / "configs" / "dataset_sd900_3class_v1.yaml",
    )
    train_ids = manifest_ids("train_fewshot")
    val_ids = manifest_ids("val")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("source train/validation overlap")

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = EXPORT_ROOT / BUNDLE_NAME
    if bundle.exists():
        raise FileExistsError(f"refusing to replace existing bundle directory: {bundle}")
    bundle.mkdir()

    static_files = (
        "pyproject.toml",
        "requirements-real-only.txt",
        "yolo11n.pt",
        "configs/real_only_final_v1.yaml",
        "scripts/train_yolo.py",
        "scripts/evaluate_yolo.py",
        "scripts/prepare_real_only_machine.py",
        "scripts/verify_transfer_bundle.py",
        "scripts/aggregate_real_only.py",
        "scripts/run_real_only_final.ps1",
        "scripts/bootstrap_real_only.ps1",
        "scripts/pack_real_only_results.py",
        "tests/test_data_lock.py",
        "tests/test_evaluate_yolo.py",
        "docs/REAL_ONLY_TRANSFER_README.md",
        "docs/REAL_ONLY_DATA_NOTE.md",
    )
    for relative in static_files:
        copy_relative(relative, bundle)
    copy_relative("docs/REAL_ONLY_TRANSFER_README.md", bundle, "README.md")

    source_package = PROJECT / "src" / "defectaug"
    for source in sorted(source_package.glob("*.py")):
        destination = bundle / "src" / "defectaug" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    portable = bundle / "data" / "splits" / "portable"
    portable.mkdir(parents=True, exist_ok=True)
    (portable / "train_fewshot_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (portable / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    for item_id in train_ids + val_ids:
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")):
            source = PROJECT / "data" / "canonical" / kind / "all" / f"{item_id}{suffix}"
            destination = bundle / "data" / "canonical" / kind / "all" / source.name
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    static_count, static_bytes = write_bundle_manifest(bundle)
    zip_path = EXPORT_ROOT / f"{BUNDLE_NAME}.zip"
    if zip_path.exists():
        raise FileExistsError(f"refusing to replace existing ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, f"{BUNDLE_NAME}/{path.relative_to(bundle).as_posix()}")
    validation = validate_relocated_bundle(zip_path)
    zip_sha256 = sha256_file(zip_path)
    checksum_path = zip_path.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{zip_sha256}  {zip_path.name}\n", encoding="utf-8")
    inventory = {
        "created_at": datetime.now().astimezone().isoformat(),
        "bundle": str(bundle),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_sha256,
        "static_manifest_files": static_count,
        "static_manifest_bytes": static_bytes,
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_in_bundle": False,
        "relocation_validation": validation,
    }
    inventory_path = EXPORT_ROOT / f"{BUNDLE_NAME}.inventory.json"
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

