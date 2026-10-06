import json
from pathlib import Path

import pytest

from defectaug.data_lock import (
    build_content_lock,
    maybe_verify_content_lock,
    verify_content_lock,
    write_content_lock,
)


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    masks = tmp_path / "masks"
    splits = tmp_path / "splits"
    for root in (images, labels, masks, splits):
        root.mkdir()
    rows = {"train_fewshot": "inclusion_1", "val": "patches_1", "test": "scratches_1"}
    for split, stem in rows.items():
        # Opaque byte fixtures: data-lock code must not attempt media decoding.
        (images / f"{stem}.jpg").write_bytes(f"image-{stem}".encode())
        (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (masks / f"{stem}.png").write_bytes(f"mask-{stem}".encode())
        (splits / f"{split}.txt").write_text(
            str(images / f"{stem}.jpg") + "\n", encoding="utf-8"
        )
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: splits/train_fewshot.txt\n"
        "val: splits/val.txt\n"
        "test: splits/test.txt\n"
        "names: [inclusion, patches, scratches]\n",
        encoding="utf-8",
    )
    return data_yaml, labels, masks


def test_content_lock_freezes_all_artifacts_and_detects_change(tmp_path: Path) -> None:
    data_yaml, labels, masks = _dataset(tmp_path)
    document = build_content_lock(
        project_root=tmp_path,
        data_yaml=data_yaml,
        labels_root=labels,
        masks_root=masks,
    )
    lock_path = tmp_path / "lock.json"
    write_content_lock(document, lock_path)

    result = verify_content_lock(
        lock_path, project_root=tmp_path, splits=("train_fewshot", "val", "test")
    )
    assert result.verified_items == 3
    assert result.verified_files == 13
    assert json.loads(lock_path.read_text())["hash_algorithm"] == "sha256-bytes-no-decode"

    (labels / "patches_1.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        verify_content_lock(lock_path, project_root=tmp_path, splits=("val",))


def test_content_lock_refuses_overwrite_and_required_mode_is_fail_closed(tmp_path: Path) -> None:
    data_yaml, labels, masks = _dataset(tmp_path)
    document = build_content_lock(
        project_root=tmp_path,
        data_yaml=data_yaml,
        labels_root=labels,
        masks_root=masks,
    )
    lock_path = tmp_path / "lock.json"
    write_content_lock(document, lock_path)
    with pytest.raises(FileExistsError, match="not overwritten"):
        write_content_lock(document, lock_path)
    with pytest.raises(FileNotFoundError, match="required dataset content lock"):
        maybe_verify_content_lock(
            tmp_path / "missing.json",
            project_root=tmp_path,
            splits=("val",),
            mode="required",
        )
    assert (
        maybe_verify_content_lock(
            tmp_path / "missing.json",
            project_root=tmp_path,
            splits=("val",),
            mode="optional",
        )
        is None
    )


def test_repeat_matched_train_requires_explicit_duplicate_mode(tmp_path: Path) -> None:
    data_yaml, labels, masks = _dataset(tmp_path)
    train_manifest = tmp_path / "splits" / "train_fewshot.txt"
    original = train_manifest.read_text(encoding="utf-8")
    train_manifest.write_text(original + original, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate image"):
        build_content_lock(
            project_root=tmp_path,
            data_yaml=data_yaml,
            labels_root=labels,
            masks_root=masks,
        )

    document = build_content_lock(
        project_root=tmp_path,
        data_yaml=data_yaml,
        labels_root=labels,
        masks_root=masks,
        allow_duplicate_train_entries=True,
    )
    train = document["splits"]["train_fewshot"]
    assert train["manifest_entry_count"] == 2
    assert train["unique_item_count"] == 1
    assert len(train["items"]) == 1


def test_content_lock_can_exclude_locked_test_from_transfer_bundle(tmp_path: Path) -> None:
    data_yaml, labels, masks = _dataset(tmp_path)
    data_yaml.write_text(
        "path: .\n"
        "train: splits/train_fewshot.txt\n"
        "val: splits/val.txt\n"
        "names: [inclusion, patches, scratches]\n",
        encoding="utf-8",
    )
    document = build_content_lock(
        project_root=tmp_path,
        data_yaml=data_yaml,
        labels_root=labels,
        masks_root=masks,
        active_splits={"train_fewshot": "train", "val": "val"},
    )
    assert set(document["splits"]) == {"train_fewshot", "val"}
    lock_path = tmp_path / "trainval-lock.json"
    write_content_lock(document, lock_path)
    result = verify_content_lock(
        lock_path,
        project_root=tmp_path,
        splits=("train_fewshot", "val"),
        expected_data_yaml=data_yaml,
    )
    assert result.verified_items == 2
    with pytest.raises(ValueError, match="has no 'test' split"):
        verify_content_lock(lock_path, project_root=tmp_path, splits=("test",))
