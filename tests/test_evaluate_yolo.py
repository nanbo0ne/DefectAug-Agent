from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from scripts.evaluate_yolo import (
    DatasetSplitLock,
    ManifestLock,
    build_parser,
    extract_detection_metrics,
    lock_dataset_split,
    main,
    runtime_dataset_document,
)


def test_default_split_is_val_and_runtime_yaml_hides_test(tmp_path) -> None:
    args = build_parser().parse_args(["--weights", "best.pt"])
    assert args.split == "val"
    assert args.content_lock_mode == "required"

    manifest = tmp_path / "val.txt"
    manifest.write_text("image.jpg\n", encoding="utf-8")
    lock = DatasetSplitLock(
        data_yaml=tmp_path / "data.yaml",
        data_yaml_sha256="a" * 64,
        logical_split="val",
        split_sha256="b" * 64,
        manifests=(ManifestLock(manifest, "b" * 64, 1),),
        class_names={0: "defect"},
    )
    runtime = runtime_dataset_document(lock)
    assert runtime["train"] == str(manifest)
    assert runtime["val"] == str(manifest)
    assert "test" not in runtime


def test_lock_opens_only_selected_manifest(tmp_path) -> None:
    val_manifest = tmp_path / "val.txt"
    val_manifest.write_text("image_a.jpg\nimage_b.jpg\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: .\nval: val.txt\ntest: deliberately_missing_test.txt\nnames: [defect]\n",
        encoding="utf-8",
    )

    lock = lock_dataset_split(data_yaml, "val")
    assert lock.logical_split == "val"
    assert lock.manifests[0].entries == 2

    with pytest.raises(FileNotFoundError, match="selected test manifest"):
        lock_dataset_split(data_yaml, "test")


def test_lock_rejects_wrong_expected_split_hash(tmp_path) -> None:
    manifest = tmp_path / "val.txt"
    manifest.write_text("image.jpg\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("val: val.txt\nnames: [defect]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        lock_dataset_split(data_yaml, "val", expected_split_sha256="0" * 64)


def test_runtime_yaml_exposes_test_only_for_explicit_test_lock(tmp_path) -> None:
    manifest = tmp_path / "test.txt"
    lock = DatasetSplitLock(
        data_yaml=tmp_path / "data.yaml",
        data_yaml_sha256="a" * 64,
        logical_split="test",
        split_sha256="b" * 64,
        manifests=(ManifestLock(manifest, "b" * 64, 3),),
        class_names={0: "a", 1: "b"},
    )
    runtime = runtime_dataset_document(lock)
    assert runtime["test"] == str(manifest)
    assert runtime["train"] == runtime["val"] == runtime["test"]


def test_test_gate_denies_before_opening_missing_test_manifest(tmp_path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"opaque-test-weights")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "test: deliberately_missing_test.txt\nnames: [defect]\n", encoding="utf-8"
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "experiment:\n"
        "  status: pilot_not_frozen\n"
        "final_test_access:\n"
        "  enabled: false\n"
        "  required_experiment_status: frozen_final\n"
        "  unlock_token: FINAL_TEST_EVALUATION\n",
        encoding="utf-8",
    )
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    audit = tmp_path / "test_access.jsonl"

    with pytest.raises(PermissionError, match="must use frozen protocol path"):
        main(
            [
                "--weights",
                str(weights),
                "--data",
                str(data_yaml),
                "--split",
                "test",
                "--protocol",
                str(protocol),
                "--expected-protocol-sha256",
                protocol_hash,
                "--final-test-unlock-token",
                "FINAL_TEST_EVALUATION",
                "--test-audit-log",
                str(audit),
                "--content-lock-mode",
                "off",
            ]
        )
    assert audit.is_file()


def test_metric_export_maps_class_ids_and_computes_worst_and_population_std() -> None:
    box = SimpleNamespace(
        ap_class_index=np.array([0, 2]),
        p=np.array([0.9, 0.6]),
        r=np.array([0.7, 0.5]),
        ap50=np.array([0.8, 0.5]),
        ap=np.array([0.6, 0.3]),
        mp=np.float64(0.75),
        mr=np.float64(0.60),
        map50=np.float64(0.65),
        map=np.float64(0.45),
    )
    raw = SimpleNamespace(
        box=box,
        nt_per_image=np.array([10, 0, 8]),
        nt_per_class=np.array([15, 0, 12]),
        speed={"inference": 1.25},
    )

    result = extract_detection_metrics(raw, {0: "head", 1: "missing", 2: "tail"})

    assert result["overall"]["ap50"] == pytest.approx(0.65)
    assert result["per_class"][1]["evaluated"] is False
    assert result["per_class"][1]["ap50"] is None
    assert result["per_class"][2]["instances"] == 12
    assert result["worst_class"]["class_name"] == "tail"
    assert result["worst_class"]["ap50"] == pytest.approx(0.5)
    assert result["class_metric_std"]["ap50"] == pytest.approx(0.15)
