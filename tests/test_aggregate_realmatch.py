import json

from scripts import aggregate_real_only_repeat_matched as aggregate


def test_aggregate_realmatch_is_validation_only_and_keeps_per_class_metrics(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(aggregate, "PROJECT", tmp_path)
    classes = ("inclusion", "patches", "scratches")
    for index, seed in enumerate(aggregate.SEEDS):
        name = f"real_only_repeat_matched_final_s{seed}_100e"
        report = tmp_path / "reports" / "yolo_eval" / f"{name}_val" / "metrics.json"
        report.parent.mkdir(parents=True)
        per_class = [
            {
                "class_name": class_name,
                "precision": 0.4 + 0.1 * class_index,
                "recall": 0.3 + 0.1 * class_index,
                "ap50": 0.2 + 0.1 * class_index + 0.01 * index,
            }
            for class_index, class_name in enumerate(classes)
        ]
        document = {
            "evaluation": {"split": "val", "test_access_authorized": False},
            "inputs": {
                "weights": {"sha256": f"weights-{seed}"},
                "content_lock": {"sha256": "same-lock"},
            },
            "metrics": {
                "overall": {
                    "precision": 0.5 + 0.01 * index,
                    "recall": 0.4 + 0.01 * index,
                    "ap50": 0.3 + 0.01 * index,
                    "ap50_95": 0.1 + 0.01 * index,
                },
                "per_class": per_class,
                "worst_class": {
                    "class_name": "inclusion",
                    "ap50": per_class[0]["ap50"],
                },
            },
        }
        report.write_text(json.dumps(document), encoding="utf-8")

    assert aggregate.main() == 0
    output = json.loads(
        (tmp_path / "reports" / "real_only_repeat_matched_final_v1" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert output["condition"] == "real_only_repeat_matched"
    assert output["test_accessed"] is False
    assert output["aggregate"]["map50"]["mean"] == 0.31
    assert set(output["per_class_aggregate"]) == set(classes)
    assert output["per_class_aggregate"]["inclusion"]["ap50"]["mean"] == 0.21

