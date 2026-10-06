"""Create a method-hidden A/B review package for the boundary-blend gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_separate(package: Path, mapping: Path) -> None:
    try:
        mapping.resolve().relative_to(package.resolve())
    except ValueError:
        return
    raise ValueError("method mapping must be outside the reviewer package")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-records",
        default="data/synthetic/patch_copy_pilot_v2/records.jsonl",
    )
    parser.add_argument(
        "--blend-records",
        default="data/synthetic/boundary_blend_gate_v2/records.jsonl",
    )
    parser.add_argument("--clone-mode", default="gaussian")
    parser.add_argument("--ring-radius", type=int, default=5)
    parser.add_argument(
        "--package",
        default="reports/blind_review/boundary_blend_v2_reviewer",
    )
    parser.add_argument(
        "--mapping",
        default="reports/review_keys/boundary_blend_v2_mapping.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    baseline_records_path = resolve_path(args.baseline_records)
    blend_records_path = resolve_path(args.blend_records)
    package = resolve_path(args.package)
    mapping = resolve_path(args.mapping)
    ensure_separate(package, mapping)
    if package.exists() and any(package.iterdir()):
        raise FileExistsError(f"review package is non-empty: {package}")
    if mapping.exists():
        raise FileExistsError(f"refusing to overwrite mapping: {mapping}")

    baseline = read_jsonl(baseline_records_path)
    blend = [
        row
        for row in read_jsonl(blend_records_path)
        if row["clone_mode"] == args.clone_mode
        and int(row["ring_radius_pixels"]) == args.ring_radius
    ]
    if len(baseline) != 9 or len(blend) != 9:
        raise RuntimeError("blind A/B gate requires exactly 9 A and 9 selected B records")
    if {str(row["id"]) for row in baseline} != {str(row["id"]) for row in blend}:
        raise RuntimeError("A and B task IDs do not match")

    candidates: list[dict[str, object]] = []
    for row in baseline:
        image = resolve_path(str(row["image"]))
        if sha256_file(image) != row["image_sha256"]:
            raise RuntimeError(f"baseline image hash mismatch: {row['id']}")
        candidates.append(
            {
                "task_id": row["id"],
                "target_class": row["class"],
                "method": "A_copy_paste",
                "source_image": str(image),
                "source_sha256": row["image_sha256"],
            }
        )
    for row in blend:
        image = resolve_path(str(row["image"]))
        if sha256_file(image) != row["image_sha256"]:
            raise RuntimeError(f"blend image hash mismatch: {row['id']}")
        candidates.append(
            {
                "task_id": row["id"],
                "target_class": row["class"],
                "method": f"B_{args.clone_mode}_r{args.ring_radius}",
                "source_image": str(image),
                "source_sha256": row["image_sha256"],
            }
        )
    random.Random(args.seed).shuffle(candidates)

    items_dir = package / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    public_items: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        review_code = f"R{index:04d}"
        destination = items_dir / f"{review_code}.png"
        shutil.copyfile(str(candidate["source_image"]), destination)
        copied_hash = sha256_file(destination)
        if copied_hash != candidate["source_sha256"]:
            raise RuntimeError(f"review copy changed bytes: {review_code}")
        public_items.append(
            {
                "review_code": review_code,
                "target_class": candidate["target_class"],
                "image": f"items/{destination.name}",
                "sha256": copied_hash,
            }
        )
        mapping_rows.append({"review_code": review_code, **candidate})

    with (package / "review_sheet.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "review_code",
            "target_class",
            "semantic_correct",
            "realistic",
            "boundary_artifact",
            "label_usable",
            "accept",
            "reviewer",
            "notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in public_items:
            writer.writerow(
                {
                    "review_code": item["review_code"],
                    "target_class": item["target_class"],
                }
            )
    public_manifest = {
        "schema_version": 1,
        "candidate_count": len(public_items),
        "seed_not_disclosed_to_reviewer": True,
        "method_identity_in_package": False,
        "instructions": (
            "Review each image independently. Use 0/1 for semantic_correct, realistic, "
            "boundary_artifact, label_usable and accept. boundary_artifact=1 means a "
            "visible paste seam or halo. Do not infer method identity."
        ),
        "items": public_items,
    }
    (package / "package_manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mapping.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in mapping_rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": len(public_items),
                "package": str(package),
                "mapping": str(mapping),
                "package_manifest_sha256": sha256_file(package / "package_manifest.json"),
                "mapping_sha256": sha256_file(mapping),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
