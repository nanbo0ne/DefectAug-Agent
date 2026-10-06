"""Create a generator-hidden review package without decoding candidate images.

The reviewer package contains only random review codes, expected defect class,
byte-identical PNG copies and an empty scoring sheet.  The origin mapping is
written to a separate path and is rejected if it sits inside the package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PENDING_REVIEW = "pending_blind_human_review"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def target_class(job: dict[str, Any]) -> str:
    extra = job.get("extra") if isinstance(job.get("extra"), dict) else {}
    value = job.get("primary_class") or extra.get("primary_class")
    if value:
        return str(value).strip().lower()
    source_id = str(job.get("source_id") or "")
    if "_" in source_id:
        return source_id.rsplit("_", maxsplit=1)[0].strip().lower()
    raise ValueError(f"cannot infer target class from job: {job}")


def load_candidates(manifests: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"generation manifest does not exist: {manifest}")
        records = read_jsonl(manifest)
        assignments: dict[tuple[str, str], str] = {}
        for record in records:
            if record.get("record_type") != "review_status_assignment":
                continue
            key = (str(record.get("run_id") or ""), str(record.get("job_id") or ""))
            status = str(record.get("review_status") or "")
            if not all(key) or status != PENDING_REVIEW:
                raise ValueError(f"invalid review status assignment: {manifest} {record}")
            if key in assignments:
                raise ValueError(f"duplicate review status assignment: {manifest} {key}")
            assignments[key] = status
        for record in records:
            if record.get("record_type") != "result" or record.get("status") != "ok":
                continue
            result = record.get("result")
            job = record.get("job")
            if not isinstance(result, dict) or not isinstance(job, dict):
                raise ValueError(f"successful result lacks job/result objects: {manifest}")
            job_id = str(job.get("job_id") or "")
            run_id = str(record.get("run_id") or "")
            review_status = result.get("review_status") or assignments.get((run_id, job_id))
            if review_status != PENDING_REVIEW:
                raise ValueError(
                    f"candidate is not explicitly pending blind review: "
                    f"{manifest} job={job_id}"
                )
            output_value = result.get("composite_output") or job.get("output")
            if not output_value:
                raise ValueError(f"candidate has no composite output: {manifest}")
            output = Path(str(output_value)).expanduser().resolve()
            if not output.is_file() or output.suffix.lower() != ".png":
                raise FileNotFoundError(f"candidate PNG does not exist: {output}")
            actual_sha256 = sha256_file(output)
            recorded_sha256 = str(
                result.get("composite_sha256") or result.get("output_sha256") or ""
            ).lower()
            if recorded_sha256 != actual_sha256:
                raise ValueError(f"candidate hash disagrees with generation record: {output}")
            candidates.append(
                {
                    "source_manifest": str(manifest.resolve()),
                    "source_manifest_sha256": sha256_file(manifest),
                    "run_id": run_id,
                    "job_id": job_id,
                    "source_id": str(job.get("source_id") or ""),
                    "target_class": target_class(job),
                    "original_output": str(output),
                    "candidate_sha256": actual_sha256,
                    "review_status_source": (
                        "result" if result.get("review_status") == PENDING_REVIEW else "append_only_assignment"
                    ),
                }
            )
    if not candidates:
        raise ValueError("no successful pending-blind-review candidates were found")
    identities = [
        (item["source_manifest"], item["run_id"], item["job_id"]) for item in candidates
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate candidate identity across generation manifests")
    return candidates


def ensure_mapping_separate(package_dir: Path, mapping: Path) -> None:
    try:
        mapping.resolve().relative_to(package_dir.resolve())
    except ValueError:
        return
    raise ValueError("origin mapping must be outside the reviewer package directory")


def prepare_package(
    *,
    manifests: list[Path],
    package_dir: Path,
    mapping: Path,
    seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifests = [path.resolve() for path in manifests]
    package_dir = package_dir.resolve()
    mapping = mapping.resolve()
    ensure_mapping_separate(package_dir, mapping)
    candidates = load_candidates(manifests)
    order = list(candidates)
    random.Random(seed).shuffle(order)
    planned: list[dict[str, Any]] = []
    for index, candidate in enumerate(order, start=1):
        code = f"R{index:04d}"
        planned.append(
            {
                **candidate,
                "review_code": code,
                "review_image": str(package_dir / "items" / f"{code}.png"),
            }
        )
    plan = {
        "seed": seed,
        "candidate_count": len(planned),
        "package_dir": str(package_dir),
        "mapping": str(mapping),
        "items": planned,
    }
    if dry_run:
        return plan
    if package_dir.exists() and any(package_dir.iterdir()):
        raise FileExistsError(f"reviewer package directory is not empty: {package_dir}")
    if mapping.exists():
        raise FileExistsError(f"refusing to overwrite origin mapping: {mapping}")
    items_dir = package_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)

    public_items: list[dict[str, Any]] = []
    for item in planned:
        source = Path(item["original_output"])
        destination = Path(item["review_image"])
        shutil.copyfile(source, destination)
        copied_sha256 = sha256_file(destination)
        if copied_sha256 != item["candidate_sha256"]:
            raise RuntimeError(f"blind-review copy changed bytes: {destination}")
        public_items.append(
            {
                "review_code": item["review_code"],
                "target_class": item["target_class"],
                "image": f"items/{destination.name}",
                "sha256": copied_sha256,
            }
        )

    sheet = package_dir / "review_sheet.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
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
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate_count": len(public_items),
        "reviewer_instructions": (
            "Do not infer or record generator identity. Score semantic correctness, "
            "realism, boundary artifacts and label usability before accept/reject."
        ),
        "items": public_items,
    }
    (package_dir / "package_manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with mapping.open("w", encoding="utf-8", newline="\n") as stream:
        for item in planned:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        action="append",
        required=True,
        help="Generation JSONL; repeat for every generator/method being blinded.",
    )
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = prepare_package(
        manifests=[project_path(value) for value in args.input_manifest],
        package_dir=project_path(args.package_dir),
        mapping=project_path(args.mapping),
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
