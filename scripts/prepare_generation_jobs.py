"""Create the frozen 10-source x 3-seed generator-comparison job table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = PROJECT / "data/manifests/sd900_three_class_v1.csv"
DEFAULT_OUTPUT = PROJECT / "data/manifests/generation_jobs_scratch_gate_v1.csv"
DEFAULT_SEEDS = (20260820, 20260821, 20260822)
DEFAULT_PROMPT = (
    "Monochrome close-up inspection image of hot-rolled steel. Inside the supplied "
    "mask, resynthesize one realistic narrow elongated scratch at the existing "
    "location and scale; preserve steel grain, viewpoint and illumination."
)
DEFAULT_NEGATIVE = (
    "text, watermark, frame, border, seam, repeated pattern, color, foreign object, "
    "multiple defects, concrete, fabric, wood, plastic"
)
FIELDNAMES = (
    "job_id",
    "source_id",
    "primary_class",
    "image",
    "mask",
    "label",
    "image_sha256",
    "mask_sha256",
    "label_sha256",
    "seed",
    "prompt",
    "negative_prompt",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative and unique")
    return seeds


def deterministic_rank(source_id: str, selection_seed: int) -> str:
    return hashlib.sha256(f"{selection_seed}:{source_id}".encode("utf-8")).hexdigest()


def load_sources(
    manifest_path: Path,
    *,
    primary_class: str,
    split: str,
    source_count: int,
    selection_seed: int,
) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidates = [
            dict(row)
            for row in csv.DictReader(stream)
            if row.get("primary_class") == primary_class and row.get("split") == split
        ]
    if len(candidates) < source_count:
        raise ValueError(
            f"requested {source_count} {primary_class!r}/{split!r} sources but found "
            f"only {len(candidates)} in {manifest_path}"
        )
    if len({row.get("stem") for row in candidates}) != len(candidates):
        raise ValueError("source manifest contains duplicate stems in the selected pool")
    candidates.sort(key=lambda row: (deterministic_rank(row["stem"], selection_seed), row["stem"]))
    return candidates[:source_count]


def build_jobs(
    sources: list[dict[str, str]],
    *,
    seeds: tuple[int, ...],
    prompt: str,
    negative_prompt: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for source in sources:
        source_id = source["stem"]
        image = Path(source["canonical_image"]).resolve(strict=True)
        mask = Path(source["canonical_mask"]).resolve(strict=True)
        label = Path(source["canonical_label"]).resolve(strict=True)
        hashes = {
            "image_sha256": sha256_file(image),
            "mask_sha256": sha256_file(mask),
            "label_sha256": sha256_file(label),
        }
        for seed in seeds:
            jobs.append(
                {
                    "job_id": f"{source_id}__s{seed}",
                    "source_id": source_id,
                    "primary_class": source["primary_class"],
                    "image": str(image),
                    "mask": str(mask),
                    "label": str(label),
                    **hashes,
                    "seed": seed,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )
    job_ids = [row["job_id"] for row in jobs]
    source_seed = [(row["source_id"], row["seed"]) for row in jobs]
    if len(set(job_ids)) != len(jobs) or len(set(source_seed)) != len(jobs):
        raise ValueError("generated job table contains duplicate jobs")
    return jobs


def write_jobs(output: Path, jobs: list[dict[str, Any]]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen job table: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(jobs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-name", default="scratches")
    parser.add_argument("--split", default="train_fewshot")
    parser.add_argument("--source-count", type=int, default=10)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--selection-seed", type=int, default=20260820)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source_count <= 0 or args.selection_seed < 0:
        raise ValueError("source count must be positive and selection seed non-negative")
    source_manifest = args.source_manifest.resolve(strict=True)
    sources = load_sources(
        source_manifest,
        primary_class=args.class_name,
        split=args.split,
        source_count=args.source_count,
        selection_seed=args.selection_seed,
    )
    jobs = build_jobs(
        sources,
        seeds=args.seeds,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )
    summary = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output": str(args.output.resolve()),
        "source_count": len(sources),
        "seeds": list(args.seeds),
        "job_count": len(jobs),
        "source_ids": [row["stem"] for row in sources],
    }
    if not args.dry_run:
        write_jobs(args.output.resolve(), jobs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
