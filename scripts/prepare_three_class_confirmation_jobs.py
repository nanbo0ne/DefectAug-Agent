"""Freeze the shared 3-class x 5-source x 2-seed generator confirmation table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_generation_jobs import (
    build_jobs,
    load_sources,
    parse_seeds,
    sha256_file,
    write_jobs,
)


PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
DEFAULT_SOURCE_MANIFEST = PROJECT / "data/manifests/sd900_three_class_v1.csv"
DEFAULT_PROMPT_LIBRARY = PROJECT / "configs/prompt_library_v1.yaml"
DEFAULT_OUTPUT = PROJECT / "data/manifests/generation_jobs_three_class_confirmation_v1.csv"
DEFAULT_SEEDS = (20260820, 20260821)


def load_prompts(path: Path) -> tuple[dict[str, str], str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("base"), str):
        raise ValueError("prompt library must contain a string base template")
    class_records = document.get("classes")
    negative = document.get("negative")
    if not isinstance(class_records, dict) or not isinstance(negative, str):
        raise ValueError("prompt library must contain classes and negative text")
    prompts: dict[str, str] = {}
    for class_name in CLASSES:
        record = class_records.get(class_name)
        if not isinstance(record, dict) or not isinstance(record.get("description"), str):
            raise ValueError(f"prompt library lacks a description for {class_name}")
        prompts[class_name] = document["base"].format(
            class_description=record["description"]
        )
    return prompts, negative


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--prompt-library", type=Path, default=DEFAULT_PROMPT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train_fewshot")
    parser.add_argument("--sources-per-class", type=int, default=5)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--selection-seed", type=int, default=20260820)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sources_per_class <= 0 or args.selection_seed < 0:
        raise ValueError("sources per class must be positive and selection seed non-negative")
    source_manifest = args.source_manifest.resolve(strict=True)
    prompt_library = args.prompt_library.resolve(strict=True)
    prompts, negative = load_prompts(prompt_library)
    jobs = []
    source_ids: dict[str, list[str]] = {}
    for class_name in CLASSES:
        sources = load_sources(
            source_manifest,
            primary_class=class_name,
            split=args.split,
            source_count=args.sources_per_class,
            selection_seed=args.selection_seed,
        )
        source_ids[class_name] = [row["stem"] for row in sources]
        jobs.extend(
            build_jobs(
                sources,
                seeds=args.seeds,
                prompt=prompts[class_name],
                negative_prompt=negative,
            )
        )

    expected = len(CLASSES) * args.sources_per_class * len(args.seeds)
    if len(jobs) != expected or len({row["job_id"] for row in jobs}) != expected:
        raise RuntimeError("three-class confirmation table has an invalid job count")
    output = args.output.resolve()
    if not args.dry_run:
        write_jobs(output, jobs)
    summary = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "prompt_library": str(prompt_library),
        "prompt_library_sha256": sha256_file(prompt_library),
        "output": str(output),
        "classes": list(CLASSES),
        "sources_per_class": args.sources_per_class,
        "seeds": list(args.seeds),
        "job_count": len(jobs),
        "source_ids": source_ids,
        "job_content_digest": hashlib.sha256(
            json.dumps(jobs, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
