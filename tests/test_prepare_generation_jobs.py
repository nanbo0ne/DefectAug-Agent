"""Tests for the frozen shared generator-comparison job table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "prepare_generation_jobs.py"
SPEC = importlib.util.spec_from_file_location("prepare_generation_jobs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_jobs_makes_same_three_seeds_for_ten_sources(tmp_path: Path) -> None:
    sources = []
    for index in range(10):
        image = tmp_path / f"s{index}.jpg"
        mask = tmp_path / f"s{index}.png"
        label = tmp_path / f"s{index}.txt"
        image.write_bytes(f"image-{index}".encode())
        mask.write_bytes(f"mask-{index}".encode())
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        sources.append(
            {
                "stem": f"s{index}",
                "primary_class": "scratches",
                "canonical_image": str(image),
                "canonical_mask": str(mask),
                "canonical_label": str(label),
            }
        )

    seeds = (20260820, 20260821, 20260822)
    jobs = MODULE.build_jobs(sources, seeds=seeds, prompt="scratch", negative_prompt="bad")

    assert len(jobs) == 30
    assert len({row["job_id"] for row in jobs}) == 30
    for source in sources:
        assert {
            row["seed"] for row in jobs if row["source_id"] == source["stem"]
        } == set(seeds)
    assert all(len(row["label_sha256"]) == 64 for row in jobs)
