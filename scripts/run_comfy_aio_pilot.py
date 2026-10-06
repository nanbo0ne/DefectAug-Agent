"""Run a small, explicitly non-formal material test with the existing Qwen AIO.

The AIO has no mask-conditioning input.  It edits the full image; this script then
resizes the raw result to the source resolution and hard-composites only the supplied
mask.  Outputs from this script are therefore exploratory candidates, not formal
Qwen Mask-Inpainting results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch

from defectaug.comfy_client import ComfyClient
from defectaug.quality import candidate_metrics, hard_composite


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "A monochrome close-up inspection image of a hot-rolled steel surface. "
    "Reconstruct the one existing narrow elongated scratch as a different realistic "
    "abrasion groove at the same location and scale. Preserve the steel texture, "
    "illumination, viewpoint, grayscale tone and every other part of the image. "
    "Natural defect boundary, no foreign object, no text, no border, no repeated pattern."
)
DEFAULT_NEGATIVE = (
    "color, text, letters, watermark, frame, border, seam, collage, repeated pattern, "
    "foreign object, changed background, changed illumination, multiple new defects"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_saved_image(history: dict[str, Any]) -> dict[str, Any]:
    for node_output in history.get("outputs", {}).values():
        images = node_output.get("images", [])
        if images:
            return images[0]
    status = history.get("status", {})
    raise RuntimeError(f"ComfyUI returned no image; status={status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, default=PROJECT / "workflows/aio_edit_api_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data/synthetic/pilot/aio")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    image_path = args.image.resolve(strict=True)
    mask_path = args.mask.resolve(strict=True)
    workflow_path = args.workflow.resolve(strict=True)
    if not 1 <= args.steps <= 10000:
        raise ValueError("steps must be within [1, 10000]")
    if not 0 <= args.denoise <= 1:
        raise ValueError("denoise must be within [0, 1]")

    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source is None or mask is None:
        raise ValueError("source image or mask is unreadable")
    if mask.shape != source.shape[:2] or not (mask > 0).any():
        raise ValueError("mask must be non-empty and match the source dimensions")

    template = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow = deepcopy(template)
    workflow["4"]["inputs"]["prompt"] = args.prompt
    workflow["5"]["inputs"]["prompt"] = args.negative_prompt
    workflow["11"]["inputs"].update(
        {"seed": args.seed, "steps": args.steps, "cfg": args.cfg, "denoise": args.denoise}
    )

    client = ComfyClient(args.base_url)
    health = client.health()
    if args.validate_only:
        print(json.dumps({"status": "validated", "health": health}, ensure_ascii=False, indent=2))
        return

    uploaded = client.upload_image(image_path, subfolder="defectaug/aio_pilot")
    workflow["2"]["inputs"]["image"] = "/".join(
        part for part in (uploaded.get("subfolder", ""), uploaded["name"]) if part
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_seed{args.seed}"
    raw_path = args.output_dir / f"{stem}_raw.png"
    final_path = args.output_dir / f"{stem}_composite.png"
    record_path = args.output_dir / f"{stem}.json"

    started = time.perf_counter()
    prompt_id = client.queue(workflow)
    history = client.wait_history(prompt_id, timeout_seconds=args.timeout_seconds)
    elapsed = time.perf_counter() - started
    output_metadata = first_saved_image(history)
    client.download_image(output_metadata, raw_path)

    generated = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if generated is None:
        raise RuntimeError("downloaded ComfyUI output is unreadable")
    if generated.shape != source.shape:
        generated = cv2.resize(
            generated, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )
    composite = hard_composite(source, generated, mask)
    if not cv2.imwrite(str(final_path), composite):
        raise RuntimeError(f"failed to save {final_path}")
    metrics = candidate_metrics(source, composite, mask).as_dict()

    record = {
        "purpose": "exploratory_aio_material_test_not_formal_mask_inpainting",
        "created_at": datetime.now().astimezone().isoformat(),
        "source": str(image_path),
        "source_sha256": sha256(image_path),
        "mask": str(mask_path),
        "mask_sha256": sha256(mask_path),
        "workflow": str(workflow_path),
        "workflow_sha256": sha256(workflow_path),
        "prompt_id": prompt_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "denoise": args.denoise,
        "elapsed_seconds": elapsed,
        "raw_output": str(raw_path),
        "composite_output": str(final_path),
        "raw_output_sha256": sha256(raw_path),
        "composite_output_sha256": sha256(final_path),
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "comfyui": health.get("system", {}).get("comfyui_version"),
            "comfyui_argv": health.get("system", {}).get("argv"),
        },
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = args.output_dir / "manifest.csv"
    write_header = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "source_id", "seed", "raw_output", "composite_output", "elapsed_seconds",
                "outside_changed_fraction", "inside_changed_fraction",
                "changed_region_mask_iou", "mean_absolute_change_inside", "review_status",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "source_id": image_path.stem,
                "seed": args.seed,
                "raw_output": raw_path,
                "composite_output": final_path,
                "elapsed_seconds": f"{elapsed:.3f}",
                **{key: f"{value:.6f}" for key, value in metrics.items()},
                "review_status": "pending_human_review",
            }
        )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
