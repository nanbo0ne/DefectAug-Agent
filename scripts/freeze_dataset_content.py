"""Create or verify the active dataset's byte-level content lock.

Image and mask files are hashed as opaque bytes.  This command never decodes,
renders or previews media.  Existing locks are not replaced unless --force is
explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defectaug.data_lock import build_content_lock, verify_content_lock, write_content_lock


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "data" / "manifests" / "dataset_content_lock_v1.json"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or verify image/label/mask SHA-256 values without decoding media."
    )
    parser.add_argument("--verify", action="store_true", help="Verify an existing lock.")
    parser.add_argument("--data", default="configs/dataset_sd900_3class_v1.yaml")
    parser.add_argument("--labels-root", default="data/canonical/labels/all")
    parser.add_argument("--masks-root", default="data/canonical/masks/all")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--split",
        action="append",
        choices=("train_fewshot", "val", "test"),
        help="Verification split; repeat as needed. Default verifies all active splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing lock when creating; never implied.",
    )
    parser.add_argument(
        "--allow-duplicate-train-entries",
        action="store_true",
        help=(
            "Allow an explicitly constructed repeat-matched train manifest; unique media "
            "are hashed once while the ordered manifest bytes and total entry count are locked."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = project_path(args.output)
    if args.verify:
        verification = verify_content_lock(
            output,
            project_root=PROJECT,
            splits=args.split or ("train_fewshot", "val", "test"),
        )
        print(json.dumps(verification.as_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.split:
        raise ValueError("--split applies only with --verify; creation always freezes all splits")
    document = build_content_lock(
        project_root=PROJECT,
        data_yaml=project_path(args.data),
        labels_root=project_path(args.labels_root),
        masks_root=project_path(args.masks_root),
        allow_duplicate_train_entries=args.allow_duplicate_train_entries,
    )
    write_content_lock(document, output, force=args.force)
    print(
        json.dumps(
            {
                "path": str(output),
                "content_digest": document["content_digest"],
                "items": {
                    split: len(record["items"])
                    for split, record in document["splits"].items()
                },
                "overwrote_existing": bool(args.force),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
