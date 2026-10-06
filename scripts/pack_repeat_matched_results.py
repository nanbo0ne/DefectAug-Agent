"""Package completed repeat-matched artifacts for return to the primary machine."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (20260820, 20260821, 20260822)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    required: list[Path] = [
        PROJECT / "configs" / "real_only_repeat_matched_final_v1.yaml",
        PROJECT / "configs" / "dataset_repeat_matched_v1.yaml",
        PROJECT / "data" / "splits" / "v1" / "train_fewshot_repeat_matched.txt",
        PROJECT / "data" / "manifests" / "dataset_repeat_matched_trainval_lock_v1.json",
        PROJECT / "reports" / "repeat_matched_machine_setup.json",
        PROJECT / "reports" / "real_only_repeat_matched_final_v1" / "summary.json",
        PROJECT / "reports" / "real_only_repeat_matched_final_v1" / "runs.csv",
        PROJECT / "reports" / "real_only_repeat_matched_final_v1" / "summary.md",
    ]
    for seed in SEEDS:
        name = f"real_only_repeat_matched_final_s{seed}_100e"
        required.extend(
            [
                PROJECT / "runs" / "yolo" / f"{name}_launch.json",
                PROJECT / "runs" / "yolo" / name / "args.yaml",
                PROJECT / "runs" / "yolo" / name / "results.csv",
                PROJECT / "runs" / "yolo" / name / "weights" / "best.pt",
                PROJECT / "runs" / "yolo" / name / "weights" / "last.pt",
                PROJECT / "reports" / "yolo_eval" / f"{name}_val" / "metrics.json",
                PROJECT / "reports" / "yolo_eval" / f"{name}_val" / "metrics.csv",
            ]
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing completed artifacts:\n" + "\n".join(map(str, missing)))

    output = PROJECT / f"RepeatMatched-results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in required:
            relative = path.relative_to(PROJECT).as_posix()
            archive.write(path, relative)
            hashes[relative] = sha256_file(path)
        archive.writestr("RESULT_FILES_SHA256.json", json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {"zip": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

