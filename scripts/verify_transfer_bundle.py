"""Verify the static transfer manifest without opening media as images."""

from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = PROJECT / "bundle_manifest.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, separator, relative = raw.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative:
            raise ValueError(f"invalid bundle manifest line {line_number}")
        path = (PROJECT / Path(relative)).resolve()
        try:
            path.relative_to(PROJECT.resolve())
        except ValueError as exc:
            raise ValueError(f"manifest path escapes bundle: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual.lower() != digest.lower():
            raise RuntimeError(f"bundle SHA-256 mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise ValueError("bundle manifest is empty")
    print(f"bundle verification passed: {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

