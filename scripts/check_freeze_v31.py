"""Verify that the v3.1 protocol and every frozen file still match protocol_freeze_v31.json."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    rec = json.loads((PROJECT / "reports" / "v3" / "protocol_freeze_v31.json").read_text(encoding="utf-8"))
    files = {rec["protocol"]: rec["protocol_sha256"], **rec["code_sha256"]}
    bad = [f for f, h in files.items() if hashlib.sha256((PROJECT / f).read_bytes()).hexdigest() != h]
    print("FREEZE OK" if not bad else f"FREEZE MISMATCH: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
