"""v3.2 dev launcher (tag dev33, execution host): the coder agent driven by API models.

Reads the API key from the first line of stdin into the process environment
(LENS_API_KEY) -- it is never written to a file or a command line -- then:
real_only for both dev seeds -> two lanes in parallel:
  lane A: agent_coder_ds_multi_round  (deepseek-v4-pro-0813, text-only) s0, s1, then formula_tuned s0, s1
  lane B: agent_coder_glm_multi_round (glm-5.3-flash, vision)            s0, s1
Each step is retried up to 3 times (finished runs are skipped).  Val only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PY = str(PROJECT / ".venv" / "Scripts" / "python.exe")
API = "https://qianfan.baidubce.com/v2/tokenplan/personal"
LOGS = PROJECT / "reports" / "v3" / "dev33"
_lock = threading.Lock()


def note(msg: str) -> None:
    with _lock, (LOGS / "driver.log").open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def run(name: str, seed: int, cond: str, model: str = "lens-main") -> bool:
    for attempt in range(1, 4):
        note(f"START {name} s{seed} attempt {attempt}")
        with (LOGS / f"{name}_s{seed}.log").open("a", encoding="utf-8") as log:
            rc = subprocess.call([PY, "scripts/run_conditions_v32.py", "--tag", "dev33", "--seeds", str(seed),
                                  "--conditions", cond, "--agent-endpoint", API, "--agent-model", model],
                                 cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        note(f"END {name} s{seed} attempt {attempt}: rc={rc}")
        if rc == 0:
            return True
        time.sleep(30)
    return False


def main() -> int:
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("API key expected on stdin")
    os.environ["LENS_API_KEY"] = key
    LOGS.mkdir(parents=True, exist_ok=True)
    r = [threading.Thread(target=run, args=("real_only", s, "real_only")) for s in (0, 1)]
    for t in r:
        t.start()
        time.sleep(20)
    for t in r:
        t.join()

    def lane_a() -> None:
        for s in (0, 1):
            run("coder_ds", s, "agent_coder_ds_multi_round", "deepseek-v4-pro-0813")
        for s in (0, 1):
            run("formula_tuned", s, "formula_tuned_multi_round")

    def lane_b() -> None:
        for s in (0, 1):
            run("coder_glm", s, "agent_coder_glm_multi_round", "glm-5.3-flash")

    lanes = [threading.Thread(target=lane_a), threading.Thread(target=lane_b)]
    for t in lanes:
        t.start()
        time.sleep(30)
    for t in lanes:
        t.join()
    note("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
