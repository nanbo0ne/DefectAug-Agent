"""Launcher for the v3.1 final runs on the execution host (tag final31, 5 final seeds, val only).

Freeze check -> real_only for every seed -> a queue of per-seed tasks
(agent_experiment first, then baselines, then the ablations) on WORKERS parallel
workers.  A task is one orchestrator process (--seeds i) and runs at most one
training at a time (agent_experiment_policy.parallel_trainings: 1), so at most
the live limit (workers.txt) of trainings run concurrently -- the host's RAM, not its GPU, is the limit
(protocol_freeze_v31.json, amendment 1).  Each task is retried up to 3 times;
finished runs are skipped on retry.  Test is never touched.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PY = str(PROJECT / ".venv" / "Scripts" / "python.exe")
EP = "http://127.0.0.1:18184/v1"   # scripts/llm_relay.py on this host -> 192.168.31.47:18182 (retries dropped links)
LOGS = PROJECT / "reports" / "v3" / "final31"
WORKERS = 3          # threads; how many may run is read live from LIMIT_FILE (default 2)
LIMIT_FILE = LOGS / "workers.txt"
SEEDS = range(5)
GROUPS = [("agent", ["agent_experiment_multi_round"]),
          ("baselines", ["real_only_repeat_matched", "uniform_fixed", "formula_one_shot",
                         "formula_multi_round", "formula_tuned_multi_round"]),
          ("ablations", ["agent_multi_round", "agent_vision_multi_round"])]
_lock = threading.Lock()


def note(msg: str) -> None:
    with _lock, (LOGS / "driver.log").open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def task(name: str, seed: int, conds: list[str]) -> bool:
    for attempt in range(1, 4):
        note(f"START {name} s{seed} attempt {attempt}")
        with (LOGS / f"{name}_s{seed}.log").open("a", encoding="utf-8") as log:
            rc = subprocess.call([PY, "scripts/run_conditions_v31e.py", "--tag", "final31", "--seeds", str(seed),
                                  "--conditions", ",".join(conds), "--agent-endpoint", EP],
                                 cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        note(f"END {name} s{seed} attempt {attempt}: rc={rc}")
        if rc == 0:
            return True
        time.sleep(20)
    return False


def _limit() -> int:
    try:
        return int(LIMIT_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 2


def run_queue(items: list[tuple[str, int, list[str]]]) -> dict:
    q: queue.Queue = queue.Queue()
    for it in items:
        q.put(it)
    results: dict = {}

    def worker(k: int) -> None:
        time.sleep(20 * k)   # stagger the first trainings
        while True:
            while k >= _limit() and not q.empty():   # parked until the live limit admits this worker
                time.sleep(30)
            try:
                name, seed, conds = q.get_nowait()
            except queue.Empty:
                return
            results[f"{name}_s{seed}"] = task(name, seed, conds)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    if subprocess.call([PY, "scripts/check_freeze_v31.py"], cwd=PROJECT) != 0:
        note("FREEZE CHECK FAILED; aborting")
        return 1
    note(f"freeze check ok; worker threads={WORKERS}, live limit={_limit()} ({LIMIT_FILE.name})")
    first = run_queue([("real_only", s, ["real_only"]) for s in SEEDS])
    note(f"real_only done {first}")
    if not all(first.values()):
        return 1
    rest = run_queue([(name, s, conds) for name, conds in GROUPS for s in SEEDS])
    note(f"ALL DONE {rest}")
    return 0 if all(rest.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
