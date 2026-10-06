"""Local v3.2 dev queue (tag dev32, dev seeds 0/1, val only) on the laptop.

Tasks run as separate orchestrator processes on WORKERS parallel workers (the laptop's
RAM fits 3 trainings); each is retried up to 3 times and resumes from its caches.
Qwen tasks go through the local retrying relay (127.0.0.1:18185 -> 192.168.31.47);
API tasks use the Qianfan endpoint with the key from this process's env LENS_API_KEY
(never written anywhere).
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PY = str(PROJECT / ".venv" / "Scripts" / "python.exe")
LOGS = PROJECT / "reports" / "v3" / "dev32"
QWEN = ("http://127.0.0.1:18185/v1", "lens-main")
API = "https://qianfan.baidubce.com/v2/tokenplan/personal"
DS, GLM = (API, "deepseek-v4-pro-0813"), (API, "glm-5.3-flash")
WORKERS = 3
TASKS = [  # (log name, seed index, condition, (endpoint, model)) in priority order
    ("formula_tuned", 0, "formula_tuned_multi_round", QWEN),
    ("coder_ds", 0, "agent_coder_ds_multi_round", DS),
    ("coder_glm", 0, "agent_coder_glm_multi_round", GLM),
    ("formula_tuned", 1, "formula_tuned_multi_round", QWEN),
    ("coder_ds", 1, "agent_coder_ds_multi_round", DS),
    ("coder_glm", 1, "agent_coder_glm_multi_round", GLM),
    ("memory", 0, "agent_memory_multi_round", QWEN),
    ("memory", 1, "agent_memory_multi_round", QWEN),
    ("coder_qwen", 1, "agent_coder_multi_round", QWEN),
    ("experiment", 0, "agent_experiment_multi_round", QWEN),
    ("experiment", 1, "agent_experiment_multi_round", QWEN),
]
_lock = threading.Lock()


def note(msg: str) -> None:
    with _lock, (LOGS / "queue.log").open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def task(name: str, seed: int, cond: str, ep: tuple[str, str]) -> bool:
    for attempt in range(1, 4):
        note(f"START {name} s{seed} attempt {attempt}")
        with (LOGS / f"q_{name}_s{seed}.log").open("a", encoding="utf-8") as log:
            rc = subprocess.call([PY, "scripts/run_conditions_v32.py", "--tag", "dev32", "--seeds", str(seed),
                                  "--conditions", cond, "--agent-endpoint", ep[0], "--agent-model", ep[1]],
                                 cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        note(f"END {name} s{seed} attempt {attempt}: rc={rc}")
        if rc == 0:
            return True
        time.sleep(60)
    return False


def main() -> int:
    if not os.environ.get("LENS_API_KEY"):
        raise SystemExit("LENS_API_KEY must be set in this process's environment")
    LOGS.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    for t in TASKS:
        q.put(t)
    results: dict = {}

    def worker(k: int) -> None:
        time.sleep(30 * k)
        while True:
            try:
                name, seed, cond, ep = q.get_nowait()
            except queue.Empty:
                return
            results[f"{name}_s{seed}"] = task(name, seed, cond, ep)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    note(f"ALL DONE {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
