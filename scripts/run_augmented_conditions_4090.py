"""Orchestrate the four augmented conditions on the local RTX 4090.

Conditions and rounds per seed (final_seeds: 20260820/21/22):

- uniform_fixed        : 1 round, alloc 50/50/50, train on 100+150
- formula_one_shot     : 1 round, alloc from real_only metrics (total 150,
                         min 10, max 70%, capacity-clamped)
- formula_multi_round  : 3 rounds x 50, alloc recomputed from previous
                         round's validation metrics (min 0, max 70%/round,
                         cumulative cap 105 and source-capacity clamp)
- agent_multi_round    : 3 rounds x 50, alloc from the LM Studio agent
                         (agent_decide.py) under the same constraints

Recovery semantics per (condition, seed, round): if metrics.json exists the
round is done; otherwise the synthetic pool is deterministically replayed
(pools are wiped and regenerated from the recorded per-round allocations,
which are deterministic functions of the same seeds/metrics), the dataset
YAML + content lock are rebuilt, then train + val-eval run.

Usage:
    python scripts/run_augmented_conditions_4090.py [--phase no_llm|agent|all]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PY = PROJECT / ".venv" / "Scripts" / "python.exe"
EVAL_ROOT = PROJECT / "reports" / "yolo_eval"
SEEDS = (20260820, 20260821, 20260822)
TAG = "4090"
EPOCHS = 100
CLASSES = ("inclusion", "patches", "scratches")
AGENT_ENDPOINT = "http://127.0.0.1:8765/v1"
AGENT_MODEL = "qwen2.5-14b-instruct"
REAL_ONLY_METRICS = {
    seed: EVAL_ROOT / f"real_only_4090_s{seed}_100e_val" / "metrics.json" for seed in SEEDS
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], log_path: Path | None = None) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    if log_path is None:
        return subprocess.call([str(c) for c in cmd], cwd=PROJECT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        return subprocess.call([str(c) for c in cmd], cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)


def _emit_manifest(pool: Path, round_index: int) -> Path:
    real_lines = [
        raw.strip()
        for raw in (PROJECT / "data" / "splits" / "v1" / "train_fewshot.txt")
        .read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    synth_paths = sorted(pool.glob("r*/images/*.png"))
    manifest = pool / f"train_manifest_r{round_index}.txt"
    manifest.write_text(
        "\n".join(real_lines + [str(p) for p in synth_paths]) + "\n",
        encoding="utf-8",
    )
    return manifest


def _alloc_total(alloc: str) -> int:
    return sum(int(part.split(":")[1]) for part in alloc.split(","))


def _generate_round(condition: str, seed: int, round_index: int, alloc: str, candidate_budget: int) -> None:
    rc = run(
        [
            PY, "scripts/run_synth_generation.py",
            "--condition", condition, "--seed", seed, "--round", round_index,
            "--alloc", alloc, "--candidate-budget", candidate_budget,
        ]
    )
    if rc not in (0, 2):
        raise RuntimeError(f"generation failed rc={rc} for {condition} s{seed} r{round_index}")
    (PROJECT / "data" / "synthetic" / f"{condition}_{seed}" / f"alloc_r{round_index}.txt").write_text(
        alloc, encoding="utf-8"
    )


def generate(condition: str, seed: int, round_index: int, alloc: str, candidate_budget: int) -> Path:
    """Generate one round idempotently and emit the cumulative manifest.

    If the round's expected image count is already present, generation is
    skipped.  A partial/inconsistent pool is wiped and deterministically
    replayed from the saved per-round allocations.
    """
    import shutil

    pool = PROJECT / "data" / "synthetic" / f"{condition}_{seed}"
    images_dir = pool / f"r{round_index}" / "images"
    existing = len(list(images_dir.glob("*.png"))) if images_dir.is_dir() else 0
    expected = _alloc_total(alloc)
    if existing == expected and expected > 0:
        log(f"GEN SKIP {condition} s{seed} r{round_index} ({existing} images present)")
        return _emit_manifest(pool, round_index)
    if existing:
        saved = {}
        for k in range(1, round_index):
            alloc_file = pool / f"alloc_r{k}.txt"
            if alloc_file.is_file():
                saved[k] = alloc_file.read_text(encoding="utf-8").strip()
        log(f"GEN REPLAY {condition} s{seed} r{round_index} ({existing}/{expected}; wiping pool)")
        shutil.rmtree(pool)
        for k in range(1, round_index):
            if k not in saved:
                raise RuntimeError(f"missing saved allocation for replay round {k}")
            _generate_round(condition, seed, k, saved[k], candidate_budget)
    _generate_round(condition, seed, round_index, alloc, candidate_budget)
    return _emit_manifest(pool, round_index)


def build_dataset(condition: str, seed: int, round_index: int) -> tuple[Path, Path]:
    rc = run(
        [PY, "scripts/build_synth_dataset.py",
         "--condition", condition, "--seed", seed, "--round", round_index]
    )
    if rc != 0:
        raise RuntimeError(f"dataset build failed for {condition} s{seed} r{round_index}")
    key = f"{condition}_{seed}_r{round_index}"
    return (
        PROJECT / "configs" / "generated" / f"dataset_{key}.yaml",
        PROJECT / "data" / "manifests" / f"dataset_content_lock_{key}.json",
    )


def train_eval(name: str, data_yaml: Path, lock: Path, seed: int) -> bool:
    """Train (if needed) + val-evaluate.  Returns True when metrics.json exists."""
    metrics = EVAL_ROOT / f"{name}_val" / "metrics.json"
    if metrics.is_file():
        log(f"SKIP  {name} (metrics.json present)")
        return True
    rundir = PROJECT / "runs" / "yolo" / name
    results = rundir / "results.csv"
    need_train = True
    if results.is_file():
        lines = sum(1 for _ in results.open("r", encoding="utf-8", errors="ignore"))
        if lines >= EPOCHS + 1:
            log(f"REUSE {name} (results.csv has {lines} lines)")
            need_train = False
        else:
            log(f"REDO  {name} (partial results.csv: {lines} lines)")
            import shutil
            shutil.rmtree(rundir, ignore_errors=True)
    elif rundir.exists():
        log(f"REDO  {name} (run dir without results.csv)")
        import shutil
        shutil.rmtree(rundir, ignore_errors=True)

    if need_train:
        rc = run(
            [
                PY, "scripts/train_yolo.py",
                "--data", data_yaml, "--model", "yolo11n.pt",
                "--epochs", EPOCHS, "--imgsz", 640, "--batch", 16,
                "--device", 0, "--workers", 4, "--seed", seed, "--name", name,
                "--optimizer", "SGD", "--lr0", 0.01, "--lrf", 0.01,
                "--momentum", 0.937, "--weight-decay", 0.0005,
                "--warmup-epochs", 3.0, "--amp",
                "--content-lock", lock, "--content-lock-mode", "required",
            ],
            EVAL_ROOT / f"{name}_train.log",
        )
        if rc != 0:
            log(f"FAIL  {name} training; see {name}_train.log")
            return False

    rc = run(
        [
            PY, "scripts/evaluate_yolo.py",
            "--weights", f"runs/yolo/{name}/weights/best.pt",
            "--data", data_yaml, "--split", "val",
            "--content-lock", lock, "--content-lock-mode", "required",
            "--imgsz", 640, "--batch", 16, "--device", 0, "--workers", 4,
            "--seed", seed, "--conf", 0.001, "--iou", 0.7, "--max-det", 300,
            "--output-root", "reports/yolo_eval", "--name", f"{name}_val", "--exist-ok",
        ],
        EVAL_ROOT / f"{name}_eval.log",
    )
    if rc != 0:
        log(f"FAIL  {name} evaluation; see {name}_eval.log")
        return False
    log(f"DONE  {name}")
    return metrics.is_file()


def formula_alloc(metrics_path: Path, total: int, minimum: int, pool_dir: Path, tag: str, seed: int, json_out: Path) -> str:
    rc_out = subprocess.run(
        [
            str(PY), "scripts/compute_formula_alloc.py",
            "--metrics", str(metrics_path), "--total", str(total), "--minimum", str(minimum),
            "--maximum-fraction", "0.70",
            "--pool-dir", str(pool_dir), "--tag", tag, "--seed", str(seed),
            "--json-out", str(json_out),
        ],
        cwd=PROJECT, capture_output=True, text=True,
    )
    if rc_out.returncode != 0:
        raise RuntimeError(f"formula alloc failed: {rc_out.stderr[-800:]}")
    return rc_out.stdout.strip().splitlines()[-1].strip()


def run_uniform(seed: int) -> None:
    name = f"uniform_fixed_{TAG}_s{seed}_{EPOCHS}e"
    metrics = EVAL_ROOT / f"{name}_val" / "metrics.json"
    if metrics.is_file():
        log(f"SKIP  {name}")
        return
    manifest = generate("uniform_fixed", seed, 1, "inclusion:50,patches:50,scratches:50", 450)
    data_yaml, lock = build_dataset("uniform_fixed", seed, 1)
    train_eval(name, data_yaml, lock, seed)


def run_formula_one_shot(seed: int) -> None:
    name = f"formula_one_shot_{TAG}_s{seed}_{EPOCHS}e"
    metrics = EVAL_ROOT / f"{name}_val" / "metrics.json"
    if metrics.is_file():
        log(f"SKIP  {name}")
        return
    pool = PROJECT / "data" / "synthetic" / f"formula_one_shot_{seed}"
    alloc = formula_alloc(
        REAL_ONLY_METRICS[seed], 150, 10, pool, "f1", seed,
        PROJECT / "reports" / "agent_decisions" / f"formula_one_shot_{seed}_alloc.json",
    )
    generate("formula_one_shot", seed, 1, alloc, 450)
    data_yaml, lock = build_dataset("formula_one_shot", seed, 1)
    train_eval(name, data_yaml, lock, seed)


def run_formula_multi_round(seed: int) -> None:
    prev_metrics = REAL_ONLY_METRICS[seed]
    for r in (1, 2, 3):
        name = f"formula_multi_round_{TAG}_s{seed}_r{r}_{EPOCHS}e"
        metrics = EVAL_ROOT / f"{name}_val" / "metrics.json"
        if metrics.is_file():
            log(f"SKIP  {name}")
            prev_metrics = metrics
            continue
        pool = PROJECT / "data" / "synthetic" / f"formula_multi_round_{seed}"
        alloc = formula_alloc(
            prev_metrics, 50, 0, pool, "fm", seed,
            PROJECT / "reports" / "agent_decisions" / f"formula_multi_round_{seed}_r{r}_alloc.json",
        )
        generate("formula_multi_round", seed, r, alloc, 150)
        data_yaml, lock = build_dataset("formula_multi_round", seed, r)
        if not train_eval(name, data_yaml, lock, seed):
            raise RuntimeError(f"{name} failed; fix and rerun")
        prev_metrics = metrics


def run_agent_multi_round(seed: int) -> None:
    prev_metrics = REAL_ONLY_METRICS[seed]
    history_path = PROJECT / "reports" / "agent_decisions" / f"agent_multi_round_{seed}_history.json"
    for r in (1, 2, 3):
        name = f"agent_multi_round_{TAG}_s{seed}_r{r}_{EPOCHS}e"
        metrics = EVAL_ROOT / f"{name}_val" / "metrics.json"
        if metrics.is_file():
            log(f"SKIP  {name}")
            prev_metrics = metrics
            continue
        pool = PROJECT / "data" / "synthetic" / f"agent_multi_round_{seed}"
        obs_path = PROJECT / "reports" / "agent_decisions" / f"obs_{seed}_r{r}.json"
        run(
            [
                PY, "scripts/build_agent_observation.py",
                "--metrics", str(prev_metrics), "--seed", str(seed), "--round", str(r),
                "--state", str(pool / "state.json"), "--history", str(history_path),
                "--out", str(obs_path),
            ]
        )
        rc_out = subprocess.run(
            [
                str(PY), "scripts/agent_decide.py",
                "--observation", str(obs_path), "--seed", str(seed), "--round", str(r),
                "--endpoint", AGENT_ENDPOINT, "--model", AGENT_MODEL,
            ],
            cwd=PROJECT, capture_output=True, text=True,
        )
        if rc_out.returncode != 0:
            raise RuntimeError(f"agent_decide failed: {rc_out.stderr[-800:]}")
        alloc = rc_out.stdout.strip().splitlines()[-1].strip()
        log(f"agent r{r} alloc: {alloc}")
        generate("agent_multi_round", seed, r, alloc, 150)
        data_yaml, lock = build_dataset("agent_multi_round", seed, r)
        if not train_eval(name, data_yaml, lock, seed):
            raise RuntimeError(f"{name} failed; fix and rerun")
        append_agent_history(history_path, r, alloc, metrics)
        prev_metrics = metrics


def append_agent_history(history_path: Path, round_index: int, alloc: str, metrics_path: Path) -> None:
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_class = {c["class_name"]: c for c in doc["metrics"]["per_class"]}
    allocation = {}
    for part in alloc.split(","):
        name, _, value = part.partition(":")
        allocation[name.strip()] = int(value)
    history.append(
        {
            "round": round_index,
            "allocation": allocation,
            "resulting_overall_ap50": doc["metrics"]["overall"]["ap50"],
            "resulting_per_class": {
                name: {"recall": per_class[name]["recall"], "ap50": per_class[name]["ap50"]}
                for name in CLASSES
            },
        }
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["no_llm", "agent", "all"])
    args = parser.parse_args()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    if args.phase in ("no_llm", "all"):
        for seed in SEEDS:
            run_uniform(seed)
            run_formula_one_shot(seed)
            run_formula_multi_round(seed)
    if args.phase in ("agent", "all"):
        for seed in SEEDS:
            run_agent_multi_round(seed)
    log("########## augmented conditions finished ##########")
    return 0


if __name__ == "__main__":
    sys.exit(main())
