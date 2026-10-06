"""v2 orchestrator: every condition driven by configs/experiment_v2.yaml.

All augmented conditions propose a per-class quota and pass it through the
shared constraint module (src/defectaug/allocation.py).  Generation shortfalls
are topped up inside the round's candidate ceiling; if the ceiling runs out
the run fails loudly.  Test data is never touched here (val only).

Usage:
    python scripts/run_conditions.py --conditions real_only,uniform_fixed --seeds all
    python scripts/run_conditions.py --smoke            # 1 seed, 3 epochs, tag smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from defectaug.allocation import (  # noqa: E402
    BudgetProtocol, ConstraintViolation, project, source_capacity_from_state,
    uniform_proposal, validate,
)
from defectaug.policy import ClassMetrics, formula_scores  # noqa: E402

PY = PROJECT / ".venv" / "Scripts" / "python.exe"
PROTOCOL_PATH = PROJECT / "configs" / "experiment_v2.yaml"
CLASSES = ("inclusion", "patches", "scratches")
SHORT = {"uniform_fixed": "u", "formula_one_shot": "f1", "formula_multi_round": "fm", "agent_multi_round": "ag"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Ctx:
    def __init__(self, protocol: dict, tag: str, epochs: int, device: str):
        self.p = protocol
        self.tag = tag
        self.epochs = epochs
        self.device = device
        self.eval_root = PROJECT / "reports" / "v2" / tag / "yolo_eval"
        self.pool_root = PROJECT / "data" / "synthetic" / "v2" / tag
        self.dec_root = PROJECT / "reports" / "v2" / tag / "decisions"
        self.splits = PROJECT / protocol["dataset"]["splits_dir"]
        self.train_manifest = self.splits / "train_fewshot.txt"
        g = protocol["generation"]
        self.max_acc = int(g["max_accepted_derivatives_per_source"])
        self.max_cand = int(g["max_candidates_per_source"])
        self.cand_total = int(g["candidate_budget_total_max"])
        for d in (self.eval_root, self.pool_root, self.dec_root):
            d.mkdir(parents=True, exist_ok=True)

    def run_name(self, cond: str, seed: int, r: int | None = None) -> str:
        return f"v2_{self.tag}_{cond}_s{seed}" + (f"_r{r}" if r else "")

    def sources_by_class(self) -> dict[str, list[str]]:
        import run_synth_generation as g
        by = g.resolve_train_sources(self.train_manifest)
        return {c: [p.stem for p in by[c]] for c in CLASSES}


def sh(cmd: list, log_path: Path | None = None) -> int:
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    if log_path is None:
        return subprocess.call(cmd, cwd=PROJECT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as s:
        return subprocess.call(cmd, cwd=PROJECT, stdout=s, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------- training
def train_eval(ctx: Ctx, name: str, data_yaml: Path, lock: Path, seed: int) -> Path:
    """Train from pretrained weights, evaluate best.pt on val. Returns metrics.json."""
    metrics = ctx.eval_root / f"{name}_val" / "metrics.json"
    if metrics.is_file():
        log(f"SKIP {name} (metrics present)")
        return metrics
    det = ctx.p["detector"]
    rundir = PROJECT / "runs" / "yolo" / name
    if rundir.exists():
        shutil.rmtree(rundir)
    rc = sh([
        PY, "scripts/train_yolo.py", "--data", data_yaml, "--model", det["model"],
        "--epochs", ctx.epochs, "--imgsz", det["imgsz"], "--batch", det["batch"],
        "--device", ctx.device, "--workers", det["workers"], "--seed", seed,
        "--name", name, "--optimizer", det["optimizer"],
        "--lr0", det["lr0"], "--lrf", det["lrf"], "--momentum", det["momentum"],
        "--weight-decay", det["weight_decay"], "--warmup-epochs", det["warmup_epochs"],
        "--amp", "--content-lock", lock, "--content-lock-mode", "required",
    ], ctx.eval_root / f"{name}_train.log")
    if rc != 0:
        raise RuntimeError(f"training failed: {name} (see {name}_train.log)")
    weights = rundir / "weights" / det["eval_weights"]
    if not weights.is_file():
        raise RuntimeError(f"missing weights {weights}")
    rc = sh([
        PY, "scripts/evaluate_yolo.py", "--weights", weights, "--data", data_yaml,
        "--split", "val", "--content-lock", lock, "--content-lock-mode", "required",
        "--imgsz", det["imgsz"], "--batch", det["batch"], "--device", ctx.device,
        "--workers", det["workers"], "--seed", seed, "--conf", 0.001, "--iou", 0.7,
        "--max-det", 300, "--output-root", ctx.eval_root.relative_to(PROJECT),
        "--name", f"{name}_val", "--exist-ok",
    ], ctx.eval_root / f"{name}_eval.log")
    if rc != 0 or not metrics.is_file():
        raise RuntimeError(f"val evaluation failed: {name}")
    return metrics


def per_class_metrics(metrics_path: Path) -> dict[str, dict]:
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {c["class_name"]: c for c in doc["metrics"]["per_class"]}


# -------------------------------------------------------------- generation
def pool_dir(ctx: Ctx, cond: str, seed: int) -> Path:
    return ctx.pool_root / f"{cond}_{seed}"


def allocated_so_far(ctx: Ctx, cond: str, seed: int) -> dict[str, int]:
    imgs = pool_dir(ctx, cond, seed).glob("r*/images/*.png")
    counts = {c: 0 for c in CLASSES}
    for p in imgs:
        counts[p.stem.split("_", 1)[0]] += 1
    return counts


def capacity(ctx: Ctx, cond: str, seed: int) -> dict[str, int]:
    state_path = pool_dir(ctx, cond, seed) / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    return source_capacity_from_state(state, ctx.sources_by_class(), ctx.max_acc, ctx.max_cand)


def generate_round(ctx: Ctx, cond: str, seed: int, r: int, alloc: dict[str, int], cand_budget: int,
                   bp: BudgetProtocol, allocated: dict[str, int], cap: dict[str, int]) -> dict:
    """Generate exactly `alloc` accepted samples for round r, topping up shortfalls.

    Top-ups stay inside this round's candidate ceiling; the per-source caps in
    state.json persist, so the generator moves on to sources that still have room.

    Exhaustion spill (same rule for every condition): if a class still short
    has no live source left (state capacity 0), it keeps what it got and the
    deficit is re-projected with the shared `project` onto classes that still
    have headroom.  The round total stays at round_budget.  Returns the
    allocation actually reached plus the spill record (None if not triggered).
    """
    pool = pool_dir(ctx, cond, seed)
    rdir = pool / f"r{r}" / "images"
    alloc = dict(alloc)
    spills: list = []
    done = {c: len(list(rdir.glob(f"{c}_*.png"))) for c in CLASSES} if rdir.is_dir() else {c: 0 for c in CLASSES}
    # Candidates already spent this round come from the receipts, so a resumed
    # round cannot exceed its ceiling.
    spent = _round_spent(pool, r)
    for attempt in range(1, 8):
        need = {c: alloc[c] - done[c] for c in CLASSES}
        live = capacity(ctx, cond, seed)
        dead = [c for c in CLASSES if need[c] > 0 and live[c] == 0]
        if dead:
            # Freeze exhausted classes at what they reached; project the rest.
            cap_spill = dict(cap)
            for c in dead:
                cap_spill[c] = done[c]
            proposal = {c: (done[c] if c in dead else alloc[c]) for c in CLASSES}
            new_alloc = project(bp, proposal, allocated, cap_spill, r)
            spills.append({"attempt": attempt, "exhausted": dead, "before": alloc, "after": new_alloc,
                           "reached": dict(done)})
            log(f"SPILL {cond} s{seed} r{r}: {dead} exhausted at {done}; {alloc} -> {new_alloc}")
            alloc = new_alloc
            need = {c: alloc[c] - done[c] for c in CLASSES}
        need = {c: alloc[c] - done[c] for c in CLASSES}
        if any(v < 0 for v in need.values()):
            raise RuntimeError(f"{cond} s{seed} r{r}: more images on disk than allocated: {done} vs {alloc}")
        if sum(need.values()) == 0:
            break
        budget_left = cand_budget - spent
        if budget_left <= 0:
            raise RuntimeError(f"{cond} s{seed} r{r}: candidate ceiling {cand_budget} exhausted, short {need}")
        out = subprocess.run([
            str(PY), "scripts/run_synth_generation.py", "--condition", cond, "--seed", str(seed),
            "--round", str(r), "--alloc", ",".join(f"{c}:{need[c]}" for c in CLASSES),
            "--candidate-budget", str(budget_left), "--train-manifest", str(ctx.train_manifest),
            "--pool-root", str(ctx.pool_root),
        ], cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
        if out.returncode not in (0, 2):
            raise RuntimeError(f"generation crashed rc={out.returncode}: {out.stderr[-1500:]}")
        spent = _round_spent(pool, r)
        done = {c: len(list(rdir.glob(f"{c}_*.png"))) for c in CLASSES}
        log(f"GEN {cond} s{seed} r{r} try{attempt}: have {done} / want {alloc}, candidates spent {spent}")
    if any(done[c] != alloc[c] for c in CLASSES):
        raise RuntimeError(f"{cond} s{seed} r{r}: could not reach allocation {alloc}, have {done}")
    (pool / f"alloc_r{r}.json").write_text(json.dumps(alloc), encoding="utf-8")
    (pool / f"candidates_r{r}.json").write_text(json.dumps({"spent": spent, "ceiling": cand_budget}), encoding="utf-8")
    return {"reached": alloc, "spills": spills}


def _round_spent(pool: Path, r: int) -> int:
    path = pool / "receipts.jsonl"
    if not path.is_file():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if int(rec["round"]) == r:
                total += sum(int(v["candidates"]) for v in rec["receipts"].values())
    return total


def build_dataset(ctx: Ctx, cond: str, seed: int, r: int) -> tuple[Path, Path]:
    pool = pool_dir(ctx, cond, seed)
    real = [l.strip() for l in ctx.train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    synth = sorted(str(p) for p in pool.glob("r*/images/*.png") if int(p.parent.parent.name[1:]) <= r)
    manifest = pool / f"train_manifest_r{r}.txt"
    manifest.write_text("\n".join(real + synth) + "\n", encoding="utf-8")
    key = f"v2_{ctx.tag}_{cond}_{seed}_r{r}"
    rc = sh([PY, "scripts/build_synth_dataset.py", "--condition", f"v2_{ctx.tag}_{cond}", "--seed", seed,
             "--round", r, "--manifest", manifest])
    if rc != 0:
        raise RuntimeError(f"dataset build failed {key}")
    return (PROJECT / "configs" / "generated" / f"dataset_{key}.yaml",
            PROJECT / "data" / "manifests" / f"dataset_content_lock_{key}.json")


# ---------------------------------------------------------------- proposers
def formula_proposal(ctx: Ctx, metrics_path: Path, bp: BudgetProtocol) -> tuple[dict, dict]:
    fp = ctx.p["formula_policy"]
    real = ctx.p["dataset"]["real_fewshot_images"]
    pcm = per_class_metrics(metrics_path)
    cm = {c: ClassMetrics(recall=float(pcm[c]["recall"]), ap50=float(pcm[c]["ap50"]), real_images=int(real[c]))
          for c in CLASSES}
    scores = formula_scores(cm, fp["recall_weight"], fp["ap50_weight"], fp["rarity_weight"])
    # Largest remainder of the round budget by score; bounds come from `project`.
    total = sum(scores.values())
    raw = {c: bp.round_budget * scores[c] / total if total > 0 else bp.round_budget / 3 for c in CLASSES}
    alloc = {c: int(raw[c]) for c in CLASSES}
    for c in sorted(CLASSES, key=lambda c: (-(raw[c] - int(raw[c])), c))[: bp.round_budget - sum(alloc.values())]:
        alloc[c] += 1
    return alloc, {"scores": scores}


def agent_proposal(ctx: Ctx, obs: dict, seed: int, r: int, bp: BudgetProtocol,
                   allocated: dict, cap: dict) -> tuple[dict, dict]:
    import agent_decide as ad
    ap = ctx.p["agent_policy"]
    endpoint = ctx.agent_endpoint
    feedback = None
    attempts = []
    last = {}
    for k in range(1, int(ap["max_attempts_per_round"]) + 1):
        prompt = ad.build_user_prompt_v2(obs, bp, feedback)
        resp = ad.call_llm(endpoint, ctx.agent_model, ad.SYSTEM_PROMPT, prompt, 900.0,
                           temperature=float(ap["temperature"]),
                           max_tokens=int(ap.get("max_tokens", 4096)))
        rec = {"attempt": k, "response": resp, "prompt": prompt}
        try:
            plan = ad.extract_json(resp["content"])
            last = plan.get("allocation", {})
            clean = validate(bp, last, allocated, cap, r)
            rec["allocation"] = clean
            attempts.append(rec)
            return clean, {"attempts": attempts, "repaired": False, "rationale": plan.get("rationale")}
        except (ConstraintViolation, ValueError, KeyError, json.JSONDecodeError) as exc:
            feedback = str(exc)
            rec["error"] = feedback
            attempts.append(rec)
    return last if isinstance(last, dict) else {}, {"attempts": attempts, "repaired": True}


def observation(ctx: Ctx, cond: str, seed: int, r: int, bp: BudgetProtocol,
                prev_metrics: Path, allocated: dict, cap: dict, history: list) -> dict:
    pcm = per_class_metrics(prev_metrics)
    receipts_path = pool_dir(ctx, cond, seed) / "receipts.jsonl"
    receipts = {c: {"candidates": 0, "accepted": 0, "rejects": {}} for c in CLASSES}
    if receipts_path.is_file():
        for line in receipts_path.read_text(encoding="utf-8").splitlines():
            for c, v in json.loads(line)["receipts"].items():
                receipts[c]["candidates"] += v["candidates"]
                receipts[c]["accepted"] += v["accepted"]
                for k, n in v["rejects"].items():
                    receipts[c]["rejects"][k] = receipts[c]["rejects"].get(k, 0) + n
    from defectaug.allocation import headroom
    return {
        "round_index": r, "total_rounds": bp.rounds, "round_budget": bp.round_budget,
        "total_budget": bp.total_budget, "real_fewshot_images": ctx.p["dataset"]["real_fewshot_images"],
        "previous_validation": {c: {k: pcm[c][k] for k in ("recall", "precision", "ap50", "ap50_95")} for c in CLASSES},
        "allocated_so_far": allocated, "headroom_this_round": headroom(bp, allocated, cap),
        "quality_receipts": receipts, "history": history,
    }


# ---------------------------------------------------------------- conditions
def run_real(ctx: Ctx, seed: int, repeat: bool) -> Path:
    cond = next(c for c in ctx.p["conditions"] if c["name"] == ("real_only_repeat_matched" if repeat else "real_only"))
    data_yaml = PROJECT / cond.get("dataset_yaml", ctx.p["dataset"]["yaml"])
    lock = PROJECT / cond.get("content_lock", ctx.p["dataset"]["content_lock"])
    return train_eval(ctx, ctx.run_name(cond["name"], seed), data_yaml, lock, seed)


def run_augmented(ctx: Ctx, cond_name: str, seed: int, real_metrics: Path) -> Path:
    cond = next(c for c in ctx.p["conditions"] if c["name"] == cond_name)
    rounds = int(cond["rounds"])
    bp = BudgetProtocol.from_protocol(ctx.p, rounds)
    cand_budget = ctx.cand_total // rounds
    prev = real_metrics
    history: list = []
    hist_path = ctx.dec_root / f"{cond_name}_{seed}_history.json"
    if hist_path.is_file():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    metrics = prev
    for r in range(1, rounds + 1):
        name = ctx.run_name(cond_name, seed, r if rounds > 1 else None)
        metrics = ctx.eval_root / f"{name}_val" / "metrics.json"
        if metrics.is_file():
            log(f"SKIP {name}")
            prev = metrics
            continue
        # Cumulative allocation before this round (excludes any partial images of round r).
        allocated = allocated_so_far(ctx, cond_name, seed)
        allocated = {c: allocated[c] - _round_count(ctx, cond_name, seed, r, c) for c in CLASSES}
        alloc_file = pool_dir(ctx, cond_name, seed) / f"alloc_r{r}.json"
        dec_file = ctx.dec_root / f"{cond_name}_{seed}_r{r}.json"
        decision = None
        if alloc_file.is_file():
            final = json.loads(alloc_file.read_text(encoding="utf-8"))
            log(f"RESUME {name} with saved allocation {final}")
        elif dec_file.is_file():
            # Decision already made (and logged) before an interruption: reuse it,
            # never re-ask the proposer.
            decision = json.loads(dec_file.read_text(encoding="utf-8"))
            final, cap = decision["final"], decision["source_capacity"]
            log(f"RESUME {name} with logged decision {final}")
        else:
            cap = capacity(ctx, cond_name, seed)
            if cond["proposer"] == "uniform":
                raw, info = uniform_proposal(bp), {}
            elif cond["proposer"] == "formula":
                raw, info = formula_proposal(ctx, prev, bp)
            else:
                obs = observation(ctx, cond_name, seed, r, bp, prev, allocated, cap, history)
                raw, info = agent_proposal(ctx, obs, seed, r, bp, allocated, cap)
                info["observation"] = obs
            try:
                final = validate(bp, raw, allocated, cap, r)
                repaired = False
            except ConstraintViolation as exc:
                final = project(bp, raw, allocated, cap, r)
                repaired = True
                info["violations"] = exc.reasons
            decision = {"condition": cond_name, "seed": seed, "round": r, "proposal": raw,
                        "final": final, "repaired": repaired, "allocated_before": allocated,
                        "source_capacity": cap, **info}
            dec_file.write_text(json.dumps(decision, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            log(f"ALLOC {name}: proposal {raw} -> final {final}{' (projected)' if repaired else ''}")
        if not alloc_file.is_file():
            gen = generate_round(ctx, cond_name, seed, r, final, cand_budget, bp, allocated, cap)
            if gen["spills"]:
                decision["generation_spill"] = gen["spills"]
                decision["final_reached"] = gen["reached"]
                dec_file.write_text(json.dumps(decision, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            final = gen["reached"]
        data_yaml, lock = build_dataset(ctx, cond_name, seed, r)
        metrics = train_eval(ctx, name, data_yaml, lock, seed)
        pcm = per_class_metrics(metrics)
        history.append({"round": r, "allocation": final,
                        "resulting": {c: {"recall": pcm[c]["recall"], "ap50": pcm[c]["ap50"]} for c in CLASSES}})
        hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        prev = metrics
    return metrics


def _round_count(ctx: Ctx, cond: str, seed: int, r: int, c: str) -> int:
    d = pool_dir(ctx, cond, seed) / f"r{r}" / "images"
    return len(list(d.glob(f"{c}_*.png"))) if d.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="real_only,real_only_repeat_matched,uniform_fixed,formula_one_shot,formula_multi_round")
    ap.add_argument("--seeds", default="all", help="'all' or comma list of indices 0..2")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="0")
    ap.add_argument("--agent-endpoint", default="")
    ap.add_argument("--agent-model", default="")
    ap.add_argument("--smoke", action="store_true", help="tag=smoke, 3 epochs, seed index 0")
    args = ap.parse_args()

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.smoke:
        args.tag, args.epochs, args.seeds = "smoke", args.epochs or 3, "0"
    epochs = args.epochs or int(protocol["detector"]["epochs_per_round"])
    if args.tag == "final" and epochs != int(protocol["detector"]["epochs_per_round"]):
        raise SystemExit("tag 'final' must use the protocol epoch count")
    ctx = Ctx(protocol, args.tag, epochs, args.device)
    ctx.agent_endpoint = args.agent_endpoint
    ctx.agent_model = args.agent_model or protocol["agent_policy"]["model"]
    seeds = protocol["experiment"]["final_seeds"]
    if args.seeds != "all":
        seeds = [seeds[int(i)] for i in args.seeds.split(",")]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if "agent_multi_round" in conds and not ctx.agent_endpoint:
        raise SystemExit("agent_multi_round needs --agent-endpoint")

    log(f"v2 run tag={ctx.tag} epochs={epochs} seeds={seeds} conditions={conds}")
    for seed in seeds:
        real = run_real(ctx, seed, repeat=False)
        for c in conds:
            if c == "real_only":
                continue
            if c == "real_only_repeat_matched":
                run_real(ctx, seed, repeat=True)
            else:
                run_augmented(ctx, c, seed, real)
    log("all requested conditions finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
