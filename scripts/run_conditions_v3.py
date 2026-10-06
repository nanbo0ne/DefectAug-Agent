"""v3 orchestrator: every condition driven by configs/experiment_v3.yaml.

v3 over v2: per-class generation params (src/defectaug/gen_params.py) are part
of every augmented action; each trained model gets a val miss analysis
(scripts/val_miss_analysis_v3.py) that feeds the formula rule and the agent;
formula_tuned_multi_round uses quota weights chosen on the tuning seed.

All augmented conditions propose a per-class quota and pass it through the
shared constraint module (src/defectaug/allocation.py).  Generation shortfalls
are topped up inside the round's candidate ceiling; if the ceiling runs out
the run fails loudly.  Test data is never touched here (val only).

Usage:
    python scripts/run_conditions_v3.py --conditions real_only,uniform_fixed --seeds all
    python scripts/run_conditions_v3.py --smoke          # 1 seed, 3 epochs, tag smoke
    python scripts/run_conditions_v3.py --tag tune --seeds tuning --formula-weights 0.8,0.1,0.1
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
from defectaug.gen_params import (  # noqa: E402
    DEFAULT, MIN_SCALE_SPAN, OVERLAP_HI, OVERLAP_LO, SCALE_HI, SCALE_LO, SOURCE_WEIGHTINGS,
    TARGET_POOLS, ParamViolation, formula_params, validate_params,
)

PY = PROJECT / ".venv" / "Scripts" / "python.exe"
PROTOCOL_PATH = PROJECT / "configs" / "experiment_v3.yaml"
CLASSES = ("inclusion", "patches", "scratches")
SHORT = {"uniform_fixed": "u", "formula_one_shot": "f1", "formula_multi_round": "fm",
         "formula_tuned_multi_round": "ft", "agent_multi_round": "ag"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Ctx:
    def __init__(self, protocol: dict, tag: str, epochs: int, device: str):
        self.p = protocol
        self.tag = tag
        self.epochs = epochs
        self.device = device
        self.eval_root = PROJECT / "reports" / "v3" / tag / "yolo_eval"
        self.pool_root = PROJECT / "data" / "synthetic" / "v3" / tag
        self.dec_root = PROJECT / "reports" / "v3" / tag / "decisions"
        self.splits = PROJECT / protocol["dataset"]["splits_dir"]
        self.train_manifest = self.splits / "train_fewshot.txt"
        g = protocol["generation"]
        self.max_acc = int(g["max_accepted_derivatives_per_source"])
        self.max_cand = int(g["max_candidates_per_source"])
        self.cand_total = int(g["candidate_budget_total_max"])
        for d in (self.eval_root, self.pool_root, self.dec_root):
            d.mkdir(parents=True, exist_ok=True)

    def run_name(self, cond: str, seed: int, r: int | None = None) -> str:
        return f"v3_{self.tag}_{cond}_s{seed}" + (f"_r{r}" if r else "")

    def sources_by_class(self) -> dict[str, list[str]]:
        import run_synth_generation_v3 as g
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


def miss_for(ctx: Ctx, metrics_path: Path) -> dict:
    """Val miss analysis of the model behind `metrics_path` (cached next to it)."""
    out = metrics_path.parent / "miss_analysis.json"
    if not out.is_file():
        name = metrics_path.parent.name.removesuffix("_val")
        weights = PROJECT / "runs" / "yolo" / name / "weights" / ctx.p["detector"]["eval_weights"]
        ma = ctx.p["miss_analysis"]
        rc = sh([PY, "scripts/val_miss_analysis_v3.py", "--weights", weights, "--output", out,
                 "--conf", ma["conf"], "--imgsz", ctx.p["detector"]["imgsz"], "--device", ctx.device],
                metrics_path.parent / "miss_analysis.log")
        if rc != 0 or not out.is_file():
            raise RuntimeError(f"miss analysis failed for {name}")
    return json.loads(out.read_text(encoding="utf-8"))["classes"]


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
                   bp: BudgetProtocol, allocated: dict[str, int], cap: dict[str, int],
                   params: dict[str, dict]) -> dict:
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
    pool.mkdir(parents=True, exist_ok=True)
    rdir = pool / f"r{r}" / "images"
    params_file = pool / f"params_r{r}.json"
    params_file.write_text(json.dumps(params, indent=1), encoding="utf-8")
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
            str(PY), "scripts/run_synth_generation_v3.py", "--condition", cond, "--seed", str(seed),
            "--round", str(r), "--alloc", ",".join(f"{c}:{need[c]}" for c in CLASSES),
            "--candidate-budget", str(budget_left), "--train-manifest", str(ctx.train_manifest),
            "--pool-root", str(ctx.pool_root), "--params", str(params_file),
            "--max-accepted-per-source", str(ctx.max_acc),
            "--max-candidates-per-source", str(ctx.max_cand),
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
    key = f"v3_{ctx.tag}_{cond}_{seed}_r{r}"
    rc = sh([PY, "scripts/build_synth_dataset.py", "--condition", f"v3_{ctx.tag}_{cond}", "--seed", seed,
             "--round", r, "--manifest", manifest, "--splits-dir", ctx.p["dataset"]["splits_dir"]])
    if rc != 0:
        raise RuntimeError(f"dataset build failed {key}")
    return (PROJECT / "configs" / "generated" / f"dataset_{key}.yaml",
            PROJECT / "data" / "manifests" / f"dataset_content_lock_{key}.json")


# ---------------------------------------------------------------- proposers
def formula_weights(ctx: Ctx, proposer: str) -> tuple[float, float, float]:
    """formula -> frozen v2 weights; formula_tuned -> weights selected on the tuning seed."""
    if ctx.formula_weights_override is not None:
        return ctx.formula_weights_override
    if proposer == "formula_tuned":
        sel = ctx.p["formula_tuned_policy"]["selected"]
        if sel is None:
            raise SystemExit("formula_tuned_policy.selected is null: run the tuning step first")
        return tuple(float(v) for v in sel)
    fp = ctx.p["formula_policy"]
    return float(fp["recall_weight"]), float(fp["ap50_weight"]), float(fp["rarity_weight"])


def formula_proposal(ctx: Ctx, metrics_path: Path, bp: BudgetProtocol,
                     weights: tuple[float, float, float]) -> tuple[dict, dict]:
    real = ctx.p["dataset"]["real_fewshot_images"]
    pcm = per_class_metrics(metrics_path)
    cm = {c: ClassMetrics(recall=float(pcm[c]["recall"]), ap50=float(pcm[c]["ap50"]), real_images=int(real[c]))
          for c in CLASSES}
    scores = formula_scores(cm, *weights)
    # Largest remainder of the round budget by score; bounds come from `project`.
    total = sum(scores.values())
    raw = {c: bp.round_budget * scores[c] / total if total > 0 else bp.round_budget / 3 for c in CLASSES}
    alloc = {c: int(raw[c]) for c in CLASSES}
    for c in sorted(CLASSES, key=lambda c: (-(raw[c] - int(raw[c])), c))[: bp.round_budget - sum(alloc.values())]:
        alloc[c] += 1
    return alloc, {"scores": scores, "weights": list(weights)}


def formula_rule_params(miss: dict) -> dict[str, dict]:
    """Fixed parameter rule (gen_params.formula_params) on the val miss analysis."""
    return {
        c: formula_params(
            c, recall=float(miss[c]["recall"] or 0.0),
            median_train_px=float(miss[c]["train_median_size_px"]),
            median_missed_px=miss[c]["median_missed_size_px"],
        ).as_dict()
        for c in CLASSES
    }


AGENT_SYSTEM_PROMPT = (
    "You are the augmentation-planning agent in a closed-loop few-shot object "
    "detection experiment. A YOLO11n detector is trained on 100 real steel-"
    "surface defect images (inclusion:10, patches:30, scratches:60) plus "
    "synthetic Copy-Paste samples you plan. After each round the detector is "
    "retrained and evaluated on a locked validation split; per-class metrics "
    "and a per-instance miss analysis (recall by object size and contrast "
    "tertile, misclassifications, false positives, median size of missed vs "
    "train objects) are fed back to you. Each round you decide, per class, "
    "(a) how many new synthetic images to generate and (b) how to generate "
    "them: paste-scale range, background pool, source weighting by defect "
    "size, and allowed overlap with existing defects. The generator "
    "(patch_copy_gauss5) copies real train defects with their masks and "
    "feathers the boundary; labels come from the transformed mask. Answer "
    "with a single JSON object and nothing else."
)


def agent_prompt(obs: dict, bp: BudgetProtocol, feedback: str | None) -> str:
    r = obs["round_index"]
    lines = [
        "OBSERVATION (JSON):",
        json.dumps(obs, ensure_ascii=False, indent=1),
        "",
        "CONSTRAINTS (checked by code; a violating plan is rejected):",
        f"- Round {r} of {bp.rounds}. Allocate exactly {bp.round_budget} new images in total.",
        f"- No class may exceed headroom_this_round: {json.dumps(obs['headroom_this_round'])}.",
        f"- Cumulative per-class total can never exceed {bp.cumulative_cap}; "
        f"after the last round every class needs >= {bp.final_min_per_class}.",
        f"- params, per class: {SCALE_LO} <= scale_min < scale_max <= {SCALE_HI}, "
        f"scale_max - scale_min >= {MIN_SCALE_SPAN}; target_pool in {list(TARGET_POOLS)}; "
        f"source_weighting in {list(SOURCE_WEIGHTINGS)}; {OVERLAP_LO} <= max_overlap <= {OVERLAP_HI}.",
        "- scale multiplies the source defect size; scales above what fits the image are capped.",
        "- Respond with one JSON object only, schema:",
        '  {"allocation": {"inclusion": int, "patches": int, "scratches": int},',
        '   "params": {"<class>": {"scale_min": float, "scale_max": float, "target_pool": str,',
        '              "source_weighting": str, "max_overlap": float}, ... all three classes},',
        '   "rationale": "short string"}',
    ]
    if feedback:
        lines += ["", f"PREVIOUS ATTEMPT REJECTED: {feedback}", "Return a corrected plan."]
    return "\n".join(lines)


def _validate_agent_params(raw: object) -> dict[str, dict]:
    if not isinstance(raw, dict) or set(raw) != set(CLASSES):
        raise ParamViolation(f"params must have exactly the keys {list(CLASSES)}")
    return {c: validate_params(c, raw[c]).as_dict() for c in CLASSES}


def agent_proposal(ctx: Ctx, obs: dict, r: int, bp: BudgetProtocol,
                   allocated: dict, cap: dict) -> tuple[dict, dict | None, dict]:
    """Returns (allocation, params or None if never valid, info)."""
    import agent_decide as ad
    ap = ctx.p["agent_policy"]
    feedback = None
    attempts = []
    last_alloc: dict = {}
    last_params = None
    for k in range(1, int(ap["max_attempts_per_round"]) + 1):
        prompt = agent_prompt(obs, bp, feedback)
        resp = ad.call_llm(ctx.agent_endpoint, ctx.agent_model, AGENT_SYSTEM_PROMPT, prompt, 1800.0,
                           temperature=float(ap["temperature"]), max_tokens=int(ap["max_tokens"]))
        if ap.get("require_served_model_match") and resp.get("served_model") != ctx.agent_model:
            raise RuntimeError(f"served_model {resp.get('served_model')!r} != {ctx.agent_model!r}; aborting")
        rec = {"attempt": k, "response": resp, "prompt": prompt}
        errors = []
        try:
            plan = ad.extract_json(resp["content"])
        except (ValueError, json.JSONDecodeError) as exc:
            plan, errors = {}, [f"unparseable reply: {exc}"]
        if plan:
            last_alloc = plan.get("allocation", {}) if isinstance(plan.get("allocation"), dict) else {}
            try:
                clean = validate(bp, last_alloc, allocated, cap, r)
            except (ConstraintViolation, ValueError) as exc:
                clean, errors = None, errors + [f"allocation: {exc}"]
            try:
                params = _validate_agent_params(plan.get("params"))
                last_params = params
            except (ParamViolation, TypeError) as exc:
                params, errors = None, errors + [f"params: {exc}"]
            if not errors:
                rec.update(allocation=clean, params=params)
                attempts.append(rec)
                return clean, params, {"attempts": attempts, "repaired": False,
                                       "rationale": plan.get("rationale")}
        feedback = "; ".join(errors)
        rec["error"] = feedback
        attempts.append(rec)
    return last_alloc, last_params, {"attempts": attempts, "repaired": True}


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
        "previous_val_miss_analysis": miss_for(ctx, prev_metrics),
        "allocated_so_far": allocated, "headroom_this_round": headroom(bp, allocated, cap),
        "default_params": DEFAULT.as_dict(),
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
        params = None
        if alloc_file.is_file():
            final = json.loads(alloc_file.read_text(encoding="utf-8"))
            log(f"RESUME {name} with saved allocation {final}")
        elif dec_file.is_file():
            # Decision already made (and logged) before an interruption: reuse it,
            # never re-ask the proposer.
            decision = json.loads(dec_file.read_text(encoding="utf-8"))
            final, cap, params = decision["final"], decision["source_capacity"], decision["params"]
            log(f"RESUME {name} with logged decision {final}")
        else:
            cap = capacity(ctx, cond_name, seed)
            proposer = cond["proposer"]
            params_source = cond["params"]
            if proposer == "uniform":
                raw, info = uniform_proposal(bp), {}
                params = {c: DEFAULT.as_dict() for c in CLASSES}
            elif proposer in ("formula", "formula_tuned"):
                raw, info = formula_proposal(ctx, prev, bp, formula_weights(ctx, proposer))
                params = formula_rule_params(miss_for(ctx, prev))
            else:
                obs = observation(ctx, cond_name, seed, r, bp, prev, allocated, cap, history)
                raw, agent_params, info = agent_proposal(ctx, obs, r, bp, allocated, cap)
                info["observation"] = obs
                if agent_params is None:
                    # Same fallback for every invalid agent action: the formula rule.
                    params, params_source = formula_rule_params(obs["previous_val_miss_analysis"]), "formula_rule_fallback"
                else:
                    params = agent_params
            try:
                final = validate(bp, raw, allocated, cap, r)
                repaired = False
            except ConstraintViolation as exc:
                final = project(bp, raw, allocated, cap, r)
                repaired = True
                info["violations"] = exc.reasons
            decision = {"condition": cond_name, "seed": seed, "round": r, "proposal": raw,
                        "final": final, "repaired": repaired, "allocated_before": allocated,
                        "source_capacity": cap, "params": params, "params_source": params_source,
                        **info}
            dec_file.write_text(json.dumps(decision, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            log(f"ALLOC {name}: proposal {raw} -> final {final}{' (projected)' if repaired else ''}")
        if not alloc_file.is_file():
            gen = generate_round(ctx, cond_name, seed, r, final, cand_budget, bp, allocated, cap, params)
            if gen["spills"]:
                decision["generation_spill"] = gen["spills"]
                decision["final_reached"] = gen["reached"]
                dec_file.write_text(json.dumps(decision, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            final = gen["reached"]
        data_yaml, lock = build_dataset(ctx, cond_name, seed, r)
        metrics = train_eval(ctx, name, data_yaml, lock, seed)
        pcm = per_class_metrics(metrics)
        history.append({"round": r, "allocation": final, "params": params,
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
    ap.add_argument("--seeds", default="all",
                    help="'all', comma list of final-seed indices, or 'tuning' (formula_tuned_policy.tuning_seed)")
    ap.add_argument("--formula-weights", default="",
                    help="recall,ap50,rarity override for formula proposers (tuning tag only)")
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
    ctx.formula_weights_override = None
    if args.formula_weights:
        if not args.tag.startswith("tune"):
            raise SystemExit("--formula-weights is only allowed with a tune* tag")
        ctx.formula_weights_override = tuple(float(v) for v in args.formula_weights.split(","))
    seeds = protocol["experiment"]["final_seeds"]
    if args.seeds == "tuning":
        seeds = [int(protocol["formula_tuned_policy"]["tuning_seed"])]
    elif args.seeds != "all":
        seeds = [seeds[int(i)] for i in args.seeds.split(",")]
    if args.tag == "final" and protocol["experiment"]["status"] != "frozen_final":
        raise SystemExit("tag 'final' requires experiment.status == frozen_final")
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if "agent_multi_round" in conds and not ctx.agent_endpoint:
        raise SystemExit("agent_multi_round needs --agent-endpoint")

    log(f"v3 run tag={ctx.tag} epochs={epochs} seeds={seeds} conditions={conds}")
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
