"""v3.1 orchestrator + agent_experiment_multi_round (the experimenting agent).

Per round the agent
  1. LOOKS (same tools as agent_vision: numbers, missed-val montages, numbered
     train-source montages) plus its LAB NOTEBOOK (every earlier candidate, its
     hypothesis, result and the lesson the agent wrote);
  2. DESIGNS K candidate plans (allocation + params + source whitelist), each a
     hypothesis;
  3. the LAB generates every candidate in its own scratch pool (starting from
     the committed pool's per-source state, same per-round candidate ceiling)
     and trains/evaluates each on val with the run's seed and full epochs;
  4. the agent KEEPS one candidate and writes a lesson.  The kept candidate's
     images, generator state and trained model become the round's result (no
     retraining: it is the same training the committed condition would run).

Extra compute vs the other conditions: K-1 additional trainings per round
(reported).  Val only; test is never touched here.  Every step is cached so an
interrupted run resumes without re-asking the LLM or retraining.

Usage:
    python scripts/run_conditions_v31e.py --tag dev_exp --seeds tuning \
        --conditions agent_experiment_multi_round --agent-endpoint http://192.168.31.47:18182/v1
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_conditions_v31 as v31  # noqa: E402  (installs the v3.1 patches on run_conditions_v3)
from defectaug import vision_agent as va  # noqa: E402

base = v31.base
CLASSES = base.CLASSES
COND = "agent_experiment_multi_round"
base.SHORT[COND] = "ae"
log = base.log


def _trial_cond(k: int) -> str:
    return f"agent_experiment_trial_{k}"


def _with_v31_generator(fn, *args):
    """Run base.generate_round with the v3.1 generator (source whitelist)."""
    real_run = base.subprocess.run

    def run(cmd, *a, **kw):
        cmd = [("scripts/run_synth_generation_v31.py" if c == "scripts/run_synth_generation_v3.py" else c)
               for c in cmd]
        return real_run(cmd, *a, **kw)
    base.subprocess.run = run
    try:
        return fn(*args)
    finally:
        base.subprocess.run = real_run


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _summary(metrics_path: Path) -> dict:
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    pc = {c["class_name"]: c for c in doc["per_class"]}
    return {"map50": round(doc["overall"]["ap50"], 4), "map50_95": round(doc["overall"]["ap50_95"], 4),
            "worst_class_ap50": round(doc["worst_class"]["ap50"], 4),
            "per_class": {c: {"ap50": round(pc[c]["ap50"], 4), "recall": round(pc[c]["recall"], 4),
                              "precision": round(pc[c]["precision"], 4)} for c in CLASSES}}


def _compact_params(params: dict) -> dict:
    out = {}
    for c, p in params.items():
        q = {k: v for k, v in p.items() if k != "sources"}
        src = p.get("sources") or []
        q["sources"] = [s.rsplit("_", 1)[1] for s in src] if src else "all"
        out[c] = q
    return out


def _remaining(ctx, seed) -> dict:
    state_path = base.pool_dir(ctx, COND, seed) / "state.json"
    per = json.loads(state_path.read_text(encoding="utf-8"))["per_source"] if state_path.is_file() else {}
    return {c: {s.stem.rsplit("_", 1)[1]: max(0, min(ctx.max_acc - per.get(s.stem, {}).get("accepted", 0),
                                                      ctx.max_cand - per.get(s.stem, {}).get("candidates", 0)))
                for s in srcs} for c, srcs in v31.sources_by_class(ctx).items()}


def _control(ctx, r, bp, allocated, cap, history) -> dict:
    """The lab's control arm: round 1 = default recipe, balanced; later = repeat the last kept plan."""
    if not history:
        return {"hypothesis": "CONTROL (added by the lab): default generation recipe, balanced allocation.",
                "allocation": base.uniform_proposal(bp), "control": True,
                "params": {c: {**base.DEFAULT.as_dict(), "sources": []} for c in CLASSES}}
    last = history[-1]
    return {"hypothesis": "CONTROL (added by the lab): repeat your last kept plan (allocation projected onto "
                          "this round's constraints).",
            "allocation": base.project(bp, last["allocation"], allocated, cap, r), "control": True,
            "params": {c: dict(last["params"][c]) for c in CLASSES}}


def _whitelist_shortfall(ctx, seed, alloc, params) -> str:
    """Whitelisted sources must still be able to supply the class's allocation."""
    state_path = base.pool_dir(ctx, COND, seed) / "state.json"
    per = json.loads(state_path.read_text(encoding="utf-8"))["per_source"] if state_path.is_file() else {}
    msgs = []
    for c in CLASSES:
        stems = params[c].get("sources") or []
        if not stems:
            continue
        left = {s.rsplit("_", 1)[1]: max(0, min(ctx.max_acc - per.get(s, {}).get("accepted", 0),
                                                  ctx.max_cand - per.get(s, {}).get("candidates", 0)))
                for s in stems}
        if alloc[c] > sum(left.values()):
            msgs.append(f"{c}: your {len(stems)} sources can still yield at most {sum(left.values())} images "
                        f"(remaining per source {left}; each source gives at most {ctx.max_acc} images over the "
                        f"whole run) but the allocation asks {alloc[c]}: add sources, use [] or lower the allocation")
    return "; ".join(msgs)


# ------------------------------------------------------------------ design
def _design(ctx, seed, r, bp, prev, allocated, cap, history, notebook, k, dec_base):
    """Ask for K candidates (cached).  Returns (candidates, messages, info)."""
    content, obs, srcs = v31.look_content(ctx, COND, seed, r, bp, prev, allocated, cap, history,
                                          schema=va.DESIGN_SCHEMA.replace("{k}", str(k)))
    content.insert(1, va.text_part(
        "LAB NOTEBOOK (every candidate you tested in earlier rounds, its result on val, the one you kept "
        "and your lesson; 'delta' = change vs the model before that round):\n"
        + (json.dumps(notebook, ensure_ascii=False) if notebook else "empty (first round)")))
    content.append(va.text_part(
        "REMAINING IMAGES PER SOURCE (source number -> how many more synthetic images it can still yield "
        f"in this run; at most {ctx.max_acc} per source in total): " + json.dumps(_remaining(ctx, seed))))
    content.append(va.text_part(f"Design exactly {k} different candidates for round {r}. Each candidate must "
                                "satisfy every constraint on its own (it may be the one kept)."))
    system = va.SYSTEM_EXPERIMENT.replace("{k}", str(k))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    path = Path(f"{dec_base}_design.json")
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        messages.append({"role": "assistant", "content": cached["reply"]})
        log(f"RESUME design r{r}: {len(cached['candidates'])} candidates")
        return cached["candidates"], messages, cached["info"]
    info = {"calls": []}
    candidates: list[dict] = []
    max_attempts = int(ctx.p["agent_vision_policy"]["max_attempts_per_step"])
    for attempt in range(max_attempts):
        resp = v31._ask(ctx, messages, info, f"design{attempt + 1}")
        errors, candidates, seen = [], [], set()
        try:
            raw = va.extract_json(resp["content"]).get("candidates")
            if not isinstance(raw, list) or not raw:
                raise ValueError("'candidates' must be a non-empty list")
            for i, cand in enumerate(raw[:k], 1):
                try:
                    alloc, params = v31._clean_plan(cand, srcs, bp, allocated, cap, r)
                except ValueError as exc:
                    errors.append(f"candidate {i}: {exc}")
                    continue
                short = _whitelist_shortfall(ctx, seed, alloc, params)
                if short and attempt + 1 < max_attempts:
                    errors.append(f"candidate {i}: {short}")
                    continue
                key = _sha([alloc, params])
                if key in seen:
                    errors.append(f"candidate {i} is identical to an earlier one")
                    continue
                seen.add(key)
                candidates.append({"hypothesis": str(cand.get("hypothesis", "")), "allocation": alloc,
                                   "params": params})
            if len(raw) != k:
                errors.append(f"expected exactly {k} candidates, got {len(raw)}")
        except (ValueError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(str(exc))
        if not errors:
            break
        if attempt + 1 < max_attempts:
            msg = "REJECTED: " + "; ".join(errors)
            if resp["finish_reason"] == "length":
                msg += " (your reply was cut off: think more briefly and end with the JSON)"
            messages.append({"role": "user", "content": msg + f"\nReturn the full corrected JSON with {k} candidates."})
    info["design_errors"] = errors
    # Proceed with whatever valid candidates exist (>= 1); none -> caller falls back.
    path.write_text(json.dumps({"candidates": candidates, "reply": messages[-1]["content"]
                                if messages[-1]["role"] == "assistant" else "", "info": info},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return candidates, messages, info


# ------------------------------------------------------------------ lab
def _prepare_trial(ctx, seed, r, k, cand, cand_budget, bp, allocated, cap) -> dict:
    """Generate candidate k in its own pool, starting from the committed pool's state."""
    tcond = _trial_cond(k)
    tpool = base.pool_dir(ctx, tcond, seed)
    stamp = tpool / f"design_r{r}.json"
    key = _sha([cand["allocation"], cand["params"], sorted(allocated.items())])
    if (tpool / f"alloc_r{r}.json").is_file() and stamp.is_file() and json.loads(
            stamp.read_text(encoding="utf-8")).get("key") == key:
        return json.loads((tpool / f"alloc_r{r}.json").read_text(encoding="utf-8"))
    shutil.rmtree(tpool, ignore_errors=True)
    tpool.mkdir(parents=True)
    main = base.pool_dir(ctx, COND, seed)
    for f in ("state.json", "receipts.jsonl", "ledger.jsonl"):
        if (main / f).is_file():
            shutil.copyfile(main / f, tpool / f)
    gen = _with_v31_generator(base.generate_round, ctx, tcond, seed, r, cand["allocation"], cand_budget,
                              bp, allocated, cap, cand["params"])
    stamp.write_text(json.dumps({"key": key, "spills": gen["spills"]}), encoding="utf-8")
    return gen["reached"]


def _trial_dataset(ctx, seed, r, k) -> tuple[Path, Path]:
    tcond = _trial_cond(k)
    tpool = base.pool_dir(ctx, tcond, seed)
    main = base.pool_dir(ctx, COND, seed)
    real = [l.strip() for l in ctx.train_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = sorted(str(p) for p in main.glob("r*/images/*.png") if int(p.parent.parent.name[1:]) < r)
    new = sorted(str(p) for p in (tpool / f"r{r}" / "images").glob("*.png"))
    manifest = tpool / f"train_manifest_r{r}.txt"
    manifest.write_text("\n".join(real + kept + new) + "\n", encoding="utf-8")
    key = f"v3_{ctx.tag}_{tcond}_{seed}_r{r}"
    rc = base.sh([base.PY, "scripts/build_synth_dataset.py", "--condition", f"v3_{ctx.tag}_{tcond}",
                  "--seed", seed, "--round", r, "--manifest", manifest,
                  "--splits-dir", ctx.p["dataset"]["splits_dir"]])
    if rc != 0:
        raise RuntimeError(f"dataset build failed {key}")
    return (PROJECT / "configs" / "generated" / f"dataset_{key}.yaml",
            PROJECT / "data" / "manifests" / f"dataset_content_lock_{key}.json")


def _train_trial(ctx, seed, r, k, data) -> Path:
    name = ctx.run_name(_trial_cond(k), seed, r)
    try:
        return base.train_eval(ctx, name, *data, seed)
    except RuntimeError as exc:   # e.g. a shared label-cache race between parallel runs
        log(f"trial {name} failed ({exc}); retrying once")
        return base.train_eval(ctx, name, *data, seed)


def _run_lab(ctx, seed, r, cands, cand_budget, bp, allocated, cap) -> list[dict]:
    results = []
    for k, cand in enumerate(cands, 1):
        reached = _prepare_trial(ctx, seed, r, k, cand, cand_budget, bp, allocated, cap)
        results.append({"k": k, "reached": reached, "data": _trial_dataset(ctx, seed, r, k)})
    workers = int(ctx.p["agent_experiment_policy"]["parallel_trainings"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {res["k"]: ex.submit(_train_trial, ctx, seed, r, res["k"], res["data"]) for res in results}
        for res in results:
            res["metrics"] = futs[res["k"]].result()
            del res["data"]
    return results


# ------------------------------------------------------------------ choose + commit
def _choose(ctx, messages, info, r, cands, results, prev, dec_base) -> tuple[int, str]:
    path = Path(f"{dec_base}_choice.json")
    if path.is_file():
        c = json.loads(path.read_text(encoding="utf-8"))
        return c["choice"], c["lesson"]
    before = _summary(prev)
    rows = []
    for cand, res in zip(cands, results):
        s = _summary(res["metrics"])
        rows.append({"candidate": res["k"], "control": bool(cand.get("control")), "hypothesis": cand["hypothesis"],
                     "allocation_reached": res["reached"],
                     "val": s, "delta_map50_vs_before": round(s["map50"] - before["map50"], 4),
                     "delta_ap50_vs_before": {c: round(s["per_class"][c]["ap50"] - before["per_class"][c]["ap50"], 4)
                                              for c in CLASSES}})
    messages.append({"role": "user", "content": [va.text_part(
        f"LAB RESULTS round {r} (validation, model trained with each candidate; before this round: "
        f"{json.dumps(before)}):\n{json.dumps(rows, ensure_ascii=False)}\n"
        f"Keep exactly one candidate (number 1..{len(cands)}). Reply schema:\n{va.CHOOSE_SCHEMA}")]})
    choice, lesson = None, ""
    for attempt in range(int(ctx.p["agent_vision_policy"]["max_attempts_per_step"])):
        resp = v31._ask(ctx, messages, info, f"choose{attempt + 1}")
        try:
            doc = va.extract_json(resp["content"])
            choice = int(doc["choice"])
            if not 1 <= choice <= len(cands):
                raise ValueError(f"choice must be 1..{len(cands)}")
            lesson = str(doc.get("lesson", ""))
            break
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            choice = None
            messages.append({"role": "user", "content": f"REJECTED: {exc}. Return the JSON again."})
    if choice is None:   # unusable reply: keep the best val mAP50 (recorded as a fallback)
        maps = [_summary(x["metrics"])["map50"] for x in results]
        choice, lesson, info["choice_fallback"] = maps.index(max(maps)) + 1, "", True
    info["lab_table"] = rows
    path.write_text(json.dumps({"choice": choice, "lesson": lesson}, ensure_ascii=False), encoding="utf-8")
    return choice, lesson


def _commit(ctx, seed, r, k, cand, reached):
    """Kept candidate -> the committed pool, run dir and val metrics of this round."""
    tcond, main = _trial_cond(k), base.pool_dir(ctx, COND, seed)
    tpool = base.pool_dir(ctx, tcond, seed)
    main.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tpool / f"r{r}", main / f"r{r}", dirs_exist_ok=True)
    for f in ("state.json", "receipts.jsonl", "ledger.jsonl", f"candidates_r{r}.json"):
        if (tpool / f).is_file():
            shutil.copyfile(tpool / f, main / f)
    (main / f"params_r{r}.json").write_text(json.dumps(cand["params"], indent=1), encoding="utf-8")
    src_name, dst_name = ctx.run_name(tcond, seed, r), ctx.run_name(COND, seed, r)
    shutil.copytree(PROJECT / "runs" / "yolo" / src_name, PROJECT / "runs" / "yolo" / dst_name, dirs_exist_ok=True)
    shutil.copytree(ctx.eval_root / f"{src_name}_val", ctx.eval_root / f"{dst_name}_val", dirs_exist_ok=True)
    (ctx.eval_root / f"{dst_name}_val" / "selected_from.json").write_text(
        json.dumps({"trial_run": src_name, "candidate": k}), encoding="utf-8")
    # Written last: base.run_augmented treats alloc_r{r}.json as "round generated".
    (main / f"alloc_r{r}.json").write_text(json.dumps(reached), encoding="utf-8")


def experiment_proposal(ctx, cond, seed, r, bp, prev, allocated, cap, history):
    pol = ctx.p["agent_experiment_policy"]
    k = int(pol["candidates_per_round"])
    dec_base = ctx.dec_root / f"{COND}_{seed}_r{r}"
    nb_path = ctx.dec_root / f"{COND}_{seed}_notebook.json"
    notebook = json.loads(nb_path.read_text(encoding="utf-8")) if nb_path.is_file() else []
    notebook = [e for e in notebook if e["round"] < r]
    cands, messages, info = _design(ctx, seed, r, bp, prev, allocated, cap, history, notebook, k, dec_base)
    if not cands:   # same fallback as the other agents: formula quota + formula params rule
        log(f"{COND} s{seed} r{r}: no valid candidate; falling back to the formula proposal")
        raw, _ = base.formula_proposal(ctx, prev, bp, base.formula_weights(ctx, "formula"))
        return raw, None, {**info, "fallback": "formula"}
    if ctx.p["agent_experiment_policy"].get("control_arm"):
        ctl = _control(ctx, r, bp, allocated, cap, history)
        if all(_sha([ctl["allocation"], ctl["params"]]) != _sha([c["allocation"], c["params"]]) for c in cands):
            cands = cands + [ctl]
    cand_budget = ctx.cand_total // bp.rounds
    results = _run_lab(ctx, seed, r, cands, cand_budget, bp, allocated, cap)
    choice, lesson = _choose(ctx, messages, info, r, cands, results, prev, dec_base)
    chosen, res = cands[choice - 1], results[choice - 1]
    before = _summary(prev)
    notebook.append({"round": r, "kept": choice, "lesson": lesson, "candidates": [
        {"candidate": rs["k"], "control": bool(c.get("control")), "hypothesis": c["hypothesis"],
         "allocation": rs["reached"],
         "params": _compact_params(c["params"]),
         "val_map50": _summary(rs["metrics"])["map50"],
         "delta_map50": round(_summary(rs["metrics"])["map50"] - before["map50"], 4),
         "val_ap50": {cl: _summary(rs["metrics"])["per_class"][cl]["ap50"] for cl in CLASSES}}
        for c, rs in zip(cands, results)]})
    nb_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    _commit(ctx, seed, r, choice, chosen, res["reached"])
    maps = [_summary(x["metrics"])["map50"] for x in results]
    info["argmax_map50"] = maps.index(max(maps)) + 1
    log(f"KEEP {COND} s{seed} r{r}: candidate {choice}/{len(cands)} (val mAP50 {maps})")
    info.update(candidates=cands, kept=choice, lesson=lesson,
                extra_trainings=len(cands) - 1)
    return res["reached"], chosen["params"], info


_v31_router = base.agent_proposal
_v31_generate_round = base.generate_round


def generate_round(ctx, cond, seed, r, *args):
    """Fallback rounds of the committed pool also use the v3.1 generator."""
    if cond == COND:
        return _with_v31_generator(v31._orig_generate_round, ctx, cond, seed, r, *args)
    return _v31_generate_round(ctx, cond, seed, r, *args)


def router(ctx, obs, r, bp, allocated, cap):
    c = v31._CTX
    if c["cond"] != COND:
        return _v31_router(ctx, obs, r, bp, allocated, cap)
    alloc, params, info = experiment_proposal(ctx, c["cond"], c["seed"], r, bp, c["prev"], allocated, cap,
                                              c["history"])
    return (alloc or {}), params, {**info, "repaired": alloc is None}


base.agent_proposal = router
base.generate_round = generate_round


def main() -> int:
    """base.main, plus: any tag starting with 'final' needs the frozen v3.1 protocol, its epoch
    count and the final seeds (base.main only guards the literal tag 'final')."""
    import argparse
    import yaml
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seeds", default="all")
    args, _ = ap.parse_known_args()
    if args.tag.startswith("final"):
        p = yaml.safe_load(base.PROTOCOL_PATH.read_text(encoding="utf-8"))
        if p["experiment"]["status"] != "frozen_final":
            raise SystemExit(f"tag {args.tag!r} requires {base.PROTOCOL_PATH.name} status frozen_final")
        if args.epochs not in (None, int(p["detector"]["epochs_per_round"])):
            raise SystemExit(f"tag {args.tag!r} must use the protocol epoch count")
        if args.seeds == "tuning":
            raise SystemExit(f"tag {args.tag!r} is for the final seeds only")
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
