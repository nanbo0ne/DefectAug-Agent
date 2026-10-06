"""v3.1 orchestrator = v3 orchestrator + agent_vision_multi_round.

Everything (training, eval, generation ceilings, constraints, formula/uniform
baselines) is inherited from run_conditions_v3.py unchanged; only the new
proposer `agent_vision` is added.  Protocol: configs/experiment_v31.yaml.

Per round the vision agent runs a short tool loop (see vision_agent.py):
  look (missed val crops + train source crops) -> plan -> preview a few
  synthetic samples in a scratch pool -> optional single revision -> commit.

Usage:
    python scripts/run_conditions_v31.py --tag dev --seeds tuning \
        --conditions agent_vision_multi_round --agent-endpoint http://192.168.31.47:18182/v1
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_conditions_v3 as base  # noqa: E402
from defectaug import vision_agent as va  # noqa: E402
from defectaug import vision_tools as vt  # noqa: E402
from defectaug.allocation import ConstraintViolation, headroom, validate  # noqa: E402
from defectaug.gen_params import ParamViolation, validate_params  # noqa: E402

CLASSES = base.CLASSES
base.PROTOCOL_PATH = PROJECT / "configs" / "experiment_v31.yaml"
base.SHORT["agent_vision_multi_round"] = "av"
MASKS = PROJECT / "data" / "canonical" / "masks" / "all"
_orig_generate_round = base.generate_round


# ------------------------------------------------------------- tool data
def missed_list(ctx, metrics_path: Path) -> list[dict]:
    """Per-instance misses of the model behind metrics_path (cached)."""
    out = metrics_path.parent / "missed_instances.json"
    if not out.is_file():
        name = metrics_path.parent.name.removesuffix("_val")
        weights = PROJECT / "runs" / "yolo" / name / "weights" / ctx.p["detector"]["eval_weights"]
        rc = base.sh([base.PY, "scripts/val_miss_analysis_v31.py", "--weights", weights,
                      "--output", metrics_path.parent / "miss_analysis_v31.json",
                      "--missed-output", out, "--conf", ctx.p["miss_analysis"]["conf"],
                      "--imgsz", ctx.p["detector"]["imgsz"], "--device", ctx.device],
                     metrics_path.parent / "miss_analysis_v31.log")
        if rc != 0 or not out.is_file():
            raise RuntimeError(f"missed-instance export failed for {name}")
    return json.loads(out.read_text(encoding="utf-8"))


def sources_by_class(ctx) -> dict[str, list[Path]]:
    import run_synth_generation_v31 as g
    return g.resolve_train_sources(ctx.train_manifest)


def num_to_stem(cls: str, nums) -> list[str]:
    return [f"{cls}_{int(n)}" for n in nums]


def _clean_plan(plan: dict, srcs: dict[str, list[Path]], bp, allocated, cap, r) -> tuple[dict, dict]:
    """Validate allocation + params + source numbers; raise with readable reasons."""
    errors = []
    try:
        alloc = validate(bp, plan.get("allocation"), allocated, cap, r)
    except (ConstraintViolation, ValueError) as exc:
        alloc, errors = None, [f"allocation: {exc}"]
    params = {}
    raw = plan.get("params")
    if not isinstance(raw, dict) or set(raw) != set(CLASSES):
        errors.append(f"params must have exactly the keys {list(CLASSES)}")
    else:
        for c in CLASSES:
            item = dict(raw[c]) if isinstance(raw[c], dict) else {}
            nums = item.pop("sources", []) or []
            try:
                p = validate_params(c, item).as_dict()
            except ParamViolation as exc:
                errors.append(f"params: {exc}")
                continue
            valid = {s.stem.rsplit("_", 1)[1] for s in srcs[c]}
            bad = [n for n in nums if str(n) not in valid]
            if bad:
                errors.append(f"{c}: unknown source numbers {bad}; valid are {sorted(valid, key=int)}")
                continue
            if nums and len(nums) < 2:
                errors.append(f"{c}: choose at least 2 sources or [] for all")
                continue
            p["sources"] = num_to_stem(c, nums)
            params[c] = p
    if errors:
        raise ValueError("; ".join(errors))
    return alloc, params


def _ask(ctx, messages, info, tag) -> dict:
    ap = ctx.p["agent_vision_policy"]
    resp = va.chat(ctx.agent_endpoint, ctx.agent_model, messages, int(ap["max_tokens"]))
    if resp.get("served_model") != ctx.agent_model:
        raise RuntimeError(f"served_model {resp.get('served_model')!r} != {ctx.agent_model!r}")
    info["calls"].append({"step": tag, **{k: resp[k] for k in ("finish_reason", "usage", "reasoning_chars",
                                                                "elapsed_seconds")},
                          "content": resp["content"][-6000:]})
    messages.append({"role": "assistant", "content": resp["content"]})
    return resp


def _plan_loop(ctx, messages, info, tag, srcs, bp, allocated, cap, r):
    """Ask until a valid plan or attempts run out; returns (alloc, params) or (None, None)."""
    for k in range(int(ctx.p["agent_vision_policy"]["max_attempts_per_step"])):
        resp = _ask(ctx, messages, info, f"{tag}{k + 1}")
        try:
            plan = va.extract_json(resp["content"])
            alloc, params = _clean_plan(plan, srcs, bp, allocated, cap, r)
            info.setdefault("plans", []).append({"step": tag, "plan": plan})
            return alloc, params
        except (ValueError, json.JSONDecodeError) as exc:
            msg = f"REJECTED: {exc}"
            if resp["finish_reason"] == "length":
                msg += " (your reply was cut off: think more briefly and end with the JSON)"
            messages.append({"role": "user", "content": msg + "\nReturn a corrected JSON plan."})
    return None, None


def _preview(ctx, cond, seed, r, params) -> list[dict]:
    """Generate a few samples per class with `params` in a scratch pool; return content parts."""
    n = int(ctx.p["agent_vision_policy"]["preview_per_class"])
    root = ctx.pool_root / "_preview"
    pool = root / f"agent_vision_preview_{seed}"
    shutil.rmtree(pool, ignore_errors=True)
    pf = root / f"params_{seed}_r{r}.json"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(params), encoding="utf-8")
    out = subprocess.run([str(base.PY), "scripts/run_synth_generation_v31.py", "--condition", "agent_vision_preview",
                          "--seed", str(seed), "--round", "1",
                          "--alloc", ",".join(f"{c}:{n}" for c in CLASSES), "--candidate-budget", str(9 * n),
                          "--train-manifest", str(ctx.train_manifest), "--pool-root", str(root),
                          "--params", str(pf)],
                         cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
    parts = [va.text_part("PREVIEW of your plan (green box = the pasted defect, numbered per class):")]
    for c in CLASSES:
        imgs = sorted((pool / "r1" / "images").glob(f"{c}_*.png"))
        sheet = vt.synthetic_montage(imgs, k=n)
        if sheet is None:
            parts.append(va.text_part(f"{c}: no sample could be generated with these params "
                                      f"(generator rc={out.returncode})."))
        else:
            parts += [va.text_part(f"{c}: {len(imgs)} preview samples"), va.image_part(vt.to_data_url(sheet))]
    return parts


def look_content(ctx, cond, seed, r, bp, prev, allocated, cap, history, schema=va.PLAN_SCHEMA):
    """Numbers + missed-val montages + source montages + action space.  Returns (content, obs, srcs)."""
    ap = ctx.p["agent_vision_policy"]
    srcs = sources_by_class(ctx)
    obs = base.observation(ctx, cond, seed, r, bp, prev, allocated, cap, history)
    missed = missed_list(ctx, prev)
    content = [va.text_part(
        f"ROUND {r} of {bp.rounds}. Budget this round: exactly {bp.round_budget} images. "
        f"headroom_this_round={json.dumps(headroom(bp, allocated, cap))}; cumulative cap per class "
        f"{bp.cumulative_cap}; after the last round every class needs >= {bp.final_min_per_class}.\n"
        "NUMBERS (previous model on val, miss analysis, history of your earlier rounds with the "
        "params you used and what happened):\n" + json.dumps(obs, ensure_ascii=False))]
    for c in CLASSES:
        sheet, n_missed = vt.missed_montage(missed, c, int(ap["missed_tiles_per_class"]))
        if sheet is not None:
            content += [va.text_part(f"{c}: {n_missed} val instances MISSED by the current detector; "
                                     f"{min(n_missed, int(ap['missed_tiles_per_class']))} shown, smallest first, "
                                     "red box = the defect ('x' = predicted as another class):"),
                        va.image_part(vt.to_data_url(sheet))]
        content += [va.text_part(f"{c}: train SOURCE defects you can copy, labelled by source number "
                                 f"(blue box):"),
                    va.image_part(vt.to_data_url(vt.source_montage(srcs[c], MASKS)))]
    content.append(va.text_part(
        "ACTION SPACE per class: scale_min/scale_max in [0.30, 1.40] with span >= 0.20 (multiplies the "
        "source defect size; capped if it would not fit); target_pool same_class|any_class "
        "(background images); source_weighting uniform|prefer_small|prefer_large; max_overlap in "
        "[0.0, 0.30]; sources = list of source numbers from the montage (>= 2) or [] for all.\n"
        "Reply schema:\n" + schema))
    return content, obs, srcs


def vision_proposal(ctx, cond, seed, r, bp, prev, allocated, cap, history):
    """Look -> plan -> preview -> (revise) -> commit.  Returns (alloc|None, params|None, info)."""
    ap = ctx.p["agent_vision_policy"]
    content, obs, srcs = look_content(ctx, cond, seed, r, bp, prev, allocated, cap, history)
    info = {"calls": [], "observation": obs}
    messages = [{"role": "system", "content": va.SYSTEM}, {"role": "user", "content": content}]
    alloc, params = _plan_loop(ctx, messages, info, "plan", srcs, bp, allocated, cap, r)
    if alloc is not None and ap.get("preview"):
        messages.append({"role": "user", "content": _preview(ctx, cond, seed, r, params) + [va.text_part(
            "Check the preview against the missed defects. If the pastes look right, repeat the SAME JSON "
            "plan. If not (wrong size, unrealistic, wrong look, wasted on a class), return a REVISED JSON "
            "plan. Same schema.")]})
        a2, p2 = _plan_loop(ctx, messages, info, "revise", srcs, bp, allocated, cap, r)
        if a2 is not None:
            info["revised"] = (a2 != alloc) or (p2 != params)
            alloc, params = a2, p2
    return alloc, params, info


def generate_round_v31(ctx, cond, seed, r, alloc, cand_budget, bp, allocated, cap, params):
    """v3 generate_round, but the agent_vision condition uses the v3.1 generator (source whitelist)."""
    if cond != "agent_vision_multi_round":
        return _orig_generate_round(ctx, cond, seed, r, alloc, cand_budget, bp, allocated, cap, params)
    real_run = subprocess.run
    def run(cmd, *a, **kw):  # swap the generator script only for this call
        cmd = [("scripts/run_synth_generation_v31.py" if c == "scripts/run_synth_generation_v3.py" else c)
               for c in cmd]
        return real_run(cmd, *a, **kw)
    base.subprocess.run = run
    try:
        return _orig_generate_round(ctx, cond, seed, r, alloc, cand_budget, bp, allocated, cap, params)
    finally:
        base.subprocess.run = real_run


_orig_agent_proposal = base.agent_proposal
_CTX = {}


def agent_proposal_router(ctx, obs, r, bp, allocated, cap):
    """Called by v3 run_augmented for proposers that are not uniform/formula."""
    cond, seed, prev, history = _CTX["cond"], _CTX["seed"], _CTX["prev"], _CTX["history"]
    if cond != "agent_vision_multi_round":
        return _orig_agent_proposal(ctx, obs, r, bp, allocated, cap)
    alloc, params, info = vision_proposal(ctx, cond, seed, r, bp, prev, allocated, cap, history)
    info.pop("observation", None)
    return (alloc or {}), params, {**info, "repaired": alloc is None}


_orig_observation = base.observation


def observation_spy(ctx, cond, seed, r, bp, prev, allocated, cap, history):
    _CTX.update(cond=cond, seed=seed, prev=prev, history=history)
    return _orig_observation(ctx, cond, seed, r, bp, prev, allocated, cap, history)


base.generate_round = generate_round_v31
base.agent_proposal = agent_proposal_router
base.observation = observation_spy

if __name__ == "__main__":
    raise SystemExit(base.main())
