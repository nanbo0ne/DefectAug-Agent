"""v3.2 DEV orchestrator: agent_experiment + cross-run experience memory.

agent_memory_multi_round = the frozen v3.1 experimenting agent (run_conditions_v31e,
imported unchanged) whose design step additionally shows the lab notebooks of
earlier complete runs (agent_memory_policy.memory_runs: the two tuning-seed dev
runs).  It has its own pools, trial pools (agent_memory_trial_k, stems y1..y4),
notebook and decisions, so it can run next to agent_experiment_multi_round on the
same seed.  agent_coder_multi_round = agent_memory + a tool to write its own
appearance operator per class (defectaug.agent_ops; sandboxed, probed on real
pastes before the lab).  Protocol: configs/experiment_v32.yaml (DEV seeds, val only, draft).

Usage:
    python scripts/run_conditions_v32.py --tag dev32 --seeds 0 --agent-endpoint http://192.168.31.47:18182/v1 \
        --conditions formula_tuned_multi_round,agent_experiment_multi_round,agent_memory_multi_round
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_conditions_v31e as e  # noqa: E402  (frozen v3.1 code, used as is)
from defectaug import vision_agent as va  # noqa: E402

base, v31 = e.base, e.v31
base.PROTOCOL_PATH = PROJECT / "configs" / "experiment_v32.yaml"
EXP, MEM, CODER = e.COND, "agent_memory_multi_round", "agent_coder_multi_round"
base.SHORT[MEM], base.SHORT[CODER] = "am", "ac"
_exp_trial = e._trial_cond
CODER_DS, CODER_GLM = "agent_coder_ds_multi_round", "agent_coder_glm_multi_round"
base.SHORT[CODER_DS], base.SHORT[CODER_GLM] = "acd", "acg"
CODERS = (CODER, CODER_DS, CODER_GLM)
TRIALS = {MEM: "agent_memory_trial_{}", CODER: "agent_coder_trial_{}",
          CODER_DS: "agent_coder_ds_trial_{}", CODER_GLM: "agent_coder_glm_trial_{}"}

CODER_TOOL = (
    "TOOL: WRITE YOUR OWN AUGMENTATION OPERATOR. Per class you may add \"augment_code\": Python source "
    "defining\n  def augment(pasted, target, mask, rng):\n"
    "pasted/target: HxWx3 uint8 RGB images after/before the paste; mask: HxW bool, the pasted defect; "
    "rng: numpy Generator. Return the new HxWx3 image. It runs after the Copy-Paste and before the "
    "boundary feather; only pixels inside mask can change, so the label stays exact. np (numpy) and cv2 "
    "are available; no imports, no file access, no while loops, at most 2 s per image. With it you can "
    "create defect appearances the plain copy cannot produce (contrast, blur, sharpness, noise, texture, "
    "brightness relative to the background, partial fading, ...), e.g.\n"
    "  def augment(pasted, target, mask, rng):\n"
    "      k = int(rng.choice([3, 5]))\n"
    "      return cv2.GaussianBlur(pasted, (k, k), 0)\n"
    "The QC gate still requires >= 50% of the defect pixels to differ by >= 3 grey levels from the "
    "background. Each operator is test-run on 8 real pastes of its class before the lab; it is rejected "
    "with the reason if it errors or most samples fail QC. Use \"augment_code\": null for a plain paste. "
    "Whether an operator helps is an empirical question: test it in the lab against a candidate without it. "
    "In the JSON, put the code in one string with \\n line breaks.")


def _with_v32_generator(fn, *args):
    """v3.2 generator = v3.1 generator with the agent_memory condition names added."""
    real_run = base.subprocess.run

    def run(cmd, *a, **kw):
        cmd = [("scripts/run_synth_generation_v32.py"
                if c in ("scripts/run_synth_generation_v3.py", "scripts/run_synth_generation_v31.py") else c)
               for c in cmd]
        return real_run(cmd, *a, **kw)
    base.subprocess.run = run
    try:
        return fn(*args)
    finally:
        base.subprocess.run = real_run


e._with_v31_generator = _with_v32_generator


def memory_text(ctx) -> str:
    runs = []
    for i, path in enumerate(ctx.p["agent_memory_policy"]["memory_runs"], 1):
        nb = json.loads((PROJECT / path).read_text(encoding="utf-8"))
        runs.append({"run": i, "rounds": [
            {"round": rd["round"], "kept": rd["kept"], "lesson": rd["lesson"],
             "candidates": [{"candidate": c["candidate"], "control": c.get("control", False),
                             "hypothesis": c["hypothesis"][:240], "allocation": c["allocation"],
                             "params": c["params"], "val_map50": c["val_map50"],
                             "delta_map50": c["delta_map50"], "val_ap50": c["val_ap50"]}
                            for c in rd["candidates"]]} for rd in nb]})
    return ("EXPERIENCE MEMORY: your lab notebooks from earlier complete runs of this same task (same "
            "real data, validation split, budget and constraints; a different training seed, so absolute "
            "numbers differ). Treat it as prior evidence, not truth -- single-run differences up to ~0.03 "
            "are noise; trust effects that were large or repeated across runs. Use it to skip hypotheses "
            "that already failed clearly and to design better candidates from round 1 on:\n"
            + json.dumps(runs, ensure_ascii=False))


_look = v31.look_content


def _api(ctx, cond) -> dict | None:
    return ctx.p.get("agent_api_models", {}).get(cond)


def source_table(srcs) -> dict:
    """Numeric stand-in for the source montage: size and contrast of every train source defect."""
    import cv2
    import numpy as np
    from PIL import Image
    out = {}
    for c, paths in srcs.items():
        rows = {}
        for pth in paths:
            m = np.asarray(Image.open(v31.MASKS / f"{pth.stem}.png").convert("L")) > 0
            g = np.asarray(Image.open(pth).convert("L"), dtype=np.float32)
            ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
            ys, xs = np.where(m)
            rows[pth.stem.rsplit("_", 1)[1]] = {
                "mask_px": int(m.sum()), "bbox_wh": [int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1] if len(xs) else [0, 0],
                "contrast": round(float(abs(g[m].mean() - g[ring].mean())), 1) if m.any() and ring.any() else None,
                "darker_than_background": bool(g[m].mean() < g[ring].mean()) if m.any() and ring.any() else None}
        out[c] = rows
    return out


def missed_table(missed: list[dict]) -> dict:
    out = {}
    for c in CLASSES:
        items = sorted((x for x in missed if x["cls"] == c), key=lambda x: x["size"])
        out[c] = {"n_missed": len(items), "instances": [
            {"size_px": round(float(x["size"]), 1), "contrast": round(float(x["contrast"]), 1),
             "box_wh": [round(float(x["xyxy"][2] - x["xyxy"][0])), round(float(x["xyxy"][3] - x["xyxy"][1]))],
             **({"misclassified_as": x["misclassified_as"]} if x.get("misclassified_as") else {})}
            for x in items]}
    return out


def look_content(ctx, cond, *args, **kw):
    content, obs, srcs = _look(ctx, cond, *args, **kw)
    if cond in (MEM,) + CODERS:
        content.insert(1, va.text_part(memory_text(ctx)))
    if cond in CODERS:
        content.insert(len(content) - 1, va.text_part(CODER_TOOL))
    api = _api(ctx, cond)
    if api and not api["vision"]:   # text-only model: numeric descriptions instead of montages
        content = [p if p.get("type") != "image_url" else va.text_part("(image not shown: no image input in this run)")
                   for p in content]
        prev = args[3]
        content.insert(len(content) - 1, va.text_part(
            "NOTE: you have no image input in this run; the montages are replaced by these numbers. "
            "MISSED VAL INSTANCES per class (size_px = sqrt of box area in px, contrast = |defect - surrounding| "
            "grey level):\n" + json.dumps(missed_table(v31.missed_list(ctx, prev)))
            + "\nTRAIN SOURCE DEFECTS per class by source number (mask_px, bbox_wh in px, contrast, darker or "
            "brighter than the background):\n" + json.dumps(source_table(srcs))))
    return content, obs, srcs


_chat = va.chat


def chat(endpoint, model, messages, max_tokens, timeout=2400.0):
    """va.chat, plus: bearer key from LENS_API_KEY for non-LAN endpoints, and per-model reasoning_effort."""
    import os
    import time
    import urllib.request
    api = next((v for k, v in (base_ctx.p.get("agent_api_models") or {}).items()
                if isinstance(v, dict) and v.get("model") == model), None)
    if api is None:
        return _chat(endpoint, model, messages, max_tokens, timeout)
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False}
    if api.get("reasoning_effort"):
        payload["reasoning_effort"] = api["reasoning_effort"]
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("LENS_API_KEY", "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    last = None
    for attempt in range(6):   # transient API errors (429/5xx, resets): wait and retry
        req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        t = time.time()
        try:
            with va._opener(endpoint).open(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"API HTTP {exc.code}: {exc.read().decode()[:300]}") from None
            last = exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
        time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"API unreachable after retries: {last!r}")
    ch = body["choices"][0]
    return {"content": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
            "served_model": body.get("model"), "usage": body.get("usage", {}),
            "reasoning_chars": len(ch["message"].get("reasoning_content") or ""),
            "elapsed_seconds": round(time.time() - t, 2)}


va.chat = chat
base_ctx = None


_clean = v31._clean_plan


def clean_plan(plan, srcs, bp, allocated, cap, r):
    """For agent_coder: strip augment_code, validate the rest as usual, then probe each operator."""
    if e.COND not in CODERS:
        return _clean(plan, srcs, bp, allocated, cap, r)
    from defectaug import agent_ops
    raw = plan.get("params")
    codes = {}
    if isinstance(raw, dict):
        raw = {c: dict(v) if isinstance(v, dict) else v for c, v in raw.items()}
        for c, v in raw.items():
            if isinstance(v, dict):
                code = v.pop("augment_code", None)
                if code:
                    codes[c] = str(code)
    alloc, params = _clean({**plan, "params": raw}, srcs, bp, allocated, cap, r)
    errors = []
    for c, code in codes.items():
        try:
            pr = agent_ops.probe(code, srcs[c], v31.MASKS, PROJECT / "data" / "canonical" / "labels" / "all",
                                 class_id=CLASSES.index(c))
        except agent_ops.OpError as exc:
            errors.append(f"{c} augment_code: {exc}")
            continue
        if pr["samples"] == 0 or pr["qc_pass"] < 0.6 * pr["samples"]:
            errors.append(f"{c} augment_code: only {pr['qc_pass']}/{pr['samples']} test pastes pass QC "
                          f"(mean defect contrast {pr['mean_contrast_before']} -> {pr['mean_contrast_after']}); "
                          "keep more of the defect visible")
            continue
        params[c]["augment_code"] = code
        PROBES[code] = pr
    if errors:
        raise ValueError("; ".join(errors))
    return alloc, params


PROBES: dict = {}
CLASSES = base.CLASSES
v31._clean_plan = clean_plan


v31.look_content = look_content
_exp_router = base.agent_proposal


def router(ctx, obs, r, bp, allocated, cap):
    """agent_memory -> the frozen experiment_proposal with the condition name and trial pools swapped."""
    global base_ctx
    base_ctx = ctx
    cond = v31._CTX["cond"]
    if cond not in TRIALS:
        e.COND, e._trial_cond = EXP, _exp_trial
        return _exp_router(ctx, obs, r, bp, allocated, cap)
    e.COND, e._trial_cond = cond, TRIALS[cond].format
    try:
        c = v31._CTX
        alloc, params, info = e.experiment_proposal(ctx, cond, c["seed"], r, bp, c["prev"], allocated, cap,
                                                    c["history"])
        info["operator_probes"] = {k[:80]: v for k, v in PROBES.items()}
    finally:
        e.COND, e._trial_cond = EXP, _exp_trial
    return (alloc or {}), params, {**info, "repaired": alloc is None}


base.agent_proposal = router
_gen = base.generate_round


def generate_round(ctx, cond, seed, r, *args):
    if cond in TRIALS:   # fallback rounds of the committed memory/coder pools
        return _with_v32_generator(v31._orig_generate_round, ctx, cond, seed, r, *args)
    return _gen(ctx, cond, seed, r, *args)


base.generate_round = generate_round

if __name__ == "__main__":
    p = base.yaml.safe_load(base.PROTOCOL_PATH.read_text(encoding="utf-8"))
    if p["experiment"]["status"] != "draft" or any(a.startswith("final") for a in sys.argv):
        raise SystemExit("v3.2 is a dev protocol: draft status and non-final tags only")
    raise SystemExit(base.main())
