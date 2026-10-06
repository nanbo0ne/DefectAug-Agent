"""v3.1 vision agent: a multi-turn tool loop per round.

Per round the agent
  1. LOOKS at the val instances the current detector misses (montage per class)
     and at every train source defect of that class (numbered montage);
  2. PROPOSES quota + generation params + a per-class source whitelist (the
     source defects that look like the misses);
  3. PREVIEWS: the orchestrator generates a few samples with that plan in a
     scratch pool and shows them back; the agent may REVISE once (repairing
     unrealistic pastes, wrong scale, bad backgrounds) before COMMITTING.
The formula baselines cannot do 1-3: they only see aggregate numbers.

The LLM endpoint is OpenAI-compatible (llama.cpp, Qwen3.8-27B + mmproj).
Reasoning effort: the server ignores reasoning_effort, so depth comes from a
large max_tokens budget and an explicit analyse->plan->check instruction.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
import urllib.parse
import urllib.request

CLASSES = ("inclusion", "patches", "scratches")

SYSTEM = (
    "You are the augmentation-planning agent in a closed-loop few-shot steel-defect detection "
    "experiment (YOLO11n; 100 real train images: inclusion 10, patches 30, scratches 60). Each round "
    "you plan synthetic Copy-Paste images: real train defects (with masks) are cut out, rescaled, "
    "rotated/flipped and pasted onto train backgrounds with a feathered boundary; labels come from "
    "the pasted mask. After the round the detector is retrained from scratch on real+all synthetic "
    "images and re-evaluated on a locked validation split, and you see the result.\n"
    "You have VISION. You will be shown (a) crops of the validation defects the detector currently "
    "MISSES, (b) every train source defect you can copy, numbered, and later (c) a preview of the "
    "synthetic images your plan produces. Use the images: match the appearance (shape, thickness, "
    "darkness/contrast, texture, size) of what is missed to the source defects that look like it, "
    "and check the preview for unrealistic or useless pastes.\n"
    "Think step by step and thoroughly before answering: (1) describe what the misses look like per "
    "class, (2) relate it to the numbers and to what happened in previous rounds, (3) decide the plan, "
    "(4) check every constraint. Then output ONE JSON object and nothing after it.\n"
    "Strategy facts you should weigh: every class shares one detector, so starving a class can lower "
    "it; giving one class far more than ~80 synthetic images has shown diminishing returns; very small "
    "scales make tiny pastes that may be unlearnable. Change what did not work last round."
)

PLAN_SCHEMA = (
    '{"analysis": "short", "allocation": {"inclusion": int, "patches": int, "scratches": int},\n'
    ' "params": {"<class>": {"scale_min": float, "scale_max": float, "target_pool": str,\n'
    '            "source_weighting": str, "max_overlap": float,\n'
    '            "sources": [int source numbers you want to copy from, or [] for all]}, ... all 3},\n'
    ' "rationale": "short"}'
)

# ------------------------------------------------------------------ experimenting agent
# The agent designs K candidate plans per round as hypotheses; the lab
# generates and trains each one (same seed, full epochs) and reports val
# results; the agent keeps one and writes a lesson into its lab notebook.
SYSTEM_EXPERIMENT = (
    "You are the augmentation-scientist agent in a closed-loop few-shot steel-defect detection "
    "experiment (YOLO11n; 100 real train images: inclusion 10, patches 30, scratches 60). Synthetic "
    "Copy-Paste images are made by cutting real train defects out with their masks, rescaling, "
    "rotating/flipping and pasting them onto train backgrounds with a feathered boundary; labels come "
    "from the pasted mask.\n"
    "You have VISION and a LAB. Each round you DESIGN {k} candidate plans, each testing a different "
    "hypothesis. The lab generates every candidate's synthetic images and trains the detector on real + "
    "all previously kept synthetic images + that candidate (full training, same seed, so the candidates "
    "are directly comparable), then shows you the validation results. You KEEP one candidate (its images "
    "stay in the training set for all later rounds) and write a LESSON into your lab notebook, which you "
    "will see in later rounds. The lab also adds a CONTROL candidate (round 1: the default recipe with a "
    "balanced allocation; later rounds: your last kept plan repeated), so every round has a baseline to "
    "beat and you may keep the control.\n"
    "The final round matters most: its kept model is the one that is tested. Plan across rounds -- e.g. "
    "do not use up a class's cumulative cap or starve a class early -- and in the final round only test "
    "variations you have evidence for; untested extreme allocations are risky.\n"
    "Objective: the highest final-round mAP50 on a held-out test set; validation is your proxy. "
    "KEEP the candidate with the highest validation mAP50; only when candidates are within 0.005 mAP50 "
    "of the best, prefer the one with the higher worst-class AP50. Single-run validation differences up "
    "to ~0.03 can be noise, so draw lessons only from large or repeated effects.\n"
    "Be a scientist: do NOT assume intuitive rules (e.g. 'generate more of what is missed', 'small "
    "misses need small pastes') are true here -- test them. Make the candidates differ clearly in one or "
    "two factors (allocation balance, scale range, source selection, background pool, overlap) so each "
    "result teaches you something. Use the images: match what the missed defects look like to the "
    "source defects. Use the notebook: build on what measurably worked, drop what did not, and keep one "
    "candidate exploring an alternative you are unsure about.\n"
    "Think step by step and thoroughly before answering, check every constraint, then output ONE JSON "
    "object and nothing after it."
)

CANDIDATE_PARAMS = (
    '{"<class>": {"scale_min": float, "scale_max": float, "target_pool": str, "source_weighting": str,\n'
    '             "max_overlap": float, "sources": [int source numbers, or [] for all]}, ... all 3}'
)

DESIGN_SCHEMA = (
    '{"analysis": "short",\n'
    ' "candidates": [   (exactly {k} different candidates)\n'
    '   {"hypothesis": "what this candidate tests and what result would confirm it",\n'
    '    "allocation": {"inclusion": int, "patches": int, "scratches": int},\n'
    '    "params": ' + CANDIDATE_PARAMS + '}, ...]}'
)

CHOOSE_SCHEMA = (
    '{"analysis": "short: compare the results with your hypotheses",\n'
    ' "choice": int candidate number to keep,\n'
    ' "lesson": "1-3 sentences for your notebook: what this round showed, what to try next"}'
)


def _opener(endpoint: str):
    host = urllib.parse.urlsplit(endpoint).hostname or ""
    try:
        private = ipaddress.ip_address(host).is_private
    except ValueError:
        private = host == "localhost"
    return urllib.request.build_opener(urllib.request.ProxyHandler({})) if private else urllib.request.build_opener()


def chat(endpoint: str, model: str, messages: list, max_tokens: int, timeout: float = 2400.0) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens,
               "stream": False, "reasoning_effort": "xhigh"}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.time()
    with _opener(endpoint).open(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    ch = body["choices"][0]
    return {"content": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
            "served_model": body.get("model"), "usage": body.get("usage", {}),
            "reasoning_chars": len(ch["message"].get("reasoning_content") or ""),
            "elapsed_seconds": round(time.time() - t, 2)}


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in reply")
    return json.loads(cleaned[start:end + 1])


def image_part(data_url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}
