"""LLM agent controller for the agent_multi_round condition.

Reads the observation (previous round's validation metrics + generation
ledger state), asks a local LM Studio model for a JSON allocation plan, and
validates it against the frozen protocol constraints:

- every class allocation is an integer >= 0 and the round sum is exactly
  the round budget (50);
- cumulative per-class allocation stays <= min(0.70 * total_budget,
  remaining source capacity) and >= final_minimum_per_class by the last
  round;
- the only whitelisted tool is patch_copy_gauss5, so the plan is a pure
  per-class image quota (no free-form actions, no code).

Invalid plans are retried (max_retries_per_seed = 2 -> 3 attempts).  Every
raw prompt/response is logged to reports/agent_decisions/*.jsonl.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
REAL_IMAGES = {"inclusion": 10, "patches": 30, "scratches": 60}
SOURCES_PER_CLASS = {"inclusion": 10, "patches": 30, "scratches": 60}
MAX_ACCEPTED_PER_SOURCE = 8
TOTAL_BUDGET = 150
ROUND_BUDGET = 50
TOTAL_ROUNDS = 3
MAX_FRACTION = 0.70
FINAL_MIN_PER_CLASS = 10
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You are the augmentation-planning agent in a closed-loop few-shot object "
    "detection experiment. A YOLO11n detector is trained on 100 real steel-"
    "surface defect images (inclusion:10, patches:30, scratches:60) plus "
    "synthetic Copy-Paste samples you allocate. After each round the detector "
    "is retrained and evaluated on a locked validation split; the per-class "
    "metrics are fed back to you. Your ONLY action each round is to decide "
    "how many new synthetic images of each class to generate this round "
    "(tool: patch_copy_gauss5, deterministic real-defect Copy-Paste with "
    "boundary feathering). Answer with a single JSON object and nothing else."
)


def build_user_prompt(observation: dict, feedback: str | None = None) -> str:
    cap_total = int(MAX_FRACTION * TOTAL_BUDGET)
    headroom = {
        name: min(cap_total, int(observation["per_class"][name]["remaining_capacity"]))
        - int(observation["per_class"][name]["accepted_so_far"])
        for name in CLASSES
    }
    lines = [
        "OBSERVATION (JSON):",
        json.dumps(observation, ensure_ascii=False, indent=2),
        "",
        "CONSTRAINTS:",
        f"- Return exactly {ROUND_BUDGET} new images this round (sum of the three classes).",
        f"- Cumulative per-class total after this round must be <= {cap_total} AND <= accepted_so_far + remaining_capacity.",
        f"- Remaining headroom this round (how much each class may still receive): {json.dumps(headroom)}.",
        "- Never allocate more to a class than its headroom.",
    ]
    if observation["round_index"] == TOTAL_ROUNDS:
        lines.append(
            f"- This is the LAST round: cumulative per-class total must be >= {FINAL_MIN_PER_CLASS}."
        )
    lines += [
        "- Respond with one JSON object only, schema:",
        '  {"allocation": {"inclusion": int, "patches": int, "scratches": int}, "rationale": "short string"}',
    ]
    if feedback:
        lines += [
            "",
            f"PREVIOUS ATTEMPT REJECTED: {feedback}",
            "Return a corrected plan that satisfies every constraint.",
        ]
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    # Reasoning models emit <think>...</think> blocks that may contain braces;
    # strip them before locating the JSON plan.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])


def validate_plan(plan: dict, accepted_so_far: dict[str, int], capacities: dict[str, int], round_index: int) -> dict[str, int]:
    if not isinstance(plan, dict):
        raise ValueError("plan is not an object")
    alloc = plan.get("allocation")
    if not isinstance(alloc, dict):
        raise ValueError("missing allocation object")
    result: dict[str, int] = {}
    for name in CLASSES:
        value = alloc.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"allocation for {name} is not an integer: {value!r}")
        if value < 0:
            raise ValueError(f"negative allocation for {name}")
        result[name] = value
    if sum(result.values()) != ROUND_BUDGET:
        raise ValueError(f"allocation sums to {sum(result.values())}, expected {ROUND_BUDGET}")
    cap_total = int(MAX_FRACTION * TOTAL_BUDGET)
    for name in CLASSES:
        cumulative = accepted_so_far[name] + result[name]
        if cumulative > min(cap_total, capacities[name]):
            raise ValueError(
                f"cumulative allocation for {name} = {cumulative} exceeds cap "
                f"{min(cap_total, capacities[name])}"
            )
    if round_index == TOTAL_ROUNDS:
        for name in CLASSES:
            if accepted_so_far[name] + result[name] < FINAL_MIN_PER_CLASS:
                raise ValueError(
                    f"final cumulative allocation for {name} below {FINAL_MIN_PER_CLASS}"
                )
    return result


def repair_plan(plan: dict, accepted_so_far: dict[str, int], capacities: dict[str, int], round_index: int) -> dict[str, int]:
    """Deterministic projection of an invalid plan onto the feasible set.

    The agent proposes; the system enforces. If every attempt violates the
    hard constraints, clip each class to its remaining headroom, satisfy the
    last-round minimums, then redistribute the leftover round budget to the
    classes with the most headroom (deterministic order). The repaired plan
    is logged alongside the raw agent output.
    """
    cap_total = int(MAX_FRACTION * TOTAL_BUDGET)
    headroom = {
        n: max(0, min(cap_total, capacities[n]) - accepted_so_far[n]) for n in CLASSES
    }
    raw = plan.get("allocation") if isinstance(plan, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    alloc: dict[str, int] = {}
    for n in CLASSES:
        v = raw.get(n, 0)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            v = 0
        alloc[n] = min(v, headroom[n])
    if round_index == TOTAL_ROUNDS:
        for n in CLASSES:
            need = FINAL_MIN_PER_CLASS - (accepted_so_far[n] + alloc[n])
            if need > 0:
                alloc[n] = min(alloc[n] + need, headroom[n])
    deficit = ROUND_BUDGET - sum(alloc.values())
    order = sorted(CLASSES, key=lambda n: -(headroom[n] - alloc[n]))
    i = 0
    while deficit > 0 and i < 10000:
        n = order[i % len(order)]
        if alloc[n] < headroom[n]:
            alloc[n] += 1
            deficit -= 1
        i += 1
    if sum(alloc.values()) != ROUND_BUDGET:
        raise ValueError("repair failed: insufficient total headroom for round budget")
    return alloc


def build_user_prompt_v2(observation: dict, protocol, feedback: str | None = None) -> str:
    """v2 prompt: every number comes from the shared BudgetProtocol.

    `observation["headroom_this_round"]` is computed by defectaug.allocation,
    the same function that validates the reply, so the prompt and the
    validator cannot disagree.
    """
    r = observation["round_index"]
    lines = [
        "OBSERVATION (JSON):",
        json.dumps(observation, ensure_ascii=False, indent=2),
        "",
        "CONSTRAINTS (checked by code; a violating plan is rejected):",
        f"- This is round {r} of {protocol.rounds}. Allocate exactly {protocol.round_budget} new images.",
        f"- No class may exceed its headroom_this_round: {json.dumps(observation['headroom_this_round'])}.",
        f"- Cumulative per-class total can never exceed {protocol.cumulative_cap}.",
        f"- After the last round every class must have >= {protocol.final_min_per_class} in total.",
        "- quality_receipts shows, per class, candidates tried, accepted, and why candidates were rejected.",
        "- Respond with one JSON object only, schema:",
        '  {"allocation": {"inclusion": int, "patches": int, "scratches": int}, "rationale": "short string"}',
    ]
    if feedback:
        lines += ["", f"PREVIOUS ATTEMPT REJECTED: {feedback}", "Return a corrected plan."]
    return "\n".join(lines)


def call_llm(endpoint: str, model: str, system: str, user: str, timeout: float,
             temperature: float = 0.2, max_tokens: int = 4096) -> dict:
    # Reasoning models (Qwen3.8) spend completion tokens on reasoning_content
    # before the answer, so the budget must cover both.  The API key is read
    # from LENS_API_KEY and never written to logs.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # Cloudflare in front of the gateway rejects urllib's default User-Agent
    # (error 1010), so send an explicit one.
    headers = {"Content-Type": "application/json", "User-Agent": "defectaug-agent/2.0"}
    api_key = os.environ.get("LENS_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    # LAN endpoints must bypass the system proxy (Clash on 127.0.0.1:7897
    # answers 502 for private addresses that NO_PROXY does not list).
    host = urllib.parse.urlsplit(endpoint).hostname or ""
    try:
        private = ipaddress.ip_address(host).is_private
    except ValueError:
        private = host == "localhost"
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if private else urllib.request.build_opener())
    started = time.time()
    with opener.open(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    choice = body["choices"][0]
    message = choice["message"].get("content") or ""
    usage = body.get("usage", {})
    return {
        "content": message,
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "served_model": body.get("model"),
        "reasoning_chars": len(choice["message"].get("reasoning_content") or ""),
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", required=True, help="Path to observation JSON.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="qwen2.5-14b-instruct")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--log-dir", default="reports/agent_decisions")
    args = parser.parse_args()

    observation = json.loads(Path(args.observation).read_text(encoding="utf-8"))
    accepted_so_far = {name: int(observation["per_class"][name]["accepted_so_far"]) for name in CLASSES}
    capacities = {name: int(observation["per_class"][name]["remaining_capacity"]) for name in CLASSES}

    log_dir = PROJECT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent_multi_round_{args.seed}_r{args.round}.jsonl"

    last_error: str | None = None
    last_plan: dict = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        record = {"seed": args.seed, "round": args.round, "attempt": attempt, "model": args.model}
        prompt = build_user_prompt(observation, feedback=last_error)
        try:
            response = call_llm(args.endpoint, args.model, SYSTEM_PROMPT, prompt, args.timeout)
        except Exception as exc:  # network/transport failure
            last_error = f"llm call failed: {exc}"
            record["error"] = last_error
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue
        record["response"] = response
        record["prompt_chars"] = len(prompt)
        try:
            plan = extract_json(response["content"])
            last_plan = plan
            allocation = validate_plan(plan, accepted_so_far, capacities, args.round)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"invalid plan: {exc}"
            record["error"] = last_error
            record["raw_plan"] = response["content"][:2000]
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue
        record["plan"] = plan
        record["allocation"] = allocation
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(",".join(f"{name}:{allocation[name]}" for name in CLASSES))
        return 0

    # Constraint enforcement fallback: project the last proposed plan onto the
    # feasible set.  The episode is logged as repaired so the audit trail shows
    # the agent's raw (invalid) proposal and the enforced allocation.
    record = {
        "seed": args.seed, "round": args.round, "attempt": "repair",
        "model": args.model, "last_error": last_error,
    }
    try:
        allocation = repair_plan(last_plan, accepted_so_far, capacities, args.round)
    except ValueError as exc:
        record["error"] = f"repair failed: {exc}"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        raise RuntimeError(f"agent produced no valid plan and repair failed: {exc}")
    record["repaired"] = True
    record["raw_plan"] = last_plan
    record["allocation"] = allocation
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(",".join(f"{name}:{allocation[name]}" for name in CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
