"""ONE-TIME v3.1 final test evaluation (tag final31, run on the execution host 192.168.31.102).

Models: every final_run_condition x 5 final seeds = 35, the final-round model
(r3 for multi-round conditions) -- exactly the weights, data yaml and content
lock recorded in that model's val metrics.json (for agent_experiment that is
the kept trial's training).  Ablations are not included.

Step 1 (freeze): scripts/check_freeze_v31.py must pass; every weights / data /
lock hash must equal the val record; the snapshot is written to
reports/v3/final31/test_freeze_v31.json (or must match it if it exists).
Step 2 (test): each model passes the test-access gate in evaluate_yolo.py
(protocol sha256 + unlock token + audit log reports/test_access_audit_v31.jsonl),
with the same evaluation parameters as its val run.  Existing test metrics are
skipped, never re-run.
Step 3 (--report, any machine): paired statistics over the 5 seeds.
--ablations: the same procedure for the protocol's ablation_conditions (run after the
primary test, never used for any decision), with its own freeze file
test_freeze_v31_ablations.json; --report then also compares them with the agent.

Usage:
    python scripts/final_test_v31.py --freeze-only
    python scripts/final_test_v31.py
    python scripts/final_test_v31.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PY = PROJECT / ".venv" / "Scripts" / "python.exe"
PROTOCOL = PROJECT / "configs" / "experiment_v31.yaml"
FINAL = PROJECT / "reports" / "v3" / "final31"
EVAL_ROOT = FINAL / "yolo_eval"
FREEZE = FINAL / "test_freeze_v31.json"
FREEZE_ABL = FINAL / "test_freeze_v31_ablations.json"
CONDS_KEY = "final_run_conditions"
TAG = "final31"
AGENT = "agent_experiment_multi_round"


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT).as_posix()


def run_name(p: dict, cond: str, seed: int) -> str:
    rounds = int({c["name"]: c for c in p["conditions"]}[cond].get("rounds", 1))
    return f"v3_{TAG}_{cond}_s{seed}" + (f"_r{rounds}" if rounds > 1 else "")


def snapshot(p: dict) -> dict:
    models = []
    for seed in p["experiment"]["final_seeds"]:
        for cond in p["experiment"][CONDS_KEY]:
            run = run_name(p, cond, seed)
            val_path = EVAL_ROOT / f"{run}_val" / "metrics.json"
            if not val_path.is_file():
                raise SystemExit(f"FATAL: missing val metrics for {run}")
            inp = json.loads(val_path.read_text(encoding="utf-8"))["inputs"]
            rec = {"condition": cond, "seed": seed, "run": run, "val_metrics": rel(val_path)}
            for key, field in (("weights", "weights"), ("data_yaml", "data_yaml"), ("content_lock", "content_lock")):
                path = Path(inp[field]["path"])
                if not path.is_file():
                    raise SystemExit(f"FATAL: missing {key} for {run}: {path}")
                h = sha256(path)
                if h != inp[field]["sha256"]:
                    raise SystemExit(f"FATAL: {run} {key} changed since val evaluation")
                rec[key], rec[f"{key}_sha256"] = rel(path), h
            models.append(rec)
    return {"protocol": rel(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
            "experiment_status": p["experiment"]["status"],
            "test_split_sha256": p["dataset"]["split_sha256"]["test"], "models": models}


def freeze(p: dict) -> dict:
    if p["experiment"]["status"] != "frozen_final":
        raise SystemExit("FATAL: protocol status is not frozen_final")
    out = subprocess.run([str(PY), "scripts/check_freeze_v31.py"], cwd=PROJECT, capture_output=True, text=True)
    if out.returncode != 0 or "FREEZE OK" not in out.stdout:
        raise SystemExit(f"FATAL: check_freeze_v31 failed:\n{out.stdout}\n{out.stderr}")
    snap = snapshot(p)
    if FREEZE.is_file():
        old = json.loads(FREEZE.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "frozen_at"} != snap:
            raise SystemExit("FATAL: current artifacts differ from test_freeze_v31.json")
        log(f"freeze verified: protocol {snap['protocol_sha256'][:12]}, {len(snap['models'])} models")
        return old
    snap["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    FREEZE.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"freeze written: protocol {snap['protocol_sha256'][:12]}, {len(snap['models'])} models")
    return snap


def run_test(p: dict, frozen: dict) -> None:
    gate = p["final_test_access"]
    for m in frozen["models"]:
        name = f"{m['run']}_test"
        if (EVAL_ROOT / name / "metrics.json").is_file():
            log(f"SKIP {name} (metrics present)")
            continue
        vp = json.loads((PROJECT / m["val_metrics"]).read_text(encoding="utf-8"))["parameters"]
        log(f"TEST {name}")
        cmd = [PY, "scripts/evaluate_yolo.py", "--weights", m["weights"], "--data", m["data_yaml"],
               "--split", "test", "--content-lock", m["content_lock"], "--content-lock-mode", "required",
               "--expected-weights-sha256", m["weights_sha256"],
               "--expected-data-sha256", m["data_yaml_sha256"],
               "--expected-split-sha256", frozen["test_split_sha256"],
               "--protocol", frozen["protocol"], "--expected-protocol-sha256", frozen["protocol_sha256"],
               "--final-test-unlock-token", gate["unlock_token"], "--test-audit-log", gate["audit_log"],
               "--imgsz", vp["imgsz"], "--batch", vp["batch"], "--device", vp["device"],
               "--workers", vp["workers"], "--seed", vp["seed"], "--conf", vp["conf"], "--iou", vp["iou"],
               "--max-det", vp["max_det"], "--output-root", rel(EVAL_ROOT), "--name", name, "--exist-ok"]
        logf = EVAL_ROOT / f"{name}.log"
        with logf.open("w", encoding="utf-8") as s:
            rc = subprocess.call([str(c) for c in cmd], cwd=PROJECT, stdout=s, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f"FATAL: test evaluation failed for {name} (see {logf.name})")
    log("final test evaluation finished")


# ---------------------------------------------------------------- report

def endpoints(doc: dict) -> dict:
    m = doc["metrics"]
    pc = {c["class_name"]: c for c in m["per_class"]}
    return {"map50": m["overall"]["ap50"], "map50_95": m["overall"]["ap50_95"],
            "worst_class_ap50": m["worst_class"]["ap50"],
            "inclusion_ap50": pc["inclusion"]["ap50"], "inclusion_recall": pc["inclusion"]["recall"],
            "interclass_ap50_std": m["class_metric_std"]["ap50"],
            **{f"ap50_{c}": pc[c]["ap50"] for c in pc}}


def paired(a: list[float], b: list[float], lower_is_better: bool = False) -> dict:
    from scipy import stats
    d = [(y - x) if lower_is_better else (x - y) for x, y in zip(a, b)]   # positive = agent better
    t = stats.ttest_1samp(d, 0.0)
    wins = sum(x > 0 for x in d)
    return {"mean_diff": statistics.mean(d), "sd_diff": statistics.stdev(d), "wins": wins, "n": len(d),
            "t": float(t.statistic), "p_two_sided": float(t.pvalue),
            "p_one_sided": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2),
            "sign_test_p_one_sided": float(stats.binomtest(wins, len(d), 0.5, alternative="greater").pvalue)}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = running
    return out


def report(p: dict) -> int:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    seeds, conds = p["experiment"]["final_seeds"], p["experiment"]["final_run_conditions"]
    res = {c: {} for c in conds}
    for m in frozen["models"]:
        doc = json.loads((EVAL_ROOT / f"{m['run']}_test" / "metrics.json").read_text(encoding="utf-8"))
        if doc["inputs"]["weights"]["sha256"] != m["weights_sha256"] or doc["evaluation"]["split"] != "test":
            raise SystemExit(f"FATAL: {m['run']} test record does not match the freeze")
        res[m["condition"]][m["seed"]] = endpoints(doc)
    col = lambda c, k: [res[c][s][k] for s in seeds]  # noqa: E731
    out = {"per_condition": {c: {"per_seed": {str(s): res[c][s] for s in seeds},
                                 "mean": {k: statistics.mean(col(c, k)) for k in res[c][seeds[0]]}} for c in conds}}
    out["primary_vs_formula_multi_round_map50"] = paired(col(AGENT, "map50"), col("formula_multi_round", "map50"))
    key = {f"vs_{b}": paired(col(AGENT, "map50"), col(b, "map50"))
           for b in ("formula_tuned_multi_round", "uniform_fixed")}
    for k, v in holm({k: v["p_one_sided"] for k, v in key.items()}).items():
        key[k]["p_holm_one_sided"] = v
    out["key_secondary_map50"] = key
    sec = {}
    for b in ("formula_multi_round", "formula_tuned_multi_round"):
        fam = {e: paired(col(AGENT, e), col(b, e), lower_is_better=(e == "interclass_ap50_std"))
               for e in ("worst_class_ap50", "inclusion_ap50", "inclusion_recall", "interclass_ap50_std")}
        for k, v in holm({k: v["p_one_sided"] for k, v in fam.items()}).items():
            fam[k]["p_holm_one_sided"] = v
        sec[f"vs_{b}"] = fam
    out["secondary"] = sec
    out["all_conditions_vs_agent_map50"] = {c: paired(col(AGENT, "map50"), col(c, "map50"))
                                           for c in conds if c != AGENT}
    if FREEZE_ABL.is_file():
        abl = json.loads(FREEZE_ABL.read_text(encoding="utf-8"))
        ares = {}
        for m in abl["models"]:
            doc = json.loads((EVAL_ROOT / f"{m['run']}_test" / "metrics.json").read_text(encoding="utf-8"))
            if doc["inputs"]["weights"]["sha256"] != m["weights_sha256"] or doc["evaluation"]["split"] != "test":
                raise SystemExit(f"FATAL: {m['run']} test record does not match the ablation freeze")
            ares.setdefault(m["condition"], {})[m["seed"]] = endpoints(doc)
        out["ablations"] = {}
        for c, per in ares.items():
            a = [per[s]["map50"] for s in seeds]
            out["ablations"][c] = {"per_seed": {str(s): per[s] for s in seeds},
                                   "mean": {k: statistics.mean(per[s][k] for s in seeds) for k in per[seeds[0]]},
                                   "agent_minus_ablation_map50": paired(col(AGENT, "map50"), a),
                                   "ablation_minus_formula_multi_map50": paired(a, col("formula_multi_round", "map50")),
                                   "agent_minus_ablation_inclusion_ap50": paired(col(AGENT, "inclusion_ap50"),
                                                                                 [per[s]["inclusion_ap50"] for s in seeds])}
    dest = FINAL / "test_results_v31.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'condition':30s} " + " ".join(f"{s:>10d}" for s in seeds) + "      mean")
    for c in conds:
        print(f"{c:30s} " + " ".join(f"{x:10.4f}" for x in col(c, "map50")) + f"  {statistics.mean(col(c, 'map50')):.4f}")
    pr = out["primary_vs_formula_multi_round_map50"]
    print(f"\nH1 agent - formula_multi: {pr['mean_diff']:+.4f} (sd {pr['sd_diff']:.4f}) wins {pr['wins']}/5 "
          f"t={pr['t']:.2f} p1={pr['p_one_sided']:.4f} sign p1={pr['sign_test_p_one_sided']:.4f}")
    for k, v in key.items():
        print(f"{k}: {v['mean_diff']:+.4f} wins {v['wins']}/5 p1={v['p_one_sided']:.4f} holm={v['p_holm_one_sided']:.4f}")
    for c, v in out.get("ablations", {}).items():
        print(f"ablation {c}: mAP50 {v['mean']['map50']:.4f}  agent-minus {v['agent_minus_ablation_map50']['mean_diff']:+.4f} "
              f"wins {v['agent_minus_ablation_map50']['wins']}/5 p1={v['agent_minus_ablation_map50']['p_one_sided']:.4f}; "
              f"vs formula_multi {v['ablation_minus_formula_multi_map50']['mean_diff']:+.4f}")
    print(f"written {rel(dest)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    args = ap.parse_args()
    global FREEZE, CONDS_KEY
    if args.ablations and not args.report:
        FREEZE, CONDS_KEY = FREEZE_ABL, "ablation_conditions"
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if args.report:
        return report(p)
    frozen = freeze(p)
    if not args.freeze_only:
        run_test(p, frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
