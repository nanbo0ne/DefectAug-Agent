#!/usr/bin/env bash
# ONE-TIME final test-set evaluation for the six frozen conditions.
# Runs only after every validation-driven decision is complete and the
# protocol is frozen (experiment.status=frozen_final, gate enabled).
# Each call passes through the test-access gate: protocol sha256 + unlock
# token + audit log.  Never re-run to "pick" a result.
set -u

PROJECT="/d/AI Paper/DefectAug-Agent-GG38"
PY="$PROJECT/.venv/Scripts/python.exe"
cd "$PROJECT" || exit 1

PROTOCOL="$PROJECT/configs/experiment_v1.yaml"
PROTOCOL_SHA="acacef312fd24ba0983b3aab44d6bdd5f159c5a5ed21ee6b86829f4d28475a7f"
# Sanity: refuse to run if the frozen protocol file changed after hashing.
ACTUAL_SHA="$("$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$PROTOCOL")"
if [ "$ACTUAL_SHA" != "$PROTOCOL_SHA" ]; then
  echo "FATAL: protocol sha256 drifted: $ACTUAL_SHA != $PROTOCOL_SHA" >&2
  exit 1
fi
TOKEN="FINAL_TEST_EVALUATION"
SEEDS="20260820 20260821 20260822"
EVAL_ROOT="$PROJECT/reports/yolo_eval"

log() { echo "[$(date +%H:%M:%S)] $*"; }

eval_test() {
  local name="$1" weights="$2" data="$3" lock="$4" seed="$5"
  local out="$EVAL_ROOT/${name}_test/metrics.json"
  if [ -f "$out" ]; then
    log "SKIP  ${name}_test (metrics.json present)"
    return 0
  fi
  log "TEST  $name"
  "$PY" scripts/evaluate_yolo.py \
    --weights "$weights" --data "$data" --split test \
    --content-lock "$lock" --content-lock-mode required \
    --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
    --final-test-unlock-token "$TOKEN" \
    --imgsz 640 --batch 16 --device 0 --workers 4 --seed "$seed" \
    --conf 0.001 --iou 0.7 --max-det 300 \
    --output-root "$EVAL_ROOT" --name "${name}_test" --exist-ok || return 1
}

for s in $SEEDS; do
  eval_test "real_only_4090_s${s}_100e" \
    "$PROJECT/runs/yolo/real_only_4090_s${s}_100e/weights/best.pt" \
    "$PROJECT/configs/dataset_sd900_3class_v1.yaml" \
    "$PROJECT/data/manifests/dataset_content_lock_v1.json" "$s" || exit 1

  eval_test "real_only_repeat_matched_4090_s${s}_100e" \
    "$PROJECT/runs/yolo/real_only_repeat_matched_4090_s${s}_100e/weights/best.pt" \
    "$PROJECT/configs/dataset_sd900_3class_realmatch_v1.yaml" \
    "$PROJECT/data/manifests/dataset_content_lock_realmatch_v1.json" "$s" || exit 1

  eval_test "uniform_fixed_4090_s${s}_100e" \
    "$PROJECT/runs/yolo/uniform_fixed_4090_s${s}_100e/weights/best.pt" \
    "$PROJECT/configs/generated/dataset_uniform_fixed_${s}_r1.yaml" \
    "$PROJECT/data/manifests/dataset_content_lock_uniform_fixed_${s}_r1.json" "$s" || exit 1

  eval_test "formula_one_shot_4090_s${s}_100e" \
    "$PROJECT/runs/yolo/formula_one_shot_4090_s${s}_100e/weights/best.pt" \
    "$PROJECT/configs/generated/dataset_formula_one_shot_${s}_r1.yaml" \
    "$PROJECT/data/manifests/dataset_content_lock_formula_one_shot_${s}_r1.json" "$s" || exit 1

  eval_test "formula_multi_round_4090_s${s}_r3_100e" \
    "$PROJECT/runs/yolo/formula_multi_round_4090_s${s}_r3_100e/weights/best.pt" \
    "$PROJECT/configs/generated/dataset_formula_multi_round_${s}_r3.yaml" \
    "$PROJECT/data/manifests/dataset_content_lock_formula_multi_round_${s}_r3.json" "$s" || exit 1

  eval_test "agent_multi_round_4090_s${s}_r3_100e" \
    "$PROJECT/runs/yolo/agent_multi_round_4090_s${s}_r3_100e/weights/best.pt" \
    "$PROJECT/configs/generated/dataset_agent_multi_round_${s}_r3.yaml" \
    "$PROJECT/data/manifests/dataset_content_lock_agent_multi_round_${s}_r3.json" "$s" || exit 1
done

log "########## final test evaluation finished ##########"
