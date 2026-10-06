#!/usr/bin/env bash
# Re-run the two real-data baselines on this machine so that all six formal
# conditions share one GPU and one software stack.  The 5090 results stay in
# place as an independent reproducibility reference.
#
# Recovery semantics, in priority order per run:
#   1. metrics.json present            -> nothing to do, skip.
#   2. results.csv has all epochs      -> training already finished, evaluate only.
#   3. anything else present           -> incomplete, delete and retrain.
# Output is written straight to per-run logs.  Never pipe this script through
# tail: that buffers everything until exit and makes a live run look idle.
set -u

PROJECT="/d/AI Paper/DefectAug-Agent-GG38"
PY="$PROJECT/.venv/Scripts/python.exe"
cd "$PROJECT" || exit 1

SEEDS="20260820 20260821 20260822"
TAG="4090"
EPOCHS=100
EVAL_ROOT="$PROJECT/reports/yolo_eval"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_condition() {
  local cond="$1" data="$2" lock="$3"
  for seed in $SEEDS; do
    local name="${cond}_${TAG}_s${seed}_${EPOCHS}e"
    local rundir="$PROJECT/runs/yolo/$name"
    local results="$rundir/results.csv"
    local metrics="$EVAL_ROOT/${name}_val/metrics.json"
    local need_train=1

    if [ -f "$metrics" ]; then
      log "SKIP  $name  (metrics.json already present)"
      continue
    fi

    if [ -f "$results" ]; then
      local lines
      lines=$(wc -l < "$results" | tr -d '[:space:]')
      if [ "$lines" -ge $((EPOCHS + 1)) ]; then
        log "REUSE $name  (results.csv has $lines lines = $EPOCHS epochs; evaluating only)"
        need_train=0
      else
        log "REDO  $name  (results.csv has only $lines lines; deleting and retraining)"
        rm -rf "$rundir" "${rundir}_launch.json"
      fi
    elif [ -e "$rundir" ]; then
      log "REDO  $name  (run dir without results.csv; deleting)"
      rm -rf "$rundir" "${rundir}_launch.json"
    fi

    if [ "$need_train" -eq 1 ]; then
      log "TRAIN $name"
      "$PY" scripts/train_yolo.py \
        --data "$data" --model yolo11n.pt \
        --epochs "$EPOCHS" --imgsz 640 --batch 16 --device 0 --workers 4 \
        --seed "$seed" --name "$name" \
        --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 \
        --weight-decay 0.0005 --warmup-epochs 3.0 --amp \
        --content-lock "$lock" --content-lock-mode required \
        > "$EVAL_ROOT/${name}_train.log" 2>&1
      if [ $? -ne 0 ]; then
        log "FAIL  $name training; see ${name}_train.log"
        continue
      fi
    fi

    log "EVAL  $name"
    "$PY" scripts/evaluate_yolo.py \
      --weights "runs/yolo/$name/weights/best.pt" \
      --data "$data" --split val \
      --content-lock "$lock" --content-lock-mode required \
      --imgsz 640 --batch 16 --device 0 --workers 4 --seed "$seed" \
      --conf 0.001 --iou 0.7 --max-det 300 \
      --output-root "reports/yolo_eval" --name "${name}_val" --exist-ok \
      > "$EVAL_ROOT/${name}_eval.log" 2>&1
    if [ $? -ne 0 ]; then
      log "FAIL  $name evaluation; see ${name}_eval.log"
      continue
    fi
    log "DONE  $name"
  done
}

mkdir -p "$EVAL_ROOT"
log "########## baseline re-run on $TAG started ##########"
run_condition "real_only" \
  "configs/dataset_sd900_3class_v1.yaml" \
  "data/manifests/dataset_content_lock_v1.json"
run_condition "real_only_repeat_matched" \
  "configs/dataset_sd900_3class_realmatch_v1.yaml" \
  "data/manifests/dataset_content_lock_realmatch_v1.json"
log "########## baseline re-run finished ##########"
