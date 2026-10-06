#!/usr/bin/env bash
# v3 unattended pipeline: tuning (4 grid points, tuning seed) -> select+freeze -> final 5 seeds.
# Test is NOT evaluated here.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
EP=http://192.168.31.47:18182/v1
W=("0.5,0.3,0.2" "0.8,0.1,0.1" "0.3,0.3,0.4" "0.2,0.7,0.1")
for i in 0 1 2 3; do
  echo "=== TUNE g$i weights ${W[$i]} $(date)"
  $PY scripts/run_conditions_v3.py --tag tune_g$i --seeds tuning \
      --conditions real_only,formula_tuned_multi_round --formula-weights "${W[$i]}"
done
echo "=== FREEZE $(date)"
$PY scripts/freeze_v3.py
echo "=== FINAL $(date)"
$PY scripts/run_conditions_v3.py --tag final --seeds all --agent-endpoint $EP \
    --conditions real_only,real_only_repeat_matched,uniform_fixed,formula_one_shot,formula_multi_round,formula_tuned_multi_round,agent_multi_round
echo "=== PIPELINE DONE $(date)"
