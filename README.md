# Detector-in-the-Loop Agentic Data Augmentation for Few-Shot Object Detection

Code, frozen protocol and experiment records for the paper *Detector-in-the-Loop Agentic Data
Augmentation for Few-Shot Object Detection* (W. Shi).

An LLM agent decides how to use a Copy-Paste generator, choosing per-class quotas, paste scale,
background pool, source weighting, overlap and a source whitelist, for a few-shot detector
(YOLO11n). In every round the agent designs three candidate plans, each with a hypothesis. A lab trains
and validates a detector for every candidate plus a control arm. The agent keeps one candidate by a
pre-registered rule and writes a lesson into a lab notebook for the next round.

## Main result (test set, 5 paired seeds, evaluated once after freezing)

| Condition | mAP50 | Inclusion AP50 |
|---|---:|---:|
| Real-only | 0.461 | 0.150 |
| Uniform-fixed | 0.573 | 0.306 |
| Formula-multi-round | 0.575 | 0.323 |
| Formula-tuned | 0.561 | 0.274 |
| Agent-plan (same LLM, no lab) | 0.572 | 0.322 |
| Agent-vision-plan (same LLM, no lab) | 0.584 | 0.348 |
| **Agent-experiment** | **0.602** | **0.405** |

The full statistics are in `reports/v3/final31/test_results_v31.json`.

## Repository layout

```text
src/defectaug/       action space (gen_params.py), agent prompts (vision_agent.py), data locks, test gate
scripts/             generation, training, evaluation and orchestration
  run_conditions_v31e.py      the experimenting agent (frozen v3.1)
  run_conditions_v31.py       the planning agents (ablations)
  run_synth_generation_v31.py Copy-Paste generator patch_copy_gauss5 + quality gate
  evaluate_yolo.py            evaluation with the test-access gate
  final_test_v31.py           one-time test evaluation and paired statistics
  freeze_v31.py, check_freeze_v31.py   protocol / code hash freeze
configs/experiment_v31.yaml  frozen protocol: hypotheses, seeds, budget, conditions
data/splits/v3/              train / val / test image lists (file names only)
data/manifests/              content locks (SHA-256 of every image, label and mask used)
reports/v3/protocol_freeze_v31.json   hashes of the protocol and 27 code/data files at freeze time
reports/v3/final31/decisions/         every agent and formula decision: prompts' outputs, candidates,
                                      choices, lab notebooks, LLM usage
reports/v3/final31/yolo_eval/         validation and test metrics.json for every trained model
reports/test_access_audit_v31.jsonl   audit log of every test-set access (45 entries)
```

`scripts/run_conditions_v32.py` and `configs/experiment_v32.yaml` hold the exploratory variants
(memory, code-writing operators, other LLMs). They run on development seeds and validation data only;
the paper reports them as a supplementary experiment.

## Data

The images are not redistributed here. Images and boxes come from **NEU-DET** (K. Song, Y. Yan, 2013),
and pixel masks come from **SD-saliency-900** (G. Song, K. Song, Y. Yan, 2020;
https://github.com/SongGuorong/MCITF). Obtain both from their authors.

- `scripts/prepare_dataset.py` builds the canonical three-class set (inclusion, patches, scratches).
- `data/manifests/dataset_content_lock_v3.json` lets you verify byte-for-byte that you have the same
  images, labels and masks.

## Environment

Python 3.10, PyTorch 2.11 (CUDA 12.8) and Ultralytics 8.4.123. The exact packages are pinned in
`requirements-lock-v31.txt`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock-v31.txt
.venv/Scripts/python -m pip install -e . --no-deps
```

The agent needs an OpenAI-compatible chat endpoint. The paper used Qwen3.8-27B (vision) served by
llama.cpp at temperature 0; pass the endpoint with `--agent-endpoint`.

## Reproducing

```bash
python scripts/check_freeze_v31.py                     # verify the frozen code and protocol hashes
python scripts/run_conditions_v31e.py --tag final31 --seeds 0 \
    --conditions agent_experiment_multi_round --agent-endpoint http://HOST:PORT/v1
python scripts/final_test_v31.py --report              # paired statistics from the stored metrics
```

Things to know before rerunning:

- Configuration files contain absolute paths of the original machine (`D:/AI Paper/...`). Adjust them
  to your setup.
- The frozen files are hash-locked, so editing them makes `check_freeze_v31.py` report a mismatch.
- Results vary with hardware. On one seed, a different GPU changed validation mAP50 by up to 0.037.
  All runs in the paper used a single RTX PRO 6000.

## Citation

```text
W. Shi. Detector-in-the-Loop Agentic Data Augmentation for Few-Shot Object Detection. 2026.
```

## License

Code: MIT (see LICENSE). The datasets keep their original licenses.
