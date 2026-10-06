"""Validate one Agent plan and its prior-round history before tool execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from defectaug.agent_plan import (
    AgentPlanPolicy,
    load_prompt_whitelist,
    load_source_whitelist,
    validate_agent_plan_semantics,
)


PROJECT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _read_plan(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError(f"Agent plan root must be an object: {path}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--history", action="append", default=[])
    parser.add_argument("--experiment", default="configs/experiment_v1.yaml")
    parser.add_argument("--prompt-library", default="configs/prompt_library_v1.yaml")
    parser.add_argument("--train-manifest", default="data/splits/v1/train_fewshot.txt")
    args = parser.parse_args()

    experiment = yaml.safe_load(project_path(args.experiment).read_text(encoding="utf-8-sig"))
    agent = experiment["agent_policy"]
    policy = AgentPlanPolicy(
        round_budget=int(agent["round_budget"]),
        total_budget=int(agent["total_budget"]),
        rounds=int(agent["rounds"]),
        final_minimum_per_class=int(agent["final_minimum_per_class"]),
        maximum_fraction_per_class=float(agent["maximum_fraction_per_class"]),
        max_seeds_per_action=int(agent["max_seeds_per_action"]),
        max_retries_per_seed=int(agent["max_retries_per_seed"]),
        candidate_budget_per_round_max=int(
            experiment["generation"]["candidate_budget_per_round_max"]
        ),
        max_candidates_per_source=int(experiment["generation"]["max_candidates_per_source"]),
        max_accepted_derivatives_per_source=int(
            experiment["generation"]["max_accepted_derivatives_per_source"]
        ),
    )
    templates, modifiers = load_prompt_whitelist(project_path(args.prompt_library))
    sources = load_source_whitelist(project_path(args.train_manifest))
    frozen_tools = experiment["generation"].get("formal_tools_by_class")
    if frozen_tools is not None and not isinstance(frozen_tools, dict):
        raise ValueError("generation.formal_tools_by_class must be null or an object")
    result = validate_agent_plan_semantics(
        _read_plan(project_path(args.plan)),
        history=[_read_plan(project_path(path)) for path in args.history],
        prompt_templates=templates,
        allowed_modifiers=modifiers,
        allowed_sources=sources,
        implemented_tools=agent["implemented_tools"],
        allowed_tools_by_class=frozen_tools,
        policy=policy,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
