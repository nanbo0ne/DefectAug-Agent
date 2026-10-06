"""Semantic validation for an Agent generation plan.

JSON Schema checks the shape and parameter ranges.  These checks enforce the
cross-field budget invariants that JSON Schema does not express conveniently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


CLASSES = ("inclusion", "patches", "scratches")
IMPLEMENTED_TOOLS = frozenset({"qwen_mask_inpaint", "sdxl_mask_inpaint"})


@dataclass(frozen=True)
class PlanBudget:
    round_index: int
    accepted_total: int
    remaining_before: int
    remaining_after: int


@dataclass(frozen=True)
class LedgerTotals:
    candidates: int
    accepted: int


def validate_generation_ledger(
    records: Iterable[Mapping[str, Any]],
    *,
    candidate_cap_total: int,
    candidate_cap_per_source: int,
    accepted_cap_per_source: int,
) -> LedgerTotals:
    """Fail closed when the central candidate/source ceilings are exceeded."""

    if min(candidate_cap_total, candidate_cap_per_source, accepted_cap_per_source) < 0:
        raise ValueError("ledger caps cannot be negative")
    candidates_by_source: dict[str, int] = {}
    accepted_by_source: dict[str, int] = {}
    candidates = 0
    accepted = 0
    for record in records:
        source_id = record.get("source_id")
        decision = record.get("accepted")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("every candidate must have a non-empty source_id")
        if not isinstance(decision, bool):
            raise ValueError("every candidate must have a boolean accepted decision")
        candidates += 1
        candidates_by_source[source_id] = candidates_by_source.get(source_id, 0) + 1
        if decision:
            accepted += 1
            accepted_by_source[source_id] = accepted_by_source.get(source_id, 0) + 1

    if candidates > candidate_cap_total:
        raise ValueError(f"candidate total {candidates} exceeds cap {candidate_cap_total}")
    for source_id, count in candidates_by_source.items():
        if count > candidate_cap_per_source:
            raise ValueError(
                f"source {source_id} candidate count {count} exceeds cap {candidate_cap_per_source}"
            )
    for source_id, count in accepted_by_source.items():
        if count > accepted_cap_per_source:
            raise ValueError(
                f"source {source_id} accepted count {count} exceeds cap {accepted_cap_per_source}"
            )
    return LedgerTotals(candidates=candidates, accepted=accepted)


def validate_plan_budget(
    plan: Mapping[str, Any],
    *,
    expected_round: int,
    expected_round_budget: int,
    remaining_accepted_budget: int,
) -> PlanBudget:
    """Validate round identity and accepted-image budget semantics.

    This function is intentionally independent of the LLM provider.  Call it
    after JSON-Schema validation and before any generation tool is invoked.
    """

    if expected_round < 1:
        raise ValueError("expected_round must be positive")
    if expected_round_budget < 0 or remaining_accepted_budget < 0:
        raise ValueError("budgets cannot be negative")
    if plan.get("round") != expected_round:
        raise ValueError(
            f"plan round {plan.get('round')!r} does not match expected round {expected_round}"
        )

    allocation = plan.get("accepted_budget")
    if not isinstance(allocation, Mapping):
        raise ValueError("accepted_budget must be an object")
    if set(allocation) != set(CLASSES):
        raise ValueError(f"accepted_budget must contain exactly {CLASSES}")

    values: dict[str, int] = {}
    for class_name in CLASSES:
        value = allocation[class_name]
        # bool is an int subclass but must not be accepted as an image count.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"accepted_budget.{class_name} must be a non-negative integer")
        values[class_name] = value

    accepted_total = sum(values.values())
    required = min(expected_round_budget, remaining_accepted_budget)
    if accepted_total != required:
        raise ValueError(
            f"round allocation must sum to {required}, got {accepted_total}"
        )
    if accepted_total > remaining_accepted_budget:
        raise ValueError("round allocation exceeds remaining accepted budget")

    actions = plan.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(CLASSES):
        raise ValueError(f"actions must contain exactly {CLASSES}")

    return PlanBudget(
        round_index=expected_round,
        accepted_total=accepted_total,
        remaining_before=remaining_accepted_budget,
        remaining_after=remaining_accepted_budget - accepted_total,
    )


class ClassBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inclusion: int = Field(ge=0, le=50)
    patches: int = Field(ge=0, le=50)
    scratches: int = Field(ge=0, le=50)


class GenerationAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    prompt_template: str = Field(min_length=1, max_length=100)
    modifier: str = Field(min_length=1, max_length=100)
    mask_expand_px: int = Field(ge=0, le=8)
    mask_blur_px: int = Field(ge=0, le=4)
    control_strength: float = Field(ge=0.5, le=1.5)
    steps: int = Field(ge=10, le=40)
    source_image_ids: list[str] = Field(default_factory=list, max_length=60)
    seeds: list[int] = Field(default_factory=list, max_length=3)
    max_retries_per_seed: int = Field(ge=0, le=2)


class ClassActions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inclusion: GenerationAction
    patches: GenerationAction
    scratches: GenerationAction


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round: int = Field(ge=1, le=3)
    accepted_budget: ClassBudget
    actions: ClassActions
    reason: str = Field(min_length=1, max_length=1000)


@dataclass(frozen=True)
class AgentPlanPolicy:
    round_budget: int = 50
    total_budget: int = 150
    rounds: int = 3
    final_minimum_per_class: int = 10
    maximum_fraction_per_class: float = 0.70
    max_seeds_per_action: int = 3
    max_retries_per_seed: int = 2
    candidate_budget_per_round_max: int = 150
    max_candidates_per_source: int = 12
    max_accepted_derivatives_per_source: int = 8

    @property
    def per_round_class_maximum(self) -> int:
        return math.floor(self.round_budget * self.maximum_fraction_per_class)

    @property
    def final_class_maximum(self) -> int:
        return math.floor(self.total_budget * self.maximum_fraction_per_class)


@dataclass(frozen=True)
class AgentPlanValidation:
    round: int
    accepted_this_round: int
    accepted_cumulative: int
    cumulative_by_class: dict[str, int]
    candidate_capacity_this_round: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": True,
            "round": self.round,
            "accepted_this_round": self.accepted_this_round,
            "accepted_cumulative": self.accepted_cumulative,
            "cumulative_by_class": self.cumulative_by_class,
            "candidate_capacity_this_round": self.candidate_capacity_this_round,
        }


def parse_agent_plan(value: Mapping[str, Any] | AgentPlan) -> AgentPlan:
    if isinstance(value, AgentPlan):
        return value
    return AgentPlan.model_validate(value)


def load_prompt_whitelist(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("prompt library root must be a mapping")
    templates = {
        key for key, value in document.items() if key == "base" and isinstance(value, str)
    }
    classes = document.get("classes")
    if not templates or not isinstance(classes, dict):
        raise ValueError("prompt library must define a string 'base' and class mappings")
    modifiers: dict[str, set[str]] = {}
    for class_name in CLASSES:
        record = classes.get(class_name)
        if not isinstance(record, dict) or not isinstance(record.get("allowed_modifiers"), list):
            raise ValueError(f"prompt library has no modifier whitelist for {class_name}")
        values = record["allowed_modifiers"]
        if not values or not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"invalid modifier whitelist for {class_name}")
        modifiers[class_name] = set(values)
    return templates, modifiers


def load_source_whitelist(train_manifest: Path) -> dict[str, set[str]]:
    """Load source IDs from manifest text without opening any listed image."""

    result = {class_name: set() for class_name in CLASSES}
    for line_number, raw in enumerate(
        train_manifest.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        stem = Path(value).stem
        class_name = stem.rsplit("_", maxsplit=1)[0]
        if class_name not in result:
            raise ValueError(
                f"unrecognized class prefix in source manifest at line {line_number}: {stem}"
            )
        if stem in result[class_name]:
            raise ValueError(f"duplicate source ID in train manifest: {stem}")
        result[class_name].add(stem)
    if any(not values for values in result.values()):
        raise ValueError("train source manifest must contain every declared class")
    return result


def _budget(plan: AgentPlan) -> dict[str, int]:
    return {class_name: int(getattr(plan.accepted_budget, class_name)) for class_name in CLASSES}


def validate_agent_plan_semantics(
    plan: Mapping[str, Any] | AgentPlan,
    *,
    history: Sequence[Mapping[str, Any] | AgentPlan] = (),
    prompt_templates: Iterable[str],
    allowed_modifiers: Mapping[str, set[str]],
    allowed_sources: Mapping[str, set[str]],
    implemented_tools: Iterable[str] = IMPLEMENTED_TOOLS,
    allowed_tools_by_class: Mapping[str, Iterable[str]] | None = None,
    policy: AgentPlanPolicy = AgentPlanPolicy(),
) -> AgentPlanValidation:
    """Enforce schema shape plus budget, tool, source, seed and retry invariants."""

    current = parse_agent_plan(plan)
    previous = [parse_agent_plan(item) for item in history]
    sequence = previous + [current]
    expected_rounds = list(range(1, current.round + 1))
    actual_rounds = [item.round for item in sequence]
    if actual_rounds != expected_rounds:
        raise ValueError(
            f"plan history must contain rounds {expected_rounds}; found {actual_rounds}"
        )
    if current.round > policy.rounds:
        raise ValueError(f"round {current.round} exceeds policy maximum {policy.rounds}")

    template_whitelist = set(prompt_templates)
    tool_whitelist = set(implemented_tools)
    if not template_whitelist or not tool_whitelist:
        raise ValueError("prompt and implemented-tool whitelists cannot be empty")
    class_tool_whitelists: dict[str, set[str]] | None = None
    if allowed_tools_by_class is not None:
        if set(allowed_tools_by_class) != set(CLASSES):
            raise ValueError(f"allowed_tools_by_class must contain exactly {CLASSES}")
        class_tool_whitelists = {}
        for class_name in CLASSES:
            allowed = set(allowed_tools_by_class[class_name])
            if not allowed:
                raise ValueError(f"frozen tool mapping is empty for {class_name}")
            unknown = allowed - tool_whitelist
            if unknown:
                raise ValueError(
                    f"frozen tool mapping for {class_name} contains unimplemented tools: "
                    f"{sorted(unknown)}"
                )
            class_tool_whitelists[class_name] = allowed

    cumulative = {class_name: 0 for class_name in CLASSES}
    used_jobs: set[tuple[str, str, int]] = set()
    capacities: dict[str, int] = {}
    for item in sequence:
        item_budget = _budget(item)
        if sum(item_budget.values()) != policy.round_budget:
            raise ValueError(
                f"round {item.round} accepted budget must total {policy.round_budget}; "
                f"found {sum(item_budget.values())}"
            )
        item_candidate_capacity = 0
        for class_name, amount in item_budget.items():
            if amount > policy.per_round_class_maximum:
                raise ValueError(
                    f"round {item.round} {class_name} budget {amount} exceeds per-round "
                    f"class maximum {policy.per_round_class_maximum}"
                )
            cumulative[class_name] += amount
            action: GenerationAction = getattr(item.actions, class_name)
            if action.tool not in tool_whitelist:
                raise ValueError(f"tool is not implemented/allowed: {action.tool}")
            if (
                class_tool_whitelists is not None
                and action.tool not in class_tool_whitelists[class_name]
            ):
                raise ValueError(
                    f"tool is not allowed by the frozen mapping for {class_name}: "
                    f"{action.tool}"
                )
            if action.prompt_template not in template_whitelist:
                raise ValueError(
                    f"prompt template is not whitelisted: {action.prompt_template}"
                )
            if action.modifier not in allowed_modifiers.get(class_name, set()):
                raise ValueError(
                    f"modifier is not whitelisted for {class_name}: {action.modifier}"
                )
            if len(action.source_image_ids) != len(set(action.source_image_ids)):
                raise ValueError(f"round {item.round} {class_name} has duplicate source IDs")
            if len(action.seeds) != len(set(action.seeds)):
                raise ValueError(f"round {item.round} {class_name} has duplicate seeds")
            if len(action.seeds) > policy.max_seeds_per_action:
                raise ValueError(
                    f"round {item.round} {class_name} uses too many seeds: {len(action.seeds)}"
                )
            if action.max_retries_per_seed > policy.max_retries_per_seed:
                raise ValueError(f"round {item.round} {class_name} retry limit exceeds policy")
            if any(not 0 <= seed <= 2**32 - 1 for seed in action.seeds):
                raise ValueError(f"round {item.round} {class_name} has invalid seeds")
            invalid_sources = set(action.source_image_ids) - set(
                allowed_sources.get(class_name, set())
            )
            if invalid_sources:
                raise ValueError(
                    f"round {item.round} {class_name} uses non-training source IDs: "
                    f"{sorted(invalid_sources)}"
                )
            if amount == 0:
                if action.source_image_ids or action.seeds:
                    raise ValueError(
                        f"round {item.round} {class_name} budget is zero, so sources/seeds must be empty"
                    )
                capacity = 0
            else:
                if not action.source_image_ids or not action.seeds:
                    raise ValueError(
                        f"round {item.round} {class_name} requires sources and seeds"
                    )
                capacity = (
                    len(action.source_image_ids)
                    * len(action.seeds)
                    * (1 + action.max_retries_per_seed)
                )
                if capacity < amount:
                    raise ValueError(
                        f"round {item.round} {class_name} candidate capacity {capacity} "
                        f"cannot satisfy accepted budget {amount}"
                    )
                candidates_per_source = len(action.seeds) * (
                    1 + action.max_retries_per_seed
                )
                if candidates_per_source > policy.max_candidates_per_source:
                    raise ValueError(
                        f"round {item.round} {class_name} allows {candidates_per_source} "
                        f"candidates per source, exceeding {policy.max_candidates_per_source}"
                    )
                accepted_capacity = (
                    len(action.source_image_ids)
                    * policy.max_accepted_derivatives_per_source
                )
                if accepted_capacity < amount:
                    raise ValueError(
                        f"round {item.round} {class_name} source derivative capacity "
                        f"{accepted_capacity} cannot satisfy accepted budget {amount}"
                    )
            item_candidate_capacity += capacity
            for source_id in action.source_image_ids:
                for seed in action.seeds:
                    job = (class_name, source_id, seed)
                    if job in used_jobs:
                        raise ValueError(
                            f"source/seed job is repeated across rounds: {class_name}/{source_id}/{seed}"
                        )
                    used_jobs.add(job)
            if item is current:
                capacities[class_name] = capacity
        if item_candidate_capacity > policy.candidate_budget_per_round_max:
            raise ValueError(
                f"round {item.round} candidate capacity {item_candidate_capacity} exceeds "
                f"per-round cap {policy.candidate_budget_per_round_max}"
            )

    cumulative_total = sum(cumulative.values())
    if cumulative_total > policy.total_budget:
        raise ValueError(
            f"cumulative accepted budget {cumulative_total} exceeds {policy.total_budget}"
        )
    if any(value > policy.final_class_maximum for value in cumulative.values()):
        raise ValueError(
            f"cumulative class budget exceeds final maximum {policy.final_class_maximum}"
        )
    if current.round == policy.rounds:
        if cumulative_total != policy.total_budget:
            raise ValueError(
                f"final cumulative budget must equal {policy.total_budget}; found {cumulative_total}"
            )
        below_minimum = {
            class_name: value
            for class_name, value in cumulative.items()
            if value < policy.final_minimum_per_class
        }
        if below_minimum:
            raise ValueError(
                f"final class budgets are below minimum {policy.final_minimum_per_class}: "
                f"{below_minimum}"
            )

    return AgentPlanValidation(
        round=current.round,
        accepted_this_round=sum(_budget(current).values()),
        accepted_cumulative=cumulative_total,
        cumulative_by_class=cumulative,
        candidate_capacity_this_round=capacities,
    )


__all__ = [
    "AgentPlan",
    "AgentPlanPolicy",
    "AgentPlanValidation",
    "CLASSES",
    "IMPLEMENTED_TOOLS",
    "LedgerTotals",
    "PlanBudget",
    "ValidationError",
    "load_prompt_whitelist",
    "load_source_whitelist",
    "parse_agent_plan",
    "validate_agent_plan_semantics",
    "validate_generation_ledger",
    "validate_plan_budget",
]
