import pytest

from defectaug.agent_plan import AgentPlanPolicy, validate_agent_plan_semantics


TEMPLATES = {"base"}
MODIFIERS = {
    "inclusion": {"faint", "strong"},
    "patches": {"faint", "strong"},
    "scratches": {"faint", "strong"},
}
SOURCES = {
    class_name: {f"{class_name}_{index}" for index in range(1, 31)}
    for class_name in MODIFIERS
}
POLICY = AgentPlanPolicy()


def _plan(round_number: int, budgets: dict[str, int], seed_offset: int = 0) -> dict:
    actions = {}
    for class_index, (class_name, budget) in enumerate(budgets.items()):
        source_count = min(30, max(0, budget))
        actions[class_name] = {
            "tool": "qwen_mask_inpaint",
            "prompt_template": "base",
            "modifier": "faint",
            "mask_expand_px": 2,
            "mask_blur_px": 1,
            "control_strength": 1.0,
            "steps": 20,
            "source_image_ids": [
                f"{class_name}_{index}" for index in range(1, source_count + 1)
            ],
            "seeds": [] if budget == 0 else [seed_offset + class_index],
            "max_retries_per_seed": 1,
        }
    return {
        "round": round_number,
        "accepted_budget": budgets,
        "actions": actions,
        "reason": "validation weakness and bounded generation cost",
    }


def _validate(plan: dict, history=()):
    return validate_agent_plan_semantics(
        plan,
        history=history,
        prompt_templates=TEMPLATES,
        allowed_modifiers=MODIFIERS,
        allowed_sources=SOURCES,
        policy=POLICY,
    )


def test_agent_plan_three_round_budget_and_bounds_are_valid() -> None:
    first = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    second = _plan(2, {"inclusion": 20, "patches": 20, "scratches": 10}, 200)
    third = _plan(3, {"inclusion": 20, "patches": 10, "scratches": 20}, 300)
    result = _validate(third, history=(first, second))
    assert result.accepted_this_round == 50
    assert result.accepted_cumulative == 150
    assert result.cumulative_by_class == {
        "inclusion": 65,
        "patches": 45,
        "scratches": 40,
    }


def test_agent_plan_rejects_wrong_round_total_and_unimplemented_tool() -> None:
    wrong_total = _plan(1, {"inclusion": 20, "patches": 15, "scratches": 10}, 100)
    with pytest.raises(ValueError, match="must total 50"):
        _validate(wrong_total)

    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    plan["actions"]["inclusion"]["tool"] = "real_patch_edge_inpaint"
    with pytest.raises(ValueError, match="not implemented/allowed"):
        _validate(plan)


def test_agent_plan_rejects_prompt_source_seed_and_retry_escape() -> None:
    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    plan["actions"]["inclusion"]["prompt_template"] = "free_form"
    with pytest.raises(ValueError, match="not whitelisted"):
        _validate(plan)

    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    plan["actions"]["inclusion"]["source_image_ids"][0] = "inclusion_999"
    with pytest.raises(ValueError, match="non-training source"):
        _validate(plan)

    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    plan["actions"]["inclusion"]["seeds"] = [1, 1]
    with pytest.raises(ValueError, match="duplicate seeds"):
        _validate(plan)

    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    plan["actions"]["inclusion"]["max_retries_per_seed"] = 3
    with pytest.raises(ValueError, match="less than or equal to 2"):
        _validate(plan)


def test_agent_plan_rejects_candidate_capacity_above_round_cap() -> None:
    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    for action in plan["actions"].values():
        action["seeds"] = [1000, 1001, 1002]
        action["max_retries_per_seed"] = 2
    with pytest.raises(ValueError, match="candidate capacity .* exceeds per-round cap 150"):
        _validate(plan)


def test_agent_plan_enforces_frozen_class_tool_mapping() -> None:
    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    plan["actions"]["patches"]["tool"] = "sdxl_mask_inpaint"
    mapping = {
        "inclusion": {"qwen_mask_inpaint"},
        "patches": {"qwen_mask_inpaint"},
        "scratches": {"qwen_mask_inpaint"},
    }
    with pytest.raises(ValueError, match="frozen mapping for patches"):
        validate_agent_plan_semantics(
            plan,
            prompt_templates=TEMPLATES,
            allowed_modifiers=MODIFIERS,
            allowed_sources=SOURCES,
            allowed_tools_by_class=mapping,
            policy=POLICY,
        )


def test_agent_plan_rejects_incomplete_or_unimplemented_frozen_mapping() -> None:
    plan = _plan(1, {"inclusion": 25, "patches": 15, "scratches": 10}, 100)
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_agent_plan_semantics(
            plan,
            prompt_templates=TEMPLATES,
            allowed_modifiers=MODIFIERS,
            allowed_sources=SOURCES,
            allowed_tools_by_class={"inclusion": {"qwen_mask_inpaint"}},
            policy=POLICY,
        )

    with pytest.raises(ValueError, match="contains unimplemented tools"):
        validate_agent_plan_semantics(
            plan,
            prompt_templates=TEMPLATES,
            allowed_modifiers=MODIFIERS,
            allowed_sources=SOURCES,
            allowed_tools_by_class={
                "inclusion": {"qwen_mask_inpaint"},
                "patches": {"unavailable_tool"},
                "scratches": {"qwen_mask_inpaint"},
            },
            policy=POLICY,
        )
