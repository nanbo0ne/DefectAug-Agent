"""Shared allocation constraints for every augmented condition.

Uniform-fixed, Formula-one-shot, Formula-multi-round and Agent-multi-round all
propose a per-class quota; every proposal goes through the same `validate`
and, if it is infeasible, the same deterministic `project`.  This is what
makes "the formula and the agent face identical constraints" a property of
the code rather than of the documentation.

Quantities
----------
allocated        cumulative quota already accepted for the class
source_capacity  how many more samples the generator can still accept for the
                 class, i.e. sum over its sources of
                 min(max_accepted - accepted, max_candidates - candidates).
                 It is already net of what was accepted.
cumulative_cap   floor(max_fraction * total_budget); applied to `allocated`.
headroom         min(cumulative_cap - allocated, source_capacity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CLASSES: tuple[str, ...] = ("inclusion", "patches", "scratches")


class ConstraintViolation(ValueError):
    """Raised when a proposal breaks one or more hard constraints."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class BudgetProtocol:
    total_budget: int
    round_budget: int
    rounds: int
    max_fraction: float
    final_min_per_class: int
    classes: tuple[str, ...] = field(default=CLASSES)

    def __post_init__(self) -> None:
        if self.round_budget * self.rounds != self.total_budget:
            raise ValueError("round_budget * rounds must equal total_budget")
        if self.final_min_per_class * len(self.classes) > self.total_budget:
            raise ValueError("final minimums exceed total budget")
        if self.cumulative_cap * len(self.classes) < self.total_budget:
            raise ValueError("max_fraction makes the total budget unreachable")

    @property
    def cumulative_cap(self) -> int:
        return math.floor(self.max_fraction * self.total_budget)

    @classmethod
    def from_protocol(cls, protocol: dict, rounds: int) -> "BudgetProtocol":
        """Build from experiment_v*.yaml. `rounds` is 1 for single-round conditions.

        One constraint set for every condition: the cap and the final minimum
        come from `allocation_constraints`, so the formula and the agent can
        never be configured apart.
        """
        total = int(protocol["generation"]["accepted_budget_total"])
        shared = protocol["allocation_constraints"]
        return cls(
            total_budget=total,
            round_budget=total // rounds,
            rounds=rounds,
            max_fraction=float(shared["maximum_fraction_per_class"]),
            final_min_per_class=int(shared["final_minimum_per_class"]),
        )


def headroom(
    protocol: BudgetProtocol, allocated: dict[str, int], source_capacity: dict[str, int]
) -> dict[str, int]:
    return {
        c: max(0, min(protocol.cumulative_cap - int(allocated[c]), int(source_capacity[c])))
        for c in protocol.classes
    }


def _check_shape(protocol: BudgetProtocol, proposal: object) -> tuple[dict[str, int], list[str]]:
    reasons: list[str] = []
    if not isinstance(proposal, dict):
        return {}, ["allocation is not an object"]
    extra = sorted(set(proposal) - set(protocol.classes))
    if extra:
        reasons.append(f"unknown classes: {extra}")
    clean: dict[str, int] = {}
    for c in protocol.classes:
        v = proposal.get(c)
        if not isinstance(v, int) or isinstance(v, bool):
            reasons.append(f"{c} is not an integer: {v!r}")
            continue
        if v < 0:
            reasons.append(f"{c} is negative: {v}")
            continue
        clean[c] = v
    return clean, reasons


def validate(
    protocol: BudgetProtocol,
    proposal: object,
    allocated: dict[str, int],
    source_capacity: dict[str, int],
    round_index: int,
) -> dict[str, int]:
    """Return the proposal unchanged if it satisfies every constraint, else raise."""
    if not 1 <= round_index <= protocol.rounds:
        raise ValueError(f"round_index {round_index} outside 1..{protocol.rounds}")
    clean, reasons = _check_shape(protocol, proposal)
    if reasons:
        raise ConstraintViolation(reasons)

    total = sum(clean.values())
    if total != protocol.round_budget:
        reasons.append(f"round sum is {total}, must be exactly {protocol.round_budget}")

    room = headroom(protocol, allocated, source_capacity)
    for c in protocol.classes:
        if clean[c] > room[c]:
            reasons.append(
                f"{c}={clean[c]} exceeds headroom {room[c]} "
                f"(cumulative cap {protocol.cumulative_cap}, already {allocated[c]}, "
                f"source capacity {source_capacity[c]})"
            )

    after = {c: int(allocated[c]) + clean[c] for c in protocol.classes}
    remaining_rounds = protocol.rounds - round_index
    if remaining_rounds == 0:
        for c in protocol.classes:
            if after[c] < protocol.final_min_per_class:
                reasons.append(
                    f"final cumulative {c}={after[c]} below minimum {protocol.final_min_per_class}"
                )
    else:
        # The remaining rounds must still be able to meet the final minimums.
        future_budget = remaining_rounds * protocol.round_budget
        need = sum(max(0, protocol.final_min_per_class - after[c]) for c in protocol.classes)
        if need > future_budget:
            reasons.append(f"final minimums would need {need} > remaining budget {future_budget}")
        future_room = sum(
            max(0, min(protocol.cumulative_cap - after[c], int(source_capacity[c]) - clean[c]))
            for c in protocol.classes
        )
        if future_room < future_budget:
            reasons.append(
                f"leaves only {future_room} headroom for the remaining budget {future_budget}"
            )
    if reasons:
        raise ConstraintViolation(reasons)
    return clean


def project(
    protocol: BudgetProtocol,
    proposal: object,
    allocated: dict[str, int],
    source_capacity: dict[str, int],
    round_index: int,
) -> dict[str, int]:
    """Deterministic projection of any proposal onto the feasible set.

    Identical for every condition: clip to headroom, raise classes to what the
    final minimum requires, then fill the remaining round budget one unit at a
    time to the class with the most spare headroom (ties by class order), or
    trim from the class with the most allocation.  Raises if no feasible
    allocation exists at all.
    """
    raw = proposal if isinstance(proposal, dict) else {}
    room = headroom(protocol, allocated, source_capacity)
    alloc = {}
    for c in protocol.classes:
        v = raw.get(c, 0)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            v = 0
        alloc[c] = min(v, room[c])

    remaining_rounds = protocol.rounds - round_index
    if remaining_rounds == 0:
        floor_need = {
            c: max(0, protocol.final_min_per_class - int(allocated[c])) for c in protocol.classes
        }
    else:
        floor_need = {c: 0 for c in protocol.classes}
    for c in protocol.classes:
        if floor_need[c] > room[c]:
            raise ValueError(f"infeasible: {c} needs {floor_need[c]} but headroom is {room[c]}")
        alloc[c] = max(alloc[c], floor_need[c])

    order = list(protocol.classes)
    while sum(alloc.values()) > protocol.round_budget:
        cands = [c for c in order if alloc[c] > floor_need[c]]
        if not cands:
            raise ValueError("infeasible: minimums exceed round budget")
        victim = max(cands, key=lambda c: (alloc[c], -order.index(c)))
        alloc[victim] -= 1
    while sum(alloc.values()) < protocol.round_budget:
        cands = [c for c in order if alloc[c] < room[c]]
        if not cands:
            raise ValueError("infeasible: total headroom below round budget")
        taker = max(cands, key=lambda c: (room[c] - alloc[c], -order.index(c)))
        alloc[taker] += 1
    return validate(protocol, alloc, allocated, source_capacity, round_index)


def source_capacity_from_state(
    state: dict,
    sources_by_class: dict[str, list[str]],
    max_accepted_per_source: int,
    max_candidates_per_source: int,
) -> dict[str, int]:
    """How many more samples each class can still accept.

    A source stays usable only while BOTH its accepted count is below
    max_accepted and its candidate count is below max_candidates.  Each
    further accept costs at least one candidate, so a source contributes
    min(max_accepted - accepted, max_candidates - candidates).  This is an
    upper bound: rejected candidates can still exhaust a source early, which
    the orchestrator handles with a logged top-up round.
    """
    per_source = state.get("per_source", {}) if state else {}
    cap: dict[str, int] = {}
    for c, stems in sources_by_class.items():
        total = 0
        for stem in stems:
            s = per_source.get(stem, {})
            acc = int(s.get("accepted", 0))
            cand = int(s.get("candidates", 0))
            total += max(0, min(max_accepted_per_source - acc, max_candidates_per_source - cand))
        cap[c] = total
    return cap


def uniform_proposal(protocol: BudgetProtocol) -> dict[str, int]:
    """Equal split of the round budget; remainder by class order."""
    n = len(protocol.classes)
    base, rem = divmod(protocol.round_budget, n)
    return {c: base + (1 if i < rem else 0) for i, c in enumerate(protocol.classes)}
