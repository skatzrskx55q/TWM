"""Public transport schema. It is NOT a general natural-language world grammar."""
from __future__ import annotations

from typing import Any, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Entity(StrictModel):
    id: str = Field(min_length=1)
    stock: int
    capacity: int
    kind: str | None = None                  # UI metadata; not neural physics
    properties: dict[str, Any] = Field(default_factory=dict)  # Display-only


class Relation(StrictModel):
    predicate: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class WorldState(StrictModel):
    entities: list[Entity] = Field(min_length=3, max_length=32)
    relations: list[Relation] = Field(default_factory=list)


class AtomNode(StrictModel):
    predicate: str = Field(min_length=1)
    left: Literal["actor", "u", "v", "w"]
    right: Literal["actor", "u", "v", "w"]
    required: bool = True


class GateNode(StrictModel):
    table: tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1], Literal[0, 1]]
    left: "RuleExpression"
    right: "RuleExpression"


RuleExpression = Annotated[Union[AtomNode, GateNode], Field(union_mode="left_to_right")]
GateNode.model_rebuild()


class LawInput(StrictModel):
    id: str = Field(min_length=1)
    priority: int = Field(ge=0, le=20)
    enabled: bool = True
    condition: RuleExpression
    mode: Literal["consume", "produce", "transfer"]
    amount: int = Field(ge=1)
    effect_predicate: str = Field(min_length=1)
    relation_add: bool


class Action(StrictModel):
    actor: str = Field(min_length=1)
    verb: str = Field(min_length=1)
    target: str = Field(min_length=1)
    instrument: str | None = None


class StepRequest(StrictModel):
    world_state: WorldState
    laws: list[LawInput]
    action: Action


class EvaluatedSubset(StrictModel):
    subset_id: int
    description: str
    effects_applied: list[str]
    # These terms cannot truthfully be isolated from the joint neural scorer.
    resource_energy: float | None = None
    relational_energy: float | None = None
    base_v79_energy: float
    joint_v710_residual_energy: float
    total_consistency_energy: float
    is_winner: bool
    conflict_detected: bool
    diagnostics: dict[str, int | bool]
    selected_candidates: list[int]


class Candidate(StrictModel):
    candidate_id: int
    law_id: str
    proposal_index: int
    binding_index: int
    binding: dict[str, str | None]
    effect_mode: str
    effect_amount: int
    effect_predicate: str
    relation_add: bool


class StepLog(StrictModel):
    action_normalized: str
    evaluated_subsets: list[EvaluatedSubset]
    candidates: list[Candidate]
    winner_subset_id: int
    energy_interpretation: str
    warnings: list[str] = Field(default_factory=list)


class StepResponse(StrictModel):
    success: bool
    step_log: StepLog
    new_world_state: WorldState
