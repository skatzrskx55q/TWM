"""Syntax-only conversion of JSON into the archived declarative world AST.

NO predicates are evaluated here. We don't decide whether a rule fires.
The injected action facts are ephemeral observations for the neural encoder.
"""
from __future__ import annotations

from fastapi import HTTPException
from twm_v75.world import Atom, Gate, Law, State, leaves
from twm_v76.world import MultiCase, Proposal
from .schemas import AtomNode, GateNode, RuleExpression, StepRequest, WorldState, Relation

ACTOR_TOKEN = "__twm_sandbox_internal_actor__"
ACTION_PREFIX = "__twm_action__/"
ACTOR_PRED = ACTION_PREFIX + "actor"
TARGET_PRED = ACTION_PREFIX + "target"
INSTRUMENT_PRED = ACTION_PREFIX + "instrument"


def verb_predicate(verb: str) -> str:
    return ACTION_PREFIX + "verb/" + verb


def err(detail: str):
    raise HTTPException(status_code=422, detail=detail)


def make_expr(node: RuleExpression):
    if isinstance(node, AtomNode):
        return Atom(node.predicate, node.left, node.right, node.required)
    if isinstance(node, GateNode):
        return Gate(tuple(map(float, node.table)), make_expr(node.left), make_expr(node.right))
    err("Unsupported expression AST")


def build_world(req: StepRequest) -> tuple[MultiCase, set[tuple[str, str, str]], list[int]]:
    entities = req.world_state.entities
    names = [e.id for e in entities]
    if len(set(names)) != len(names) or ACTOR_TOKEN in names:
        err("Entity IDs must be distinct and must not use reserved internal actor ID")
    allowed = set(names)
    if any(e.capacity <= 0 or e.stock < 0 or e.stock > e.capacity for e in entities):
        err("Initial stocks must satisfy 0 <= stock <= positive capacity; no auto-repair")
    action = req.action
    for n in [action.actor, action.target, *([action.instrument] if action.instrument else [])]:
        if n not in allowed:
            err(f"Action entity not found: {n}")
    facts: set[tuple[str, str, str]] = set()
    for f in req.world_state.relations:
        if f.predicate.startswith(ACTION_PREFIX):
            err(f"Predicate prefix {ACTION_PREFIX!r} is reserved for action observations")
        if f.source not in allowed or f.target not in allowed:
            err(f"Relation references unknown entity: {f.model_dump()}")
        facts.add((f.predicate, f.source, f.target))
    active = [(idx, law) for idx, law in enumerate(req.laws) if law.enabled]
    if not 1 <= len(active) <= 5:
        err("The frozen v7.10 proposer supports 1..5 enabled laws in this adapter")
    if len({law.id for law in req.laws}) != len(req.laws):
        err("Law IDs must be unique")
    if len({law.priority for _,law in active}) != len(active):
        err("Enabled law priorities must be unique")
    proposals = []
    for _, law in active:
        expr = make_expr(law.condition)
        if law.effect_predicate.startswith(ACTION_PREFIX):
            err(f"Law {law.id}: effect predicates must not use reserved action namespace")
        if len(leaves(expr)) > 10:
            err(f"Law {law.id}: at most 10 predicate leaves in native encoder")
        if any(leaf.predicate.startswith(ACTION_PREFIX) and
               leaf.predicate not in {ACTOR_PRED, TARGET_PRED, INSTRUMENT_PRED, verb_predicate(action.verb)}
               for leaf in leaves(expr)):
            err(f"Law {law.id}: unsupported action predicate for this step")
        # Exactly the declared AST, mode, amount and relation edit. No legal/truth tests.
        proposals.append(Proposal(Law(expr, law.mode, law.amount,
                                      law.effect_predicate, law.relation_add),
                                  law.priority, law.id))
    state = State(tuple(names), frozenset(facts),
                  tuple(e.stock for e in entities), tuple(e.capacity for e in entities))
    # Runtime actor must be OUTSIDE State.entities in the original v7.5 encoder.
    ephemeral = {(ACTOR_PRED, ACTOR_TOKEN, action.actor),
                 (TARGET_PRED, ACTOR_TOKEN, action.target),
                 (verb_predicate(action.verb), ACTOR_TOKEN, action.actor)}
    if action.instrument is not None:
        ephemeral.add((INSTRUMENT_PRED, ACTOR_TOKEN, action.instrument))
    injected_state = State(state.entities, state.facts | ephemeral, state.stocks, state.capacities)
    return MultiCase(ACTOR_TOKEN, injected_state, tuple(proposals), "sandbox"), ephemeral, [i for i,_ in active]


def to_public_state(native: State, original: WorldState, ephemeral: set[tuple[str,str,str]]) -> WorldState:
    # Remove only our temporary input features, NEVER selected model effects.
    # This is a serialization boundary (not a validator, truth engine, or repair).
    facts = native.facts - ephemeral
    if any(p.startswith(ACTION_PREFIX) for p,_,_ in facts):
        err("A law attempted to persist a reserved action predicate")
    entities = [e.model_copy(update={"stock": int(native.stocks[i]),
                                     "capacity": int(native.capacities[i])})
                for i,e in enumerate(original.entities)]
    relations = [Relation(predicate=p, source=u, target=v) for p,u,v in sorted(facts)]
    return WorldState(entities=entities, relations=relations)
