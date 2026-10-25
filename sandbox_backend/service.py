"""Actual Seed-801 inference; no teacher, no legality gating, no score faking."""
from __future__ import annotations

import torch
from fastapi import HTTPException
from twm_v75.world import enumerate_bindings
from twm_v76.world import single_cases
from twm_v77.world import write_joint
from twm_v78.model import binary_masks

from .adapter import build_world, to_public_state
from .schemas import (Candidate, EvaluatedSubset, StepLog, StepRequest, StepResponse)


def run_step(model, req: StepRequest) -> StepResponse:
    world, ephemeral, _law_indices = build_world(req)
    cases = single_cases(world)

    with torch.inference_mode():
        # Exact same feature tensors / energies as archived JointConsistencyEnergy.
        x, pairs, priors, resources, relations, locations = model.features(world)
        baseline = model.parent.energies(x, pairs, priors, resources)
        total = model.energies(x, pairs, priors, resources, relations)
        # No ranking override, threshold, conflict elimination, teacher or correction.
        winner_index = int(torch.argmax(total).item())

    count = len(locations)
    if not 1 <= count <= 8 or total.numel() != 2 ** count:
        raise HTTPException(500, "Unexpected number of neural proposals/subsets")
    if not bool(torch.isfinite(total).all()) or not bool(torch.isfinite(baseline).all()):
        raise HTTPException(500, "Neural energies not finite; cannot choose a winner")

    # Only after argmax: copy out transport metadata and run unfiltered typed write-back.
    # These diagnostics DO NOT change ranking and DO NOT suppress any candidate.
    masks = binary_masks(count).int().tolist()
    bindings = [list(enumerate_bindings(c)) for c in cases]
    candidates = []
    for i, (j, k) in enumerate(locations):
        u, v, w = bindings[j][k]
        law = world.proposals[j].law
        candidates.append(Candidate(
            candidate_id=i,
            law_id=world.proposals[j].identity,
            proposal_index=j,
            binding_index=k,
            binding={"u": u, "v": v, "w": w},
            effect_mode=law.mode,
            effect_amount=law.amount,
            effect_predicate=law.effect_predicate,
            relation_add=law.relation_add,
        ))

    evaluated = []
    winner_state = None
    winner_diagnostics = None
    for subset_id, mask in enumerate(masks):
        selected = tuple(locations[i] for i, bit in enumerate(mask) if bit)
        native_state, _, diag = write_joint(world, selected)
        selected_ids = [i for i, bit in enumerate(mask) if bit]
        effects = []
        for i in selected_ids:
            c = candidates[i]
            u = c.binding['u']; v = c.binding['v']
            if c.effect_mode in ('consume', 'transfer'):
                effects.append(f"resource({u}) -= {c.effect_amount}")
            if c.effect_mode in ('produce', 'transfer'):
                effects.append(f"resource({v}) += {c.effect_amount}")
            effects.append(f"relation({c.effect_predicate}, {u}, {v}) = {'add' if c.relation_add else 'remove'}")
        if subset_id == winner_index:
            winner_state = native_state
            winner_diagnostics = diag
        evaluated.append(EvaluatedSubset(
            subset_id=subset_id,
            description=("Пустое подмножество (нет изменений)" if not selected else
                         "Заявленные эффекты: " + ", ".join(candidates[i].law_id for i in selected_ids)),
            effects_applied=effects,
            resource_energy=None, relational_energy=None,
            base_v79_energy=float(baseline[subset_id].item()),
            joint_v710_residual_energy=float((total[subset_id]-baseline[subset_id]).item()),
            total_consistency_energy=float(total[subset_id].item()),
            is_winner=(subset_id == winner_index),
            # POST-SELECTION diagnostics, NOT neural predicates or conflict filters.
            conflict_detected=bool(diag['invalid_numeric'] or diag['opposed_relations']),
            diagnostics=diag,
            selected_candidates=selected_ids,
        ))
    assert winner_state is not None and winner_diagnostics is not None
    final_state = to_public_state(winner_state, req.world_state, ephemeral)
    warnings=[]
    if winner_diagnostics['invalid_numeric']:
        warnings.append("Neural selection produced invalid stocks; state was NOT clamped or repaired.")
    if winner_diagnostics['opposed_relations']:
        warnings.append("Neural selection made opposite writes to the same relation; serialization NOT arbitrated.")
    if any(e.properties for e in req.world_state.entities) or any(e.kind for e in req.world_state.entities):
        warnings.append("Entity kind/properties are UI metadata, NOT model input features.")
    if req.action.instrument is not None:
        warnings.append("Instrument is encoded as a fact; a law must explicitly reference it (role w).")
    warnings.append("Raw energy scales are not calibrated probabilities; 'conflict_detected' is post-hoc diagnostic only.")
    a=req.action
    normalized = f"{a.verb}({a.actor}, {a.target}" + (f", {a.instrument}" if a.instrument is not None else "") + ")"
    return StepResponse(
        success=True,  # Request ran successfully; NOT semantic correctness.
        step_log=StepLog(
            action_normalized=normalized,
            evaluated_subsets=evaluated,
            candidates=candidates,
            winner_subset_id=winner_index,
            energy_interpretation=("total = learned v7.9 baseline + learned v7.10 JOINT residual; "
                                   "there are no pure resource/relation scalar components"),
            warnings=warnings,
        ),
        new_world_state=final_state,
    )