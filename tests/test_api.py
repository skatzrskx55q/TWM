"""Contract + model-integrity tests, Seed 801 (real weights, no stub)."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from sandbox_backend.main import app
from sandbox_backend.adapter import build_world
from sandbox_backend.schemas import StepRequest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def example():
    return json.loads((ROOT / "example_step.json").read_text(encoding="utf-8"))


def test_checkpoint_booted(client):
    assert client.get("/api/health").json()["seed"] == 801


def test_subsets_really_scored_and_max_selected(client, example):
    response = client.post("/api/step", json=example)
    assert response.status_code == 200, response.text
    payload = response.json()
    subs = payload["step_log"]["evaluated_subsets"]
    candidates = payload["step_log"]["candidates"]
    assert len(subs) == 2 ** len(candidates) == 4
    assert [s["subset_id"] for s in subs] == list(range(len(subs)))
    assert sum(s["is_winner"] for s in subs) == 1
    best = max(subs, key=lambda s: s["total_consistency_energy"])
    assert best["is_winner"] and best["subset_id"] == payload["step_log"]["winner_subset_id"]
    assert all(s["resource_energy"] is None and s["relational_energy"] is None for s in subs)
    assert all(s["total_consistency_energy"] == pytest.approx(
        s["base_v79_energy"] + s["joint_v710_residual_energy"], abs=1e-5) for s in subs)
    assert payload["step_log"]["action_normalized"] == "открыть(Анна, северная_дверь, красный_ключ)"
    assert not any(r["predicate"].startswith("__twm_action__/") for r in payload["new_world_state"]["relations"])

    # Compare exposed scores with actual forward of the checkpoint, no substitutes.
    native, _, _ = build_world(StepRequest.model_validate(example))
    with torch.inference_mode():
        real_scores, loc = client.app.state.model(native)
    assert list(loc) == [(c["proposal_index"], c["binding_index"]) for c in candidates]
    for i, s in enumerate(subs):
        assert s["total_consistency_energy"] == pytest.approx(float(real_scores[i]), abs=1e-6)


def test_cannot_call_oracle_during_api_inference(monkeypatch, client, example):
    import twm_v77.world as reference
    import twm_v75.world as old
    def blocked(*args, **kwargs):
        raise AssertionError("Oracle executed at inference")
    monkeypatch.setattr(reference, "reference_joint", blocked)
    monkeypatch.setattr(reference, "reference_transition", blocked)
    monkeypatch.setattr(old, "truth", blocked)
    monkeypatch.setattr(old, "numeric_legal", blocked)
    assert client.post("/api/step", json=example).status_code == 200


def test_bad_input_rejected_as_transport_not_predictions(client, example):
    bad = deepcopy(example)
    bad["world_state"]["entities"][0]["stock"] = -1
    assert client.post("/api/step", json=bad).status_code == 422
    bad = deepcopy(example)
    bad["laws"][1]["priority"] = bad["laws"][0]["priority"]
    assert client.post("/api/step", json=bad).status_code == 422
    bad = deepcopy(example)
    bad["world_state"]["relations"].append({"predicate": "__twm_action__/target", "source": "Анна", "target": "комната"})
    assert client.post("/api/step", json=bad).status_code == 422


def test_disabled_laws_are_not_neural_candidates(client, example):
    example["laws"][1]["enabled"] = False
    response = client.post("/api/step", json=example)
    assert response.status_code == 200, response.text
    assert len(response.json()["step_log"]["evaluated_subsets"]) == 2


def test_wrong_neural_prediction_is_not_repaired(monkeypatch, client, example):
    """Force argmax to select the 2 most damaging effects; serializer must expose it."""
    example["laws"][0]["amount"] = 20
    example["laws"][1]["mode"] = "consume"
    example["laws"][1]["amount"] = 20
    model = client.app.state.model
    def force_all(x, p, prior, resources, relations):
        return torch.arange(1 << x.size(0), dtype=torch.float32)
    monkeypatch.setattr(model, "energies", force_all)
    res = client.post("/api/step", json=example)
    assert res.status_code == 200, res.text
    d = res.json()
    winner = d["step_log"]["evaluated_subsets"][-1]
    assert winner["is_winner"]
    assert winner["conflict_detected"]
    assert winner["diagnostics"]["opposed_relations"] == 1
    assert winner["diagnostics"]["invalid_numeric"]
    stocks = {e["id"]: e["stock"] for e in d["new_world_state"]["entities"]}
    assert stocks["Анна"] == -33  # intentionally INVALID; no clamping
