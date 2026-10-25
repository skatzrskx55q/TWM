"""Streamlit Cloud host for the *real* TWM v7.10 / Seed 801 inference.

This module is deliberately a transport/controller, never a physics engine.
All subset scores and the next state come exclusively from run_step.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import streamlit as st
import torch
from fastapi import HTTPException
from pydantic import ValidationError

from sandbox_backend.schemas import StepRequest
from sandbox_backend.service import run_step
from twm_v710.load import load

ROOT = Path(__file__).resolve().parent
EXAMPLE = json.loads((ROOT / "example_step.json").read_text(encoding="utf-8"))


def make_presets():
    """Alternative *inputs*, not canned answers or computed world transitions."""
    baseline = copy.deepcopy(EXAMPLE)
    baseline["name"] = "Дверь и ключ"
    baseline["subtitle"] = "Совместная оценка ресурса и реляционных правок"

    wrong_key = copy.deepcopy(EXAMPLE)
    wrong_key["name"] = "Ключ не подходит"
    wrong_key["subtitle"] = "Контрфактическое изменение одного исходного отношения"
    wrong_key["world_state"]["relations"] = [
        r for r in wrong_key["world_state"]["relations"] if r["predicate"] != "подходит"
    ]

    conflict = copy.deepcopy(EXAMPLE)
    conflict["name"] = "Конкурирующие законы"
    conflict["subtitle"] = "Два противоречащих реляционных предложения"
    conflict["laws"][1]["priority"] = 6
    conflict["laws"][1]["amount"] = 2

    return [baseline, wrong_key, conflict]


PRESETS = make_presets()


@st.cache_resource(show_spinner="Загрузка обученных весов Seed 801…")
def model_801():
    # Archived chained checkpoints reference project-relative parent paths.
    os.chdir(ROOT)
    torch.set_num_threads(1)
    model = load("checkpoints_v710/checkpoint_801.pt")
    model.eval()
    return model


def initial_session(index=0):
    preset = PRESETS[index]
    return {
        "preset_index": index,
        "world_state": copy.deepcopy(preset["world_state"]),
        "laws": copy.deepcopy(preset["laws"]),
        "action": copy.deepcopy(preset["action"]),
        "history": [],
        "error": None,
        "revision": 0,
    }


def ensure_session():
    if "twm_lab" not in st.session_state:
        st.session_state.twm_lab = initial_session()
    return st.session_state.twm_lab


def handle_command(command: dict, state: dict, model):
    """Only orchestration, request validation, snapshots and history."""
    kind = command.get("type")
    if kind == "reset":
        preset_index = int(state.get("preset_index", 0))
        state.clear()
        state.update(initial_session(preset_index))
    elif kind == "preset":
        index = int(command["index"])
        if not 0 <= index < len(PRESETS):
            raise ValueError("Неизвестный сценарий")
        state.clear()
        state.update(initial_session(index))
    elif kind == "step":
        body = command.get("request")
        request = StepRequest.model_validate(body)
        before = copy.deepcopy(request.world_state.model_dump(mode="json"))
        output = run_step(model, request)  # THE ONLY WORLD TRANSITION
        record = {
            "number": len(state["history"]) + 1,
            "action": request.action.model_dump(mode="json"),
            "world_state_before": before,
            "world_state_after": output.new_world_state.model_dump(mode="json"),
            "laws": copy.deepcopy(body["laws"]),
            "response": output.model_dump(mode="json"),
        }
        state["history"].append(record)
        state["world_state"] = copy.deepcopy(record["world_state_after"])
        state["laws"] = copy.deepcopy(body["laws"])
        state["action"] = request.action.model_dump(mode="json")
    elif kind == "branch":
        number = int(command["number"])
        if not 0 <= number <= len(state["history"]):
            raise ValueError("Некорректный номер исторического шага")
        if number == 0:
            state["world_state"] = copy.deepcopy(PRESETS[state["preset_index"]]["world_state"])
            state["laws"] = copy.deepcopy(PRESETS[state["preset_index"]]["laws"])
            state["action"] = copy.deepcopy(PRESETS[state["preset_index"]]["action"])
        else:
            saved = state["history"][number - 1]
            state["world_state"] = copy.deepcopy(saved["world_state_after"])
            state["laws"] = copy.deepcopy(saved["laws"])
            state["action"] = copy.deepcopy(saved["action"])
        state["history"] = state["history"][:number]
    else:
        raise ValueError("Неизвестная команда интерфейса")
    state["error"] = None
    state["revision"] += 1


def russian_error(exc):
    if isinstance(exc, ValidationError):
        parts = []
        for item in exc.errors()[:3]:
            parts.append(f"{'.'.join(map(str, item['loc']))}: {item['msg']}")
        return "Ошибка формата входных данных: " + "; ".join(parts)
    if isinstance(exc, HTTPException):
        return "Запрос не принят адаптером TWM (без изменения мира): " + str(exc.detail)
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "Ошибка запроса: " + str(exc)
    # A real unexpected error should also be logged on the server, not masked as success.
    raise exc


def main():
    st.set_page_config(
        page_title="TWM — Лаборатория мира",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""
      <style>
      html, body, [data-testid="stApp"], .stApp {background:#081016!important;color:#e4eef2!important;}
      [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
      [data-testid="stSidebar"], footer, #MainMenu {display:none!important;}
      .block-container {max-width:100%!important;padding:0!important;margin:0!important;}
      [data-testid="stMainBlockContainer"] {max-width:100%!important;padding:0!important;}
      [data-testid="stVerticalBlock"] {gap:0!important;}
      [data-testid="stElementContainer"] {margin:0!important;width:100%!important;}
      [data-testid="stAppViewContainer"] {overflow-x:hidden!important;}
      </style>
    """, unsafe_allow_html=True)

    state = ensure_session()
    try:
        model = model_801()
    except Exception as exc:
        st.error(f"Не удалось загрузить настоящие веса Seed 801: {exc}")
        st.stop()

    bundle = ROOT / "ui" / "dist" / "twm.js"
    styles = ROOT / "ui" / "dist" / "twm.css"
    if not bundle.exists() or not styles.exists():
        st.error("Не найдена готовая React-сборка: ui/dist. Запустите `cd ui && npm ci && npm run build`.")
        st.stop()

    # File contents passed as trusted component source (never from user input).
    lab = st.components.v2.component(
        "twm_world_sandbox_v710",
        html='<div id="twm-sandbox-react-root"></div>',
        js=bundle.read_text(encoding="utf-8"),
        css=styles.read_text(encoding="utf-8"),
        isolate_styles=True,
    )
    data = {
        "revision": state["revision"],
        "presetIndex": state["preset_index"],
        "presets": [{"name": p["name"], "subtitle": p["subtitle"]} for p in PRESETS],
        "world_state": state["world_state"],
        "laws": state["laws"],
        "action": state["action"],
        "history": state["history"],
        "error": state["error"],
        "model": "TWM v7.10",
        "seed": 801,
    }
    result = lab(data=data, key="twm_lab_widget", on_command_change=lambda: None)
    if result.command:
        try:
            command = json.loads(result.command)
            handle_command(command, state, model)
        except Exception as exc:
            state["error"] = russian_error(exc)
            state["revision"] += 1
        st.rerun()


if __name__ == "__main__":
    main()
