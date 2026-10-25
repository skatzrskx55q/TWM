"""FastAPI entry point. Run `uvicorn sandbox_backend.main:app` from project root.

The archive's original chained checkpoint loader uses paths relative to CWD.
Do not silently substitute random weights when loading fails.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from twm_v710.load import load

from .schemas import StepRequest, StepResponse
from .service import run_step

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("checkpoints_v710/checkpoint_801.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if Path.cwd().resolve() != PROJECT_ROOT:
        raise RuntimeError(
            f"Run from TWM sandbox project root: cd {PROJECT_ROOT}; "
            "archived checkpoint metadata uses project-relative parent paths."
        )
    torch.set_num_threads(1)
    app.state.model = load(str(MODEL_PATH))  # weights_only=True in archived loader
    app.state.model.eval()
    yield
    del app.state.model


app = FastAPI(
    title="TWM World Sandbox / Seed 801",
    version="0.1.0",
    description="Real v7.10 neural joint-subset energy, with no reference-world inference.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health(request: Request):
    return {"status": "ready", "model": "TWM v7.10", "seed": 801,
            "checkpoint": str(MODEL_PATH), "backend": "torch-cpu"}


@app.post("/api/step", response_model=StepResponse)
def step(body: StepRequest, request: Request) -> StepResponse:
    return run_step(request.app.state.model, body)
