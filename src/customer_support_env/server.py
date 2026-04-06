from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .environment import CustomerSupportEnv
from .models import SupportAction, SupportObservation, SupportState, StepResult, TaskId

app = FastAPI(title="Customer Support OpenEnv", version="0.1.0")

_SESSIONS: Dict[str, CustomerSupportEnv] = {}


def _get_or_create_env(session_id: str) -> CustomerSupportEnv:
    env = _SESSIONS.get(session_id)
    if env is None:
        env = CustomerSupportEnv()
        _SESSIONS[session_id] = env
    return env


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    task_id: Optional[TaskId] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    action: SupportAction


class StateRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/reset", response_model=SupportObservation)
def reset(req: ResetRequest) -> SupportObservation:
    env = _get_or_create_env(req.session_id)
    return env.reset(task_id=req.task_id, seed=req.seed)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    env = _get_or_create_env(req.session_id)
    return env.step(req.action)


@app.post("/state", response_model=SupportState)
def state(req: StateRequest) -> SupportState:
    env = _get_or_create_env(req.session_id)
    return env.state()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    env = CustomerSupportEnv()
    await websocket.accept()
    await websocket.send_json({"type": "hello", "message": "customer_support_openenv"})
    try:
        while True:
            msg = await websocket.receive_json()
            kind = msg.get("type")
            if kind == "reset":
                obs = env.reset(task_id=msg.get("task_id"), seed=msg.get("seed"))
                await websocket.send_json({"type": "reset_result", "observation": obs.model_dump()})
            elif kind == "step":
                action = SupportAction.model_validate(msg.get("action", {}))
                res = env.step(action)
                await websocket.send_json({"type": "step_result", **res.model_dump()})
            elif kind == "state":
                await websocket.send_json({"type": "state_result", "state": env.state().model_dump()})
            else:
                await websocket.send_json({"type": "error", "message": f"unknown message type: {kind}"})
    except WebSocketDisconnect:
        return