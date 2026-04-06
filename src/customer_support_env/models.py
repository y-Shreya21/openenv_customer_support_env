from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class TaskId(str, Enum):
    EASY = "easy_login_triage"
    MEDIUM = "medium_refund_reply"
    HARD = "hard_queue_planning"


class TicketPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(..., description="Ticket identifier")
    
    category: Optional[str] = Field(default=None, description="Support category")
    priority: Optional[int] = Field(default=None, ge=1, le=5)
    owner_team: Optional[str] = Field(default=None, description="Owning team")
    reply: Optional[str] = Field(default=None, description="Draft reply to the customer")
    escalate: bool = Field(default=False, description="Whether to escalate")
    eta_hours: Optional[int] = Field(default=None, ge=0, description="Expected handling time")
    status: Optional[str] = Field(default=None, description="Proposed ticket status")


class SupportAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: TaskId = Field(..., description="Which task this action targets")
    step_type: Literal["classify", "route", "respond"] = "respond"
    plans: List[TicketPlan] = Field(default_factory=list, description="One or more plans for the current tickets")
    finalize: bool = Field(default=False, description="Mark the episode complete")
    freeform_note: Optional[str] = Field(default=None, description="Optional extra note")


class SupportObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: TaskId
    task_title: str
    instruction: str
    tickets: List[Dict[str, str]]
    remaining_steps: int
    step_count: int
    best_score: float
    latest_score: float
    last_feedback: str
    done: bool


class SupportState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: TaskId
    task_title: str
    step_count: int
    max_steps: int
    done: bool
    best_score: float
    latest_score: float
    last_feedback: str
    progress: Dict[str, float]
    history: List[Dict[str, object]]
    current_tickets: List[Dict[str, str]]



class SupportReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)
