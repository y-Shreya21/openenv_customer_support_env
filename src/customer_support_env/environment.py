from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import re
import random

from .models import SupportAction, SupportObservation, SupportReward, SupportState, StepResult, TaskId


@dataclass(frozen=True)
class TaskSpec:
    task_id: TaskId
    title: str
    instruction: str
    tickets: List[Dict[str, str]]
    expected_order: List[str]
    expected_category: Dict[str, str]
    expected_priority: Dict[str, int]
    expected_team: Dict[str, str]
    expected_escalate: Dict[str, bool]
    required_keywords: List[str]
    forbidden_keywords: List[str]
    max_steps: int


TASKS: Dict[TaskId, TaskSpec] = {
    TaskId.EASY: TaskSpec(
        task_id=TaskId.EASY,
        title="Login access triage",
        instruction=(
            "A user cannot log in after resetting their password. "
            "Classify the ticket, set the priority, route it to the correct team, and draft a short helpful reply."
        ),
        tickets=[{
            "ticket_id": "T-1001",
            "subject": "Cannot log in after password reset",
            "customer_message": "I reset my password twice, but the app still says invalid credentials. I need access today.",
        }],
        expected_order=["T-1001"],
        expected_category={"T-1001": "login_access"},
        expected_priority={"T-1001": 2},
        expected_team={"T-1001": "identity_support"},
        expected_escalate={"T-1001": False},
        required_keywords=["password", "reset", "help"],
        forbidden_keywords=["guarantee", "never"],
        max_steps=3,
    ),
    TaskId.MEDIUM: TaskSpec(
        task_id=TaskId.MEDIUM,
        title="Refund delay response drafting",
        instruction=(
            "A customer is waiting on a refund that has not posted yet. "
            "Write a reply that acknowledges the delay, explains the next step, and stays policy-safe."
        ),
        tickets=[{
            "ticket_id": "T-2001",
            "subject": "Refund still missing",
            "customer_message": "You said my refund would arrive by Friday, but I still do not see it in my bank account.",
        }],
        expected_order=["T-2001"],
        expected_category={"T-2001": "billing_refund"},
        expected_priority={"T-2001": 3},
        expected_team={"T-2001": "billing_ops"},
        expected_escalate={"T-2001": False},
        required_keywords=["sorry", "refund", "timeline", "check"],
        forbidden_keywords=["guaranteed", "instant", "done"],
        max_steps=3,
    ),
    TaskId.HARD: TaskSpec(
        task_id=TaskId.HARD,
        title="Queue planning with escalation",
        instruction=(
            "Three tickets arrived at once. Order them sensibly, assign the right teams, and mark escalation only where needed."
        ),
        tickets=[
            {
                "ticket_id": "T-3001",
                "subject": "Checkout outage for enterprise account",
                "customer_message": "Enterprise users cannot complete checkout right now; revenue is blocked.",
            },
            {
                "ticket_id": "T-3002",
                "subject": "Duplicate charge on card",
                "customer_message": "I was charged twice for the same order and need this fixed today.",
            },
            {
                "ticket_id": "T-3003",
                "subject": "Feature request for invoice export",
                "customer_message": "Can the dashboard export invoices as CSV?",
            },
        ],
        expected_order=["T-3001", "T-3002", "T-3003"],
        expected_category={
            "T-3001": "incident_outage",
            "T-3002": "billing_refund",
            "T-3003": "product_feedback",
        },
        expected_priority={"T-3001": 1, "T-3002": 2, "T-3003": 4},
        expected_team={"T-3001": "infra_oncall", "T-3002": "billing_ops", "T-3003": "product_ops"},
        expected_escalate={"T-3001": True, "T-3002": False, "T-3003": False},
        required_keywords=["outage", "refund", "feature"],
        forbidden_keywords=["ignore", "sometime"],
        max_steps=4,
    ),
}


class CustomerSupportEnv:
    def __init__(self, task_id: TaskId | str = TaskId.EASY, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)
        self._task_id = TaskId(task_id)
        self._spec = TASKS[self._task_id]
        self._state = self._new_state()

    def _new_state(self) -> SupportState:
        return SupportState(
            episode_id=str(uuid4()),
            task_id=self._spec.task_id,
            task_title=self._spec.title,
            step_count=0,
            max_steps=self._spec.max_steps,
            done=False,
            best_score=0.0,
            latest_score=0.0,
            last_feedback="Environment ready.",
            progress={
                "classification": 0.0,
                "priority": 0.0,
                "routing": 0.0,
                "reply_quality": 0.0,
                "ordering": 0.0,
                "escalation": 0.0,
            },
            history=[],
            current_tickets=self._tickets_as_state(),
        )

    def _tickets_as_state(self) -> List[Dict[str, str]]:
        variants = [
            "I reset my password twice but still can't log in",
            "Password reset not working, locked out",
            "Unable to login even after changing password",
        ]

        tickets = [
            {
                "ticket_id": t["ticket_id"],
                "subject": t["subject"],
                "customer_message": self._rng.choice(variants) if "password" in t["customer_message"].lower() else t["customer_message"],
            }
            for t in self._spec.tickets
        ]
        return tickets

    def reset(self, task_id: Optional[TaskId | str] = None, seed: Optional[int] = None) -> SupportObservation:
        if task_id is not None:
            self._task_id = TaskId(task_id)
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self._spec = TASKS[self._task_id]
        self._state = self._new_state()
        return self._observation("New episode started.")

    def state(self) -> SupportState:
        return self._state.model_copy(deep=True)

    def step(self, action: SupportAction) -> StepResult:
        if self._state.done:
            return StepResult(
                observation=self._observation("Episode already finished."),
                reward=SupportReward(value=0.0, components={}),
                done=True,
                info={"reason": "episode_done"},
            )

        if action.task_id != self._task_id:
            self._state.step_count += 1
            self._state.last_feedback = f"Action targeted {action.task_id}, but current task is {self._task_id}."
            self._state.history.append({"action": action.model_dump(), "score": 0.0, "note": "wrong_task"})
            done = self._maybe_finish()
            return self._result(0.0, done, {"error": "wrong_task"})

        step_type = action.step_type

        score, breakdown, feedback = self._grade(action)

        # optional: small reward shaping only
        step_bonus = 0.0
        if step_type == "classify":
            step_bonus = 0.02
        elif step_type == "route":
            step_bonus = 0.02
        elif step_type == "respond":
            step_bonus = 0.03
        # penalty for incorrect escalation
        for plan in action.plans:
            expected = self._spec.expected_escalate.get(plan.ticket_id, False)
            if plan.escalate and not expected:
                score -= 0.05

        shaped_reward = max(0.0, score - self._state.best_score) + step_bonus
        if action.finalize:
            shaped_reward += 0.1 * score
        shaped_reward = round(min(1.0, shaped_reward), 4)

        self._state.step_count += 1
        self._state.latest_score = round(score, 4)
        self._state.best_score = round(max(self._state.best_score, score), 4)
        self._state.last_feedback = feedback
        self._state.history.append({
            "action": action.model_dump(),
            "score": round(score, 4),
            "breakdown": breakdown,
        })

        done = action.finalize or self._state.step_count >= self._state.max_steps or score >= 0.999
        self._state.done = done
        info = {
            "task_score": round(self._state.best_score, 4),
            "component_scores": breakdown,
            "feedback": feedback,
        }
        return self._result(shaped_reward, done, info)

    def _maybe_finish(self) -> bool:
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True
        return self._state.done

    def _result(self, reward: float, done: bool, info: Dict[str, object]) -> StepResult:
        return StepResult(
            observation=self._observation(self._state.last_feedback),
            reward=SupportReward(value=reward, components=info.get("component_scores", {})),
            done=done,
            info=info,
        )

    def _observation(self, feedback: str) -> SupportObservation:
        return SupportObservation(
            task_id=self._spec.task_id,
            task_title=self._spec.title,
            instruction=self._spec.instruction,
            tickets=self._tickets_as_state(),
            remaining_steps=max(0, self._spec.max_steps - self._state.step_count),
            step_count=self._state.step_count,
            best_score=self._state.best_score,
            latest_score=self._state.latest_score,
            last_feedback=feedback,
            done=self._state.done,
        )

    def _grade(self, action: SupportAction) -> Tuple[float, Dict[str, float], str]:
        plans = action.plans
        total = 0.0
        breakdown: Dict[str, float] = {}
        expected_ids = [t["ticket_id"] for t in self._spec.tickets]

        coverage = 0.0
        if len(plans) == len(expected_ids):
            coverage = 0.1
        elif len(plans) == 1:
            coverage = 0.05
        breakdown["coverage"] = coverage
        total += coverage

        order_score = self._score_order(plans, expected_ids)
        breakdown["ordering"] = order_score
        total += order_score

        category_score = self._score_field(plans, self._spec.expected_category, "category", 0.22)
        priority_score = self._score_priority(plans, self._spec.expected_priority)
        team_score = self._score_field(plans, self._spec.expected_team, "owner_team", 0.18)
        escalate_score = self._score_bool(plans, self._spec.expected_escalate, "escalate", 0.12)
        reply_score = self._score_reply(plans)

        breakdown.update({
            "classification": round(category_score, 4),
            "priority": round(priority_score, 4),
            "routing": round(team_score, 4),
            "escalation": round(escalate_score, 4),
            "reply_quality": round(reply_score, 4),
        })
        total += category_score + priority_score + team_score + escalate_score + reply_score

        total = round(min(1.0, total), 4)
        feedback = self._make_feedback(total, breakdown, action)
        return total, breakdown, feedback

    def _score_order(self, plans: List, expected_ids: List[str]) -> float:
        if not plans:
            return 0.0
        actual = [p.ticket_id for p in plans]
        if len(actual) == 1:
            return 0.05 if actual[0] == expected_ids[0] else 0.0
        score = 0.0
        for idx, tid in enumerate(actual[: len(expected_ids)]):
            if idx < len(expected_ids) and tid == expected_ids[idx]:
                score += 0.08
        return round(min(0.24, score), 4)

    def _score_field(self, plans: List, expected: Dict[str, object], attr: str, max_score: float) -> float:
        if not plans:
            return 0.0
        per_ticket = max_score / max(1, len(expected))
        score = 0.0
        for plan in plans:
            truth = expected.get(plan.ticket_id)
            got = getattr(plan, attr)
            if truth is not None and got is not None and str(got).lower() == str(truth).lower():
                score += per_ticket
        return round(min(max_score, score), 4)

    def _score_priority(self, plans: List, expected: Dict[str, int]) -> float:
        if not plans:
            return 0.0
        score = 0.0
        per_ticket = 0.18 / max(1, len(expected))
        for plan in plans:
            truth = expected.get(plan.ticket_id)
            if truth is None or plan.priority is None:
                continue
            if plan.priority == truth:
                score += per_ticket
            elif abs(plan.priority - truth) == 1:
                score += per_ticket * 0.5
        return round(min(0.18, score), 4)

    def _score_bool(self, plans: List, expected: Dict[str, bool], attr: str, max_score: float) -> float:
        if not plans:
            return 0.0
        per_ticket = max_score / max(1, len(expected))
        score = 0.0
        for plan in plans:
            truth = expected.get(plan.ticket_id)
            got = getattr(plan, attr)
            if truth is not None and bool(got) == bool(truth):
                score += per_ticket
        return round(min(max_score, score), 4)

    def _score_reply(self, plans: List) -> float:
        if not plans:
            return 0.0
        score = 0.0
        text = " ".join([p.reply or "" for p in plans]).lower()
        for kw in self._spec.required_keywords:
            if kw.lower() in text:
                score += 0.08
        for bad in self._spec.forbidden_keywords:
            if bad.lower() in text:
                score -= 0.06
        if len(text.strip()) >= 40:
            score += 0.06
        return round(min(0.26, max(0.0, score)), 4)

    def _make_feedback(self, score: float, breakdown: Dict[str, float], action: SupportAction) -> str:
        if score >= 0.95:
            return "Strong solution. The ticket handling is close to production-ready."
        if score >= 0.7:
            return "Good progress. A few details still need tightening."
        if breakdown.get("reply_quality", 0.0) < 0.08:
            return "The reply is missing key support language or policy-safe wording."
        if breakdown.get("classification", 0.0) < 0.1:
            return "The ticket classification is off. Re-check the customer intent."
        if len(action.plans) != len(self._spec.tickets):
            return "The plan is incomplete for the current ticket set."
        return "Partial progress. Refine priority, routing, and response details."
